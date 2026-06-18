"""Pseudo-real-time, expanding-window backtest — the credibility core.

For each evaluation month tau we re-fit every model using ONLY data visible at tau
(via the VintageStore), then predict P(recession in tau+h). No future information
is used; tests/test_no_lookahead.py asserts this. We score Brier / log-loss / AUC /
PR-AUC + calibration (reliability, ECE) against the term-spread probit and Sahm
benchmarks, attach stationary-block-bootstrap CIs, and run Diebold-Mariano /
Giacomini-White predictive-ability tests.

We run three configurations to support the vintage-vs-final refutation and the
second market:
  * US real-time  (publication-lag store)            -> headline
  * US final-data (no lag; more recent data visible)  -> upper bound on skill
  * Canada real-time (C.D. Howe labels)               -> second market

Writes data/processed/backtest_<country>_<mode>.json and the per-model probability
paths consumed by the dashboard.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import INTERIM, PROCESSED, load_config, read_json, write_json  # noqa: E402
import metrics as M  # noqa: E402
from models import (FactorProbit, GbmFactors, MidasLogit, SahmLogit,  # noqa: E402
                    SpreadProbit, pca_em_factors)
from panel import VintageStore, transform_panel  # noqa: E402

COVID = ("2020-02", "2020-08")


# --- feature construction --------------------------------------------------
def build_feature_sets(store: VintageStore, tau, cfg, k):
    """Return {model_name: X_df} of features for months up to tau, plus factors."""
    bm = cfg["us"]["benchmarks"]
    raw = store.as_of(tau)                       # visible raw levels
    trans = transform_panel(raw, store.specs)     # stationary panel as-of tau
    panel_ids = [d["id"] for d in store.specs if d["id"] in trans.columns]
    trans = trans[panel_ids].dropna(how="all")

    factors, *_ = pca_em_factors(trans, k)

    feats = {}
    # benchmark A — Estrella-Mishkin term spread. The curve LEADS by ~12 months,
    # so the predictor for recession at t is the spread 12 months earlier (an
    # inverted curve a year ago -> recession now). Using spread_{t-12} is both the
    # canonical specification and look-ahead-safe (it only uses older data).
    sp_col = bm["term_spread"]
    if sp_col in raw.columns:
        feats["spread_probit"] = raw[[sp_col]].shift(12).rename(columns={sp_col: "spread"})
    else:
        feats["spread_probit"] = pd.DataFrame(index=raw.index)
    # benchmark B — real-time Sahm value
    sa_col = bm["sahm_realtime"]
    feats["sahm"] = raw[[sa_col]].rename(columns={sa_col: "sahm"}) \
        if sa_col in raw.columns else pd.DataFrame(index=raw.index)
    # primary — dynamic factors
    feats["dfm"] = factors
    # penalised MIDAS — indicators at lags 0,1,2
    lags = [trans.shift(L).add_suffix(f"_l{L}") for L in (0, 1, 2)]
    feats["midas"] = pd.concat(lags, axis=1)
    # GBM — factors + lagged spread + sahm
    gbm = factors.copy()
    if sp_col in raw.columns:
        gbm["spread_l12"] = raw[sp_col].shift(12)
    if sa_col in raw.columns:
        gbm["sahm"] = raw[sa_col]
    feats["gbm"] = gbm
    return feats, factors


def make_models(seed):
    return {
        "spread_probit": SpreadProbit(C=1.0),
        "sahm": SahmLogit(C=1.0),
        "dfm": FactorProbit(C=1.0),
        "midas": MidasLogit(C=0.3),
        "gbm": GbmFactors(seed=seed),
    }


def run_one(store: VintageStore, labels: pd.Series, cfg, h: int,
            mode: str, country: str):
    seed = cfg["seed"]
    k = cfg["factors"]["n_factors"]
    start = pd.Timestamp(cfg["eval"]["start"] + "-01")
    min_train = cfg["eval"]["min_train_months"]
    refit_every = 1 if country == "us" else 1

    months = store.levels.index
    months = months[(months >= start) & (months <= months.max() - pd.DateOffset(months=h))]

    preds = {m: [] for m in make_models(seed)}
    asof_used, target_used = [], []
    max_release_log = []
    t0 = time.time()
    cached_feats = None
    for i, tau in enumerate(months):
        target = tau + pd.DateOffset(months=h)
        if (i % refit_every == 0) or cached_feats is None:
            feats, _ = build_feature_sets(store, tau, cfg, k)
            cached_feats = feats
        feats = cached_feats
        # y_target at month t = recession_{t+h}
        y_shift = labels.shift(-h)
        models = make_models(seed)
        ok = False
        for name, mdl in models.items():
            X = feats[name]
            if X.empty:
                preds[name].append(np.nan)
                continue
            # train on months with known target and target month <= tau
            train_idx = X.index[(X.index <= tau) &
                                (X.index + pd.DateOffset(months=h) <= tau)]
            ytr = y_shift.reindex(train_idx)
            mdl.fit(X.loc[train_idx], ytr)
            # Ragged edge: at tau the latest months may not be released yet, so
            # carry forward the last available value per series (real-time-correct)
            # before selecting the prediction row. Without this the tau row is all
            # NaN and standardisation maps it to a garbage point.
            Xf = X.ffill()
            pred_row = Xf.loc[[tau]] if tau in Xf.index else Xf.iloc[[-1]]
            p = float(mdl.predict_proba(pred_row)[0])
            preds[name].append(p)
            ok = True
        if ok:
            asof_used.append(tau)
            target_used.append(target)
            max_release_log.append((tau, store.max_release_date(tau)))
    dt = time.time() - t0
    print(f"  [{country}/{mode}/h{h}] {len(months)} months in {dt:.1f}s")

    # assemble prediction frame
    df = pd.DataFrame({m: preds[m] for m in preds}, index=pd.DatetimeIndex(asof_used))
    df["target_month"] = pd.DatetimeIndex(target_used)
    df["y"] = labels.reindex(df["target_month"]).to_numpy()
    df = df.dropna(subset=["y"])
    return df, max_release_log


def score(df: pd.DataFrame, cfg, model_names) -> dict:
    nb = cfg["bootstrap"]["n_boot"]
    bl = cfg["bootstrap"]["block_len"]
    seed = cfg["seed"]
    y = df["y"].to_numpy(int)
    covid_mask = ((df["target_month"] >= pd.Timestamp(COVID[0] + "-01")) &
                  (df["target_month"] <= pd.Timestamp(COVID[1] + "-01"))).to_numpy()
    out = {"models": {}, "n_obs": int(len(df)),
           "eval_start": df["target_month"].min().strftime("%Y-%m"),
           "eval_end": df["target_month"].max().strftime("%Y-%m"),
           "base_rate": float(y.mean())}
    bench = df["spread_probit"].to_numpy()
    for name in model_names:
        p = df[name].to_numpy()
        m_all = M.all_metrics(y, p)
        m_excovid = M.all_metrics(y[~covid_mask], p[~covid_mask])
        ci = {met: M.bootstrap_ci(y, p, getattr(M, met), nb, bl, seed)
              for met in ("brier", "log_loss", "auc")}
        centers, obs, pred, cnt = M.reliability(y, p)
        rec = {"metrics": m_all, "metrics_ex_covid": m_excovid, "ci": ci,
               "reliability": {"centers": centers, "obs": obs,
                               "pred": pred, "count": cnt}}
        if name != "spread_probit":
            rec["dm_vs_spread"] = M.diebold_mariano(y, p, bench, "brier")
            rec["gw_vs_spread"] = M.giacomini_white(y, p, bench, "brier")
            sahm = df["sahm"].to_numpy()
            rec["dm_vs_sahm"] = M.diebold_mariano(y, p, sahm, "brier")
        out["models"][name] = rec
    return out


def main(offline: bool = False) -> None:
    cfg = load_config()
    levels = pd.read_parquet(INTERIM / "levels_us.parquet")
    labels_us = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    labels_ca = pd.read_parquet(PROCESSED / "labels_ca.parquet")["recession"]
    specs = cfg["us"]["panel"]
    h = cfg["eval"]["horizons"][0]  # headline nowcast horizon
    model_names = list(make_models(cfg["seed"]).keys())

    data_mode = "live"
    dm_path = PIPE.parent / "data" / "facts" / "data_mode.json"
    if dm_path.exists():
        data_mode = read_json(dm_path).get("data_mode", "live")

    configs = [
        ("us", "realtime", labels_us, False),
        ("us", "final", labels_us, True),
        ("ca", "realtime", labels_ca, False),
    ]
    for country, mode, labels, final in configs:
        store = VintageStore.from_levels(levels, specs)
        if final:
            store.lags = {c: 0 for c in store.levels.columns}  # no publication lag
        df, rel_log = run_one(store, labels, cfg, h, mode, country)
        results = score(df, cfg, model_names)
        results.update({"country": country, "mode": mode, "horizon": h,
                        "data_mode": data_mode, "seed": cfg["seed"]})
        # persist probability paths for the dashboard (real-time US headline)
        paths = {"asof": [d.strftime("%Y-%m") for d in df.index],
                 "target": [d.strftime("%Y-%m") for d in df["target_month"]],
                 "y": df["y"].astype(int).tolist()}
        for name in model_names:
            paths[name] = [round(float(v), 5) for v in df[name].to_numpy()]
        results["paths"] = paths
        write_json(PROCESSED / f"backtest_{country}_{mode}.json", results)
        # headline summary line
        dfm = results["models"]["dfm"]["metrics"]
        sp = results["models"]["spread_probit"]["metrics"]
        print(f"    {country}/{mode}: DFM Brier={dfm['brier']:.3f} AUC={dfm['auc']:.3f}"
              f" | spread Brier={sp['brier']:.3f} AUC={sp['auc']:.3f}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
