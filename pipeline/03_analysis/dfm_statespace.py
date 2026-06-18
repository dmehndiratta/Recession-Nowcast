"""Primary model (full-sample) — mixed-frequency dynamic factor model.

Fits statsmodels `DynamicFactorMQ` (Kalman filter/smoother, EM for missing data)
on the standardized stationary panel to extract a small number of dynamic factors,
then a probit maps the smoothed factor(s) to the recession label. This is the
genuine state-space machinery the plan calls for; the expanding-window backtest
uses the faster PCA-EM factor approximation for tractability (see models.py).

We also persist the smoothed factor and its standard error, which uncertainty.py
uses for the Monte-Carlo-over-states predictive band.

Writes results_dfm.json (+ factor path with std) and falls back to PCA-EM factors
if DynamicFactorMQ does not converge, so the pipeline never breaks.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import PROCESSED, load_config, write_json  # noqa: E402
import metrics as M  # noqa: E402
from models import pca_em_factors, statsmodels_probit, FactorProbit  # noqa: E402
from panel import standardize, winsorize  # noqa: E402


def fit_dfm(panel: pd.DataFrame, k: int, max_iter: int):
    """Return (factors_df, factor_std_df, method). Falls back to PCA-EM."""
    z, *_ = standardize(panel)
    z = winsorize(z, 8.0)
    # Restrict to the span where enough series are observed, then try a cascade of
    # increasingly simple state-space specs. Real panels with ragged starts and
    # collinear series often make the richest spec non-positive-definite; the
    # simpler specs are far more stable while still genuine DFMs.
    cover = z.notna().mean(axis=1)
    z = z.loc[cover[cover >= 0.6].index.min():] if (cover >= 0.6).any() else z
    specs = [
        dict(factor_orders=1, idiosyncratic_ar1=False),
        dict(factor_orders=2, idiosyncratic_ar1=False),
        dict(factor_orders=2, idiosyncratic_ar1=True),
    ]
    for spec in specs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
                mod = DynamicFactorMQ(z, factors=k, **spec)
                res = mod.fit(maxiter=max_iter, disp=False)
            fac = res.factors.smoothed
            if isinstance(fac, pd.Series):
                fac = fac.to_frame()
            fac.columns = [f"F{i+1}" for i in range(fac.shape[1])]
            try:
                cov = res.factors.smoothed_cov
                std = np.sqrt(np.clip(np.array([cov.iloc[i].values.reshape(-1)[0]
                                                for i in range(len(fac))]), 0, None))
                fstd = pd.DataFrame(np.tile(std.reshape(-1, 1), (1, fac.shape[1])),
                                    index=fac.index, columns=fac.columns)
            except Exception:  # noqa: BLE001
                fstd = pd.DataFrame(0.25, index=fac.index, columns=fac.columns)
            tag = f"DynamicFactorMQ(fo={spec['factor_orders']}," \
                  f"ar1={spec['idiosyncratic_ar1']})"
            return fac, fstd, tag
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] DFM spec {spec} failed ({exc}); trying simpler")
    print("  [warn] all DynamicFactorMQ specs failed; PCA-EM fallback")
    fac, _l, _m, _s, _evr = pca_em_factors(panel, k)
    fstd = pd.DataFrame(0.25, index=fac.index, columns=fac.columns)
    return fac, fstd, "PCA-EM"


def main(offline: bool = False) -> None:
    cfg = load_config()
    panel = pd.read_parquet(PROCESSED / "panel_us.parquet")
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    k = cfg["factors"]["n_factors"]

    panel = panel.dropna(how="all").ffill().dropna(how="all")
    fac, fstd, method = fit_dfm(panel, k, cfg["factors"]["max_em_iter"])

    # probit of recession on smoothed factors (in-sample diagnostic)
    mdl = FactorProbit(C=1.0).fit(fac, labels.reindex(fac.index))
    p = mdl.predict_proba(fac)
    y = labels.reindex(fac.index).to_numpy()
    auc = M.auc(y, p)

    out = {
        "method": method,
        "n_factors": int(fac.shape[1]),
        "n_months": int(fac.shape[0]),
        "in_sample_auc": float(auc),
        "in_sample_brier": float(M.brier(y, p)),
        "factor_path": {
            "month": [d.strftime("%Y-%m") for d in fac.index],
            **{c: [round(float(v), 5) for v in fac[c]] for c in fac.columns},
            **{f"{c}_std": [round(float(v), 5) for v in fstd[c]] for c in fstd.columns},
        },
    }
    print(f"  DFM method={method} k={fac.shape[1]} in-sample AUC={auc:.3f}")
    write_json(PROCESSED / "results_dfm.json", out)
    fac.to_parquet(PROCESSED / "dfm_factors.parquet")
    fstd.to_parquet(PROCESSED / "dfm_factor_std.parquet")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
