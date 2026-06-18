"""Deterministic SYNTHETIC fallback panel (used only when live fetch is blocked).

This is NOT real data. It exists so the full pipeline — vintage store, pseudo-
real-time backtest, calibration, uncertainty bands, dashboard — runs and is
testable in environments where St. Louis Fed's endpoints are unreachable. It uses
the *real* FRED series IDs and the *real* NBER recession months (public historical
fact), then generates macro series with an embedded recession signal so the models
have something genuine to learn. Every artefact it feeds is stamped
`data_mode = "demo-synthetic"`; the dashboard and report show a prominent banner.

Run the live fetchers (fetch_fred_series.py) wherever FRED is reachable to replace
this with genuine data — no downstream code changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import RAW, load_config, snapshot_dir, write_json  # noqa: E402

# Real NBER recession months since 1959 (peak -> trough inclusive).
NBER = [
    ("1960-04", "1961-02"), ("1969-12", "1970-11"), ("1973-11", "1975-03"),
    ("1980-01", "1980-07"), ("1981-07", "1982-11"), ("1990-07", "1991-03"),
    ("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04"),
]
START, END = "1959-01", "2025-12"


def usrec_series(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(0, index=index, dtype=int)
    for peak, trough in NBER:
        s[(index >= pd.Timestamp(peak + "-01")) & (index <= pd.Timestamp(trough + "-01"))] = 1
    return s


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def build(seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="MS")
    n = len(idx)
    rec = usrec_series(idx).to_numpy(float)

    # Latent monthly growth-pressure factor: negative in/around recessions.
    pressure = _smooth(rec, 5)
    f = -pressure + 0.35 * rng.standard_normal(n)
    f = _smooth(f, 3)

    cfg = load_config()
    cols: dict[str, np.ndarray] = {}

    def level_from_dlog(mu, beta, vol):
        dlog = mu + beta * f + vol * rng.standard_normal(n)
        return 100.0 * np.exp(np.cumsum(dlog))

    for spec in cfg["us"]["panel"]:
        sid, tcode, grp = spec["id"], int(spec["tcode"]), spec.get("group")
        if grp == "labour" and sid == "UNRATE":
            # unemployment: rises (positive dx) when pressure high
            dx = -0.02 + 0.9 * pressure + 0.15 * rng.standard_normal(n)
            cols[sid] = 4.0 + np.cumsum(dx) - np.cumsum(dx).min() + 3.0
        elif tcode in (5, 6):           # growth-rate series
            cols[sid] = level_from_dlog(0.0025, 0.012, 0.006)
        elif tcode == 2:                # differenced level (rates, util)
            base = 5.0 if grp == "rates" else 80.0
            dx = -0.4 * f + 0.2 * rng.standard_normal(n)
            cols[sid] = base + np.cumsum(dx) * (0.1 if grp == "rates" else 0.3)
        elif tcode == 4:                # log-level (housing): falls in recession
            log_lvl = 7.0 + 1.2 * _smooth(-pressure, 6) + 0.04 * np.cumsum(rng.standard_normal(n)) * 0.02
            cols[sid] = np.exp(log_lvl)
        else:                           # level (hours, sentiment)
            cols[sid] = 40.0 + 5.0 * f + 1.0 * rng.standard_normal(n)

    # benchmarks ------------------------------------------------------------
    bm = cfg["us"]["benchmarks"]
    # Term spread inverts ~12 months before recessions -> probit predictive power,
    # but with a realistic signal-to-noise (autocorrelated noise + a slow level
    # cycle) so the lag-12 benchmark is strong yet beatable, not deterministic.
    future_rec = pd.Series(rec, index=idx).shift(-12).fillna(0).to_numpy()
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.8 * noise[t - 1] + rng.standard_normal()  # AR(1) noise
    cycle = 0.6 * np.sin(np.arange(n) / 40.0)                   # slow level drift
    spread = 1.6 - 2.0 * _smooth(future_rec, 6) + 0.9 * noise + cycle
    cols[bm["term_spread"]] = spread
    cols[bm["term_spread_alt"]] = spread * 0.7 + 0.3 + 0.2 * rng.standard_normal(n)
    # Real-time Sahm: 3m-avg unemployment minus trailing-12m min.
    u = pd.Series(cols["UNRATE"], index=idx)
    sahm = (u.rolling(3).mean() - u.rolling(12).min()).clip(lower=0).fillna(0)
    cols[bm["sahm_realtime"]] = sahm.to_numpy()
    # Fed smoothed probability (0-100), reference line.
    prob = 100.0 / (1.0 + np.exp(-(6.0 * pressure - 2.0)))
    cols[bm["fed_smoothed_prob"]] = _smooth(prob, 3)

    df = pd.DataFrame(cols, index=idx)
    df["USREC"] = usrec_series(idx).to_numpy()
    df.index.name = "date"
    return df, usrec_series(idx)


def main(seed: int | None = None) -> None:
    cfg = load_config()
    seed = seed or cfg["seed"]
    df, _ = build(seed)
    snap = snapshot_dir("fred_api")
    df.to_csv(snap / "fred_monthly.csv")
    df.to_csv(RAW / "fred_api" / "fred_monthly_latest.csv")
    write_json(PIPE.parent / "data" / "facts" / "data_mode.json",
               {"data_mode": "demo-synthetic",
                "reason": "Live FRED endpoints unreachable in this environment; "
                          "synthetic panel with real NBER dates and embedded signal.",
                "seed": seed, "n_months": int(df.shape[0]),
                "span": [df.index.min().strftime("%Y-%m"),
                         df.index.max().strftime("%Y-%m")]})
    print(f"  SYNTHETIC demo panel {df.shape} written ({df.index.min():%Y-%m}.."
          f"{df.index.max():%Y-%m}); data_mode=demo-synthetic")


if __name__ == "__main__":
    main()
