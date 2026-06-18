"""Predictive uncertainty bands for the headline (DFM) probability.

Per the brief, the deliverable is a probability *band*, not a point. We combine two
sources of uncertainty:

  (i) parameter uncertainty — stationary block-bootstrap the training months and
      refit the factor->recession probit, giving a distribution of coefficients;
  (ii) factor-state uncertainty — Monte-Carlo draw the smoothed factors from their
      Kalman-smoother standard errors (from dfm_statespace.py).

Each Monte-Carlo draw combines a bootstrapped parameter vector with a perturbed
factor path; the band is the [5%, 95%] quantile envelope of the resulting
probabilities. Writes results_uncertainty.json (recent path band + current gauge).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import PROCESSED, load_config, write_json  # noqa: E402
from metrics import stationary_bootstrap_indices  # noqa: E402
from models import FactorProbit  # noqa: E402


def main(offline: bool = False) -> None:
    cfg = load_config()
    seed = cfg["seed"]
    n_mc = cfg["uncertainty"]["n_mc"]
    lo_q, hi_q = cfg["uncertainty"]["band"]
    block = cfg["bootstrap"]["block_len"]

    fac = pd.read_parquet(PROCESSED / "dfm_factors.parquet")
    fstd = pd.read_parquet(PROCESSED / "dfm_factor_std.parquet").reindex_like(fac)
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    y = labels.reindex(fac.index)

    train = fac.dropna()
    ytr = y.reindex(train.index)
    rng = np.random.default_rng(seed)

    # point estimate
    base = FactorProbit(C=1.0).fit(train, ytr)
    p_point = base.predict_proba(fac)

    draws = np.empty((n_mc, len(fac)))
    n = len(train)
    for d in range(n_mc):
        idx = stationary_bootstrap_indices(n, block, rng)          # param uncertainty
        Xb = train.iloc[idx]
        yb = ytr.iloc[idx]
        mdl = FactorProbit(C=1.0).fit(Xb, yb)
        noise = rng.standard_normal(fac.shape) * fstd.to_numpy()    # state uncertainty
        fac_draw = fac + noise
        draws[d] = mdl.predict_proba(fac_draw)

    med = np.nanmedian(draws, axis=0)
    lo = np.nanquantile(draws, lo_q, axis=0)
    hi = np.nanquantile(draws, hi_q, axis=0)

    months = [d.strftime("%Y-%m") for d in fac.index]
    recent = slice(max(0, len(months) - 180), len(months))
    out = {
        "band": [lo_q, hi_q],
        "n_mc": int(n_mc),
        "month": months[recent],
        "point": [round(float(v), 5) for v in p_point[recent]],
        "median": [round(float(v), 5) for v in med[recent]],
        "lo": [round(float(v), 5) for v in lo[recent]],
        "hi": [round(float(v), 5) for v in hi[recent]],
        "current": {
            "month": months[-1],
            "point": round(float(p_point[-1]), 5),
            "lo": round(float(lo[-1]), 5),
            "hi": round(float(hi[-1]), 5),
            "median": round(float(med[-1]), 5),
        },
        "mean_band_width": round(float(np.mean(hi - lo)), 5),
    }
    print(f"  uncertainty: current P={out['current']['point']:.3f} "
          f"[{out['current']['lo']:.3f}, {out['current']['hi']:.3f}], "
          f"mean width={out['mean_band_width']:.3f}")
    write_json(PROCESSED / "results_uncertainty.json", out)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
