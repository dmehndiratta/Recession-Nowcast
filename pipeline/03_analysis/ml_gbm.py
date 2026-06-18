"""Nonlinear comparator (full-sample diagnostics) — gradient-boosted trees.

XGBoost on the dynamic factors (+ spread, Sahm) to show ML vs econometrics
honestly. Reports feature importances. All out-of-sample evaluation is in
backtest.py; this fit is in-sample and for interpretation only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import PROCESSED, load_config, write_json  # noqa: E402
import metrics as M  # noqa: E402
from models import GbmFactors, pca_em_factors  # noqa: E402


def main(offline: bool = False) -> None:
    cfg = load_config()
    panel = pd.read_parquet(PROCESSED / "panel_us.parquet").dropna(how="all")
    bench = pd.read_parquet(PROCESSED / "benchmarks_us.parquet")
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    k = cfg["factors"]["n_factors"]

    fac, *_ = pca_em_factors(panel, k)
    X = fac.copy()
    X["spread"] = bench[cfg["us"]["benchmarks"]["term_spread"]].reindex(X.index)
    X["sahm"] = bench[cfg["us"]["benchmarks"]["sahm_realtime"]].reindex(X.index)

    mdl = GbmFactors(seed=cfg["seed"]).fit(X, labels)
    p = mdl.predict_proba(X)
    y = labels.reindex(X.index).to_numpy()

    imp = {}
    if mdl.const_p is None:
        for name, v in zip(mdl.cols, mdl.model.feature_importances_):
            imp[name] = round(float(v), 4)
        imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    out = {
        "estimator": "XGBClassifier",
        "feature_importance": imp,
        "in_sample_auc": float(M.auc(y, p)),
        "in_sample_brier": float(M.brier(y, p)),
        "note": "In-sample only; OOS skill is reported by backtest.py.",
    }
    print(f"  GBM: in-sample AUC={out['in_sample_auc']:.3f} "
          f"(in-sample; OOS in backtest)")
    write_json(PROCESSED / "results_gbm.json", out)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
