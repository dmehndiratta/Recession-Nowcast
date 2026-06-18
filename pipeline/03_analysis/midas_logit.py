"""Penalised MIDAS logistic (full-sample diagnostics).

Elastic-net logistic regression mapping the stationary indicators at several
monthly lags (a discrete MIDAS-style polynomial-free lag structure) to the
recession label. Reports the non-zero selected features so the report can show
which indicators carry the signal. OOS scoring is in backtest.py.
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
from models import MidasLogit  # noqa: E402


def main(offline: bool = False) -> None:
    cfg = load_config()
    panel = pd.read_parquet(PROCESSED / "panel_us.parquet").dropna(how="all")
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]

    lags = [panel.shift(L).add_suffix(f"_l{L}") for L in (0, 1, 2)]
    X = pd.concat(lags, axis=1)
    mdl = MidasLogit(C=0.3).fit(X, labels)
    p = mdl.predict_proba(X)
    y = labels.reindex(X.index).to_numpy()

    selected = {}
    if mdl.clf is not None:
        coefs = mdl.clf.coef_.ravel()
        for name, c in zip(mdl.cols, coefs):
            if abs(c) > 1e-4:
                selected[name] = round(float(c), 4)
        selected = dict(sorted(selected.items(), key=lambda kv: -abs(kv[1]))[:15])

    out = {
        "n_features": int(X.shape[1]),
        "n_selected": len(selected),
        "top_selected": selected,
        "in_sample_auc": float(M.auc(y, p)),
        "in_sample_brier": float(M.brier(y, p)),
    }
    print(f"  MIDAS: {out['n_selected']}/{out['n_features']} features selected, "
          f"in-sample AUC={out['in_sample_auc']:.3f}")
    write_json(PROCESSED / "results_midas.json", out)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
