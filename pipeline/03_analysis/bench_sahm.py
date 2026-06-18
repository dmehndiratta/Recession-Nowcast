"""Benchmark B — real-time Sahm rule (full-sample diagnostics).

The Sahm rule triggers when the 3-month-average unemployment rate rises >= 0.5pp
above its trailing-12-month minimum. We report the trigger's contemporaneous
classification of NBER recession months and map the Sahm value to a probability
(logistic) for a like-for-like comparison in the backtest.
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


def main(offline: bool = False) -> None:
    cfg = load_config()
    bench = pd.read_parquet(PROCESSED / "benchmarks_us.parquet")
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    sa_col = cfg["us"]["benchmarks"]["sahm_realtime"]
    sahm = bench[sa_col]

    df = pd.concat([sahm.rename("sahm"), labels.rename("y")], axis=1).dropna()
    trigger = (df["sahm"] >= 0.5).astype(int)
    y = df["y"].to_numpy()
    tp = int(((trigger == 1) & (df["y"] == 1)).sum())
    fp = int(((trigger == 1) & (df["y"] == 0)).sum())
    fn = int(((trigger == 1).eq(False) & (df["y"] == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {
        "series": sa_col,
        "threshold": 0.5,
        "trigger_precision": float(prec),
        "trigger_recall": float(rec),
        "trigger_auc": M.auc(y, df["sahm"].to_numpy()),
        "n": int(len(df)),
        "recession_months": int(df["y"].sum()),
    }
    print(f"  Sahm trigger: precision={prec:.2f} recall={rec:.2f} "
          f"AUC={out['trigger_auc']:.3f}")
    write_json(PROCESSED / "results_bench_sahm.json", out)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
