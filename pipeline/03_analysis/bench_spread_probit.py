"""Benchmark A — Estrella-Mishkin term-spread probit (full-sample diagnostics).

Fits P(recession_{t+h}) = Phi(a + b * spread_t) with a genuine statsmodels Probit
so the report can show the canonical inverted-curve -> recession pattern (b < 0,
significant). The out-of-sample scoring lives in backtest.py; this writes the
in-sample coefficient table to results_bench_spread.json.
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
from models import statsmodels_probit  # noqa: E402


def main(offline: bool = False) -> None:
    cfg = load_config()
    bench = pd.read_parquet(PROCESSED / "benchmarks_us.parquet")
    labels = pd.read_parquet(PROCESSED / "labels_us.parquet")["recession"]
    sp_col = cfg["us"]["benchmarks"]["term_spread"]
    spread = bench[sp_col]

    out = {"series": sp_col, "horizons": {}}
    for h in (cfg["eval"]["horizons"] + [12]):
        y = labels.shift(-h)
        fit = statsmodels_probit(spread, y)
        if fit is None:
            out["horizons"][str(h)] = {"status": "insufficient"}
            continue
        res, summ = fit
        df = pd.concat([spread.rename("s"), y.rename("y")], axis=1).dropna()
        p = res.predict(pd.DataFrame({"const": 1.0, "spread": df["s"]}))
        summ["in_sample_auc"] = M.auc(df["y"].to_numpy(), p.to_numpy())
        summ["inverted_curve_signal"] = bool(summ["spread_coef"] < 0)
        out["horizons"][str(h)] = summ
        print(f"  h={h}: spread coef={summ['spread_coef']:+.3f} "
              f"(p={summ['spread_pvalue']:.3g}) AUC={summ['in_sample_auc']:.3f}")

    write_json(PROCESSED / "results_bench_spread.json", out)


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
