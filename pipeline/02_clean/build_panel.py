"""Build the analysis-ready monthly panel + labels + vintage store inputs.

Reads the fetched FRED monthly levels, applies the tcode stationarity transforms,
attaches the USREC label (US) and C.D. Howe label (Canada), and writes:

  data/interim/levels_us.parquet      raw levels (vintage-store backend)
  data/processed/panel_us.parquet     stationary transformed panel
  data/processed/labels_us.parquet    USREC monthly 0/1
  data/processed/labels_ca.parquet    C.D. Howe monthly 0/1 (on US panel index)
  data/processed/panel_meta.json      coverage + transform summary

Guards: no level/growth mixing (tcodes from config), no NA in labels over the
covered span.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import (INTERIM, MANUAL, PROCESSED, RAW, load_config,  # noqa: E402
                    to_month_start, write_json)
from panel import (cdhowe_labels, transform_panel, usrec_labels)  # noqa: E402


def load_fred_levels() -> pd.DataFrame:
    """Prefer the latest dated snapshot; fall back to the stable copy."""
    cand = sorted((RAW / "fred_api").glob("*/fred_monthly.csv"))
    stable = RAW / "fred_api" / "fred_monthly_latest.csv"
    path = cand[-1] if cand else (stable if stable.exists() else None)
    if path is None:
        raise SystemExit("No fetched FRED panel found; run fetch_fred_series.py first.")
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index = to_month_start(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"  loaded levels {df.shape} from {path.parent.name}/{path.name}")
    return df


def main(offline: bool = False) -> None:
    cfg = load_config()
    levels = load_fred_levels()

    label_id = cfg["us"]["label_series"]
    bench_ids = list(cfg["us"]["benchmarks"].values())
    panel_specs = cfg["us"]["panel"]
    panel_ids = [d["id"] for d in panel_specs]

    # raw levels for the vintage store: panel series + benchmark inputs.
    keep = [c for c in panel_ids + bench_ids if c in levels.columns]
    levels_store = levels[keep].copy()
    levels_store.to_parquet(INTERIM / "levels_us.parquet")

    # stationary transformed panel
    panel = transform_panel(levels, panel_specs)
    panel = panel.dropna(how="all")
    panel.to_parquet(PROCESSED / "panel_us.parquet")

    # labels
    labels = usrec_labels(levels, label_id)
    labels = labels.reindex(panel.index).ffill().fillna(0).astype(int)
    if labels.reindex(panel.index).isna().any():
        raise SystemExit("NA in US labels over panel span.")
    labels.to_frame().to_parquet(PROCESSED / "labels_us.parquet")

    ca = cdhowe_labels(MANUAL / "cdhowe_recessions.csv", panel.index)
    ca.to_frame().to_parquet(PROCESSED / "labels_ca.parquet")

    # benchmark inputs (term spreads, sahm, fed prob) kept raw on month index
    bench = levels[[c for c in bench_ids if c in levels.columns]].copy()
    bench.to_parquet(PROCESSED / "benchmarks_us.parquet")

    cov = {
        "n_months": int(panel.shape[0]),
        "n_series": int(panel.shape[1]),
        "panel_start": panel.index.min().strftime("%Y-%m"),
        "panel_end": panel.index.max().strftime("%Y-%m"),
        "series": panel_ids,
        "us_recession_months": int(labels.sum()),
        "ca_recession_months": int(ca.sum()),
        "transforms": {d["id"]: int(d["tcode"]) for d in panel_specs},
    }
    write_json(PROCESSED / "panel_meta.json", cov)
    print(f"  panel {panel.shape}, US recession months {labels.sum()}, "
          f"CA recession months {ca.sum()}")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
