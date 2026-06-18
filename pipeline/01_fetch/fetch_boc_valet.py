"""Fetch Bank of Canada yields/spreads via the Valet API (keyless JSON/CSV).

We pull the GoC benchmark yields needed to build a Canadian term spread, the
analogue of the US 10y-3m used in the term-spread probit. Cached under
data/raw/boc/<date>/ with last-good fallback.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fetch_text, http_session, load_config, snapshot_dir  # noqa: E402

VALET = "https://www.bankofcanada.ca/valet/observations/{series}/csv?start_date=1980-01-01"


def fetch_series(session, series: str, offline: bool) -> pd.Series | None:
    snap = snapshot_dir("boc")
    dest = snap / f"{series.replace('.', '_')}.csv"
    text = fetch_text(session, VALET.format(series=series), dest, offline=offline)
    if text is None:
        return None
    # Valet CSV has a metadata preamble; the OBSERVATIONS block starts after a
    # line beginning with "date".
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith('"date"')
                  or ln.lower().startswith("date")), None)
    if start is None:
        return None
    body = "\n".join(lines[start:])
    df = pd.read_csv(io.StringIO(body))
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    val_col = df.columns[1]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.dropna(subset=["date"]).set_index("date")[val_col]
    return s.rename(series)


def main(offline: bool = False) -> None:
    cfg = load_config()
    session = http_session()
    series = list(cfg["ca"]["boc_series"].values())
    n_ok = 0
    for sid in series:
        s = fetch_series(session, sid, offline)
        if s is not None and not s.dropna().empty:
            n_ok += 1
            print(f"  BoC {sid}: {s.dropna().shape[0]} obs")
        else:
            print(f"  [warn] BoC {sid}: unavailable")
    print(f"  BoC Valet: {n_ok}/{len(series)} series fetched")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
