"""Fetch FRED-MD: current.csv + dated monthly vintages (real-time discipline).

FRED-MD ships one CSV per month with retained vintages, the canonical real-time
US macro panel. The first data row encodes the per-series stationarity transform
(`tcode`); we keep it. Files live behind St. Louis Fed's CDN; some networks block
the bulk endpoint with an Akamai bot challenge (AccessDenied), in which case this
falls back to the last-good cache and the pipeline proceeds on the keyless
per-series panel (clearly labelled "final-data") built by fetch_fred_series.py.

Usage:
    python fetch_fredmd.py                 # current + a few recent vintages
    python fetch_fredmd.py --vintage 2026-05
    python fetch_fredmd.py --offline       # cache only
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fetch_text, http_session, snapshot_dir, write_json  # noqa: E402

BASE = "https://files.stlouisfed.org/files/htdocs/fred-md"
CURRENT = f"{BASE}/monthly/current.csv"
VINTAGE = BASE + "/monthly/{ym}.csv"


def _parse_fredmd(text: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (data indexed by month, tcode Series). Row 0 holds tcodes."""
    raw = pd.read_csv(io.StringIO(text))
    raw = raw.rename(columns={raw.columns[0]: "sasdate"})
    tcode = raw.iloc[0].drop(labels=["sasdate"])
    tcode = pd.to_numeric(tcode, errors="coerce").astype("Int64")
    data = raw.iloc[1:].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"], errors="coerce")
    data = data.dropna(subset=["sasdate"]).set_index("sasdate")
    data.index = data.index.to_period("M").to_timestamp()  # month start
    data = data.apply(pd.to_numeric, errors="coerce")
    return data, tcode


def fetch_vintage(session, ym: str | None, offline: bool) -> bool:
    url = CURRENT if ym is None else VINTAGE.format(ym=ym)
    name = "current" if ym is None else ym
    snap = snapshot_dir("fredmd")
    dest = snap / f"{name}.csv"
    text = fetch_text(session, url, dest, offline=offline)
    if text is None:
        print(f"  [warn] FRED-MD {name} unavailable (blocked or offline)")
        return False
    try:
        data, tcode = _parse_fredmd(text)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not parse FRED-MD {name}: {exc}")
        return False
    # persist tcodes once (they change rarely; freeze per fredmdchanges notes)
    tdest = snapshot_dir("fredmd").parent.parent / "manual" / "fredmd_tcodes.csv"
    if not tdest.exists():
        tcode.rename("tcode").to_csv(tdest, header=True)
    print(f"  FRED-MD {name}: {data.shape[0]} months x {data.shape[1]} series "
          f"({data.index.min():%Y-%m}..{data.index.max():%Y-%m})")
    write_json(snap / f"{name}_meta.json",
               {"vintage": name, "n_months": int(data.shape[0]),
                "n_series": int(data.shape[1])})
    return True


def main(offline: bool = False, vintage: str | None = None) -> None:
    session = http_session()
    ok = fetch_vintage(session, None, offline)  # current.csv
    if vintage:
        fetch_vintage(session, vintage, offline)
    if not ok:
        print("  [note] FRED-MD bulk CSV not reachable here; the pipeline uses the "
              "keyless per-series FRED panel as the (final-data) fallback.")


if __name__ == "__main__":
    vt = None
    if "--vintage" in sys.argv:
        vt = sys.argv[sys.argv.index("--vintage") + 1]
    main(offline="--offline" in sys.argv, vintage=vt)
