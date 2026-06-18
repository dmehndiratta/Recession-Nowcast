"""Fetch individual FRED series (keyless CSV, or FRED API if FRED_API_KEY set).

Covers: USREC label, term spreads, real-time Sahm, the Fed smoothed probability,
and every monthly panel series listed in config.yaml `us.panel`. Idempotent: each
series cached under data/raw/fred_api/<date>/<id>.csv with last-good fallback.

The keyless graph endpoint (fredgraph.csv) returns the *latest revised* series.
This is the "final-data" path used for the pseudo-real-time backtest where genuine
FRED-MD vintages are unavailable; the vintage-vs-final refutation quantifies the
difference. With FRED_API_KEY set, ALFRED vintages can be layered in later.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (SITE_DATA, fetch_text, http_session, load_config,  # noqa: E402
                    snapshot_dir, to_month_start, write_json)

GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
API = ("https://api.stlouisfed.org/fred/series/observations"
       "?series_id={sid}&api_key={key}&file_type=json")


def _parse_two_col(text: str, sid: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"][: len(df.columns)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")["value"]
    return df.rename(sid)


def fetch_one(session, sid: str, snap: Path, offline: bool) -> pd.Series | None:
    dest = snap / f"{sid}.csv"
    # Idempotent: reuse today's cached snapshot (avoids re-hitting any host).
    if dest.exists():
        try:
            return _parse_two_col(dest.read_text(encoding="utf-8"), sid)
        except Exception:  # noqa: BLE001 - corrupt cache -> refetch below
            pass
    if offline:
        return None
    key = os.environ.get("FRED_API_KEY")
    # Prefer the FRED API host when a key is present: it is built for automation
    # and isn't burst-throttled like the public graph CSV endpoint. On success we
    # cache as a uniform 2-column CSV and return WITHOUT touching the graph host.
    if key and not offline and not dest.exists():
        try:
            r = session.get(API.format(sid=sid, key=key), timeout=45)
            r.raise_for_status()
            obs = r.json()["observations"]
            df = pd.DataFrame(obs)[["date", "value"]]
            tmp = dest.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(dest)
            return _parse_two_col(dest.read_text(encoding="utf-8"), sid)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] API fetch {sid} failed ({exc}); trying keyless CSV")
    # Keyless graph CSV (or cached file). Used when no key, or as API fallback.
    text = fetch_text(session, GRAPH.format(sid=sid), dest, offline=offline)
    if text is None:
        print(f"  [warn] no data for {sid}")
        return None
    return _parse_two_col(text, sid)


def to_monthly(s: pd.Series) -> pd.Series:
    """Collapse daily/weekly series to month-start by month-average."""
    if s.empty:
        return s
    freq = pd.infer_freq(s.index[:50]) or ""
    if freq.startswith(("D", "B", "W")):
        s = s.resample("MS").mean()
    else:
        s.index = to_month_start(s.index)
        s = s[~s.index.duplicated(keep="last")]
    return s


def main(offline: bool = False) -> None:
    cfg = load_config()
    session = http_session()
    snap = snapshot_dir("fred_api")

    ids = [cfg["us"]["label_series"]]
    ids += list(cfg["us"]["benchmarks"].values())
    ids += [d["id"] for d in cfg["us"]["panel"]]
    ids = list(dict.fromkeys(ids))  # dedupe, keep order

    cols = {}
    meta = {}
    have_key = bool(os.environ.get("FRED_API_KEY"))
    for i, sid in enumerate(ids):
        # Polite spacing only matters for the burst-throttled keyless graph host;
        # the API host doesn't need it.
        if i and not offline and not have_key and not (snap / f"{sid}.csv").exists():
            time.sleep(1.0)
        s = fetch_one(session, sid, snap, offline)
        if s is None or s.dropna().empty:
            meta[sid] = {"rows": 0, "status": "missing"}
            continue
        sm = to_monthly(s).dropna()
        cols[sid] = sm
        meta[sid] = {"rows": int(sm.shape[0]),
                     "start": sm.index.min().strftime("%Y-%m"),
                     "end": sm.index.max().strftime("%Y-%m"),
                     "status": "ok"}
        print(f"  {sid}: {meta[sid]['rows']} obs "
              f"{meta[sid].get('start')}..{meta[sid].get('end')}")

    if not cols:
        raise SystemExit("No FRED series fetched; aborting (check network).")

    panel = pd.concat(cols, axis=1)
    panel.index.name = "date"
    out = snapshot_dir("fred_api") / "fred_monthly.csv"
    panel.to_csv(out)
    # also drop a stable copy for the cleaner to pick up
    panel.to_csv(snap.parent / "fred_monthly_latest.csv")
    write_json(SITE_DATA.parent.parent / "data" / "interim" / "fred_fetch_meta.json",
               {"vintage_date": snap.name, "series": meta})
    # Stamp live mode (overrides any prior demo-synthetic marker) so downstream
    # artefacts and the dashboard banner reflect real data.
    n_ok = sum(1 for m in meta.values() if m.get("status") == "ok")
    used_api = bool(os.environ.get("FRED_API_KEY"))
    write_json(SITE_DATA.parent.parent / "data" / "facts" / "data_mode.json",
               {"data_mode": "live",
                "reason": f"Fetched {n_ok}/{len(meta)} FRED series via "
                          f"{'FRED API' if used_api else 'keyless CSV'}.",
                "vintage_date": snap.name, "n_series_ok": n_ok})
    print(f"  panel shape {panel.shape}; saved {out.name}; data_mode=live "
          f"({n_ok}/{len(meta)} series ok)")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
