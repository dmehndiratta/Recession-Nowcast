"""Shared helpers: paths, config, caching HTTP, validate-then-promote, JSON guard.

Every stage imports from here so conventions (dated snapshots, last-good
fallback, deterministic browser-parseable JSON) are applied uniformly.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
FACTS = DATA / "facts"
MANUAL = DATA / "manual"
SITE_DATA = ROOT / "site" / "data"

for _p in (RAW, INTERIM, PROCESSED, FACTS, SITE_DATA):
    _p.mkdir(parents=True, exist_ok=True)

TODAY = _dt.date.today().isoformat()

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# --- config ----------------------------------------------------------------
def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seeds(seed: int) -> None:
    """Deterministic numpy RNG; record seed in results JSON elsewhere."""
    np.random.seed(seed)


# --- HTTP with dated snapshot caching + last-good fallback -----------------
def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": _BROWSER_UA, "Accept": "*/*"})
    return s


def snapshot_dir(source: str, when: str | None = None) -> Path:
    d = RAW / source / (when or TODAY)
    d.mkdir(parents=True, exist_ok=True)
    return d


def last_good(source: str) -> Path | None:
    """Most recent dated snapshot directory for a source, or None."""
    base = RAW / source
    if not base.exists():
        return None
    dated = sorted(p for p in base.iterdir() if p.is_dir())
    return dated[-1] if dated else None


def fetch_text(session: requests.Session, url: str, dest: Path,
               offline: bool = False, retries: int = 4, backoff: float = 3.0,
               **kwargs) -> str | None:
    """Fetch `url` to `dest` with validate-then-promote semantics.

    Retries with exponential backoff (hosts like FRED throttle bursts). Returns
    the text on success (live or cached). On a failed/empty fetch the existing
    cached file is kept (never overwritten) and its content returned; if nothing
    cached and offline, returns None.
    """
    if dest.exists() and (offline or os.environ.get("NOWCAST_USE_CACHE") == "1"):
        return dest.read_text(encoding="utf-8")
    if offline:
        return dest.read_text(encoding="utf-8") if dest.exists() else None
    last_exc = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=45, **kwargs)
            r.raise_for_status()
            text = r.text
            if not text or len(text) < 8:
                raise ValueError("empty/short response")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(dest)  # promote atomically only after validation
            return text
        except Exception as exc:  # noqa: BLE001 - last-good fallback is the point
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    print(f"  [warn] fetch failed for {url}: {last_exc}")
    if dest.exists():
        print(f"  [warn] using last-good cache {dest}")
        return dest.read_text(encoding="utf-8")
    return None


def fetch_bytes(session: requests.Session, url: str, dest: Path,
                offline: bool = False, **kwargs) -> bytes | None:
    if dest.exists() and (offline or os.environ.get("NOWCAST_USE_CACHE") == "1"):
        return dest.read_bytes()
    if offline:
        return dest.read_bytes() if dest.exists() else None
    try:
        r = session.get(url, timeout=120, **kwargs)
        r.raise_for_status()
        if not r.content:
            raise ValueError("empty response")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(dest)
        return r.content
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] fetch failed for {url}: {exc}")
        if dest.exists():
            return dest.read_bytes()
        return None


# --- deterministic, browser-parseable JSON ---------------------------------
def _reject_nonfinite(x):
    raise ValueError(f"non-finite value not allowed in JSON: {x}")


def _clean(obj):
    """Recursively convert numpy types and replace NaN/Inf with None so the
    emitted JSON is strictly parseable by browsers (no NaN/Infinity tokens)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, _dt.date, _dt.datetime)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    if obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    return obj


def write_json(path: Path, obj, validate: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _clean(obj)
    text = json.dumps(cleaned, indent=2, sort_keys=False, ensure_ascii=False)
    if validate:
        # The exact guard the CI runs: reject NaN/Infinity tokens.
        json.loads(text, parse_constant=_reject_nonfinite)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)} ({len(text)} bytes)")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def month_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def to_month_start(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M").to_timestamp()
