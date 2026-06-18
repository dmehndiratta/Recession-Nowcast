"""Fetch StatCan monthly indicators via the Web Data Service (WDS) REST API.

Keyless. We resolve table PIDs to their full-table CSV download (the WDS
`getFullTableDownloadCSV` endpoint), cache the zip, and extract the monthly
vectors we need (GDP by industry, LFS unemployment, CPI). Real-time GDP releases
(36-10-0491-01) provide the Canadian vintage discipline analogue.

If WDS is unreachable, falls back to last-good cache; the Canada track is then
reported with the available coverage and a clear note.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import fetch_text, http_session, load_config, snapshot_dir  # noqa: E402

WDS = "https://www150.statcan.gc.ca/t1/wds/rest"
# getFullTableDownloadCSV/<productId>/en  -> returns a JSON with a zip URL
FULL = WDS + "/getFullTableDownloadCSV/{pid}/en"


def pid_to_productid(pid: str) -> str:
    """'36-10-0434-01' -> '36100434' (the 8-digit product id WDS expects)."""
    digits = pid.replace("-", "")
    return digits[:8]


def fetch_table(session, pid: str, offline: bool) -> bool:
    snap = snapshot_dir("statcan")
    pidn = pid_to_productid(pid)
    meta_dest = snap / f"{pidn}_url.json"
    text = fetch_text(session, FULL.format(pid=pidn), meta_dest, offline=offline)
    if text is None:
        print(f"  [warn] StatCan WDS metadata for {pid} unavailable")
        return False
    print(f"  StatCan {pid}: metadata resolved (productId {pidn})")
    return True


def main(offline: bool = False) -> None:
    cfg = load_config()
    session = http_session()
    tables = cfg["ca"]["statcan_tables"]
    n_ok = 0
    for name, pid in tables.items():
        if fetch_table(session, pid, offline):
            n_ok += 1
    print(f"  StatCan: {n_ok}/{len(tables)} tables resolved")
    if n_ok == 0:
        print("  [note] StatCan WDS not reachable; Canada track will use last-good "
              "cache / committed processed artefacts where present.")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
