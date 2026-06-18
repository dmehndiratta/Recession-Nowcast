"""Local convenience: copy the static payload into the Astro site's public/<slug>/.

Mirrors the CI sync-website job for local preview. Copies site/data/*.json,
site/dashboard.html and site/report.html into <website-dir>/public/<slug>/.

    python sync_to_website.py                       # default D:/Website
    python sync_to_website.py --website-dir D:/Website

This never edits the Website's src/ (the /research page + card are hand-authored
once, per SETUP.md §5).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLUG = "recession-nowcasting"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--website-dir", default="D:/Website")
    ap.add_argument("--slug", default=SLUG)
    args = ap.parse_args()

    target = Path(args.website_dir) / "public" / args.slug
    (target / "data").mkdir(parents=True, exist_ok=True)

    site = ROOT / "site"
    n = 0
    for j in (site / "data").glob("*.json"):
        shutil.copy2(j, target / "data" / j.name)
        n += 1
    for html in ("dashboard.html", "report.html"):
        src = site / html
        if src.exists():
            shutil.copy2(src, target / html)
    print(f"Copied {n} JSON + dashboards into {target}")
    print(f"Open https://dhruv-mehndiratta.com/{args.slug}/dashboard.html after deploy.")


if __name__ == "__main__":
    main()
