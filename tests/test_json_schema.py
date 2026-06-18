"""site/data/*.json must be browser-parseable (no NaN/Infinity) with key fields."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site" / "data"


def _reject(x):
    raise ValueError(f"non-finite token: {x}")


def _all_jsons():
    return sorted(SITE.glob("*.json"))


@pytest.mark.skipif(not _all_jsons(), reason="run the pipeline export first")
def test_browser_parseable():
    for p in _all_jsons():
        text = p.read_text(encoding="utf-8")
        json.loads(text, parse_constant=_reject)  # rejects NaN/Infinity


@pytest.mark.skipif(not (SITE / "nowcast_us.json").exists(),
                    reason="run the pipeline export first")
def test_nowcast_us_shape():
    d = json.loads((SITE / "nowcast_us.json").read_text(encoding="utf-8"))
    for key in ("country", "data_mode", "metrics_table", "paths", "reliability"):
        assert key in d, f"missing {key}"
    assert d["country"] == "us"
    models = {r["model"] for r in d["metrics_table"]}
    assert {"spread_probit", "sahm", "dfm"}.issubset(models)
    # every probability path is within [0, 1]
    for name in ("dfm", "spread_probit", "sahm"):
        vals = d["paths"][name]
        assert all(0.0 <= v <= 1.0 for v in vals)


@pytest.mark.skipif(not (SITE / "facts.json").exists(),
                    reason="run the pipeline export first")
def test_facts_present():
    d = json.loads((SITE / "facts.json").read_text(encoding="utf-8"))
    assert d["data_mode"] in ("live", "demo-synthetic")
    assert "last_updated" in d
