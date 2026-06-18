"""Assemble site/data/*.json — the only thing the dashboard/report read.

Pulls the committed processed artefacts (backtests, uncertainty, benchmark and
model diagnostics, data-mode + panel meta) into a small set of browser-parseable
JSON files. Numbers in report.html must trace back to these.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPE))
from common import (FACTS, PROCESSED, SITE_DATA, load_config,  # noqa: E402
                    read_json, write_json)

MODELS = ["spread_probit", "sahm", "dfm", "midas", "gbm"]
LABELS = {"spread_probit": "Term-spread probit", "sahm": "Sahm rule",
          "dfm": "Dynamic factor (DFM)", "midas": "Penalised MIDAS",
          "gbm": "Gradient boosting"}


def _maybe(path: Path):
    return read_json(path) if path.exists() else None


def metrics_table(bt: dict) -> list[dict]:
    rows = []
    for name in MODELS:
        if name not in bt["models"]:
            continue
        m = bt["models"][name]["metrics"]
        mx = bt["models"][name]["metrics_ex_covid"]
        ci = bt["models"][name]["ci"]
        row = {
            "model": name, "label": LABELS[name],
            "brier": m["brier"], "brier_lo": ci["brier"]["lo"], "brier_hi": ci["brier"]["hi"],
            "log_loss": m["log_loss"], "auc": m["auc"],
            "auc_lo": ci["auc"]["lo"], "auc_hi": ci["auc"]["hi"],
            "pr_auc": m["pr_auc"], "ece": m["ece"],
            "brier_ex_covid": mx["brier"], "auc_ex_covid": mx["auc"],
        }
        if "dm_vs_spread" in bt["models"][name]:
            row["dm_vs_spread_p"] = bt["models"][name]["dm_vs_spread"]["p_value"]
            row["dm_favors"] = bt["models"][name]["dm_vs_spread"]["favors"]
        rows.append(row)
    return rows


def build_country(country: str, cfg) -> dict:
    rt = _maybe(PROCESSED / f"backtest_{country}_realtime.json")
    if rt is None:
        return {}
    paths = rt["paths"]
    out = {
        "country": country,
        "data_mode": rt.get("data_mode", "live"),
        "horizon": rt.get("horizon", 0),
        "eval_start": rt["eval_start"], "eval_end": rt["eval_end"],
        "base_rate": rt["base_rate"], "n_obs": rt["n_obs"],
        "metrics_table": metrics_table(rt),
        "reliability": {n: rt["models"][n]["reliability"] for n in MODELS
                        if n in rt["models"]},
        "paths": paths,
    }
    if country == "us":
        fin = _maybe(PROCESSED / "backtest_us_final.json")
        unc = _maybe(PROCESSED / "results_uncertainty.json")
        if fin:
            out["vintage_vs_final"] = {
                "realtime": {n: rt["models"][n]["metrics"] for n in MODELS
                             if n in rt["models"]},
                "final": {n: fin["models"][n]["metrics"] for n in MODELS
                          if n in fin["models"]},
            }
        if unc:
            out["uncertainty"] = unc
            out["current"] = unc.get("current")
    return out


def main(offline: bool = False) -> None:
    cfg = load_config()
    data_mode = _maybe(FACTS / "data_mode.json") or {"data_mode": "live"}

    us = build_country("us", cfg)
    ca = build_country("ca", cfg)

    write_json(SITE_DATA / "nowcast_us.json", us)
    write_json(SITE_DATA / "nowcast_ca.json", ca)

    facts = {
        "title": "Recession Nowcast",
        "last_updated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_mode": data_mode.get("data_mode", "live"),
        "data_mode_reason": data_mode.get("reason", ""),
        "seed": cfg["seed"],
        "panel_meta": _maybe(PROCESSED / "panel_meta.json"),
        "models": {
            "dfm": _maybe(PROCESSED / "results_dfm.json"),
            "spread": _maybe(PROCESSED / "results_bench_spread.json"),
            "sahm": _maybe(PROCESSED / "results_bench_sahm.json"),
            "midas": _maybe(PROCESSED / "results_midas.json"),
            "gbm": _maybe(PROCESSED / "results_gbm.json"),
        },
        "current_us": us.get("current"),
    }
    write_json(SITE_DATA / "facts.json", facts)
    print(f"  exported site/data: nowcast_us, nowcast_ca, facts "
          f"(data_mode={facts['data_mode']})")


if __name__ == "__main__":
    main(offline="--offline" in sys.argv)
