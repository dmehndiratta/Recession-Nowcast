"""Recession-Nowcast pipeline orchestrator.

Stages:
  1  fetch     FRED series (+ FRED-MD vintages, StatCan, BoC); demo fallback
  2  clean     build the stationary panel, labels, vintage-store inputs
  3  analysis  benchmarks, DFM, MIDAS, GBM, backtest, uncertainty
  4  export    assemble site/data/*.json

Flags:
  --offline       rebuild from last-good cache / committed artefacts (no network)
  --export-only   run only stage 4
  --stage N       run a single stage
  --country {us,ca}   (informational; both are computed by the backtest)
  --demo          force the synthetic panel (skip live fetch)

Examples:
  python run_pipeline.py
  python run_pipeline.py --offline
  python run_pipeline.py --stage 3
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"
sys.path.insert(0, str(PIPE))


def _run(rel: str, **kwargs):
    path = PIPE / rel
    name = rel.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    print(f"\n>>> {rel}")
    spec.loader.exec_module(mod)
    mod.main(**kwargs)


def stage_fetch(offline: bool, demo: bool):
    if demo:
        _run("01_fetch/make_demo_panel.py")
        return
    _run("01_fetch/fetch_fred_series.py", offline=offline)
    _run("01_fetch/fetch_fredmd.py", offline=offline)
    _run("01_fetch/fetch_statcan.py", offline=offline)
    _run("01_fetch/fetch_boc_valet.py", offline=offline)
    # Fall back to the synthetic panel if the live fetch produced nothing.
    latest = ROOT / "data" / "raw" / "fred_api" / "fred_monthly_latest.csv"
    if not latest.exists():
        print("\n[fallback] live FRED panel empty -> generating synthetic demo panel")
        _run("01_fetch/make_demo_panel.py")


def stage_clean(offline: bool):
    _run("02_clean/build_panel.py", offline=offline)


def stage_analysis(offline: bool):
    _run("03_analysis/bench_spread_probit.py", offline=offline)
    _run("03_analysis/bench_sahm.py", offline=offline)
    _run("03_analysis/dfm_statespace.py", offline=offline)
    _run("03_analysis/midas_logit.py", offline=offline)
    _run("03_analysis/ml_gbm.py", offline=offline)
    _run("03_analysis/backtest.py", offline=offline)
    _run("03_analysis/uncertainty.py", offline=offline)


def stage_export(offline: bool):
    _run("04_export/export_json.py", offline=offline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--country", choices=["us", "ca"], default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.export_only:
        stage_export(args.offline)
        return
    if args.stage:
        {1: lambda: stage_fetch(args.offline, args.demo),
         2: lambda: stage_clean(args.offline),
         3: lambda: stage_analysis(args.offline),
         4: lambda: stage_export(args.offline)}[args.stage]()
        return

    stage_fetch(args.offline, args.demo)
    stage_clean(args.offline)
    stage_analysis(args.offline)
    stage_export(args.offline)
    print("\nPipeline complete. See site/data/*.json and site/dashboard.html")


if __name__ == "__main__":
    main()
