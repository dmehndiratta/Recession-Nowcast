# Recession-Nowcast

Real-time **recession nowcasting** for the US (and Canada) with honest,
out-of-sample, **calibrated probabilities and uncertainty bands** — built so the
backtest only ever sees data *as it was available at each historical date* (no
look-ahead via revisions), and benchmarked against transparent baselines (the
term-spread probit and the Sahm rule).

> **Thesis.** Can a mixed-frequency factor model on real-time vintages beat the
> term-spread probit and the Sahm rule out-of-sample on Brier / log-loss / AUC,
> while staying well-calibrated — and be served as a live, weekly-refreshing
> probability *band*, not a single number? It is falsified if the extra machinery
> does not beat the benchmarks with block-bootstrap intervals, or if skill
> vanishes once vintage (not final-revised) data is used.

## What's here

| Piece | Where |
|---|---|
| Stationary panel + **vintage store** (no-look-ahead) | `pipeline/panel.py`, `pipeline/02_clean/build_panel.py` |
| Benchmarks: term-spread probit, Sahm rule | `pipeline/03_analysis/bench_*.py` |
| Primary: **mixed-frequency DFM** (statsmodels `DynamicFactorMQ`) + probit | `pipeline/03_analysis/dfm_statespace.py` |
| Penalised **MIDAS** logistic; **GBM** comparator | `pipeline/03_analysis/midas_logit.py`, `ml_gbm.py` |
| **Pseudo-real-time backtest** + block bootstrap + DM/GW | `pipeline/03_analysis/backtest.py`, `pipeline/metrics.py` |
| **Uncertainty bands** (param bootstrap × MC over factor states) | `pipeline/03_analysis/uncertainty.py` |
| Live **dashboard** + long-form **report** | `site/dashboard.html`, `site/report.html` |

## Quickstart

```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# full pipeline (fetch -> clean -> analysis -> export)
.venv\Scripts\python run_pipeline.py

# rebuild charts/JSON from committed artefacts, no network
.venv\Scripts\python run_pipeline.py --offline

# just one stage (1 fetch, 2 clean, 3 analysis, 4 export)
.venv\Scripts\python run_pipeline.py --stage 3
```

Open `site/dashboard.html` in a browser (loads `site/data/*.json`).

## Data mode

The pipeline runs in one of two modes, stamped into every artefact and shown as a
banner on the dashboard/report:

- **`live`** — real series fetched from FRED (keyless CSV or `FRED_API_KEY`),
  StatCan WDS, and Bank of Canada Valet.
- **`demo-synthetic`** — a deterministic synthetic panel (real FRED series IDs +
  real NBER recession dates + embedded signal) used **only** when the live
  endpoints are unreachable, so the full method is runnable and testable anywhere.
  Synthetic numbers are illustrative; never quote them as empirical results.

See [`CLAUDE.md`](CLAUDE.md) for conventions and [`SOURCES.md`](SOURCES.md) for the
primary data sources.

## Data sources (all free / keyless)

- **FRED-MD** real-time monthly vintages (St. Louis Fed) + **FRED** series
  (`USREC`, `T10Y3M`, `SAHMREALTIME`, `RECPROUSM156N`, the panel indicators).
- **NBER** recession dates (`USREC`) — US label.
- **C.D. Howe Institute** Business Cycle Council chronology — Canada label
  (`data/manual/cdhowe_recessions.csv`, one source URL per row).
- **Statistics Canada** WDS (monthly GDP, LFS, CPI) and **Bank of Canada** Valet
  (yields) for the Canada track.

## Honesty notes

- Evaluation is **pseudo-real-time** and **expanding-window**; metrics carry
  **block-bootstrap** CIs and **Diebold-Mariano / Giacomini-White** tests.
- Results are reported **with and without COVID** (2020-02..2020-08).
- The **vintage-vs-final** comparison shows real-time skill is not a revision
  artefact.
- Recession episodes are few (≈9 US, ≈4–5 Canada since 1960): intervals are wide
  by construction and reported honestly. **Not investment advice.**

## License / status

Research code. Living analysis (weekly refresh via GitHub Actions). See
[`plan.md`](plan.md) for the full execution plan and acceptance criteria.
