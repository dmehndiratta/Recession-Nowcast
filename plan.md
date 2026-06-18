# plan-recession-nowcast.md

> Execution plan for Claude Code. Self-contained. See SETUP.md for the shared
> website mechanism and house conventions.
>
> **Repo:** `dmehndiratta/Recession-Nowcast` (new, public, independent)
> **Local path:** `D:\Python Projects\Recession-Nowcast`
> **Site slug:** `recession-nowcasting` → `Website/public/recession-nowcasting/`,
> page `/research/recession-nowcasting`. **Living analysis** (weekly cron).

---

## 1. Thesis

**Question:** Can a mixed-frequency model built only from data *as it was actually
available at each historical date* (real-time vintages) produce recession
probabilities for the US (and Canada) that beat simple, transparent benchmarks
out-of-sample — and can it be served as an honest, weekly-refreshing probability
with explicit uncertainty rather than a single number?

**Hypothesis:** A dynamic-factor / penalised mixed-frequency model on FRED-MD
(+ StatCan for Canada) will, in **pseudo-real-time backtests**, deliver a lower
log-loss / Brier and a higher AUC for classifying NBER (US) and C.D. Howe (Canada)
recession months than (i) the term-spread-only probit and (ii) the Sahm-rule
trigger, while remaining well-calibrated.

**What would falsify it:** If, in strict out-of-sample evaluation on the real-time
vintages, the model does **not** beat the term-spread probit and the Sahm rule on
Brier/log-loss with non-overlapping block-bootstrap intervals, then "the extra
machinery earns its keep" is false and the honest deliverable is the simple
benchmark with calibrated bands. Equally falsifying: if probabilities are
miscalibrated (reliability curve far off-diagonal) or if apparent skill vanishes
once we use vintage data instead of final-revised data (a classic look-ahead
artefact), the contribution is negative and must be reported as such.

---

## 2. Why it is in the portfolio

**Audience:** capital-markets macro/rates strategists, bank economics desks,
asset-allocation and risk teams who consume recession-probability signals.

**Skill demonstrated:** real-time macro-econometrics done correctly — vintage
discipline (no look-ahead), mixed-frequency modelling (MIDAS / dynamic factor /
state-space), honest out-of-sample evaluation, **calibrated probabilistic
forecasting with uncertainty bands**, and production engineering (a signal that
refreshes weekly via CI). This is precisely the workflow a markets economist is
hired to own, and the "served as a live dashboard" piece shows data-engineering
maturity.

---

## 3. Data

### US — primary
- **FRED-MD** (St. Louis Fed): 128 monthly macro/financial series back to 1959, in
  a single CSV, **with retained monthly vintages** (real-time). **Verified live
  2026-06-15.** Access: `https://www.stlouisfed.org/research/economists/mccracken/
  fred-databases`; current file `current.csv` and dated vintages
  `monthly/<YYYY-MM>.csv` (e.g. `2026-05.csv`); historical vintages as zip
  (`1999-08`→`2014-12`, `2015-01`→`2024-12`). The first data row encodes the
  recommended stationarity transform per series (`tcode`) — use it.
- **FRED-QD** (quarterly companion) for real GDP / GDP growth as the low-frequency
  anchor.
- **NBER recession dates** → the label. Available as the FRED series `USREC` (NBER-
  based recession indicator, monthly 0/1) and from NBER directly. Also pull
  `USRECM`/peak-trough. Use `USREC` for the monthly classification target.
- **Benchmarks/targets for comparison:** term spread (`T10Y3M`/`T10Y2Y`),
  **Sahm rule real-time** (`SAHMREALTIME`), and the Fed's smoothed recession
  probabilities (`RECPROUSM156N`, Chauvet–Piger) as a published reference line —
  all free on FRED via the FRED API (free key) or keyless CSV download.

### Canada — second market
- **C.D. Howe Institute Business Cycle Council** recession chronology (the NBER
  analogue for Canada), peaks/troughs back to 1926; recent rulings: 2008–09 =
  Nov 2008→May 2009; COVID = Mar 2020 peak, Apr 2020 trough; **recession avoided in
  2022 and 2023**. **Verified live 2026-06-15** (`https://cdhowe.org/council/
  business-cycle-council/`). These dates are published in press releases/PDFs;
  encode them into a small committed `data/manual/cdhowe_recessions.csv` with the
  source URL per row (the chronology is the label; it updates rarely).
- **Statistics Canada monthly indicators** (public, WDS/REST API + CSV):
  monthly real **GDP by industry** (Table **36-10-0434-01**; real-time/vintage
  releases Table **36-10-0491-01**), **Labour Force Survey** (unemployment rate,
  employment), CPI, retail/manufacturing, housing starts. Plus Canadian financial
  series from FRED/Bank of Canada (yield curve, spreads).
- **Bank of Canada** Valet API for rates/yields (free, no key).

### Access methods, cadence, size
- FRED API (free key `FRED_API_KEY`) or keyless CSV endpoints; FRED-MD via direct
  CSV; StatCan via the **Web Data Service** (`https://www150.statcan.gc.ca/t1/wds/`)
  with table PIDs; Bank of Canada Valet JSON. All small (MB-scale).
- Cadence: FRED-MD monthly; FRED daily series daily; StatCan monthly GDP ~2-month
  lag; LFS monthly; labels (NBER/C.D. Howe) update only at turning points.

### Gotchas to guard against
1. **Look-ahead via revisions (the cardinal sin):** real GDP, employment, etc. are
   heavily revised. Backtests MUST use the **vintage available at the decision
   date**, not today's revised series. FRED-MD vintages and StatCan real-time table
   36-10-0491-01 exist precisely for this — use them; do not use `current.csv` for
   historical evaluation.
- 2. **Label timing / publication lag:** NBER/C.D. Howe *declare* recessions with long
  lags (often 6–18 months). The target is the *eventual* peak/trough dating, but the
  model is judged on predicting it from contemporaneous data — that asymmetry is the
  whole point of nowcasting and must be framed, not hidden.
3. **Few events / class imbalance:** monthly recessions are rare (US ~15% of months
  since 1959; Canada fewer turning points). Evaluate with Brier/log-loss/PR-AUC, use
  block/structured CV (no random k-fold across time), and be explicit about the
  small number of independent recession episodes (≈8 US since 1960; Canada ≈5).
4. **COVID outlier:** Feb–Apr 2020 are massive outliers that can dominate fits and
  metrics. Report results both including and excluding 2020; never let 2020 silently
  drive the headline.
5. **Series availability across vintages:** some FRED-MD series start late or are
  discontinued; the panel is unbalanced and changes composition over vintages.
  Handle missingness in the factor/state-space step (EM), and freeze the series set
  per the `fredmdchanges.pdf` notes.
6. **Stationarity transforms:** apply the `tcode` transforms; mixing levels and
  growth rates corrupts the factors.

---

## 4. Method

**Target:** monthly binary recession indicator (US: `USREC`; Canada: C.D. Howe
chronology), and the *probability* of recession in month `t` given data through `t`
(nowcast) and `t+h` (short-horizon forecast, h = 1–3).

**Estimators (ladder from transparent to richer; comparison is the deliverable):**
1. **Benchmark A — term-spread probit** (Estrella–Mishkin style): `P(recession_{t+12})`
   on the 10y–3m spread. The canonical baseline.
2. **Benchmark B — Sahm rule** (real-time unemployment trigger) as a rule-based
   comparator.
3. **Primary — mixed-frequency dynamic factor model** (state-space, Kalman filter/
   smoother; EM for missing data) extracting a small number of factors from the
   monthly panel, feeding a **dynamic-factor Markov-switching / probit** classifier;
   **and/or** a **penalised MIDAS logistic** (elastic-net) mapping mixed-frequency
   indicators to the recession label. Provide both a parsimonious linear version and
   one nonlinear comparator (gradient-boosted classifier on the factors) to show ML
   vs econometrics honestly.

**Identification / honest evaluation — "done right":**
- **Pseudo-real-time, expanding-window out-of-sample:** for each evaluation month
  `τ`, re-fit using ONLY the vintage available at `τ`; produce `P(recession)`; never
  peek forward. This is the core credibility move.
- **Metrics:** Brier score, log-loss, AUC, PR-AUC, and the **calibration**
  (reliability curve, ECE) of the probabilities — vs Benchmarks A and B — with
  **block bootstrap** / stationary bootstrap CIs that respect serial dependence.
- **Uncertainty (mandatory, per brief):** report probability **bands**, not point
  probabilities — e.g., predictive intervals from (i) parameter uncertainty via
  bootstrap/Bayesian posterior, and (ii) a small **Monte Carlo** over factor-state
  uncertainty from the Kalman smoother. The dashboard shows a shaded band.
- **Robustness:** results with/without COVID 2020; alternative factor counts;
  alternative spread definitions; US and Canada separately; sensitivity to the
  label's declaration lag (evaluate at the *contemporaneous* truth vs *final* truth).
- **Refutation:** (a) **vintage vs final-data test** — show that skill measured on
  final-revised data is *inflated* relative to true real-time skill (demonstrates
  the model isn't just exploiting revisions); (b) placebo — scramble the recession
  label in time, confirm skill collapses; (c) horse-race significance — Diebold–
  Mariano / Giacomini–White conditional predictive-ability tests vs the benchmarks.

**What makes it NOT credible:** evaluating on revised (`current.csv`) data;
random k-fold CV; quoting AUC without calibration; ignoring the ~8-episode reality
and reporting tight CIs; letting COVID drive the headline; no benchmark comparison.

---

## 5. Architecture

```
Recession-Nowcast/
├── README.md
├── CLAUDE.md
├── plan.md
├── requirements.txt              # pinned
├── run_pipeline.py               # --offline --export-only --stage N --country {us,ca}
├── sync_to_website.py
├── .gitignore
├── SOURCES.md
├── config.yaml                   # series lists, factor count, horizons, seed, eval start
├── data/
│   ├── manual/cdhowe_recessions.csv     # Canada labels + source URLs (committed)
│   ├── raw/fredmd/<YYYY-MM-DD>/          # downloaded vintages (gitignored)
│   ├── raw/statcan/<YYYY-MM-DD>/
│   ├── raw/fred_api/<YYYY-MM-DD>/
│   ├── interim/
│   └── processed/                        # backtest results, prob series (committed, small)
├── pipeline/
│   ├── 01_fetch/fetch_fredmd.py          # current + dated vintages; cache; print vintage
│   ├── 01_fetch/fetch_fred_series.py     # USREC, T10Y3M, SAHMREALTIME, RECPROUSM156N (API/keyless)
│   ├── 01_fetch/fetch_statcan.py         # WDS API: 36-10-0434/0491, LFS, CPI
│   ├── 01_fetch/fetch_boc_valet.py       # Canada yields/spreads
│   ├── 02_clean/build_panel.py           # apply tcode transforms; align; vintage-aware
│   ├── 03_analysis/bench_spread_probit.py# results_bench_spread.json
│   ├── 03_analysis/bench_sahm.py         # results_bench_sahm.json
│   ├── 03_analysis/dfm_statespace.py     # Kalman/EM factors → results_dfm.json
│   ├── 03_analysis/midas_logit.py        # penalised MIDAS logistic → results_midas.json
│   ├── 03_analysis/ml_gbm.py             # GBM on factors (nonlinear comparator)
│   ├── 03_analysis/backtest.py           # pseudo-real-time loop; metrics; block bootstrap; DM/GW tests
│   ├── 03_analysis/uncertainty.py        # MC over states + param bootstrap → bands
│   └── 04_export/export_json.py
├── site/{report.html,dashboard.html,data/*.json}
├── tests/                        # vintage-no-lookahead assert, label join, JSON schema
└── .github/workflows/recession-nowcast-update.yml
```

- **Environment:** **Python 3.11**. Pin: `pandas==2.2.*`, `numpy==1.26.*`,
  `statsmodels==0.14.*` (state-space `DynamicFactorMQ`, `MarkovRegression`),
  `scikit-learn==1.5.*`, `xgboost==2.1.*`, `fredapi==0.5.*` (or keyless requests),
  `requests`, `arch==7.*` (bootstrap utilities) or a custom block bootstrap,
  `matplotlib==3.9.*`, `pyyaml`, `tqdm`. `statsmodels.DynamicFactorMQ` natively
  supports **mixed-frequency** monthly/quarterly with EM — use it for the DFM.
- **Seeds:** `SEED=20260615` for bootstrap/MC; recorded in results JSON.
- **Vintage discipline enforced in code:** the backtest loop takes a vintage-keyed
  data store; a test asserts that for evaluation date `τ` no observation with a
  release date > `τ` is used.
- Reuse SETUP.md §7 conventions (validate-then-promote, last-good fallback so the
  weekly CI never publishes on a failed fetch, dated snapshots, JSON guard).

---

## 6. Deliverables

- **Repo** as above; `python run_pipeline.py --country us` and `--country ca`
  reproduce all results; `--offline` rebuilds from committed processed artefacts.
- **Report** `site/report.html`: the real-time problem → data & vintage method →
  benchmark ladder → primary models → **pseudo-real-time backtest** with calibrated
  metrics and DM/GW tests → uncertainty construction → US vs Canada → robustness
  (COVID in/out, vintage-vs-final) → limitations.
- **Interactive dashboard** `site/dashboard.html` (living; static; Plotly + D3 from
  CDN, data from JSON): (a) **current recession-probability gauge** with a shaded
  uncertainty band, US & Canada toggle; (b) historical probability line vs NBER/
  C.D. Howe shaded recessions and vs the benchmark lines; (c) **reliability/
  calibration** panel; (d) factor/indicator contributions for the latest month;
  (e) "last updated" stamp from a facts JSON.
- **Figures/tables:** backtest metrics table (model vs benchmarks, with CIs),
  reliability curves, probability-vs-recessions chart, vintage-vs-final comparison.
- **Project CLAUDE.md:** vintage rules, series list, label sources, COVID handling,
  seed, "never evaluate on current.csv."

---

## 7. Website integration (Pattern A; see SETUP.md §3–§8)

**Secrets in this repo:** `WEBSITE_REPO_TOKEN` (required). `FRED_API_KEY` (optional —
the pipeline falls back to FRED's keyless CSV endpoints if absent; StatCan and BoC
are keyless).

**Workflow:** `.github/workflows/recession-nowcast-update.yml` — same two-job shape
as §7 of plan-credit-default-pd.md, with these differences:
- Add a **weekly schedule** (`cron: '0 11 * * 1'`, Mondays 11:00 UTC — after the
  US Monday data and ahead of the week) plus `workflow_dispatch`; this is a *living*
  analysis.
- The `update` job runs the **full** pipeline (fetch → model → export), passing
  `FRED_API_KEY: ${{ secrets.FRED_API_KEY }}`; it commits refreshed `site/data/`
  and `data/processed/` back to the repo, with validate-then-promote so a failed
  fetch keeps the last-good JSON.
- `sync-website` copies the full payload into
  `website/public/recession-nowcasting/{data/,dashboard.html,report.html}` and
  pushes (triggering Cloudflare Pages). Target path uses slug `recession-nowcasting`.
- Keep the verbatim **browser-parseable JSON guard** step before committing.

**Human action items (Dhruv):** create repo `Recession-Nowcast`; add
`WEBSITE_REPO_TOKEN` (and optionally `FRED_API_KEY`) to Actions secrets; one-time
Website edit — `/research` card + `src/pages/research/recession-nowcasting.mdx`
(status `LIVE`, tags `MACRO`, `CAPITAL MARKETS`, `TIME SERIES`, `NOWCASTING`)
embedding `/recession-nowcasting/dashboard.html`.

**Verify the deploy:** trigger `workflow_dispatch`; confirm both jobs green; open
`https://dhruv-mehndiratta.com/recession-nowcasting/dashboard.html`; confirm the
probability band renders and `/recession-nowcasting/data/` JSON parses; confirm a
subsequent scheduled run updates the "last updated" stamp.

---

## 8. Acceptance criteria

- [ ] FRED-MD vintages, FRED benchmark series, StatCan, and BoC all fetched
      idempotently with vintage dates logged.
- [ ] `tcode` stationarity transforms applied; panel vintage-aware.
- [ ] Labels: US `USREC`; Canada C.D. Howe CSV with per-row source URLs.
- [ ] **No-look-ahead test passes**: backtest at date τ uses only data released ≤ τ.
- [ ] Term-spread probit and Sahm-rule benchmarks implemented and scored.
- [ ] DFM (state-space, mixed-frequency) and penalised MIDAS logistic implemented;
      one nonlinear comparator (GBM) included.
- [ ] Pseudo-real-time expanding-window backtest produces Brier/log-loss/AUC/PR-AUC
      and calibration, with block-bootstrap CIs and DM/GW tests vs benchmarks.
- [ ] **Uncertainty bands** produced (MC over states + parameter bootstrap) and shown.
- [ ] Robustness: COVID in/out; **vintage-vs-final** comparison demonstrating real-
      time skill is not a revision artefact; US and Canada both reported.
- [ ] Living dashboard shows current probability + band + calibration + "last updated".
- [ ] JSON guard passes; report numbers trace to JSON; limitations present.

---

## 9. Task sequence

1. Scaffold repo, `config.yaml` (series lists, factor count, horizons, eval-start,
   SEED), pinned `requirements.txt`, README/CLAUDE skeletons, `SOURCES.md` (FRED-MD,
   FRED, StatCan WDS, BoC Valet, NBER, C.D. Howe URLs). **Verify:** imports work;
   links resolve.
2. `fetch_fredmd.py`: download `current.csv` + needed dated vintages; cache to
   `data/raw/fredmd/<date>/`; print vintage. **Verify:** ≥1 historical vintage and
   current both load; idempotent.
3. `fetch_fred_series.py`: `USREC`, `T10Y3M`, `T10Y2Y`, `SAHMREALTIME`,
   `RECPROUSM156N` via API or keyless CSV. **Verify:** series non-empty; keyless
   fallback works with no `FRED_API_KEY`.
4. `fetch_statcan.py` + `fetch_boc_valet.py`: monthly GDP (36-10-0434/0491), LFS,
   CPI, BoC yields. **Verify:** WDS returns vectors for the PIDs; row counts logged.
5. `data/manual/cdhowe_recessions.csv`: encode peaks/troughs with source URLs.
   **Verify:** dates match the press-release chronology (2008-11→2009-05; 2020-03→
   2020-04; no 2022/2023 recession).
6. `02_clean/build_panel.py`: apply tcodes, align monthly panel, attach labels,
   build a vintage-keyed store. **Verify:** no level/growth mixing; no NA in labels.
7. `bench_spread_probit.py` + `bench_sahm.py` → benchmark JSONs. **Verify:** spread
   probit reproduces the well-known inverted-curve→recession pattern.
8. `dfm_statespace.py` (DynamicFactorMQ + classifier) and `midas_logit.py`
   (elastic-net) → results JSONs. **Verify:** factors explain sensible variance;
   models fit on a training vintage.
9. `ml_gbm.py`: nonlinear comparator on factors. **Verify:** OOS only.
10. `backtest.py`: pseudo-real-time expanding-window loop for all models + benchmarks;
    Brier/log-loss/AUC/PR-AUC; calibration; block bootstrap; DM/GW. **Verify:**
    no-look-ahead test passes; metrics table emitted.
11. `uncertainty.py`: MC over Kalman states + parameter bootstrap → probability
    bands. **Verify:** bands cover; widths sensible.
12. Robustness runs: COVID in/out; vintage-vs-final; Canada. **Verify:** vintage
    skill ≤ final skill (documented).
13. `export_json.py` → `site/data/*.json` (+ facts/last-updated). **Verify:** JSON
    guard passes.
14. Build `report.html` + living `dashboard.html`. **Verify:** `file://` load; band
    renders; numbers match JSON.
15. `tests/`: no-look-ahead, label join, JSON schema, reproducibility. **Verify:**
    `pytest` green.
16. Add weekly workflow (§7); commit processed artefacts for `--offline`. **Verify:**
    `workflow_dispatch` both jobs green.
17. (Dhruv) secrets + one-time Website page/card; **verify deploy**.

---

## 10. Limitations and caveats

- **Tiny sample of recessions:** ≈8 US and ≈5 Canadian independent episodes since
  1960 mean wide, honestly-reported uncertainty; no model can be "validated" on so
  few events the way a high-frequency forecaster can. We quantify, not pretend.
- **Regime change:** relationships (e.g., the yield curve's signal) can shift;
  out-of-sample skill is conditional on the future resembling the past.
- **Label lag and revision:** official dating arrives long after the fact; the
  nowcast is judged against an evolving truth, and Canadian dates are sparse and
  annual-meeting-driven.
- **COVID:** a non-economic shock that no macro model "predicts"; we report with and
  without it and never claim to have called it.
- **Not investment advice:** a recession probability is a research signal, not a
  trade. State this plainly.

---

## 11. Open questions and risks

1. **Canada label granularity:** C.D. Howe dates are coarser/less frequent than NBER.
   *Decision:* lead with US (more episodes, cleaner monthly target), present Canada as
   a second market with appropriately wider caveats — agree?
2. **Primary model choice:** DFM-Markov-switching vs penalised MIDAS as the headline.
   Default: present both; lead with the DFM state-space (most defensible, native
   mixed-frequency). Confirm preference.
3. **FRED-MD vintage coverage:** pre-2015 vintages come as zips; confirm we backtest
   from ~1999 (vintage start) forward, accepting that pre-1999 evaluation uses
   pseudo-vintages (final data) clearly labelled.
4. **Refresh cadence:** weekly Monday cron proposed; most inputs are monthly, so
   weekly mostly refreshes daily financial series and the "as-of" framing — confirm
   weekly (matches your other living projects) vs monthly.
5. **Scope guard:** keep to recession *probability*; do NOT expand into full GDP
   density nowcasting this round (it's a natural v2).
