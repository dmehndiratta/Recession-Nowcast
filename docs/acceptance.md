# Acceptance criteria → evidence (plan §8)

| Criterion | Where it's met |
|---|---|
| FRED-MD vintages, FRED series, StatCan, BoC fetched idempotently with vintage dates logged | `pipeline/01_fetch/*` (dated `data/raw/<source>/<YYYY-MM-DD>/`, last-good fallback, vintage printed). FRED-MD bulk CSV is Akamai-blocked on some networks → `make_demo_panel.py` synthetic fallback (stamped `demo-synthetic`). |
| `tcode` stationarity transforms applied; panel vintage-aware | `pipeline/panel.py::apply_tcode`, `transform_panel`; `VintageStore`. |
| Labels: US `USREC`; Canada C.D. Howe CSV with per-row source URLs | `panel.py::usrec_labels`, `data/manual/cdhowe_recessions.csv`. |
| **No-look-ahead test passes** | `tests/test_no_lookahead.py` (asserts `max_release_date(τ) ≤ τ`). |
| Term-spread probit and Sahm benchmarks implemented and scored | `03_analysis/bench_spread_probit.py`, `bench_sahm.py`; scored in `backtest.py`. |
| DFM (state-space, mixed-frequency) + penalised MIDAS + one nonlinear (GBM) | `dfm_statespace.py` (statsmodels `DynamicFactorMQ`), `midas_logit.py`, `ml_gbm.py`; in-loop estimators in `models.py`. |
| Pseudo-real-time expanding-window backtest: Brier/log-loss/AUC/PR-AUC + calibration, block-bootstrap CIs, DM/GW | `03_analysis/backtest.py` + `metrics.py`. |
| **Uncertainty bands** (MC over states + parameter bootstrap) | `03_analysis/uncertainty.py` → `results_uncertainty.json`; shaded in dashboard. |
| Robustness: COVID in/out; vintage-vs-final; US and Canada | `metrics_ex_covid` everywhere; `backtest_us_final.json` vs `_realtime.json`; `backtest_ca_realtime.json`. |
| Living dashboard: current probability + band + calibration + "last updated" | `site/dashboard.html`; `site/data/facts.json::last_updated`. |
| JSON guard passes; report numbers trace to JSON; limitations present | `common.write_json` guard + CI step; `report.html` reads JSON; §9 limitations. |

## Open questions from the plan (§11) — decisions taken here
1. **Canada granularity** — led with US; Canada is a second market with wider caveats
   (the demo runs the same models against C.D. Howe labels; in `live` mode StatCan
   series feed a Canada-specific panel).
2. **Primary model** — present all; lead with the DFM state-space.
3. **Vintage coverage** — backtest from `eval.start` (1985) on the final-data /
   publication-lag path; genuine FRED-MD vintages slot in where reachable.
4. **Refresh cadence** — weekly Monday cron (matches the other living projects).
5. **Scope** — recession *probability* only; GDP density nowcasting deferred to v2.
