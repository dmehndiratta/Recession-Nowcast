# SOURCES.md — primary data sources & traceability

Every headline figure traces to a primary source below and to a value in
`site/data/*.json`. Verified live 2026-06-15 (per the project plan).

## US

| Source | What | Access | Notes |
|---|---|---|---|
| **FRED-MD** (McCracken, St. Louis Fed) | 128 monthly macro/financial series, 1959– , with retained monthly **vintages** (real-time) | `https://www.stlouisfed.org/research/economists/mccracken/fred-databases` — `current.csv`, `monthly/<YYYY-MM>.csv`, historical zips | First data row encodes `tcode` stationarity transform; some networks block the bulk CDN (Akamai). |
| **FRED** series | `USREC` (NBER label), `T10Y3M`, `T10Y2Y` (term spread), `SAHMREALTIME`, `RECPROUSM156N` (Chauvet–Piger reference), panel indicators | FRED API (`FRED_API_KEY`) or keyless `fredgraph.csv?id=<ID>` | Keyless graph endpoint returns latest-revised values (the "final-data" path). |
| **NBER** | Business-cycle peak/trough dating → `USREC` | `https://www.nber.org/research/business-cycle-dating` | Declared with long lags (6–18 months). |

## Canada

| Source | What | Access | Notes |
|---|---|---|---|
| **C.D. Howe Institute** Business Cycle Council | Canadian recession chronology (NBER analogue) | `https://cdhowe.org/council/business-cycle-council/` | Encoded in `data/manual/cdhowe_recessions.csv`, source URL per row. 2008-11→2009-05; COVID 2020-03→2020-04; no 2022/2023 recession. |
| **Statistics Canada** WDS | Monthly real GDP by industry (`36-10-0434-01`; real-time `36-10-0491-01`), LFS unemployment (`14-10-0287-01`), CPI (`18-10-0004-01`) | `https://www150.statcan.gc.ca/t1/wds/` REST | Keyless; ~2-month GDP lag. |
| **Bank of Canada** Valet | GoC yields / spreads | `https://www.bankofcanada.ca/valet/` | Keyless JSON/CSV. |

## Labels

- **US:** `USREC` monthly 0/1 (NBER-based recession indicator).
- **Canada:** `data/manual/cdhowe_recessions.csv` — peak/trough rows with
  `granularity` (month/quarter), `label`, and `source_url`.

## Reproducibility

- `SEED = 20260615` (config.yaml), recorded in every results JSON.
- Raw snapshots cached under `data/raw/<source>/<YYYY-MM-DD>/` (gitignored);
  committed artefacts under `data/processed/`, `data/facts/`, `data/manual/`.
- `data_mode` (`live` / `demo-synthetic`) is stamped on all outputs.
