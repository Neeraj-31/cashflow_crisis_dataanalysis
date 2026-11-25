# Early Warning of Cash-Flow Crises

A layered, self-referential monitor: each firm is judged against its own history. Course project
(Foundations of Data Science). Case studies NMC Health and Jyoti Structures, an India cohort, a US EDGAR panel
and a simulation study.

## Start here

| If you want to… | Open |
|---|---|
| Understand the whole project in plain words | `reports/explainer.html` (or `.pdf`) |
| Read the short report (5 pages) | `reports/report_short.pdf` (or `.md` / `.html`) |
| Read the full write-up | `reports/report.md` |
| Present it | `reports/slides.html` (arrow keys to navigate) |
| See what was promised before the results | `docs/PREREGISTRATION.md` |
| See why each judgement call was made | `docs/decision_log.md` |
| Change a threshold, window or seed | `configs/config.yaml` (the only place settings live) |

## What is where

```
configs/config.yaml        every setting: seed, paths, thresholds, windows (pre-registered blocks + dated addenda)

data/
  raw/                     inputs as obtained, never edited by code
    nmc_*.csv              NMC Health annual accounts
    jyoti_screener.csv     Jyoti Structures annual accounts
    cohort/                India cohort: firm list, per-firm CSVs, control matching, cached Screener pages
    edgar/                 US SEC EDGAR: bankruptcy filings list, XBRL facts per firm
  processed/               derived data and run logs (simulation runs, EDGAR robustness runs)

src/                       all code (run as python -m src.<module>)
  config.py                load_config(), rng(), paths
  ingest/                  raw data -> standard columns (EDGAR, Screener, Jyoti)
  ratios/                  Layer 0: oriented ratios + accounting-identity checks
  detectors/               Layers 1-4: event time, rule flags, Kalman model, CUSUM / MCUSUM
  calibration/             Layer 5: bootstrap thresholds and the three corrections
  masking/                 Layer 6: masking index and archetype rule
  simulate/                simulation study (H3-H5)
  run_nmc.py, run_jyoti.py case-study EDA      run_layers.py  Layers 1-2 for both cases
  run_real.py              hypotheses H1-H6    run_edgar.py   full method on the US panel
  robustness.py            sensitivity checks  post_hoc.py    post-hoc review checks
  eda.py                   design-choice EDA   evaluation.py  shared bootstrap CIs
  sanity.py                automated sanity checks (incl. "every reported number traces to a table")
  reporting/               builds report.md, explainer.html/.pdf, slides.html and the figures

reports/                   everything generated; nothing here is edited by hand
  report.md, explainer.*, slides.html
  figures/                 all PNG figures
  tables/                  every number used anywhere, grouped by file prefix:
    sim_*                    simulation (H3, H4, H5, masking index)
    nmc_*, jyoti_*, ratios_*, layer*  case studies (legacy EDA, Layers 0-2)
    real_*, hypotheses.csv   archetypes and the H1-H6 verdicts
    cohort_*                 India cohort
    edgar_*                  US EDGAR panel results
    robust_*                 robustness checks
    posthoc_*                post-hoc review checks (exploratory)
    eda_*                    design-choice EDA
    sanity_checks.csv        result of python -m src.sanity

docs/
  PREREGISTRATION.md       hypotheses, rules, and the deviations log (s.10)
  decision_log.md          every analysis choice with its reason
  PREREG_FINGERPRINT.txt, prereg_snapshot/   proof the frozen settings were not changed

tests/                     pytest suite; legacy/ + golden/ prove the refactor reproduces the original scripts exactly
notebooks/                 empty (all analysis is in src/)
```

## Rebuild

`make all` runs the full pipeline in order. Without `make` (Windows), run the steps in this order:

```
python -m src.run_jyoti --tables ; python -m src.run_nmc --tables ; python -m src.run_layers --tables
python -m src.simulate.run ; python -m src.simulate.report      # 1-2 hours
python -m src.run_real
python -m src.ingest.edgar ; python -m src.run_edgar             # download needs network
python -m src.robustness ; python -m src.post_hoc ; python -m src.eda
python -m src.sanity                                              # also runs the tests
python -m src.reporting.write_report ; python -m src.reporting.write_short_report ; python -m src.reporting.write_explainer ; python -m src.reporting.write_slides
```

Rules: settings only in `configs/config.yaml`; randomness only via `src.config.rng()`; no number is typed into a report
(every one is read from `reports/tables/`).
