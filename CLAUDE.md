# Early Warning of Cash-Flow Crises: A Layered, Self-Referential Monitoring Framework

Course project. Case studies: NMC Health, Jyoti Structures; plus a validation cohort.

## Non-negotiable rules

1. **Global random seed is `20261121`.** It lives in `configs/config.yaml` (`seed`). Get randomness only via
   `src.config.rng()` (optionally with an `offset` for independent streams). Never call `np.random.seed`,
   `default_rng()` without a seed, or pass literal seeds.
2. **All thresholds come from `configs/config.yaml` only.** No threshold, cut-off, window, event date or
   band boundary may be a literal in `src/`. Read them through `src.config.load_config()`. Labels that
   mention a threshold (e.g. "Debtor days > 270") are generated from the config value.
3. **No number in any report may be hand-typed.** Every number that appears in the write-up/slides must
   come from a CSV written by a script into `reports/tables/*.csv` (figures go to `reports/figures/`).
   Figure annotations are computed from data, not typed. If you need a new number, add it to a table.
4. **Ratios are oriented so lower = worse.** Use `src.ratios.compute_ratios` (registry: `ORIENTATION`;
   higher-is-worse quantities are negated with a `neg_` prefix). Near-zero denominators are masked to NaN,
   gap years are never interpolated, and `check_identities` must pass before any analysis. The legacy EDA
   tables (`reports/tables/nmc_*`, `reports/tables/jyoti_*`) are in native units; do not feed them to
   detectors.
5. **Pre-registration.** `docs/PREREGISTRATION.md` (dated 2026-11-21) mirrors the pre-registered blocks of
   the config (`cohort`, `archetype`, `altman_z2_em`, `layer2`, `detectors`). Do not change those values
   without adding a row to its deviations log. NMC and Jyoti are exploratory; the cohort is confirmatory.

## Layout

```
configs/config.yaml        seed, paths, every threshold
data/raw/                  inputs as obtained (never edited by code)
data/processed/            derived datasets written by src/ingest
src/config.py              load_config(), rng(), Paths
src/ingest/                raw -> processed (jyoti_screener.py; a data-pull module would go here); canonical.py maps sources to one schema
src/ratios/                core.py = oriented ratio families + identity checks; nmc.py, jyoti.py = legacy EDA
src/detectors/             event_time.py (L1), layer2_rules.py (L2), statespace.py + cusum.py (L3/L4 models and
                           CUSUM/Crosier MCUSUM), monitor.py (L3-L5 on one real firm), rule_flags.py (legacy EDA)
src/calibration/           thresholds.py: block-bootstrap calibrator + fixes (inflation, parametric, conformal)
src/masking/               index.py (masking index, fingerprint), archetype.py (pre-registered rule)
src/simulate/              generator.py, experiment.py (per-firm worker), run.py (parallel), analysis.py, report.py
src/ingest/edgar.py        EDGAR US panel (exploratory external transfer; config `edgar`)
src/ingest/screener.py     Screener.in pages -> cohort data files; cohort_controls.py = mechanical control matching
data/raw/cohort/           cohort.csv, <firm>.csv, PROPOSAL.md (frame, rule, amendments), control_matching.csv, html/ cache
src/run_real.py            archetypes, Layers 3-5 where data allow, cohort, hypotheses.csv (H1-H6)
src/run_edgar.py           Layers 0-6 on the EDGAR panel
src/robustness.py          sensitivity analyses (config `robustness`) -> reports/tables/robust_*.csv
src/sanity.py              sanity / common-sense checks -> reports/tables/sanity_checks.csv
docs/prereg_snapshot/      exact copies of the fingerprinted config + pre-registration (hashes match)
src/reporting/             figure code + write_report.py (report.md) + write_explainer.py (explainer.html/.pdf)
src/run_nmc.py, run_jyoti.py   legacy EDA entry points (--tables / --figures / --out-root)
src/run_layers.py          Layer 1 + Layer 2 for all firms in config (--tables / --figures / --firms)
docs/PREREGISTRATION.md    pre-registration (H1-H6 verbatim from the blueprint; deviations log in s.10)
reports/tables/, reports/figures/   all report outputs (script-generated)
tests/                     test_reproduces_legacy.py + frozen legacy scripts and golden outputs
```

## Commands

- `make real` – tables for the case studies; `make figures` – all figures; `make sim` – simulation;
  `make all`; `make test`.
- Without `make` (e.g. Windows): `python -m src.run_jyoti --tables`, `python -m src.run_nmc --figures`,
  `python -m pytest`.

## Refactoring guard

`tests/test_reproduces_legacy.py` runs the frozen originals in `tests/legacy/` and the refactored code
side by side: CSVs must equal the goldens in `tests/golden/` exactly and figures must be pixel-identical.
Run it after any change to `src/`. If you *intend* to change an output, update the golden deliberately
and say so. Do not edit `tests/legacy/`.
