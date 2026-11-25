PYTHON ?= python

all: cohort sim real edgar robust eda figures sanity report explainer

.PHONY: all cohort sim real edgar robust eda figures sanity report explainer test clean

## Simulation and calibration study (about 1-2 hours on 9 workers) -> data/processed/sim_runs.pkl.gz + sim_* tables
sim:
	$(PYTHON) -m src.simulate.run
	$(PYTHON) -m src.simulate.report --tables

## Cohort data from Screener.in (uses the cached pages in data/raw/cohort/html; network only if a page is missing)
cohort:
	$(PYTHON) -m src.ingest.screener

## Real case studies (+ cohort if data/raw/cohort/cohort.csv exists) and the hypotheses table
real:
	$(PYTHON) -m src.run_jyoti --tables
	$(PYTHON) -m src.run_nmc --tables
	$(PYTHON) -m src.run_layers --tables
	$(PYTHON) -m src.run_real

## EDGAR external-transfer panel: download (network, ~15 min) then analyse
edgar:
	$(PYTHON) -m src.ingest.edgar
	$(PYTHON) -m src.run_edgar --tables

## All figures -> reports/figures
figures:
	$(PYTHON) -m src.run_jyoti --figures
	$(PYTHON) -m src.run_nmc --figures
	$(PYTHON) -m src.run_layers --figures
	$(PYTHON) -m src.simulate.report --figures
	$(PYTHON) -m src.run_edgar --figures

## Robustness / sensitivity analyses (EDGAR variants re-run in parallel, ~5-35 min) -> reports/tables/robust_*.csv
robust:
	$(PYTHON) -m src.robustness

## Design-choice EDA (training quarters only, descriptive) -> reports/tables/eda_*.csv
eda:
	$(PYTHON) -m src.eda

## Sanity and common-sense checks (incl. the test suite) -> reports/tables/sanity_checks.csv
sanity:
	$(PYTHON) -m src.sanity

## Report (every number read from reports/tables) -> reports/report.md
report:
	$(PYTHON) -m src.reporting.write_report
	$(PYTHON) -m src.reporting.write_short_report

## Plain-language explainer -> reports/explainer.html and reports/explainer.pdf
explainer:
	$(PYTHON) -m src.reporting.write_explainer

test:
	$(PYTHON) -m pytest

clean:
	$(PYTHON) -c "import pathlib; [p.unlink() for d in ('reports/tables','reports/figures') for p in pathlib.Path(d).glob('*') if p.name != '.gitkeep']"
