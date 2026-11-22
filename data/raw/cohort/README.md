# Pre-registered validation cohort: how to add the data

`python -m src.run_real` picks this folder up automatically once `cohort.csv` exists. It runs Layers 0–2
on every firm, applies the archetype rule, and writes `reports/tables/cohort_*.csv`.

## 1. `cohort.csv`: one row per firm

| column | meaning |
|---|---|
| firm_id | short id, also the data file name (`<firm_id>.csv`) |
| name | company name |
| role | `failed`, `control` or `negative_control` |
| matched_to | for controls: the `firm_id` of the failed firm it is matched to |
| event_date | YYYY-MM-DD, verified against NCLT / IBBI / exchange filings (controls: their failed firm's date) |
| fy_end_month | 3 for an Indian April–March year, 12 for a calendar year |
| country, sector, size_band | as used for matching |
| source | where the numbers came from (e.g. Screener.in export, BSE annual report) |

The selection rule is in `configs/config.yaml` (`cohort`) and in `docs/PREREGISTRATION.md` §2. Choose
firms by that rule **before** opening their numbers. Jet Airways and Reliance Communications go in as
`negative_control`.

## 2. `<firm_id>.csv`: one row per fiscal year

The file needs a column `fy` (the fiscal year, e.g. 2016 for the year ending March 2016). Use any of
these canonical columns, in one currency unit per firm, as originally reported:

`revenue, cogs, ebitda, ebit, interest, net_income, depreciation, other_income, pbt, ocf, capex, cfi, cff,
cash, bank_deposits, receivables, inventory, payables, current_assets, current_liabilities, total_assets,
total_liabilities, total_debt, st_debt, equity, retained_earnings, dso, dio, dpo, ccc`

- Leave a cell blank if the number is not reported. Never estimate it.
- Accounting identities are checked before anything runs, and the run stops with the firm, year and
  column if one fails.
- Mapping from a Screener.in export: Sales→revenue, Operating Profit→ebitda, Interest→interest,
  Depreciation→depreciation, Other Income→other_income, Profit before tax→pbt, Net Profit→net_income,
  Cash from Operating Activity→ocf, Investing→cfi, Financing→cff, Borrowings→total_debt,
  Equity Capital + Reserves→equity, Reserves→retained_earnings, Total Assets→total_assets,
  Borrowings + Other Liabilities→total_liabilities, Other Assets→current_assets (proxy),
  Other Liabilities→current_liabilities (proxy), Debtor/Inventory/Payable Days and Cash Conversion
  Cycle→dso/dio/dpo/ccc.
