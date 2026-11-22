# Raw data

- `nmc_raw.csv`, `nmc_ratios.csv` – NMC Health annual financials and precomputed ratios, FY2013–FY2018 (US$ '000).
- `jyoti_screener.csv` – Jyoti Structures consolidated financials, INR crore, from Screener.in
  (vendor-standardised). FY2017 is not published on Screener and is absent. FY2013 has cash-flow items
  only, from a secondary source (cross-checked: CFO+CFI+CFF = −21 = Screener's FY2013 net cash flow).
  Values were transcribed into the original `jyoti_build_full.py` and extracted here verbatim.
  `src/ingest/jyoti_screener.py` checks the balance-sheet, P&L and CCC identities (±3 cr).
