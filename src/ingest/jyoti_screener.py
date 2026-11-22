"""Jyoti Structures full-lifecycle dataset (consolidated, INR crore) from Screener.in (vendor-standardised).

Refactored from jyoti_build_full.py. The transcribed figures now live in data/raw/jyoti_screener.csv.
- FY2017 is NOT published on the Screener page -> left as a gap.
- FY2013 has cash-flow items only (secondary source, cross-checked: CFO+CFI+CFF = -21 = Screener's
  FY2013 net cash flow).
"""
from __future__ import annotations

import pandas as pd

from src.config import Paths

RAW_FILE = "jyoti_screener.csv"
OUT_FILE = "jyoti_full.csv"
YEARS = range(2013, 2027)
TOL = 3  # INR crore rounding tolerance on accounting identities (data-validation, not a model threshold)


def check_identities(d: pd.DataFrame) -> None:
    chk = d.dropna(subset=["total_assets"])
    assert ((chk.equity_capital + chk.reserves + chk.borrowings + chk.other_liabilities - chk.total_assets).abs() <= TOL).all()
    assert ((chk.op_profit + chk.other_income - chk.interest - chk.depreciation - chk.pbt).abs() <= TOL).all()
    assert (chk.debtor_days + chk.inventory_days - chk.payable_days - chk.ccc_days).dropna().abs().le(TOL).all()


def build(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.reindex(YEARS)
    d.index.name = "fy"
    check_identities(d)
    d["noncurrent_assets"] = d.total_assets - d.other_assets  # fixed assets + CWIP + investments
    return d


def load_raw(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.raw / RAW_FILE, index_col="fy")


def main(paths: Paths) -> pd.DataFrame:
    d = build(load_raw(paths))
    d.to_csv(paths.processed / OUT_FILE)
    print(f"saved {OUT_FILE}; balance sheet, P&L and CCC identities all reconcile")
    return d
