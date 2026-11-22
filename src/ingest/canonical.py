"""Map each source onto one canonical firm-year schema used by src.ratios.

Index: integer fiscal year `fy`. Every source column is optional; ratios needing an absent input are NaN.
Proxy columns are listed in `PROXIES` so reports can say which numbers rest on approximations.

Canonical columns
  revenue, cogs, ebitda, ebit, interest, net_income, depreciation, other_income, pbt
  ocf, capex (positive = spending), cfi, cff
  cash, bank_deposits, receivables, inventory, payables
  current_assets, current_liabilities, total_assets, total_liabilities, total_debt, st_debt
  equity, retained_earnings
  dso, dio, dpo, ccc            (vendor-supplied days; used only if components are absent)
  fy_end                         (Timestamp of fiscal year end)
"""
from __future__ import annotations

import pandas as pd

from src.config import Paths

PROXIES: dict[str, dict[str, str]] = {
    "nmc": {},
    "jyoti": {
        "current_assets": "Screener 'other assets' (all assets except fixed assets, CWIP, investments)",
        "current_liabilities": "Screener 'other liabilities' (excludes short-term borrowings => WC overstated)",
        "capex": "-CFI (net investing outflow)",
        "ebit": "PBT + interest",
        "retained_earnings": "reserves",
        "dso/dio/dpo/ccc": "Screener vendor days",
    },
}


def fy_end(fy: pd.Index, month: int) -> pd.DatetimeIndex:
    return pd.to_datetime([f"{y}-{month:02d}-01" for y in fy]) + pd.offsets.MonthEnd(0)


def nmc(paths: Paths, fy_end_month: int = 12) -> pd.DataFrame:
    raw = pd.read_csv(paths.raw / "nmc_raw.csv", parse_dates=["period_end"])
    out = pd.DataFrame({
        "revenue": raw.revenue, "cogs": raw.cogs, "ebitda": raw.ebitda, "ebit": raw.ebit,
        "interest": raw.interest_expense, "net_income": raw.net_income,
        "ocf": raw.ocf_ytd, "capex": raw.capex_ytd,
        "cash": raw.cash, "bank_deposits": raw.bank_deposits, "receivables": raw.receivables,
        "inventory": raw.inventory, "payables": raw.payables,
        "current_assets": raw.current_assets, "current_liabilities": raw.current_liabilities,
        "total_assets": raw.total_assets, "total_liabilities": raw.total_liabilities,
        "total_debt": raw.total_debt, "st_debt": raw.st_debt,
        "equity": raw.equity, "retained_earnings": raw.retained_earnings,
    })
    out.index = pd.Index(raw.period_end.dt.year, name="fy")
    out["fy_end"] = fy_end(out.index, fy_end_month)
    return out


def jyoti(paths: Paths, fy_end_month: int = 3) -> pd.DataFrame:
    """Reads data/processed/jyoti_full.csv (built by src.ingest.jyoti_screener). FY2017 row stays NaN."""
    d = pd.read_csv(paths.processed / "jyoti_full.csv", index_col=0)
    out = pd.DataFrame({
        "revenue": d.sales, "ebitda": d.op_profit, "ebit": d.pbt + d.interest,
        "interest": d.interest, "net_income": d.net_profit, "depreciation": d.depreciation,
        "other_income": d.other_income, "pbt": d.pbt,
        "ocf": d.cfo, "capex": -d.cfi, "cfi": d.cfi, "cff": d.cff,
        "current_assets": d.other_assets, "current_liabilities": d.other_liabilities,
        "total_assets": d.total_assets, "total_liabilities": d.borrowings + d.other_liabilities,
        "total_debt": d.borrowings, "equity": d.equity_capital + d.reserves, "retained_earnings": d.reserves,
        "dso": d.debtor_days, "dio": d.inventory_days, "dpo": d.payable_days, "ccc": d.ccc_days,
    }, index=pd.Index(d.index, name="fy"))
    out["fy_end"] = fy_end(out.index, fy_end_month)
    return out


def from_csv(path, fy_end_month: int) -> pd.DataFrame:
    """Generic loader for cohort firms: a CSV already in the canonical schema with an `fy` column
    (see data/raw/cohort/README.md). Unknown columns are ignored."""
    d = pd.read_csv(path).set_index("fy")
    d.index = d.index.astype(int)
    keep = [c for c in CANONICAL if c in d.columns]
    out = d[keep].astype(float)
    out["fy_end"] = fy_end(out.index, fy_end_month)
    return out


CANONICAL = ["revenue", "cogs", "ebitda", "ebit", "interest", "net_income", "depreciation", "other_income", "pbt",
             "ocf", "capex", "cfi", "cff", "cash", "bank_deposits", "receivables", "inventory", "payables",
             "current_assets", "current_liabilities", "total_assets", "total_liabilities", "total_debt", "st_debt",
             "equity", "retained_earnings", "dso", "dio", "dpo", "ccc"]

LOADERS = {"nmc": nmc, "jyoti": jyoti}
