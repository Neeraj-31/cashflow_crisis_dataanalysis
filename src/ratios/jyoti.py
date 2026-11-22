"""Jyoti Structures derived metrics (refactored from jyoti_eda_full.py).

Values are in native units/sign (NOT yet oriented lower = worse); see CLAUDE.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PREFAILURE_COLS = ["op_margin", "ebitda_minus_cfo", "gap_to_sales", "cfo_to_ebitda", "ebitda_interest_cover",
                   "interest_to_sales", "implied_interest_rate", "net_worth", "debt_to_net_worth",
                   "borrowings_to_assets", "borrowings_growth", "external_funding_share_of_burn",
                   "other_assets_to_sales", "sloan_accruals", "debtor_days", "payable_days", "ccc_days"]
RECURRENCE_COLS = ["sales_growth", "op_margin", "cfo_to_sales", "sloan_accruals", "borrowings_growth", "debtor_days"]


def metrics(d: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame(index=d.index)
    sales = d.sales.where(d.sales > 0)
    m["sales_growth"] = d.sales.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    m["op_margin"] = d.op_profit / sales
    m["net_margin"] = d.net_profit / sales
    m["ebitda"] = d.op_profit
    m["cfo"] = d.cfo
    m["ebitda_minus_cfo"] = d.op_profit - d.cfo
    m["gap_to_sales"] = m.ebitda_minus_cfo / sales
    m["cfo_to_ebitda"] = d.cfo / d.op_profit.where(d.op_profit > 0)
    m["cfo_to_sales"] = d.cfo / sales
    m["cfo_to_borrowings"] = d.cfo / d.borrowings
    m["ebitda_interest_cover"] = d.op_profit / d.interest.where(d.interest > 0)
    m["interest_to_sales"] = d.interest / sales
    m["interest_growth"] = d.interest.pct_change(fill_method=None)
    m["avg_borrowings"] = (d.borrowings + d.borrowings.shift(1)) / 2
    m["implied_interest_rate"] = d.interest / m.avg_borrowings
    m["net_worth"] = d.equity_capital + d.reserves
    m["debt_to_net_worth"] = (d.borrowings / m.net_worth).where(m.net_worth > 0)
    m["borrowings_to_assets"] = d.borrowings / d.total_assets
    m["borrowings_growth"] = d.borrowings.pct_change(fill_method=None)
    m["other_liab_to_assets"] = d.other_liabilities / d.total_assets
    m["other_assets_to_sales"] = d.other_assets / sales
    m["fcf"] = d.cfo + d.cfi
    m["external_funding_share_of_burn"] = d.cff / (-m.fcf).where(m.fcf < 0)
    m["sloan_accruals"] = (d.net_profit - d.cfo) / d.total_assets
    for c in ["debtor_days", "inventory_days", "payable_days", "ccc_days"]:
        m[c] = d[c]
    return m
