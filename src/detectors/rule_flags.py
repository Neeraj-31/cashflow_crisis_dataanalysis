"""Rule-based early-warning flags (refactored from jyoti_eda_full.py).

All thresholds come from configs/config.yaml (jyoti.flags); flag labels are generated from them.
Each flag is 1.0 (tripped), 0.0 (not tripped) or NaN (inputs missing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fl(cond: pd.Series, valid: pd.Series, index) -> pd.Series:
    return pd.Series(np.where(valid, cond.astype(float), np.nan), index=index)


def jyoti_flags(d: pd.DataFrame, m: pd.DataFrame, t: dict) -> pd.DataFrame:
    i = d.index
    cover = (d.op_profit / d.interest.where(d.interest > 0)).fillna(np.inf)  # no interest => never tripped
    return pd.DataFrame({
        f"Operating cash flow < {t['cfo_max']:g}":
            _fl(d.cfo < t["cfo_max"], d.cfo.notna(), i),
        f"Cash conversion < {t['cash_conversion_min']:.0%} of EBITDA":
            _fl(d.cfo < t["cash_conversion_min"] * d.op_profit.clip(lower=0), d.cfo.notna() & d.op_profit.notna(), i),
        f"EBITDA / interest < {t['interest_cover_min']:g}":
            _fl(cover < t["interest_cover_min"], d.op_profit.notna(), i),
        f"Debtor days > {t['debtor_days_max']:g}":
            _fl(d.debtor_days > t["debtor_days_max"], d.debtor_days.notna(), i),
        f"Cash conversion cycle > {t['ccc_days_max']:g} days":
            _fl(d.ccc_days > t["ccc_days_max"], d.ccc_days.notna(), i),
        f"Borrowings growth > {t['borrowings_growth_max']:.0%}":
            _fl(m.borrowings_growth > t["borrowings_growth_max"], m.borrowings_growth.notna(), i),
        f"Sales fall > {t['sales_fall_max']:.0%}":
            _fl(m.sales_growth < -t["sales_fall_max"], m.sales_growth.notna(), i),
        f"Net worth < {t['net_worth_min']:g}":
            _fl(m.net_worth < t["net_worth_min"], m.net_worth.notna(), i),
    })


def first_trip_year(F: pd.DataFrame) -> pd.Series:
    """First index value at which each flag equals 1 (NaN if never)."""
    return pd.Series({c: next((y for y in F.index if F.loc[y, c] == 1), np.nan) for c in F.columns}, name="first_year")


def flag_counts(F: pd.DataFrame) -> pd.Series:
    return F.sum(axis=1, min_count=1).rename("flags_tripped")
