"""NMC Health annual red-flag metrics (refactored from nmc_eda.py).

Values are in native units/sign (NOT yet oriented lower = worse); see CLAUDE.md.
"""
from __future__ import annotations

import pandas as pd

ZSCORE_COLS = ["ocf_to_cl", "ocf_margin", "current_ratio", "ccc", "dso", "interest_cov", "netdebt_ebitda", "altman_z2"]


def redflag_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=raw.index)
    x["rev_growth"] = raw.revenue.pct_change()
    x["recv_growth"] = raw.receivables.pct_change()
    x["recv_minus_rev_growth"] = x.recv_growth - x.rev_growth
    x["debt_growth"] = raw.total_debt.pct_change()
    x["ebitda_growth"] = raw.ebitda.pct_change()
    x["ocf_growth"] = raw.ocf_ytd.pct_change()
    x["ocf_to_ebitda"] = raw.ocf_ytd / raw.ebitda
    x["fcf_to_debt"] = (raw.ocf_ytd - raw.capex_ytd) / raw.total_debt
    x["liquid_to_st_debt"] = (raw.cash + raw.bank_deposits) / raw.st_debt
    x["st_debt_share"] = raw.st_debt / raw.total_debt
    x["goodwill_proxy_equity"] = raw.equity / raw.total_assets
    x["net_debt_to_equity"] = (raw.total_debt - raw.cash - raw.bank_deposits) / raw.equity
    return x


def ratio_summary(r: pd.DataFrame) -> pd.DataFrame:
    return r.describe().T[["mean", "std", "min", "max"]]


def baseline_zscores(r: pd.DataFrame, baseline_end: str) -> pd.DataFrame:
    base = r.loc[:baseline_end]
    return ((r - base.mean()) / base.std())[ZSCORE_COLS]


def correlations_with(r: pd.DataFrame, target: str = "ocf_to_cl") -> pd.Series:
    return r.corr()[target].drop(target).sort_values()
