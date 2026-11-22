"""Oriented ratio families (lower = worse) on the canonical firm-year schema (src.ingest.canonical).

Conventions
- Every output ratio is oriented so that LOWER = WORSE. Higher-is-worse quantities are negated and
  carry a `neg_` prefix; `ORIENTATION` records the natural quantity and the sign applied.
- Near-zero denominators are masked to NaN, never divided: |den| <= eps * |total assets|, with eps from
  config `ratios.denominator_eps_rel_assets` (`activity` for revenue/COGS denominators, `default` otherwise). Ratios that are meaningless for a non-positive
  denominator (e.g. net debt / EBITDA with EBITDA < 0) are also masked ("sign-domain" mask).
- Gap years (no filing) stay NaN; growth rates need two consecutive fiscal years. Nothing is
  interpolated unless `ratios.fill_missing: linear`.
- DPO is oriented higher = worse (neg_dpo): stretched payables signal supplier financing under stress.
  This is deliberately the opposite of its sign inside the CCC, which is why CCC components are
  monitored separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 365

# oriented name -> (family, natural quantity, sign applied)
ORIENTATION: dict[str, tuple[str, str, int]] = {
    "ocf_to_revenue": ("cash_flow", "OCF / revenue", +1),
    "ocf_to_current_liabilities": ("cash_flow", "OCF / current liabilities", +1),
    "ocf_to_total_debt": ("cash_flow", "OCF / total debt", +1),
    "ocf_to_ebitda": ("cash_flow", "OCF / EBITDA (EBITDA > 0 only)", +1),
    "fcf_to_total_debt": ("cash_flow", "(OCF - capex) / total debt", +1),
    "fcf_to_assets": ("cash_flow", "(OCF - capex) / total assets", +1),
    "neg_dso": ("ccc", "debtor days (DSO)", -1),
    "neg_dio": ("ccc", "inventory days (DIO)", -1),
    "neg_dpo": ("ccc", "payable days (DPO)", -1),
    "neg_ccc": ("ccc", "cash conversion cycle (days)", -1),
    "current_ratio": ("liquidity", "current assets / current liabilities", +1),
    "quick_ratio": ("liquidity", "(current assets - inventory) / current liabilities", +1),
    "cash_ratio": ("liquidity", "cash / current liabilities", +1),
    "liquid_to_st_debt": ("liquidity", "(cash + deposits) / short-term debt", +1),
    "ebitda_interest_cover": ("leverage_coverage", "EBITDA / interest", +1),
    "ebit_interest_cover": ("leverage_coverage", "EBIT / interest", +1),
    "equity_to_assets": ("leverage_coverage", "book equity / total assets", +1),
    "neg_debt_to_assets": ("leverage_coverage", "total debt / total assets", -1),
    "neg_debt_to_equity": ("leverage_coverage", "total debt / equity (equity > 0 only)", -1),
    "neg_net_debt_to_ebitda": ("leverage_coverage", "net debt / EBITDA (EBITDA > 0 only)", -1),
    "neg_sloan_accruals": ("accruals", "(net income - OCF) / total assets", -1),
    "altman_z2_em": ("altman", "Altman Z''-EM score", +1),
    "revenue_growth": ("growth_funding", "revenue growth y/y", +1),
    "neg_debt_growth": ("growth_funding", "total debt growth y/y", -1),
    "neg_receivables_minus_revenue_growth": ("growth_funding", "receivables growth - revenue growth", -1),
    "neg_financing_to_burn": ("growth_funding", "CFF / cash burn (-(OCF - capex)), burn years only", -1),
}
FLAG_COLS = ["missing_filing", "anomaly_ocf_gt_revenue"]
SOURCE_COLS = ["revenue", "ebitda", "ebit", "net_income", "ocf", "total_assets", "total_debt", "equity"]


class IdentityError(AssertionError):
    pass


# ------------------------------------------------------------------ helpers
def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df else pd.Series(np.nan, index=df.index, dtype=float)


def safe_div(num, den, scale, eps: float, positive_den: bool = False) -> pd.Series:
    """num / den, NaN where den is missing, |den| <= eps*|scale| (or den == 0 if scale missing),
    or (if positive_den) den <= that bound."""
    num, den = pd.Series(num, dtype=float), pd.Series(den, dtype=float)
    bound = (eps * pd.Series(scale, dtype=float).abs()).fillna(0.0)
    bad = den.isna() | (den.abs() <= bound)
    if positive_den:
        bad |= den <= bound
    return (num / den.where(~bad)).astype(float)


def _growth(x: pd.Series, scale: pd.Series, eps: float) -> pd.Series:
    """y/y growth between consecutive fiscal years only (index must be consecutive integers)."""
    prev = x.shift(1)
    return safe_div(x - prev, prev, scale, eps, positive_den=True)


def _days(df, num, den, vendor, eps):  # eps: the activity epsilon
    """Days from components when reported; else vendor days, masked where revenue (their implicit
    denominator) is near zero, e.g. Jyoti FY2022 DSO 158,009 on sales of 4."""
    ta = _col(df, "total_assets")
    if num in df and den in df:
        return DAYS * safe_div(df[num], df[den], ta, eps, positive_den=True)
    rev_ok = safe_div(pd.Series(1.0, index=df.index), _col(df, "revenue"), ta, eps, positive_den=True).notna()
    return _col(df, vendor).where(rev_ok)


# ------------------------------------------------------------------ identities
def check_identities(df: pd.DataFrame, cfg: dict, firm: str = "") -> None:
    """Raise IdentityError naming firm, fiscal year (row) and columns if any accounting identity fails.
    Rows with a missing input are skipped for that identity."""
    rel, abs_, dtol = cfg["rel_tol"], cfg["abs_tol"], cfg["days_tol"]
    ta = _col(df, "total_assets").abs()
    tol = np.maximum(abs_, rel * ta.fillna(0))
    checks = []
    if {"total_assets", "total_liabilities", "equity"} <= set(df):
        checks.append(("assets = liabilities + equity", ["total_assets"], ["total_liabilities", "equity"],
                       df.total_assets - (df.total_liabilities + df.equity), tol))
    if {"ebitda", "other_income", "interest", "depreciation", "pbt"} <= set(df):
        checks.append(("EBITDA + other income - interest - depreciation = PBT",
                       ["ebitda", "other_income", "interest", "depreciation"], ["pbt"],
                       df.ebitda + df.other_income - df.interest - df.depreciation - df.pbt, tol))
    if {"ebitda", "ebit"} <= set(df) and "other_income" not in df:  # D&A >= 0 (only if EBIT excludes other income)
        checks.append(("EBITDA >= EBIT", ["ebitda"], ["ebit"], (df.ebit - df.ebitda).clip(lower=0), tol))
    if {"dso", "dio", "dpo", "ccc"} <= set(df):
        checks.append(("DSO + DIO - DPO = CCC", ["dso", "dio", "dpo"], ["ccc"],
                       df.dso + df.dio - df.dpo - df.ccc, pd.Series(dtol, index=df.index)))
    errors = []
    for name, lhs, rhs, diff, t in checks:
        bad = diff.notna() & (diff.abs() > t)
        for fy in df.index[bad]:
            vals = ", ".join(f"{c}={df.at[fy, c]:,.4g}" for c in lhs + rhs)
            errors.append(f"[{firm}] identity '{name}' fails at row fy={fy}: {vals} "
                          f"(diff {diff[fy]:,.4g}, tol {t[fy]:,.4g}; columns {lhs} vs {rhs})")
    if errors:
        raise IdentityError("\n".join(errors))


# ------------------------------------------------------------------ main
def compute_ratios(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """cfg is the full config dict (uses `ratios` and `altman_z2_em`)."""
    rc, az = cfg["ratios"], cfg["altman_z2_em"]
    eps_d = rc["denominator_eps_rel_assets"]
    eps, eps_act = eps_d["default"], eps_d["activity"]
    full = range(int(df.index.min()), int(df.index.max()) + 1)
    df = df.reindex(full)
    df.index.name = "fy"
    money = [c for c in df.columns if c != "fy_end"]
    missing = df[[c for c in SOURCE_COLS if c in df]].isna().all(axis=1)
    if rc["fill_missing"] == "linear":
        df[money] = df[money].interpolate(limit_area="inside")
    elif rc["fill_missing"] != "none":
        raise ValueError(f"ratios.fill_missing must be 'none' or 'linear', got {rc['fill_missing']!r}")

    g = lambda c: _col(df, c)  # noqa: E731
    ta = g("total_assets")
    div = lambda n, d, pos=False: safe_div(n, d, ta, eps, positive_den=pos)  # noqa: E731
    div_act = lambda n, d: safe_div(n, d, ta, eps_act, positive_den=True)  # noqa: E731

    fcf = g("ocf") - g("capex")
    net_debt = g("total_debt") - g("cash").fillna(0) - g("bank_deposits").fillna(0)
    net_debt = net_debt.where(g("cash").notna() | g("bank_deposits").notna())
    dso = _days(df, "receivables", "revenue", "dso", eps_act)
    dio = _days(df, "inventory", "cogs", "dio", eps_act)
    dpo = _days(df, "payables", "cogs", "dpo", eps_act)
    # computed from components when the firm reports them; otherwise the vendor CCC (if supplied)
    ccc = dso + dio - dpo if ("receivables" in df or "ccc" not in df) else _days(df, "-", "-", "ccc", eps_act)

    x1, x2, x3 = div(g("current_assets") - g("current_liabilities"), ta), div(g("retained_earnings"), ta), div(g("ebit"), ta)
    x4 = div(g("equity"), g("total_liabilities"))
    c = az["coef"]
    z = az["constant"] + c["wc_ta"] * x1 + c["re_ta"] * x2 + c["ebit_ta"] * x3 + c["bve_tl"] * x4

    rg = _growth(g("revenue"), ta, eps_act)
    burn = (-fcf).where(fcf < 0)

    r = pd.DataFrame(index=df.index)
    r["fy_end"] = df["fy_end"] if "fy_end" in df else pd.NaT
    r["ocf_to_revenue"] = div_act(g("ocf"), g("revenue"))
    r["ocf_to_current_liabilities"] = div(g("ocf"), g("current_liabilities"), True)
    r["ocf_to_total_debt"] = div(g("ocf"), g("total_debt"), True)
    r["ocf_to_ebitda"] = div(g("ocf"), g("ebitda"), True)
    r["fcf_to_total_debt"] = div(fcf, g("total_debt"), True)
    r["fcf_to_assets"] = div(fcf, ta, True)
    r["neg_dso"], r["neg_dio"], r["neg_dpo"], r["neg_ccc"] = -dso, -dio, -dpo, -ccc
    r["current_ratio"] = div(g("current_assets"), g("current_liabilities"), True)
    r["quick_ratio"] = div(g("current_assets") - g("inventory"), g("current_liabilities"), True)
    r["cash_ratio"] = div(g("cash"), g("current_liabilities"), True)
    r["liquid_to_st_debt"] = div(g("cash") + g("bank_deposits"), g("st_debt"), True)
    r["ebitda_interest_cover"] = div(g("ebitda"), g("interest"), True)
    r["ebit_interest_cover"] = div(g("ebit"), g("interest"), True)
    r["equity_to_assets"] = div(g("equity"), ta, True)
    r["neg_debt_to_assets"] = -div(g("total_debt"), ta, True)
    r["neg_debt_to_equity"] = -div(g("total_debt"), g("equity"), True)
    r["neg_net_debt_to_ebitda"] = -div(net_debt, g("ebitda"), True)
    r["neg_sloan_accruals"] = -div(g("net_income") - g("ocf"), ta, True)
    r["altman_x1_wc_ta"], r["altman_x2_re_ta"], r["altman_x3_ebit_ta"], r["altman_x4_bve_tl"] = x1, x2, x3, x4
    r["altman_z2_em"] = z
    r["revenue_growth"] = rg
    r["neg_debt_growth"] = -_growth(g("total_debt"), ta, eps)
    r["neg_receivables_minus_revenue_growth"] = -(_growth(g("receivables"), ta, eps) - rg)
    r["neg_financing_to_burn"] = -div(g("cff"), burn, True)
    r["missing_filing"] = missing.astype(bool)
    anomaly = g("ocf") > rc["anomaly"]["ocf_to_revenue_max"] * g("revenue")
    r["anomaly_ocf_gt_revenue"] = anomaly.astype("boolean").where(g("ocf").notna() & g("revenue").notna())
    return r


def altman_zone(z: pd.Series, az: dict) -> pd.Series:
    out = pd.Series(np.where(z < az["distress"], "distress", np.where(z > az["safe"], "safe", "grey")), index=z.index)
    return out.where(z.notna())
