"""Ratio engine tests on the real NMC and Jyoti numbers in data/raw."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Paths, load_config
from src.ingest import canonical, jyoti_screener
from src.ratios import ORIENTATION, IdentityError, check_identities, compute_ratios, safe_div

CFG = load_config()
PATHS = Paths.from_config(CFG)


@pytest.fixture(scope="module")
def nmc():
    df = canonical.nmc(PATHS, CFG["firms"]["nmc"]["fy_end_month"])
    return df, compute_ratios(df, CFG)


@pytest.fixture(scope="module")
def jyoti(tmp_path_factory):
    out = Paths.from_config(CFG, out_root=tmp_path_factory.mktemp("j")).ensure()
    jyoti_screener.main(out)
    df = canonical.jyoti(out, CFG["firms"]["jyoti"]["fy_end_month"])
    return df, compute_ratios(df, CFG)


# ------------------------------------------------------------------ NMC: cross-check vs independently computed ratios
def test_nmc_matches_precomputed_ratios(nmc):
    _, r = nmc
    ref = pd.read_csv(PATHS.raw / "nmc_ratios.csv", index_col=0, parse_dates=True)
    ref.index = ref.index.year
    pairs = {"ocf_to_revenue": ("ocf_margin", 1), "ocf_to_current_liabilities": ("ocf_to_cl", 1),
             "ocf_to_total_debt": ("cf_to_debt", 1), "current_ratio": ("current_ratio", 1),
             "quick_ratio": ("quick_ratio", 1), "cash_ratio": ("cash_ratio", 1), "neg_dso": ("dso", -1),
             "neg_dio": ("dio", -1), "neg_dpo": ("dpo", -1), "neg_ccc": ("ccc", -1),
             "ebit_interest_cover": ("interest_cov", 1), "neg_debt_to_equity": ("debt_to_equity", -1)}
    for mine, (theirs, sign) in pairs.items():
        np.testing.assert_allclose(r[mine], sign * ref[theirs], rtol=1e-9, err_msg=mine)


def test_nmc_altman_em_is_z2_plus_constant(nmc):
    _, r = nmc
    ref = pd.read_csv(PATHS.raw / "nmc_ratios.csv", index_col=0, parse_dates=True)
    np.testing.assert_allclose(r.altman_z2_em.values, ref.altman_z2.values + CFG["altman_z2_em"]["constant"], atol=1e-3)


def test_nmc_liquidity_cover_matches_legacy_redflags(nmc):
    _, r = nmc
    g = pd.read_csv("tests/golden/nmc_redflags.csv", index_col=0, parse_dates=True)
    np.testing.assert_allclose(r.liquid_to_st_debt.values, g.liquid_to_st_debt.values, rtol=1e-12)
    np.testing.assert_allclose(r.neg_debt_growth.values[1:], -g.debt_growth.values[1:], rtol=1e-12)


def test_nmc_identities_hold(nmc):
    df, _ = nmc
    check_identities(df, CFG["identities"], "nmc")


# ------------------------------------------------------------------ Jyoti
def test_jyoti_fy2017_missing_not_interpolated(jyoti):
    _, r = jyoti
    assert bool(r.loc[2017, "missing_filing"]) is True
    assert r.loc[[y for y in r.index if y != 2017], "missing_filing"].sum() == 0
    num = r.drop(columns=["fy_end", "missing_filing", "anomaly_ocf_gt_revenue"]).loc[2017]
    assert num.isna().all()
    # growth across the gap must not bridge FY2016 -> FY2018
    assert np.isnan(r.loc[2018, "revenue_growth"]) and np.isnan(r.loc[2018, "neg_debt_growth"])


def test_jyoti_fy2018_ocf_anomaly(jyoti):
    df, r = jyoti
    assert (df.loc[2018, "ocf"], df.loc[2018, "revenue"]) == (1190, 324)
    assert r.anomaly_ocf_gt_revenue.fillna(False).tolist() == [y == 2018 for y in r.index]


def test_jyoti_matches_legacy_metrics(jyoti):
    _, r = jyoti
    g = pd.read_csv("tests/golden/jyoti_metrics_full.csv", index_col=0)
    np.testing.assert_allclose(r.ocf_to_total_debt, g.cfo_to_borrowings, atol=6e-4)
    np.testing.assert_allclose(r.neg_sloan_accruals, -g.sloan_accruals, atol=6e-4)
    both = g.ebitda_interest_cover.notna() & r.ebitda_interest_cover.notna()
    np.testing.assert_allclose(r.ebitda_interest_cover[both], g.ebitda_interest_cover[both], atol=6e-4)
    burn = r.neg_financing_to_burn.notna() & g.external_funding_share_of_burn.notna()
    np.testing.assert_allclose(r.neg_financing_to_burn[burn], -g.external_funding_share_of_burn[burn], atol=6e-4)


def test_jyoti_near_zero_denominators_masked(jyoti):
    df, r = jyoti
    eps = CFG["ratios"]["denominator_eps_rel_assets"]["activity"]
    tiny = df.revenue.abs() <= eps * df.total_assets.abs()      # FY2020 (16), FY2021 (0), FY2022 (4)
    assert set(df.index[tiny]) == {2020, 2021, 2022}
    for c in ["ocf_to_revenue", "neg_dso", "neg_ccc"]:
        assert r.loc[tiny[tiny].index, c].isna().all(), c
    assert np.isfinite(r.select_dtypes("number").to_numpy(dtype=float)[~np.isnan(r.select_dtypes("number").to_numpy(dtype=float))]).all()
    assert r.loc[2016, "neg_debt_to_equity"] != r.loc[2016, "neg_debt_to_equity"]  # equity < 0 => masked


def test_jyoti_identities_hold(jyoti):
    df, _ = jyoti
    check_identities(df, CFG["identities"], "jyoti")


@pytest.mark.parametrize("col,msg", [("total_assets", "assets = liabilities + equity"),
                                     ("pbt", "PBT"), ("ccc", "DSO + DIO - DPO = CCC")])
def test_identity_failure_names_row_and_column(jyoti, col, msg):
    df, _ = jyoti
    bad = df.copy(); bad.loc[2015, col] += 500
    with pytest.raises(IdentityError) as e:
        check_identities(bad, CFG["identities"], "jyoti")
    s = str(e.value)
    assert "fy=2015" in s and col in s and msg in s and "[jyoti]" in s


# ------------------------------------------------------------------ orientation & helpers
def test_orientation_registry_complete(nmc):
    _, r = nmc
    for name, (_, _, sign) in ORIENTATION.items():
        assert name in r.columns
        assert (sign < 0) == name.startswith("neg_"), name


def test_orientation_lower_is_worse_on_synthetic():
    """Make a firm strictly worse on every dimension; every oriented ratio must not increase."""
    base = dict(revenue=1000., cogs=600., ebitda=200., ebit=150., interest=20., net_income=100., ocf=150.,
                capex=50., cff=-10., cash=100., bank_deposits=50., receivables=200., inventory=100.,
                payables=80., current_assets=500., current_liabilities=250., total_assets=2000.,
                total_liabilities=1000., total_debt=600., st_debt=200., equity=1000., retained_earnings=500.)
    worse = dict(base, ebitda=120., ebit=80., interest=40., net_income=110., ocf=20., cash=40., bank_deposits=10.,
                 receivables=400., inventory=180., payables=200., current_assets=560., current_liabilities=450.,
                 total_liabilities=1400., total_debt=1000., st_debt=500., equity=600., retained_earnings=200.,
                 cff=150., capex=200.)
    df = pd.DataFrame([base, base, worse], index=pd.Index([2020, 2021, 2022], name="fy"))
    r = compute_ratios(df, CFG)
    skip = {"revenue_growth", "neg_receivables_minus_revenue_growth", "neg_debt_growth", "neg_financing_to_burn"}
    for c in ORIENTATION:
        if c in skip or np.isnan(r.loc[2022, c]):
            continue
        assert r.loc[2022, c] <= r.loc[2021, c] + 1e-12, c
    assert r.loc[2022, "neg_debt_growth"] < 0 and r.loc[2022, "neg_receivables_minus_revenue_growth"] < 0


def test_safe_div_masks_near_zero():
    out = safe_div([1, 1, 1, 1], [0.0, 0.5, 5.0, -5.0], [100] * 4, eps=0.01)
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == 0.2 and out[3] == -0.2
    assert np.isnan(safe_div([1], [-5.0], [100], eps=0.01, positive_den=True)[0])


def test_linear_fill_is_opt_in(jyoti):
    df, _ = jyoti
    cfg = {**CFG, "ratios": {**CFG["ratios"], "fill_missing": "linear"}}
    r = compute_ratios(df, cfg)
    assert np.isfinite(r.loc[2017, "ocf_to_total_debt"]) and bool(r.loc[2017, "missing_filing"])
