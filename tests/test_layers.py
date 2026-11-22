"""Layer 1 (event time) and Layer 2 (rules) tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Paths, load_config
from src.detectors import event_time, layer2_rules
from src.run_layers import firm_results

CFG = load_config()


@pytest.fixture(scope="module")
def out_paths(tmp_path_factory):
    return Paths.from_config(CFG, out_root=tmp_path_factory.mktemp("layers")).ensure()


def test_event_fy_alignment():
    assert event_time.event_fy("2017-07-04", 3) == 2018      # Indian FY Apr-Mar
    assert event_time.event_fy("2017-03-31", 3) == 2017
    assert event_time.event_fy("2020-04-09", 12) == 2020
    assert event_time.event_fraction("2017-04-01", 3) == pytest.approx(0, abs=.01)
    assert event_time.event_fraction("2020-07-01", 12) == pytest.approx(.5, abs=.01)


def test_panel_window_and_gaps(out_paths):
    res = firm_results("jyoti", CFG, out_paths)
    p = res["panel"]
    assert list(p.index) == list(range(CFG["layer1"]["window"][0], CFG["layer1"]["window"][1] + 1))
    assert list(p.fy) == [2013, 2014, 2015, 2016, 2017, 2018]
    assert p.loc[-1].drop("fy").isna().all()                     # FY2017 gap stays a gap


def test_nmc_panel_has_no_data_after_fy2018(out_paths):
    res = firm_results("nmc", CFG, out_paths)
    assert res["efy"] == 2020 and list(res["panel"].fy) == list(range(2015, 2021))
    assert res["panel"].loc[[-1, 0]].drop(columns="fy").isna().all().all()
    assert {2019, 2020} <= res["missing"]


def test_jyoti_first_trips_and_lead(out_paths):
    first = firm_results("jyoti", CFG, out_paths)["first"]
    t = CFG["layer2"]["rules"]
    for rule in [f"OCF < {t['ocf_max']:g}", f"EBITDA / interest < {t['ebitda_interest_cover_min']:g}",
                 f"Debtor days > {t['debtor_days_max']:g}", "Any rule"]:
        assert first.loc[rule, "first_trip_fy"] == 2014
        assert first.loc[rule, "lead_fy"] == 4
        assert first.loc[rule, "lead_years"] == pytest.approx((pd.Timestamp("2017-07-04") - pd.Timestamp("2014-03-31")).days / 365.25, abs=.01)


def test_nmc_no_rule_trips(out_paths):
    first = firm_results("nmc", CFG, out_paths)["first"]
    assert first.first_trip_fy.isna().all()
    assert (first.pre_event_years_evaluable == 6).all()


def test_interest_rule_ignores_zero_interest():
    idx = pd.Index([1, 2, 3], name="fy")
    canon = pd.DataFrame({"ocf": [1., 1., 1.], "ebitda": [-5., 5., 1.], "interest": [0., 10., 1.]}, index=idx)
    ratios = pd.DataFrame({"neg_dso": [-10., -10., -10.], "altman_z2_em": [9., 9., 9.]}, index=idx)
    F = layer2_rules.rule_flags(canon, ratios, CFG)
    col = [c for c in F if c.startswith("EBITDA")][0]
    assert F[col].tolist() == [0.0, 1.0, 1.0]


def test_control_band():
    a = pd.DataFrame({"x": [1., 2., 3.]}, index=pd.Index([-2, -1, 0], name="tau"))
    b = pd.DataFrame({"x": [3., 4., np.nan]}, index=a.index)
    band = event_time.control_band([a, b], ["x"], (0.25, 0.75))["x"]
    assert band.loc[-2, "median"] == 2 and band.loc[0, "n"] == 1
    assert event_time.control_band([], ["x"], (0.25, 0.75)) is None
