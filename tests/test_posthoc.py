"""Checks for the shared evaluation helpers and the post-hoc self-starting statistics (added 2026-11-22)."""
import numpy as np
import pandas as pd

from src import evaluation as ev
from src.config import rng
from src.post_hoc import _first_run
from src.simulate.self_starting import q_ar1, q_iid


def test_rate_and_diff_ci_are_deterministic_and_bracket_the_estimate():
    a, b = np.r_[np.ones(30), np.zeros(10)], np.r_[np.ones(10), np.zeros(10)]
    r1, r2 = ev.rate_ci(a, rng(1), 500), ev.rate_ci(a, rng(1), 500)
    assert r1 == r2 and r1[1] <= r1[0] <= r1[2]
    d, lo, hi = ev.diff_ci(a, b, rng(2), 500)
    assert np.isclose(d, a.mean() - b.mean()) and lo <= d <= hi


def test_cluster_bootstrap_keeps_controls_with_their_failed_firm():
    f = pd.DataFrame({"cik": [1, 2, 3]}); c = pd.DataFrame({"cik": [10, 20], "matched_to": [1, 2]})
    Fi, Ci = ev.cluster_indices(f, c, rng(3), 50)
    for fi, ci in zip(Fi, Ci):
        # every drawn failed firm 1 or 2 brings exactly its own control (positions 0, 1); firm 3 has none
        assert sorted(ci.tolist()) == sorted([{0: 0, 1: 1}[i] for i in fi if i in (0, 1)])


def test_paired_difference_of_a_method_with_itself_is_zero():
    f = pd.DataFrame({"cik": range(20)}); c = pd.DataFrame({"cik": range(100, 110), "matched_to": range(10)})
    Fi, Ci = ev.cluster_indices(f, c, rng(4), 100)
    x = ev.paired_stats(np.arange(20) % 2, np.arange(10) % 3 == 0, Fi, Ci, 1, [5])
    assert np.allclose(x["expected_cost_cm5"] - x["expected_cost_cm5"], 0)


def test_self_starting_statistics_are_standard_normal_under_their_model():
    g = rng(5)
    X = g.standard_normal((4000, 30))
    U = q_iid(X, 10)
    assert abs(U.mean()) < 0.03 and abs(U.std() - 1) < 0.03
    Y = np.empty_like(X); Y[:, 0] = X[:, 0]
    for t in range(1, X.shape[1]):
        Y[:, t] = 0.5 * Y[:, t - 1] + X[:, t]
    V = q_ar1(Y, 12)
    assert abs(V.mean()) < 0.05 and abs(V.std() - 1) < 0.05


def test_first_run_needs_consecutive_quarters():
    assert _first_run(np.array([1, 0, 1, 1, 0], bool), 2) == 3
    assert _first_run(np.array([1, 0, 1, 0], bool), 2) is None
    assert _first_run(np.array([0, 1], bool), 1) == 1
