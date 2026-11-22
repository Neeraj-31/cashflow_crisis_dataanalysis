"""Detector, calibrator and masking-index unit tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration import thresholds as th
from src.detectors import cusum as cs
from src.detectors import statespace as ss
from src.masking.index import fingerprint, masking_index


def test_cusum_recursion_by_hand():
    z = np.array([0.0, -1.0, -2.0, np.nan, 3.0])
    # k = 0.5: S = max(0, S - z - k)
    assert cs.cusum(z, 0.5).tolist() == [0.0, 0.5, 2.0, 2.0, 0.0]


def test_mcusum_one_ratio_equals_cusum_up_to_sign():
    rng = np.random.default_rng(0)
    z = rng.normal(-0.6, 1, 200)
    m = cs.mcusum(z[:, None], np.array([[1.0]]), 0.5)
    # Crosier is two-sided; on a series that keeps deteriorating it coincides with the lower CUSUM.
    # Exact identity holds on the absolute value whenever S never changes sign:
    neg = -np.abs(rng.normal(0, 1, 50)) - 0.1
    assert np.allclose(cs.mcusum(neg[:, None], np.eye(1), 0.5), cs.cusum(neg, 0.5))
    assert np.corrcoef(m, cs.cusum(z, 0.5))[0, 1] > 0.95


def test_vectorised_paths_match_loops():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(7, 16))
    assert np.allclose(cs.cusum_paths(Z, 0.5), np.vstack([cs.cusum(r, 0.5) for r in Z]))
    Z3 = rng.normal(size=(5, 16, 3)); Sinv = np.linalg.inv(np.eye(3) + 0.3)
    assert np.allclose(cs.mcusum_paths(Z3, Sinv, 0.5), np.vstack([cs.mcusum(r, Sinv, 0.5) for r in Z3]))


def test_no_drift_rarely_alarms_and_drift_does():
    rng = np.random.default_rng(2)
    h = 4.0
    healthy = [cs.cusum(rng.normal(size=16), 0.5).max() > h for _ in range(2000)]
    drift = [cs.cusum(rng.normal(size=16) - 1.0, 0.5).max() > h for _ in range(2000)]
    assert np.mean(healthy) < 0.05 and np.mean(drift) > 0.9


def test_kalman_innovations_standardised_and_handle_nan():
    rng = np.random.default_rng(3)
    y = np.zeros(60)
    for t in range(1, 60):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    y[40] = np.nan
    z, p = ss.kalman_z(y, 40)
    assert np.isnan(z[40]) and np.isnan(z[:ss.BURN]).all()
    assert 0.6 < np.nanstd(z[ss.BURN:40]) < 1.4


def test_circular_blocks_and_conformal():
    idx = th.circular_block_indices(10, 16, 4, 50, np.random.default_rng(4))
    assert idx.shape == (50, 16) and idx.max() < 10
    d = np.diff(idx[:, :4], axis=1) % 10
    assert (d == 1).all()                                       # contiguous (circular) within a block
    assert th.block_length(28, 2) == 4 and th.block_length(5, 2) == 2
    assert th.conformal_h(np.arange(1, 20), 0.1) == 18          # ceil(20*0.9)=18th smallest
    assert th.conformal_h(np.arange(1, 5), 0.1) == np.inf


def test_inflation_fit_recovers_curve():
    ns = np.array([12, 16, 20, 28, 40]); c = np.exp(0.1 + 3 / ns)
    ab = th.fit_inflation(ns, c)
    assert ab == pytest.approx((0.1, 3.0), abs=1e-9)


def test_masking_index_and_fingerprint():
    assert masking_index([2.0], 4.0, [1.0], 2.0)[0] == 0.0
    assert fingerprint(None, None) == "neither" and fingerprint(3, None) == "cash_only"
    assert fingerprint(5, 2) == "structural_first" and fingerprint(2, 2) == "both_together"
    assert fingerprint(1, 4) == "cash_first" and fingerprint(None, 4) == "structural_only"
