"""Alarm-threshold calibration (blueprint s.7 Layer 5 and s.8).

Baseline calibrator (the one whose small-sample behaviour H4 measures):
  refit on the first 60% of the training window, filter the rest -> pseudo-out-of-sample innovations,
  centre the pool, circular block bootstrap monitoring paths of length H, h = (1-alpha) quantile of
  max_t S_t.
Fixes (H5): (a) inflation factor c(n) on h, (b) parametric bootstrap with refit, (c) conformal
threshold from calibration firms with finite-sample quantile ceil((m+1)(1-alpha))/m.
"""
from __future__ import annotations

import math

import numpy as np

from src.detectors import cusum as cs
from src.detectors import statespace as ss


MIN_POOL = 3


def block_length(n_train: int, min_block: int) -> int:
    return max(min_block, math.ceil(n_train ** (1 / 3)))


def circular_block_indices(m: int, H: int, b: int, B: int, rng: np.random.Generator) -> np.ndarray:
    nblk = math.ceil(H / b)
    starts = rng.integers(0, m, size=(B, nblk))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % m
    return idx.reshape(B, nblk * b)[:, :H]


def _quantiles(maxima: np.ndarray, alphas) -> dict:
    return {a: float(np.quantile(maxima, 1 - a)) for a in alphas}


# ------------------------------------------------------------------ pseudo-OOS pools
def pool_univariate(y_train, frac: float) -> np.ndarray:
    y_train = np.asarray(y_train, float)
    n1 = int(math.floor(frac * len(y_train)))
    z = ss.uc_innovations(y_train, ss.fit_uc(y_train[:n1]))[n1:]
    z = z[np.isfinite(z)]
    return z - z.mean() if len(z) else z


def pool_multivariate(Y_train, frac: float) -> np.ndarray:
    Y_train = np.asarray(Y_train, float)
    n1 = int(math.floor(frac * len(Y_train)))
    Z = np.column_stack([ss.uc_innovations(Y_train[:, j], ss.fit_uc(Y_train[:n1, j]))[n1:] for j in range(Y_train.shape[1])])
    Z = Z[np.isfinite(Z).all(axis=1)]
    return Z - Z.mean(axis=0) if len(Z) else Z


# ------------------------------------------------------------------ block bootstrap (baseline calibrator)
def bootstrap_h(pool, alphas, k, H, B, b, rng, Sinv=None) -> dict:
    """Univariate if pool is 1-D, multivariate (needs Sinv) if 2-D."""
    pool = np.asarray(pool, float)
    if len(pool) < MIN_POOL:                      # degenerate fit: threshold undefined, reported as missing
        return {a: float("nan") for a in alphas}
    idx = circular_block_indices(len(pool), H, b, B, rng)
    if pool.ndim == 1:
        mx = cs.cusum_paths(pool[idx], k).max(axis=1)
    else:
        mx = cs.mcusum_paths(pool[idx], Sinv, k).max(axis=1)
    return _quantiles(mx, alphas)


# ------------------------------------------------------------------ fix (b): parametric bootstrap with refit
def parametric_h(params, n_train, alphas, k, H, B, rng) -> dict:
    """Simulate B series from the fitted model, REFIT each on its training part, filter its monitoring
    part with the refitted parameters, take max CUSUM. Captures parameter-estimation error."""
    sims = ss.simulate_uc(params, n_train + H, rng, B)
    mx = np.empty(B)
    for i, y in enumerate(sims):
        z = ss.uc_innovations(y, ss.fit_uc(y[:n_train]))[n_train:]
        mx[i] = cs.cusum(z, k).max()
    return _quantiles(mx, alphas)


# ------------------------------------------------------------------ fix (a): inflation factor
def required_inflation(max_stats, h_boot, alpha) -> float:
    """c such that P(max S > c * h_boot) = alpha over a set of healthy calibration firms."""
    return float(np.quantile(np.asarray(max_stats) / np.asarray(h_boot), 1 - alpha))


def fit_inflation(ns, cs_) -> tuple[float, float]:
    """OLS of log c on 1/n: log c = a + b / n. Returns (a, b)."""
    X = np.column_stack([np.ones(len(ns)), 1 / np.asarray(ns, float)])
    a, b = np.linalg.lstsq(X, np.log(np.asarray(cs_, float)), rcond=None)[0]
    return float(a), float(b)


def inflation_factor(n, ab) -> float:
    return float(np.exp(ab[0] + ab[1] / n))


# ------------------------------------------------------------------ fix (c): conformal
def conformal_h(scores, alpha) -> float:
    """Finite-sample conformal quantile: the ceil((m+1)(1-alpha))-th smallest score (inf if > m)."""
    s = np.sort(np.asarray(scores, float)); m = len(s)
    r = math.ceil((m + 1) * (1 - alpha))
    return float(s[r - 1]) if r <= m else math.inf
