"""Synthetic firms: p AR(1) ratios (oriented lower = worse) with equicorrelated innovations and a known onset.

Ratio 0 is the target (cash) ratio; ratios 1..p-1 are structural.
Scenarios (deterioration = subtracting a linear ramp from onset):
  H   healthy
  H4  healthy, target reported exponentially smoothed (reference for S4 at matched false-alarm rate)
  S1  drift in the target ratio
  S2  target healthy, other ratios drift
  S3  small drift in all ratios
  S4  concealment: all ratios drift, target reported exponentially smoothed over the whole series
"""
from __future__ import annotations

import numpy as np


def base_series(rng: np.random.Generator, T: int, p: int, phi: float, rho: float) -> np.ndarray:
    cov = np.full((p, p), rho) + (1 - rho) * np.eye(p)
    E = rng.multivariate_normal(np.zeros(p), cov, size=T)
    X = np.empty((T, p)); X[0] = E[0] / np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        X[t] = phi * X[t - 1] + E[t]
    return X


def ramp(T: int, t0: int, d: float) -> np.ndarray:
    return np.where(np.arange(T) >= t0, d * (np.arange(T) - t0 + 1), 0.0)


def ewma(x: np.ndarray, s: float) -> np.ndarray:
    out = np.empty_like(x); out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = s * x[t] + (1 - s) * out[t - 1]
    return out


def scenario(X: np.ndarray, name: str, t0: int, d: float, s_smooth: float) -> np.ndarray:
    T = len(X); r = ramp(T, t0, d); Y = X.copy()
    if name == "H":
        pass
    elif name == "H4":
        Y[:, 0] = ewma(X[:, 0], s_smooth)
    elif name == "S1":
        Y[:, 0] -= r
    elif name == "S2":
        Y[:, 1:] -= r[:, None]
    elif name == "S3":
        Y -= r[:, None]
    elif name == "S4":
        Y -= r[:, None]
        Y[:, 0] = ewma(Y[:, 0], s_smooth)
    else:
        raise ValueError(name)
    return Y
