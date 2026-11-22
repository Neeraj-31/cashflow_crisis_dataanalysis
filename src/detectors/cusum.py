"""CUSUM statistics (blueprint s.7).

Univariate, one-sided toward deterioration (ratios oriented lower = worse):
    S_t = max(0, S_{t-1} - z_t - k), alarm when S_t > h.
Multivariate, Crosier (1988):
    C_t = sqrt((S_{t-1}+z_t)' Sigma^-1 (S_{t-1}+z_t));  S_t = 0 if C_t <= k else (S_{t-1}+z_t)(1 - k/C_t)
    monitored statistic: ||S_t|| = sqrt(S_t' Sigma^-1 S_t), alarm when > h.
Missing z_t (NaN) carries the statistic forward unchanged.
"""
from __future__ import annotations

import numpy as np


def cusum(z, k: float) -> np.ndarray:
    z = np.asarray(z, float)
    out = np.empty(len(z)); s = 0.0
    for t, v in enumerate(z):
        if not np.isnan(v):
            s = max(0.0, s - v - k)
        out[t] = s
    return out


def cusum_paths(Z, k: float) -> np.ndarray:
    """Vectorised over rows: Z (B, H) -> S (B, H). No NaN handling (bootstrap paths are complete)."""
    Z = np.asarray(Z, float)
    S = np.empty_like(Z); s = np.zeros(Z.shape[0])
    for t in range(Z.shape[1]):
        s = np.maximum(0.0, s - Z[:, t] - k); S[:, t] = s
    return S


def mcusum(Z, Sinv, k: float) -> np.ndarray:
    Z = np.asarray(Z, float)
    out = np.empty(len(Z)); s = np.zeros(Z.shape[1])
    for t, z in enumerate(Z):
        if not np.isnan(z).any():
            v = s + z
            c = float(np.sqrt(v @ Sinv @ v))
            s = np.zeros_like(v) if c <= k else v * (1 - k / c)
        out[t] = float(np.sqrt(s @ Sinv @ s))
    return out


def mcusum_paths(Z, Sinv, k: float) -> np.ndarray:
    """Z (B, H, p) -> ||S|| (B, H)."""
    Z = np.asarray(Z, float)
    B, H, p = Z.shape
    S = np.zeros((B, p)); out = np.empty((B, H))
    for t in range(H):
        V = S + Z[:, t, :]
        C = np.sqrt(np.einsum("bi,ij,bj->b", V, Sinv, V))
        f = np.where(C > k, 1 - k / np.where(C > 0, C, 1), 0.0)
        S = V * f[:, None]
        out[:, t] = np.sqrt(np.einsum("bi,ij,bj->b", S, Sinv, S))
    return out


def first_alarm(stat, h: float, start: int = 0):
    """Index (relative to `start`) of the first t >= start with stat_t > h, or None."""
    idx = np.flatnonzero(np.asarray(stat)[start:] > h)
    return int(idx[0]) if len(idx) else None
