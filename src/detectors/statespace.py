"""Forecast models that turn a ratio series into standardised one-step innovations z_t.

Layer 3 model (blueprint s.7): local level + AR(1) noise (statsmodels UnobservedComponents), fitted on
the training window, then the FULL series is filtered with the parameters FROZEN. Missing values are
handled natively by the Kalman filter (z_t = NaN there).
Baselines: ARIMA(1,0,0) with constant, and a naive random walk.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

BURN = 2  # first innovations under diffuse initialisation carry no information


def _uc(y):
    return UnobservedComponents(np.asarray(y, float), level="local level", autoregressive=1)


def fit_uc(y_train) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_uc(y_train).fit(disp=False, maxiter=200).params)


def uc_innovations(y, params) -> np.ndarray:
    y = np.asarray(y, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _uc(y).filter(params)
    z = np.array(res.filter_results.standardized_forecasts_error[0], dtype=float)
    z[np.isnan(y)] = np.nan
    z[:BURN] = np.nan
    return z


def simulate_uc(params, T: int, rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw `size` series of length T from a fitted local level + AR(1) model. Returns (size, T)."""
    s_irr, s_lvl, s_ar, phi = (float(v) for v in params)
    phi = float(np.clip(phi, -0.99, 0.99))
    lvl = np.cumsum(rng.normal(0, np.sqrt(max(s_lvl, 0)), (size, T)), axis=1)
    e = rng.normal(0, np.sqrt(max(s_ar, 0)), (size, T))
    ar = np.empty((size, T)); ar[:, 0] = e[:, 0] / np.sqrt(max(1 - phi ** 2, 1e-6))
    for t in range(1, T):
        ar[:, t] = phi * ar[:, t - 1] + e[:, t]
    return lvl + ar + rng.normal(0, np.sqrt(max(s_irr, 0)), (size, T))


def _arima(y):
    return SARIMAX(np.asarray(y, float), order=(1, 0, 0), trend="c")


def fit_arima(y_train) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_arima(y_train).fit(disp=False).params)


def arima_innovations(y, params) -> np.ndarray:
    y = np.asarray(y, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _arima(y).filter(params)
    z = np.array(res.filter_results.standardized_forecasts_error[0], dtype=float)
    z[np.isnan(y)] = np.nan
    z[:1] = np.nan
    return z


def rw_innovations(y, n_train: int) -> np.ndarray:
    """Naive random-walk forecast y_hat_t = y_{t-1}; errors scaled by the training SD of first differences."""
    y = np.asarray(y, float)
    d = np.r_[np.nan, np.diff(y)]
    sd = np.nanstd(d[1:n_train], ddof=1)
    return d / sd if sd > 0 else d * np.nan


def kalman_z(y, n_train: int):
    """Fit on y[:n_train], filter all of y. Returns (z, params)."""
    p = fit_uc(np.asarray(y, float)[:n_train])
    return uc_innovations(y, p), p


def multivariate_z(Y, n_train: int):
    """Layer 4 input: per-ratio Kalman innovations (T, p) and Ledoit-Wolf Sigma^-1 from training innovations."""
    Y = np.asarray(Y, float)
    Z = np.column_stack([kalman_z(Y[:, j], n_train)[0] for j in range(Y.shape[1])])
    return Z, ledoit_wolf_inv(Z[:n_train])


def ledoit_wolf_inv(Ztrain) -> np.ndarray:
    Zt = Ztrain[~np.isnan(Ztrain).any(axis=1)]
    return np.linalg.inv(LedoitWolf().fit(Zt).covariance_)
