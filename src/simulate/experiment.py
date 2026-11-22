"""Per-firm simulation worker. Everything stochastic is seeded from (global seed, set, n, firm).

For one synthetic firm and training length n it fits every model ONCE on the healthy training window
(drift only enters after training, except the concealed target in H4/S4, which gets its own fit),
computes all thresholds, and stores the monitored CUSUM paths for every scenario x detector, so all
downstream analyses (calibration curve, fixes, matched false-alarm comparison, power curves, masking
index) are computed from the same stored runs.
"""
from __future__ import annotations

import numpy as np

from src.calibration import thresholds as th
from src.detectors import cusum as cs
from src.detectors import statespace as ss
from src.simulate import generator as gen

SETS = {"train": 0, "heldout": 1}
ALPHAS = (0.05, 0.10)


def firm_rng(seed: int, set_name: str, n: int, firm: int, offset: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, offset, SETS[set_name], n, firm]))


def run_firm(args) -> dict:
    seed, set_name, n, firm, cfg = args
    sc, dc = cfg["simulation"], cfg["detectors"]
    k, H, B = dc["cusum"]["k"], sc["monitor_length"], dc["bootstrap"]["n_resamples"]
    frac = dc["training"]["train_fraction"]
    b = th.block_length(n, dc["bootstrap"]["min_block"])
    offset = sc["seed_offset_train"] if set_name == "train" else sc["seed_offset_heldout"]
    rng = firm_rng(seed, set_name, n, firm, offset)
    T, t0, p = n + H, n + sc["onset_in_monitor"], sc["n_ratios"]
    X = gen.base_series(rng, T, p, sc["ar_phi"], sc["innovation_corr"])
    s = sc["concealment_smoothing"]

    # ---- fits on the healthy training window (and on the concealed target for H4/S4)
    Yh, Yc = X, gen.scenario(X, "H4", t0, 0.0, s)
    th_par = [ss.fit_uc(Yh[:n, j]) for j in range(p)]
    th0_c = ss.fit_uc(Yc[:n, 0])
    ar_par, ar_par_c = ss.fit_arima(Yh[:n, 0]), ss.fit_arima(Yc[:n, 0])

    def zcols(Y, conceal):
        cols = [ss.uc_innovations(Y[:, 0], th0_c if conceal else th_par[0])]
        cols += [ss.uc_innovations(Y[:, j], th_par[j]) for j in range(1, p)]
        return np.column_stack(cols)

    Zh, Zc = zcols(Yh, False), zcols(Yc, True)
    Sinv_all, Sinv_all_c = ss.ledoit_wolf_inv(Zh[:n]), ss.ledoit_wolf_inv(Zc[:n])
    Sinv_st = ss.ledoit_wolf_inv(Zh[:n, 1:])

    # ---- baseline calibrator: pseudo-OOS pools + circular block bootstrap
    pool0, pool0_c = th.pool_univariate(Yh[:n, 0], frac), th.pool_univariate(Yc[:n, 0], frac)
    poolM, poolM_c = th.pool_multivariate(Yh[:n], frac), th.pool_multivariate(Yc[:n], frac)
    hb = lambda pool, Sinv=None: th.bootstrap_h(pool, ALPHAS, k, H, B, b, rng, Sinv)  # noqa: E731
    out = {"set": set_name, "n": n, "firm": firm, "block": b}
    for name, h in [("kalman", hb(pool0)), ("kalman_c", hb(pool0_c)), ("mcusum", hb(poolM, Sinv_all)),
                    ("mcusum_c", hb(poolM_c, Sinv_all_c)), ("struct", hb(poolM[:, 1:], Sinv_st))]:
        for a, v in h.items():
            out[f"h_{name}_{a:.2f}"] = v
    if set_name == "heldout" and cfg.get("_parametric_B", 0):
        for a, v in th.parametric_h(th_par[0], n, ALPHAS, k, H, cfg["_parametric_B"], rng).items():
            out[f"h_parametric_{a:.2f}"] = v

    # ---- monitored CUSUM paths for every scenario x detector
    paths = {}

    def monitor(tag, Y, conceal):
        Z = zcols(Y, conceal)
        Sinv = Sinv_all_c if conceal else Sinv_all
        y0 = Y[:, 0]
        paths[(tag, "kalman")] = cs.cusum(Z[n:, 0], k)
        paths[(tag, "arima")] = cs.cusum(ss.arima_innovations(y0, ar_par_c if conceal else ar_par)[n:], k)
        paths[(tag, "rw")] = cs.cusum(ss.rw_innovations(y0, n)[n:], k)
        paths[(tag, "mcusum")] = cs.mcusum(Z[n:], Sinv, k)
        paths[(tag, "struct")] = cs.mcusum(Z[n:, 1:], Sinv_st, k)

    d1, d3 = sc["drift_sd_per_period"], sc["small_drift_sd_per_period"]
    for tag, d in [("H", 0), ("H4", 0), ("S1", d1), ("S2", d1), ("S3", d3), ("S4", d1)]:
        monitor(tag, gen.scenario(X, tag, t0, d, s), tag in ("H4", "S4"))
    for d in sc["power_drift_grid"]:
        for tag in ("S1", "S2"):
            monitor(f"{tag}@{d:g}", gen.scenario(X, tag, t0, d, s), False)
    out["paths"] = paths
    return out
