"""Layers 3-5 on one real firm: univariate cash detector, multivariate detector, structural detector,
block-bootstrap thresholds (uncorrected and n-corrected), alarm tiers and the masking index.

Runs only when the training window has >= detectors.min_periods_layer3 usable cash-ratio values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration import thresholds as th
from src.detectors import cusum as cs
from src.detectors import statespace as ss
from src.masking.index import fingerprint, masking_index

ALPHAS = (0.05, 0.10)


def usable_training_periods(ratios: pd.DataFrame, col: str, n_train: int) -> int:
    return int(ratios[col].iloc[:n_train].notna().sum())


def monitor_firm(ratios: pd.DataFrame, rules: pd.DataFrame, n_train: int, cfg: dict, rng: np.random.Generator,
                 inflation: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """ratios: oriented ratios indexed by period; rules: Layer 2 flags on the same index.
    inflation: {alpha: (a, b)} from reports/tables/sim_inflation_fit.csv, or None."""
    dc, mk = cfg["detectors"], cfg["masking"]
    k, H, B = dc["cusum"]["k"], dc["monitor_horizon"], dc["bootstrap"]["n_resamples"]
    frac, b = dc["training"]["train_fraction"], th.block_length(n_train, dc["bootstrap"]["min_block"])
    cash, struct = mk["cash_ratio"], mk["structural_ratios"]
    need = dc["min_periods_layer3"]
    # structural ratios enter the multivariate/structural detectors only with >= `need` usable training values
    avail = [c for c in struct if c != cash and ratios[c].iloc[:n_train].notna().sum() >= need]
    Y = ratios[[cash] + avail].to_numpy(float)
    nanh = {a: np.nan for a in ALPHAS}

    z_cash, _ = ss.kalman_z(Y[:, 0], n_train)
    h = {"cash": th.bootstrap_h(th.pool_univariate(Y[:n_train, 0], frac), ALPHAS, k, H, B, b, rng)}
    idx = ratios.index[n_train:]
    S_cash = cs.cusum(z_cash[n_train:], k)
    S_multi = S_struct = np.full(len(idx), np.nan); h["multi"] = h["struct"] = nanh
    if avail:
        Z = np.column_stack([z_cash] + [ss.kalman_z(Y[:, j], n_train)[0] for j in range(1, Y.shape[1])])
        poolM = th.pool_multivariate(Y[:n_train], frac)
        if (~np.isnan(Z[:n_train]).any(axis=1)).sum() >= 4 and len(poolM) >= 4:
            Sinv_m, Sinv_s = ss.ledoit_wolf_inv(Z[:n_train]), ss.ledoit_wolf_inv(Z[:n_train, 1:])
            h["multi"] = th.bootstrap_h(poolM, ALPHAS, k, H, B, b, rng, Sinv_m)
            h["struct"] = th.bootstrap_h(poolM[:, 1:], ALPHAS, k, H, B, b, rng, Sinv_s)
            S_multi, S_struct = cs.mcusum(Z[n_train:], Sinv_m, k), cs.mcusum(Z[n_train:, 1:], Sinv_s, k)
    # a threshold <= 0 comes from a degenerate fit (no innovation variance): undefined, never "alarm on anything"
    degenerate = sorted({d for d, hh in h.items() for v in hh.values() if not np.isnan(v) and v <= 0})
    h = {d: {a: (v if not np.isnan(v) and v > 0 else np.nan) for a, v in hh.items()} for d, hh in h.items()}
    c = {a: (th.inflation_factor(n_train, inflation[a]) if inflation else np.nan) for a in ALPHAS}
    rule_any = (rules.reindex(idx) == 1).any(axis=1).to_numpy()
    out = pd.DataFrame({"S_cash": S_cash, "S_multi": S_multi, "S_struct": S_struct, "rule_any": rule_any}, index=idx)
    for tag, mult in [("uncorrected", {a: 1.0 for a in ALPHAS}), ("corrected", c)]:
        amber = (S_cash > h["cash"][0.10] * mult[0.10]) | (S_multi > h["multi"][0.10] * mult[0.10])
        red = (S_multi > h["multi"][0.05] * mult[0.05]) & rule_any
        out[f"amber_{tag}"], out[f"red_{tag}"] = amber, red
    out["masking_index"] = masking_index(S_struct, h["struct"][0.10], S_cash, h["cash"][0.10])

    def first(mask):
        hits = np.flatnonzero(np.asarray(mask, bool))
        return int(hits[0]) if len(hits) else None

    t_cash = first(S_cash > h["cash"][0.10]) if not np.isnan(h["cash"][0.10]) else None
    t_struct = first(S_struct > h["struct"][0.10]) if avail and not np.isnan(h["struct"][0.10]) else None
    summ = {"n_train": n_train, "structural_ratios_used": "; ".join(avail) or "none",
            "degenerate_thresholds": "; ".join(degenerate) or "none", "block_length": b, "usable_training_periods": usable_training_periods(ratios, cash, n_train),
            **{f"h_{d}_{a:.2f}": v for d, hh in h.items() for a, v in hh.items()},
            **{f"inflation_{a:.2f}": c[a] for a in ALPHAS},
            "peak_masking_index": float(np.nanmax(out.masking_index)) if out.masking_index.notna().any() else np.nan,
            "fingerprint": fingerprint(t_cash, t_struct) if avail else "structural not available"}
    for tag in ("uncorrected", "corrected"):
        fa, fr = first(out[f"amber_{tag}"]), first(out[f"red_{tag}"])
        summ[f"first_amber_{tag}"] = idx[fa] if fa is not None else None
        summ[f"first_red_{tag}"] = idx[fr] if fr is not None else None
        summ[f"red_without_prior_amber_{tag}"] = bool(fr is not None and (fa is None or fr < fa))
    return out, summ


def load_inflation(tables_dir) -> dict | None:
    f = tables_dir / "sim_inflation_fit.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f).drop_duplicates("nominal")
    return {round(r.nominal, 2): (r.a, r.b) for r in d.itertuples()}
