"""Self-starting CUSUM benchmark for the H4 / H5 setting (ADDED POST HOC, 2026-11-22; config `post_hoc.self_starting`).

    python -m src.simulate.self_starting

Short-history comparator in the spirit of Hawkins (1987) self-starting CUSUM and Quesenberry (1991) Q-charts:
each new value is standardised by the mean / SD (or AR(1) regression) estimated from ALL values before it, the
resulting Student-t statistic is mapped to N(0,1) through its exact distribution under the model, and a one-sided
CUSUM (same k, same direction) runs on the result. Because the transformed values are N(0,1) under the model, the
threshold h needs no estimated parameter: h is the (1 - alpha) quantile of the 16-step CUSUM maximum of iid N(0,1)
noise (Monte Carlo). Two versions:
  q_iid    Q-statistics assuming iid normal values (the textbook self-starting chart)
  q_ar1    recursive (Brown-Durbin-Evans) residuals of an AR(1) regression x_t = a + phi x_{t-1} + e_t, studentised
The held-out simulated firms are regenerated exactly from their seeds (checked against the stored Kalman paths), so
rates are on the same firms as sim_fixes.csv. Detection uses the pre-registered definition (first alarm at or after
onset). Output: reports/tables/posthoc_sim_self_starting.csv
"""
from __future__ import annotations

import gzip
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from src import evaluation as ev
from src.config import Paths, load_config, rng
from src.detectors import cusum as cs
from src.detectors import statespace as ss
from src.simulate import generator as gen
from src.simulate.experiment import firm_rng

ALPHAS = (0.05, 0.10)


def q_iid(X: np.ndarray, start: int) -> np.ndarray:
    """X (F, T): rows are firms. Returns U (F, T - start)."""
    out = []
    for t in range(start, X.shape[1]):
        past = X[:, :t]; m = past.shape[1]
        T = (X[:, t] - past.mean(axis=1)) / (past.std(axis=1, ddof=1) * np.sqrt(1 + 1 / m))
        out.append(stats.norm.ppf(stats.t.cdf(T, df=m - 1)))
    return np.column_stack(out)


def q_ar1(X: np.ndarray, start: int) -> np.ndarray:
    out = []
    for t in range(start, X.shape[1]):
        y, x = X[:, 1:t], X[:, :t - 1]; m = y.shape[1]                       # pairs (x_{j-1}, x_j), j < t
        Z = np.stack([np.ones_like(x), x], axis=2)                          # (F, m, 2)
        G = np.einsum("fmi,fmj->fij", Z, Z); beta = np.linalg.solve(G, np.einsum("fmi,fm->fi", Z, y)[..., None])[..., 0]
        rss = ((y - np.einsum("fmi,fi->fm", Z, beta)) ** 2).sum(axis=1); s = np.sqrt(rss / (m - 2))
        z0 = np.stack([np.ones(len(X)), X[:, t - 1]], axis=1)
        v = 1 + np.einsum("fi,fij,fj->f", z0, np.linalg.inv(G), z0)
        T = (X[:, t] - (z0 * beta).sum(axis=1)) / (s * np.sqrt(v))
        out.append(stats.norm.ppf(stats.t.cdf(T, df=m - 2)))
    return np.column_stack(out)


def main():
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    sc, dc, ph = cfg["simulation"], cfg["detectors"], cfg["post_hoc"]
    k, H = dc["cusum"]["k"], sc["monitor_length"]; onset = sc["onset_in_monitor"]
    rg = rng(ph["seed_offsets"]["self_starting"])
    null = cs.cusum_paths(rg.standard_normal((ph["self_starting"]["null_paths"], H)), k).max(axis=1)
    h = {a: float(np.quantile(null, 1 - a)) for a in ALPHAS}
    with gzip.open(paths.processed / "sim_runs.pkl.gz", "rb") as fh:
        runs = pickle.load(fh)["runs"]
    stored = {(r["n"], r["firm"]): r for r in runs if r.get("set") == "heldout" and "error" not in r}
    rows, checked = [], 0
    for n in sc["training_lengths"]:
        firms = sorted(f for (nn, f) in stored if nn == n)
        Xs = {"H": [], "S1": []}
        for f in firms:
            g = firm_rng(cfg["seed"], "heldout", n, f, sc["seed_offset_heldout"])
            X = gen.base_series(g, n + H, sc["n_ratios"], sc["ar_phi"], sc["innovation_corr"])
            if f < 3:   # the regenerated firm must be the stored firm: its Kalman CUSUM path must match exactly
                path = cs.cusum(ss.uc_innovations(X[:, 0], ss.fit_uc(X[:n, 0]))[n:], k)
                assert np.allclose(path, stored[(n, f)]["paths"][("H", "kalman")]), (n, f)
                checked += 1
            for s in Xs:
                Xs[s].append(gen.scenario(X, s, n + onset, sc["drift_sd_per_period"] if s == "S1" else 0.0,
                                          sc["concealment_smoothing"])[:, 0])
        for meth, fn in [("self-starting Q-chart CUSUM (iid)", q_iid), ("self-starting recursive-residual CUSUM (AR(1))", q_ar1)]:
            S = {s: cs.cusum_paths(fn(np.vstack(v), n), k) for s, v in Xs.items()}
            for a in ALPHAS:
                fa = (S["H"].max(axis=1) > h[a]).astype(float)
                hit = S["S1"] > h[a]; first = np.where(hit.any(axis=1), hit.argmax(axis=1), np.inf)
                det = (np.isfinite(first) & (first >= onset)).astype(float)
                m, lo, hi = ev.rate_ci(fa, rg, sc["ci_resamples"]); dm, dlo, dhi = ev.rate_ci(det, rg, sc["ci_resamples"])
                rows.append(dict(n=n, nominal=a, method=meth, empirical_fa=m, ci_lo=lo, ci_hi=hi,
                                 within_tolerance=abs(m - a) <= sc["h5_tolerance_points"] / 100, detection_S1=dm,
                                 det_ci_lo=dlo, det_ci_hi=dhi, h=h[a], firms=len(firms)))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_sim_self_starting.csv", index=False)
    print(f"regenerated firms checked against stored paths: {checked}")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
