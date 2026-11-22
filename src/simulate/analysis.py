"""Turn stored simulation runs into the report tables (reports/tables/sim_*.csv) and the H3-H5 verdicts.

Definitions (pre-registered):
- healthy false alarm: any alarm in the 16 monitored periods of scenario H (H4 for the concealed target)
- detection: first alarm at or after onset (monitored period `onset_in_monitor`) with no earlier alarm
- all rates are over held-out firms; train firms are used only to fit the inflation factor and the
  conformal threshold
- CIs: percentile bootstrap over firms (simulation.ci_resamples), paired across scenarios
"""
from __future__ import annotations

import gzip
import pickle

import numpy as np
import pandas as pd

from src.calibration import thresholds as th
from src.masking.index import FINGERPRINTS, fingerprint

ALPHAS = (0.05, 0.10)
UNI = ["arima", "kalman", "rw"]


def load(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


class Runs:
    def __init__(self, runs, cfg):
        self.cfg, self.sc = cfg, cfg["simulation"]
        self.onset = self.sc["onset_in_monitor"]
        self.rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 777]))
        self.errors = [r for r in runs if "error" in r]
        self.by = {}
        for r in runs:
            if "error" in r:
                continue
            self.by.setdefault((r["set"], r["n"]), []).append(r)

    def firms(self, s, n):
        return self.by[(s, n)]

    def maxs(self, s, n, scen, det):
        return np.array([r["paths"][(scen, det)].max() for r in self.firms(s, n)])

    def h(self, s, n, key):
        return np.array([r[key] for r in self.firms(s, n)], float)

    def first(self, s, n, scen, det, h):
        """First alarm index per firm (np.inf if none). h scalar or per-firm array."""
        P = np.vstack([r["paths"][(scen, det)] for r in self.firms(s, n)])
        hh = np.broadcast_to(np.asarray(h, float).reshape(-1, 1) if np.ndim(h) else h, P.shape)
        hit = P > hh
        return np.where(hit.any(axis=1), hit.argmax(axis=1), np.inf)

    def detected(self, s, n, scen, det, h):
        f = self.first(s, n, scen, det, h)
        return (f >= self.onset) & np.isfinite(f)

    def ci(self, x):
        x = np.asarray(x, float); m = len(x)
        bs = x[self.rng.integers(0, m, (self.sc["ci_resamples"], m))].mean(axis=1)
        return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def _row(**kw):
    return kw


# ------------------------------------------------------------------ Experiment A: calibration curve
def calibration_curve(R: Runs) -> pd.DataFrame:
    rows = []
    for n in R.sc["training_lengths"]:
        for a in ALPHAS:
            for det, key, scen in [("kalman_cusum", "kalman", "H"), ("mcusum", "mcusum", "H")]:
                hh = R.h("heldout", n, f"h_{key}_{a:.2f}"); ok = np.isfinite(hh)
                fa = R.maxs("heldout", n, scen, det if det == "mcusum" else "kalman")[ok] > hh[ok]
                m, lo, hi = R.ci(fa)
                rows.append(_row(n=n, nominal=a, detector=det, empirical_fa=m, ci_lo=lo, ci_hi=hi,
                                 excess_points=100 * (m - a), firms=len(fa), excluded_undefined_h=int((~ok).sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fixes (H5) with detection cost
def fixes(R: Runs):
    ns = R.sc["training_lengths"]; rows = []; infl = []
    for a in ALPHAS:
        c_req = []
        for n in ns:
            ht = R.h("train", n, f"h_kalman_{a:.2f}"); ok = np.isfinite(ht) & (ht > 0)   # h = 0: degenerate fit
            c_req.append(th.required_inflation(R.maxs("train", n, "H", "kalman")[ok], ht[ok], a))
        ab = th.fit_inflation(ns, c_req)
        for n, c in zip(ns, c_req):
            ht = R.h("train", n, f"h_kalman_{a:.2f}")
            infl.append(_row(nominal=a, n=n, c_required=c, c_fitted=th.inflation_factor(n, ab), a=ab[0], b=ab[1],
                             train_firms_used=int((np.isfinite(ht) & (ht > 0)).sum()), train_firms_degenerate_h=int((ht <= 0).sum())))
        for n in ns:
            hb = R.h("heldout", n, f"h_kalman_{a:.2f}")
            methods = {"block_bootstrap": hb, "inflation_factor": hb * th.inflation_factor(n, ab),
                       "conformal": np.full(len(hb), th.conformal_h(R.maxs("train", n, "H", "kalman"), a))}
            if f"h_parametric_{a:.2f}" in R.firms("heldout", n)[0]:
                methods["parametric_bootstrap"] = R.h("heldout", n, f"h_parametric_{a:.2f}")
            mh = R.maxs("heldout", n, "H", "kalman")
            ok = np.isfinite(hb) & ~np.any([np.isnan(v) for v in methods.values()], axis=0)   # same firms for every method
            base_det = R.detected("heldout", n, "S1", "kalman", hb)[ok].mean()
            for meth, h in methods.items():
                m, lo, hi = R.ci(mh[ok] > h[ok])
                det = R.detected("heldout", n, "S1", "kalman", h)[ok]
                dm, dlo, dhi = R.ci(det)
                rows.append(_row(n=n, nominal=a, method=meth, empirical_fa=m, ci_lo=lo, ci_hi=hi,
                                 within_3_points=abs(m - a) <= 0.03, detection_S1=dm, det_ci_lo=dlo, det_ci_hi=dhi,
                                 detection_cost_points=100 * (base_det - dm), median_h=float(np.median(h[ok])), firms=int(ok.sum()),
                                 degenerate_h0_firms=int((hb <= 0).sum())))
    return pd.DataFrame(rows), pd.DataFrame(infl)


# ------------------------------------------------------------------ matched false-alarm comparison
def _matched_h(R, n, det, ref, fa):
    return float(np.quantile(R.maxs("heldout", n, ref, det), 1 - fa))


def matched_comparison(R: Runs, n=None, fa=0.10) -> pd.DataFrame:
    n = n or R.sc["h5_training_length"]; rows = []
    for scen in ["S1", "S2", "S3", "S4"]:
        ref = "H4" if scen == "S4" else "H"
        for det in UNI + ["mcusum"]:
            h = _matched_h(R, n, det, ref, fa)
            m, lo, hi = R.ci(R.detected("heldout", n, scen, det, h))
            rows.append(_row(n=n, matched_fa=fa, scenario=scen, detector=det, h_matched=h,
                             detection=m, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def power_curves(R: Runs, fa=0.10):
    n28 = R.sc["h5_training_length"]; drift, by_n = [], []
    for d in R.sc["power_drift_grid"]:
        for scen in ["S1", "S2"]:
            for det in UNI + ["mcusum"]:
                h = _matched_h(R, n28, det, "H", fa)
                m, lo, hi = R.ci(R.detected("heldout", n28, f"{scen}@{d:g}", det, h))
                drift.append(_row(n=n28, scenario=scen, drift=d, detector=det, detection=m, ci_lo=lo, ci_hi=hi))
    for n in R.sc["training_lengths"]:
        for scen in ["S1", "S2", "S4"]:
            ref = "H4" if scen == "S4" else "H"
            for det in ["kalman", "mcusum"]:
                h = _matched_h(R, n, det, ref, fa)
                m, lo, hi = R.ci(R.detected("heldout", n, scen, det, h))
                by_n.append(_row(n=n, scenario=scen, detector=det, detection=m, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(drift), pd.DataFrame(by_n)


# ------------------------------------------------------------------ masking index in simulation
def masking(R: Runs, a=0.10):
    n = R.sc["h5_training_length"]; F = R.firms("heldout", n)
    hs = R.h("heldout", n, f"h_struct_{a:.2f}")
    peak, fp = {}, []
    for scen in ["H", "S1", "S2", "S3", "S4"]:
        hc = R.h("heldout", n, f"h_kalman{'_c' if scen == 'S4' else ''}_{a:.2f}")
        Sc = np.vstack([r["paths"][(scen, "kalman")] for r in F]); Ss = np.vstack([r["paths"][(scen, "struct")] for r in F])
        MI = Ss / hs[:, None] - Sc / hc[:, None]
        peak[scen] = MI.max(axis=1)
        tc, ts = R.first("heldout", n, scen, "kalman", hc), R.first("heldout", n, scen, "struct", hs)
        for c, s_ in zip(tc, ts):
            fp.append({"scenario": scen, "fingerprint": fingerprint(None if np.isinf(c) else int(c), None if np.isinf(s_) else int(s_))})
    summ = pd.DataFrame([_row(scenario=s, n=n, mean_peak_mi=float(np.nanmean(v)), sd_peak_mi=float(np.nanstd(v, ddof=1)),
                              median_peak_mi=float(np.nanmedian(v)), firms=int(np.isfinite(v).sum())) for s, v in peak.items()])
    fpt = (pd.DataFrame(fp).value_counts().unstack(fill_value=0).reindex(columns=FINGERPRINTS, fill_value=0))
    fpt = fpt.div(fpt.sum(axis=1), axis=0).reset_index()
    sep = []
    for lo_s, hi_s in R.cfg["masking"]["separation_pairs"]:
        a_, b_ = peak[lo_s.split("_")[0]], peak[hi_s.split("_")[0]]
        keep = np.isfinite(a_) & np.isfinite(b_); a_, b_ = a_[keep], b_[keep]
        d, auc = _effects(a_, b_)
        idx = R.rng.integers(0, len(a_), (R.sc["ci_resamples"], len(a_)))
        bs = np.array([_effects(a_[i], b_[i]) for i in idx])
        sep.append(_row(low=lo_s, high=hi_s, n=n, cohens_d=d, d_ci_lo=float(np.quantile(bs[:, 0], .025)),
                        d_ci_hi=float(np.quantile(bs[:, 0], .975)), auc=auc,
                        auc_ci_lo=float(np.quantile(bs[:, 1], .025)), auc_ci_hi=float(np.quantile(bs[:, 1], .975))))
    return summ, fpt, pd.DataFrame(sep), peak


def _effects(a, b):
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
    allv = np.r_[a, b]; ranks = pd.Series(allv).rank().to_numpy()
    auc = (ranks[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))
    return float(d), float(auc)


# ------------------------------------------------------------------ hypothesis verdicts H3-H5
def verdicts(cal, fix, matched, cfg) -> pd.DataFrame:
    n28 = cfg["simulation"]["h5_training_length"]; tol = cfg["simulation"]["h5_tolerance_points"] / 100
    out = []
    c = cal[(cal.n == n28) & (cal.detector == "kalman_cusum")]
    over = (c.ci_lo > c.nominal)
    ev4 = "; ".join(f"nominal {r.nominal:.0%}: empirical {r.empirical_fa:.1%} [{r.ci_lo:.1%}, {r.ci_hi:.1%}]" for r in c.itertuples())
    out.append(dict(id="H4", verdict="SUPPORTED" if over.all() else ("REFUTED" if (c.empirical_fa <= c.nominal).all() else "MIXED"),
                    evidence=f"n={n28}, Kalman+CUSUM, block-bootstrap calibrator: {ev4}", source="sim_calibration_curve.csv"))
    f = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    hits = [m for m in ("inflation_factor", "conformal") if m in f.index and abs(f.loc[m, "empirical_fa"] - 0.10) <= tol]
    ev5 = "; ".join(f"{m}: {f.loc[m, 'empirical_fa']:.1%} [{f.loc[m, 'ci_lo']:.1%}, {f.loc[m, 'ci_hi']:.1%}]" for m in f.index)
    out.append(dict(id="H5", verdict="SUPPORTED" if hits else "REFUTED",
                    evidence=f"n={n28}, nominal 10%, held-out firms: {ev5}. Pre-registered fixes within +/-3 points: "
                             f"{', '.join(hits) or 'none'}" + ("; the parametric bootstrap (added, not pre-registered) is also within"
                             if "parametric_bootstrap" in f.index and abs(f.loc["parametric_bootstrap", "empirical_fa"] - 0.10) <= tol else ""),
                    source="sim_fixes.csv"))
    m = matched.set_index(["scenario", "detector"]).detection
    s1, s2, s4 = (m[(s, "mcusum")] - m[(s, "kalman")] for s in ("S1", "S2", "S4"))
    pattern = (s1 < 0) and (s2 > 0) and (s4 > 0)
    opposite = (s1 > 0) and (s2 < 0) and (s4 < 0)
    out.append(dict(id="H3", verdict="SUPPORTED" if pattern else ("REFUTED" if opposite else "MIXED"),
                    evidence=f"matched 10% FA, n={n28}, multivariate minus Kalman detection: S1 {100*s1:+.1f} pts, "
                             f"S2 {100*s2:+.1f} pts, S4 {100*s4:+.1f} pts",
                    source="sim_matched_fa.csv"))
    return pd.DataFrame(out)
