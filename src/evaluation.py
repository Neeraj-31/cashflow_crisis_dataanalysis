"""Shared EDGAR evaluation helpers: ONE method registry, ONE analysed-sample rule and ONE set of bootstrap
functions, used by src.run_edgar (main table), src.robustness, src.post_hoc and src.sanity.

Why this module exists (decision log #34): the explainer and the sanity table used to quote different 95% CIs
for the same quantity because run_edgar and sanity each drew their own bootstrap with a different seed stream.
Now every CI is computed here, once, written to a table, and every other place reads that table.

Bootstrap designs
- rate_ci / diff_ci: independent percentile bootstrap of the failed group and the control group (the design of
  the main EDGAR table, edgar_detection_summary.csv).
- cluster_indices: matched-unit bootstrap for PAIRED comparisons of two methods (post hoc). A unit is one failed
  firm together with its analysed control (if any); the same resampled units are used for both methods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = {"amber (uncorrected)": "first_amber_uncorrected", "red (uncorrected)": "first_red_uncorrected",
           "amber (n-corrected)": "first_amber_corrected", "red (n-corrected)": "first_red_corrected",
           "Altman Z''-EM distress": "altman_alarm_q", "any Layer 2 rule": "rule_alarm_q"}


def ci_resamples(cfg) -> int:
    return int(cfg["evaluation"]["ci_resamples"])


def analysed(R: pd.DataFrame):
    """Analysed EDGAR sample: every failed firm whose run is ok, and every ok control whose failed firm is ok."""
    ok = R[R.status == "ok"]
    f = ok[ok.role == "failed"]
    return f, ok[(ok.role == "control") & ok.matched_to.isin(f.cik)]


def rate_ci(x, rg: np.random.Generator, B: int):
    x = np.asarray(x, float); m = len(x)
    if m == 0:
        return np.nan, np.nan, np.nan
    bs = x[rg.integers(0, m, (B, m))].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def diff_ci(a, b, rg: np.random.Generator, B: int):
    """mean(a) - mean(b) with an independent bootstrap of each group (draws interleaved a, b, a, b, ...)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [a[rg.integers(0, len(a), len(a))].mean() - b[rg.integers(0, len(b), len(b))].mean() for _ in range(B)]
    return float(a.mean() - b.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def auc(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def cluster_indices(f: pd.DataFrame, c: pd.DataFrame, rg: np.random.Generator, B: int):
    """Matched-unit bootstrap. Returns (Fi, Ci): for each of B resamples, the row positions of the failed firms
    and of the controls in the resample (a control enters whenever its failed firm's unit is drawn)."""
    ctrl_of = {m: i for i, m in enumerate(c.matched_to.to_numpy())}
    units = np.arange(len(f)); cik = f.cik.to_numpy()
    Fi, Ci = [], []
    for _ in range(B):
        u = units[rg.integers(0, len(units), len(units))]
        Fi.append(u)
        Ci.append(np.array([ctrl_of[cik[i]] for i in u if cik[i] in ctrl_of], int))
    return Fi, Ci


def paired_stats(det_f, fa_c, Fi, Ci, c_fa, c_miss):
    """Per-resample detection, false-alarm rate and expected costs for one method (arrays over resamples)."""
    det_f, fa_c = np.asarray(det_f, float), np.asarray(fa_c, float)
    d = np.array([det_f[i].mean() for i in Fi]); fa = np.array([fa_c[i].mean() if len(i) else np.nan for i in Ci])
    return {"detection": d, "false_alarm": fa, **{f"expected_cost_cm{cm}": c_fa * fa + cm * (1 - d) for cm in c_miss}}
