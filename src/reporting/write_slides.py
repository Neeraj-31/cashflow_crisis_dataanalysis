"""Presentation deck -> reports/slides.html (full page) and, with --fragment PATH, a body fragment for publishing.

    python -m src.reporting.write_slides [--fragment PATH]

Like the report and the explainer, every number on a slide is read from reports/tables/*.csv or configs/config.yaml at
build time; none is typed (CLAUDE.md rule 3; src.sanity traces slides.html too). The two explanatory animations (how a
CUSUM alarms; how an adaptive level absorbs a slow decline) are drawn from a deterministic, labelled illustration
series, not from data, and carry no numbers. Arrow keys / space / click to navigate.
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Paths, load_config

cfg = load_config(); P = Paths.from_config(cfg)
W, H = 1280, 720


def T(name):
    return pd.read_csv(P.tables / f"{name}.csv")


def pct(x, d=0):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def e(x):
    return html.escape(str(x))


# ------------------------------------------------------------------ charts (inline SVG, tokens only)
def svg(w, h, body, label):
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{e(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def bars_h4(cal, nominal):
    g = cal[(cal.detector == "kalman_cusum") & (cal.nominal == nominal)].sort_values("n")
    w, h, l, r, t, b = 620, 340, 40, 20, 34, 46
    top = float(g.ci_hi.max()) * 1.12
    y = lambda v: t + (h - t - b) * (1 - v / top)  # noqa: E731
    bw = (w - l - r) / len(g)
    out = [f'<line class="axis" x1="{l}" x2="{w - r}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>']
    for i, r_ in enumerate(g.itertuples()):
        x = l + i * bw + bw * 0.22; ww = bw * 0.56
        out.append(f'<g class="grow" style="--d:{0.15 + 0.12 * i:.2f}s"><rect class="m-coral" x="{x:.1f}" y="{y(r_.empirical_fa):.1f}" '
                   f'width="{ww:.1f}" height="{y(0) - y(r_.empirical_fa):.1f}" rx="4"><title>n = {r_.n}: {pct(r_.empirical_fa, 1)} actual '
                   f'(95% CI {pct(r_.ci_lo, 1)}–{pct(r_.ci_hi, 1)})</title></rect></g>')
        out.append(f'<line class="ci" x1="{x + ww / 2:.1f}" x2="{x + ww / 2:.1f}" y1="{y(r_.ci_lo):.1f}" y2="{y(r_.ci_hi):.1f}"/>')
        out.append(f'<text class="val" x="{x + ww / 2:.1f}" y="{y(r_.ci_hi) - 8:.1f}" text-anchor="middle">{pct(r_.empirical_fa)}</text>')
        out.append(f'<text class="tick" x="{x + ww / 2:.1f}" y="{h - b + 22}" text-anchor="middle">{r_.n}</text>')
    out.append(f'<line class="promise" x1="{l}" x2="{w - r}" y1="{y(nominal):.1f}" y2="{y(nominal):.1f}"/>')
    out.append(f'<text class="lab-teal" x="{w - r}" y="{y(nominal) - 8:.1f}" text-anchor="end">promised {pct(nominal)}</text>')
    out.append(f'<text class="tick" x="{(w + l - r) / 2:.0f}" y="{h - 6}" text-anchor="middle">training history n (periods)</text>')
    return svg(w, h, "".join(out), "Actual false-alarm rate by history length against the promised rate")


def bars_h5(fix, n28, tol):
    g = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    order = [("block_bootstrap", "No correction"), ("inflation_factor", "Inflation factor"),
             ("conformal", "Conformal threshold"), ("parametric_bootstrap", "Parametric bootstrap")]
    w, h, l, r, t, rowh = 620, 300, 190, 70, 30, 62
    top = float(g.empirical_fa.max()) * 1.1
    x = lambda v: l + (w - l - r) * v / top  # noqa: E731
    lo, hi = 0.10 - tol / 100, 0.10 + tol / 100
    out = [f'<rect class="band" x="{x(lo):.1f}" y="{t - 10}" width="{x(hi) - x(lo):.1f}" height="{rowh * 4 + 4}"/>',
           f'<text class="lab-teal" x="{x(0.10):.1f}" y="{t - 16}" text-anchor="middle">target {pct(0.10)} ± {tol} pts</text>']
    for i, (k, lab) in enumerate(order):
        r_ = g.loc[k]; yy = t + i * rowh
        out.append(f'<text class="rowlab" x="{l - 14}" y="{yy + 25}" text-anchor="end">{lab}</text>')
        out.append(f'<g class="growx" style="--d:{0.2 + 0.15 * i:.2f}s"><rect class="{"m-coral" if k == "block_bootstrap" else "m-paper"}" '
                   f'x="{l}" y="{yy + 8}" width="{x(r_.empirical_fa) - l:.1f}" height="26" rx="4"><title>{lab}: {pct(r_.empirical_fa, 1)} '
                   f'false alarms; S1 detection {pct(r_.detection_S1, 1)}</title></rect></g>')
        out.append(f'<text class="val" x="{x(r_.empirical_fa) + 8:.1f}" y="{yy + 27}">{pct(r_.empirical_fa, 1)}</text>')
    return svg(w, h, "".join(out), "False-alarm rate after each correction at n = 28")


def dumbbell(ed):
    order = ["any Layer 2 rule", "Altman Z''-EM distress", "amber (uncorrected)", "amber (n-corrected)", "red (uncorrected)", "red (n-corrected)"]
    w, h, l, r, t, rowh = 640, 360, 200, 40, 44, 50
    x = lambda v: l + (w - l - r) * v  # noqa: E731
    out = [f'<line class="grid" x1="{x(0)}" x2="{x(0)}" y1="{t - 14}" y2="{t + rowh * len(order) - 20}"/>',
           f'<line class="grid" x1="{x(1)}" x2="{x(1)}" y1="{t - 14}" y2="{t + rowh * len(order) - 20}"/>',
           f'<text class="tick" x="{x(0)}" y="{t - 22}" text-anchor="middle">0%</text>',
           f'<text class="tick" x="{x(1)}" y="{t - 22}" text-anchor="middle">{pct(1)}</text>']
    for i, m in enumerate(order):
        r_ = ed.loc[m]; yy = t + i * rowh
        out.append(f'<text class="rowlab" x="{l - 16}" y="{yy + 5}" text-anchor="end">{e(m)}</text>')
        out.append(f'<g class="fade" style="--d:{0.15 + 0.12 * i:.2f}s"><line class="link" x1="{x(r_.false_alarm):.1f}" x2="{x(r_.detection):.1f}" y1="{yy}" y2="{yy}"/>'
                   f'<circle class="m-teal ring" cx="{x(r_.false_alarm):.1f}" cy="{yy}" r="8"><title>{e(m)}: healthy flagged {pct(r_.false_alarm, 1)}</title></circle>'
                   f'<circle class="m-coral ring" cx="{x(r_.detection):.1f}" cy="{yy}" r="8"><title>{e(m)}: failed flagged {pct(r_.detection, 1)}</title></circle></g>')
        lo, hi = sorted([(r_.false_alarm, "teal"), (r_.detection, "coral")])
        out.append(f'<text class="val small" x="{x(lo[0]) - 13:.1f}" y="{yy + 4}" text-anchor="end">{pct(lo[0])}</text>')
        out.append(f'<text class="val small" x="{x(hi[0]) + 13:.1f}" y="{yy + 4}">{pct(hi[0])}</text>')
    return svg(w, h, "".join(out), "Failed and healthy firms flagged, by method")


def event_lines(et, M):
    w, h, l, r, t, b = 600, 320, 46, 96, 24, 44
    xs = et.tau_quarters.to_numpy(float); lo_x, hi_x = xs.min(), xs.max()
    vals = np.r_[et.failed_median, et.control_median]; top = float(vals.max()) * 1.15; bot = min(0.0, float(vals.min()))
    x = lambda v: l + (w - l - r) * (v - lo_x) / (hi_x - lo_x)  # noqa: E731
    y = lambda v: t + (h - t - b) * (1 - (v - bot) / (top - bot))  # noqa: E731
    out = [f'<rect class="band" x="{x(-M + 1):.1f}" y="{t}" width="{x(hi_x) - x(-M + 1):.1f}" height="{h - t - b}"/>',
           f'<text class="tick" x="{(x(-M + 1) + x(hi_x)) / 2:.1f}" y="{t + 16}" text-anchor="middle">monitored window</text>',
           f'<line class="axis" x1="{l}" x2="{w - r}" y1="{y(bot):.1f}" y2="{y(bot):.1f}"/>']
    for v in xs:
        out.append(f'<text class="tick" x="{x(v):.1f}" y="{h - b + 22}" text-anchor="middle">{int(v)}</text>')
    out.append(f'<text class="tick" x="{(l + w - r) / 2:.0f}" y="{h - 4}" text-anchor="middle">quarters before the bankruptcy filing</text>')
    for col, cls, lab in [("control_median", "teal", "controls"), ("failed_median", "coral", "failed")]:
        pts = " ".join(f"{x(a):.1f},{y(v):.1f}" for a, v in zip(xs, et[col]))
        out.append(f'<polyline class="draw s-{cls}" points="{pts}" pathLength="1"/>')
        for a, v in zip(xs, et[col]):
            out.append(f'<circle class="m-{cls} ring fade" style="--d:1.1s" cx="{x(a):.1f}" cy="{y(v):.1f}" r="5"><title>{lab}, {int(a)} quarters: '
                       f'median {v:.2f}</title></circle>')
        last = et[col].iloc[-1]
        out.append(f'<text class="lab-{cls}" x="{x(hi_x) + 12:.1f}" y="{y(last) + 5:.1f}">{lab}</text>')
    return svg(w, h, "".join(out), "Median cash ratio of failed firms and controls by quarter before the event")


# ------------------------------------------------------------------ illustrations (no data, no numbers)
def _illustration():
    """Deterministic illustration: noise-like wiggle around zero, then a steady decline (not data)."""
    t = np.arange(40)
    wig = 0.55 * np.sin(1.7 * t) + 0.35 * np.sin(0.63 * t + 1.1) + 0.2 * np.cos(2.9 * t)
    drift = np.where(t >= 22, -0.16 * (t - 21), 0.0)
    return t, wig, drift


def cusum_anim(k):
    t, wig, drift = _illustration()
    z = wig + drift
    S = np.zeros_like(z)
    for i in range(1, len(z)):
        S[i] = max(0.0, S[i - 1] - z[i] - k)
    hthr = 0.55 * S.max(); hit = int(np.argmax(S > hthr))
    w, h = 1000, 330; l, r = 30, 30
    x = lambda i: l + (w - l - r) * i / (len(z) - 1)  # noqa: E731
    zy0 = 60; zs = 64 / float(np.abs(z).max()); sy0, ss = 318, 140 / S.max()
    out = [f'<text class="tick" x="{l}" y="20">surprise z<tspan baseline-shift="sub" font-size="70%">t</tspan> each quarter (bad = down)</text>',
           f'<line class="axis" x1="{l}" x2="{w - r}" y1="{zy0}" y2="{zy0}"/>']
    for i, v in enumerate(z):
        y0, y1 = sorted([zy0, zy0 - zs * v])
        cls = "m-coral" if v < -k else "m-muted"
        out.append(f'<rect class="{cls} pop" style="--d:{0.05 * i:.2f}s" x="{x(i) - 7:.1f}" y="{y0:.1f}" width="14" height="{max(1.5, y1 - y0):.1f}" rx="3"/>')
    out.append(f'<text class="tick" x="{l}" y="{sy0 - 160}">running total S<tspan baseline-shift="sub" font-size="70%">t</tspan> = max(0, S<tspan baseline-shift="sub" font-size="70%">t−1</tspan> − z<tspan baseline-shift="sub" font-size="70%">t</tspan> − k)</text>')
    out.append(f'<line class="axis" x1="{l}" x2="{w - r}" y1="{sy0}" y2="{sy0}"/>')
    out.append(f'<line class="thr" x1="{l}" x2="{w - r}" y1="{sy0 - ss * hthr:.1f}" y2="{sy0 - ss * hthr:.1f}"/>')
    out.append(f'<text class="lab-amber" x="{l}" y="{sy0 - ss * hthr - 8:.1f}" ">threshold h</text>')
    pts = " ".join(f"{x(i):.1f},{sy0 - ss * v:.1f}" for i, v in enumerate(S))
    out.append(f'<polyline class="draw slow s-paper" points="{pts}" pathLength="1"/>')
    out.append(f'<g class="alarm" style="--d:{0.05 * hit + 0.4:.2f}s"><circle class="m-amber" cx="{x(hit):.1f}" cy="{sy0 - ss * S[hit]:.1f}" r="9"/>'
               f'<circle class="pulse" cx="{x(hit):.1f}" cy="{sy0 - ss * S[hit]:.1f}" r="9"/>'
               f'<text class="lab-amber" x="{x(hit) - 14:.1f}" y="{sy0 - ss * S[hit] - 16:.1f}" text-anchor="end">alarm</text></g>')
    out.append(f'<text class="tick" x="{x(22) - 10:.1f}" y="{zy0 - 30}" text-anchor="end">slow drift starts →</text>')
    return svg(w, h, "".join(out), "Illustration: small surprises are absorbed, a run of bad ones accumulates until the threshold")


def adaptive_anim():
    t, wig, drift = _illustration()
    y = 0.25 * wig + drift
    lev = np.zeros_like(y); a = 0.45
    for i in range(1, len(y)):
        lev[i] = lev[i - 1] + a * (y[i - 1] - lev[i - 1])
    w, h, l, r = 640, 300, 20, 190
    lo, hi = y.min() - 0.3, y.max() + 0.4
    X = lambda i: l + (w - l - r) * i / (len(y) - 1)  # noqa: E731
    Y = lambda v: 20 + (h - 50) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    mean = float(y[:20].mean())
    out = [f'<line class="axis" x1="{l}" x2="{w - r}" y1="{h - 30}" y2="{h - 30}"/>',
           f'<text class="tick" x="{l}" y="{h - 10}">time →</text>']
    i0 = 30
    out.append(f'<line class="gapline fade" style="--d:2.2s" x1="{X(i0):.1f}" x2="{X(i0):.1f}" y1="{Y(mean):.1f}" y2="{Y(y[i0]):.1f}"/>')
    out.append(f'<line class="gapline-c fade" style="--d:2.4s" x1="{X(i0) + 6:.1f}" x2="{X(i0) + 6:.1f}" y1="{Y(lev[i0]):.1f}" y2="{Y(y[i0]):.1f}"/>')
    for cls, ser, lab, d in [("s-paper", y, "the ratio", "0s"), ("s-coral dash", lev, "adaptive level", ".6s")]:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ser))
        out.append(f'<polyline class="draw {cls}" style="--d:{d}" points="{pts}" pathLength="1"/>')
    out.append(f'<line class="draw s-teal" style="--d:1.2s" x1="{l}" x2="{w - r}" y1="{Y(mean):.1f}" y2="{Y(mean):.1f}" pathLength="1"/>')
    out.append(f'<text class="lab-teal" x="{w - r + 10}" y="{Y(mean) + 5:.1f}">fixed training mean</text>')
    out.append(f'<text class="lab-coral" x="{w - r + 10}" y="{Y(lev[-1]) - 6:.1f}">adaptive level</text>')
    out.append(f'<text class="lab-paper" x="{w - r + 10}" y="{Y(y[-1]) + 18:.1f}">the ratio</text>')
    return svg(w, h, "".join(out), "Illustration: the adaptive level follows a slow decline, so its surprises stay small")


# ------------------------------------------------------------------ deck
def slide(body, cls="", notes=""):
    return f'<section class="slide {cls}" aria-roledescription="slide">{body}</section>'


def chip(v):
    return f'<span class="chip {v.lower()}">{e(v)}</span>'


def build_slides():
    Hy = T("hypotheses").set_index("id"); cal = T("sim_calibration_curve"); fix = T("sim_fixes"); mt = T("sim_matched_fa")
    ed = T("edgar_detection_summary").set_index("method"); au = T("robust_edgar_auc").set_index("score")
    l2 = T("layer2_first_trip"); rn = pd.read_csv(P.tables / "ratios_nmc.csv", index_col=0)
    sw = T("posthoc_static_weakness").set_index(["method", "role"]); pr = T("posthoc_paired_rules")
    hl, rp, se, dn, cv, et = (T(f"eda_{k}") for k in ["history_length", "ratio_properties", "seasonality", "denominators", "covariance", "event_time"])
    lc = T("robust_edgar_level_vs_change"); lc = lc[lc["sample"] == "analysed"].set_index(["ratio", "role"])
    w_ = T("posthoc_winsor"); tt = T("posthoc_ttm_overlap")
    sc, dc = cfg["simulation"], cfg["detectors"]; n28 = sc["h5_training_length"]; tol = sc["h5_tolerance_points"]
    M = cfg["edgar"]["monitor_quarters"]; cash = cfg["masking"]["cash_ratio"]; k = dc["cusum"]["k"]
    c28 = cal[(cal.n == n28) & (cal.detector == "kalman_cusum")].set_index("nominal")
    f28 = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    m = mt.set_index(["scenario", "detector"]).detection
    jf = l2[l2.firm == "jyoti"].set_index("rule"); jo = jf.loc["OCF < 0"]; ja = jf.loc["Altman Z''-EM < 4.35 (distress)"]
    lcov = rn.liquid_to_st_debt
    am, rule, alt = ed.loc["amber (uncorrected)"], ed.loc["any Layer 2 rule"], ed.loc["Altman Z''-EM distress"]
    a = rp[rp.role == "all analysed"].set_index("ratio"); s4 = se[se.lag_quarters == 4].set_index("input")
    d = dn.set_index(["item", "role"]); c = cv.set_index("dimension_p").loc["all"]
    anf = hl[hl["sample"] == "EDGAR: analysed failed firms"].iloc[0]
    rl = sw.loc[("any Layer 2 rule", "failed")]
    pa = pr[pr.other == "Altman Z''-EM distress"].set_index("metric")
    stat = [x for x in pr.other.unique() if x != "Altman Z''-EM distress"]
    beat_stats = all(pr[(pr.other == x) & pr.metric.str.startswith("expected_cost")].verdict.str.startswith("rules better").all() for x in stat)
    alt_indist = all(v.startswith("not") for v in pa.loc[[f"expected_cost_cm{x}" for x in cfg["cost"]["c_miss"]], "verdict"])
    wmax = float(max((w_.detection - w_.base_detection).abs().max(), (w_.false_alarm - w_.base_false_alarm).abs().max()))
    a_k, a_s, a_l = (au.loc[x] for x in ["cash CUSUM on Kalman errors (pre-registered)",
                                          "cash CUSUM on deviations from the training mean (non-adaptive)",
                                          "lowest cash ratio in the window (level)"])
    ci = lambda r: f"{r.auc:.2f} <span class=\"dim\">({r.auc_ci_lo:.2f}–{r.auc_ci_hi:.2f})</span>"  # noqa: E731
    S = []

    # 1 title
    S.append(slide(f"""
<div class="kicker">Foundations of Data Science · course project</div>
<h1 class="title">Judging a firm against <em>its own</em> history</h1>
<p class="lede">Early warning of cash-flow crises with a layered, self-referential monitor: simple rules, state-space
change detectors, calibrated alarm thresholds and a masking index.</p>
<div class="hero-anim">{cusum_anim(k)}</div>
<div class="meta">Pre-registered {e(cfg['prereg_date'])} · seed {cfg['seed']} · every number on these slides is read from the project's tables</div>""", "title-slide"))

    # 2 motivation
    S.append(slide(f"""
<div class="kicker">Why</div><h2>Two collapses, two different warnings</h2>
<div class="two">
 <div class="card"><div class="firm">Jyoti Structures <span class="dim">India · insolvency {e(cfg['firms']['jyoti']['event_date'])}</span></div>
  <div class="big coral">FY{int(jo.first_trip_fy)}</div><p>Operating cash flow turned negative <b>{jo.lead_years:.1f} years</b> before the
  insolvency admission. Altman Z''-EM only reached distress in FY{int(ja.first_trip_fy)}, after the event.</p>
  <p class="tag">Cash warned first.</p></div>
 <div class="card"><div class="firm">NMC Health <span class="dim">UAE / UK · administration {e(cfg['firms']['nmc']['event_date'])}</span></div>
  <div class="big amber">{lcov.loc[lcov.index.max()]:.2f}</div><p>Reported cash flow looked healthy to the end. Only liquidity cover
  (cash ÷ short-term debt) fell below 1, in {', '.join('FY' + str(y) for y in lcov.index[lcov < 1])}; billions of debt were never reported.</p>
  <p class="tag">The balance sheet warned; cash hid it.</p></div>
</div>
<p class="foot">Classic distress models compare a firm with many peers and need big samples of failures. Mid-sized firms in India or the
Gulf have few peers and short histories. <b>So compare each firm with its own past.</b></p>"""))

    # 3 layers
    l2r = cfg["layer2"]["rules"]
    layers = [("0", "Ratios &amp; audit", "oriented so lower = worse; identities checked; gaps never filled"),
              ("1", "Event time", "line firms up by quarters before the event"),
              ("2", "Rule flags", f"OCF &lt; {l2r['ocf_max']} · EBITDA/interest &lt; {l2r['ebitda_interest_cover_min']:g} · debtor days &gt; {l2r['debtor_days_max']} · Z''-EM &lt; {cfg['altman_z2_em']['distress']:g}"),
              ("3", "Cash detector", "Kalman model on OCF ÷ current liabilities + one-sided CUSUM"),
              ("4", "Multivariate detector", "cash + structural ratios, Ledoit–Wolf covariance, Crosier MCUSUM"),
              ("5", "Thresholds &amp; tiers", f"block bootstrap h; <span class='amber'>amber</span> at {pct(dc['tiers']['amber_alpha'])}, <span class='coral'>red</span> at {pct(dc['tiers']['red_alpha'])} + a rule"),
              ("6", "Masking index", "is the balance sheet alarming while cash looks calm?")]
    S.append(slide("""<div class="kicker">How</div><h2>Seven layers, cheapest first</h2>
<p class="sub">Each layer works on its own; a lender can stop early. Running them side by side tests whether the expensive layers add anything.</p>
<ol class="stack">""" + "".join(f'<li style="--d:{0.08 * i:.2f}s"><span class="ln">L{n}</span><b>{t}</b><span class="lx">{x}</span></li>'
                     for i, (n, t, x) in enumerate(layers)) + "</ol>"))

    # 4 CUSUM explainer
    S.append(slide(f"""<div class="kicker">The core idea</div><h2>An alarm that adds up small bad surprises</h2>
<div class="wide-anim">{cusum_anim(k)}</div>
<div class="three">
 <div><b>1 · Learn normal.</b> A state-space model fitted on the firm's training years predicts next quarter's ratio.</div>
 <div><b>2 · Score surprises.</b> z = (actual − predicted) ÷ its usual size. Surprises smaller than k = {k} are forgiven.</div>
 <div><b>3 · Accumulate.</b> A run of bad surprises pushes S past the threshold h, set by bootstrapping the firm's own errors.</div>
</div><p class="note">Animation: an illustration, not data.</p>"""))

    # 5 EDA
    S.append(slide(f"""<div class="kicker">Exploratory data analysis</div><h2>What the data look like</h2>
<div class="eda">
 <div class="tiles">
  <div class="tile"><div class="tv">{anf['median']:g}</div><div class="tl">median training quarters, analysed US failed firms; only {pct(anf.share_at_least_h5_n)} reach the n = {n28} used in simulation</div></div>
  <div class="tile"><div class="tv">{a.median_acf1_level.min():.2f}–{a.median_acf1_level.max():.2f}</div><div class="tl">lag-1 autocorrelation of every ratio: persistent, not independent draws</div></div>
  <div class="tile"><div class="tv">{s4.iloc[1].median_acf_of_change:+.2f} → {s4.iloc[0].median_acf_of_change:+.2f}</div><div class="tl">lag-4 seasonality of cash-ratio changes: single quarters → trailing twelve months</div></div>
  <div class="tile"><div class="tv">{pct(d.loc[('interest', 'control'), 'share_below_activity_eps'])}</div><div class="tl">of healthy-firm quarters have interest ≤ {pct(cfg['ratios']['denominator_eps_rel_assets']['activity'])} of assets</div></div>
 </div>
 <div class="chartbox"><div class="ct">Median OCF ÷ current liabilities</div>{event_lines(et, M)}
  <p class="cap">Failed firms start lower and slide steadily: weak all along, then gradually worse.</p></div>
</div><p class="note">US EDGAR panel, training quarters only (event-time chart excepted). Added after the results: retrospective.</p>"""))

    # 6 design choices
    rows = [("Short histories are the norm", "judge every threshold method at short n (H4, H5)", "ok"),
            ("Ratios are persistent", "local level + AR(1) model, not iid", "ok"),
            ("Quarterly cash is seasonal", "trailing-twelve-month flows", "ok"),
            (f"OCF ÷ current liabilities: coverage {pct(a.loc[cash, 'share_quarters_observed'])}, light tails", "the cash ratio of Layer 3", "ok"),
            (f"Interest often ≤ {pct(cfg['ratios']['denominator_eps_rel_assets']['activity'])} of assets", f"ε = {cfg['ratios']['denominator_eps_rel_assets']['default']:g} for interest, not one ε", "ok"),
            ("Decline is gradual over years", f"cumulative CUSUM over {M} quarters", "ok"),
            (f"Ledoit–Wolf shrinkage median {c.median_shrinkage:.2f}", "MCUSUM effectively ignores correlations", "warn")]
    S.append(slide("""<div class="kicker">From EDA to design</div><h2>What the data told us to build</h2>
<table class="map"><thead><tr><th>What the data show</th><th></th><th>Design choice</th></tr></thead><tbody>""" +
                   "".join(f'<tr class="{s}" style="--d:{0.07 * i:.2f}s"><td>{x}</td><td class="arrow">→</td><td>{y}</td></tr>' for i, (x, y, s) in enumerate(rows)) +
                   """</tbody></table><p class="note">The last row qualifies a choice: shrinkage is stable but discards the correlations a joint detector is meant to use.</p>"""))

    # 7 scoreboard
    plain = {"H1": "Jyoti: does the statistical detector warn ≥ 2 years early?", "H2": "NMC: cash silent while balance-sheet flags fire?",
             "H3": "Many ratios beat one when cash is masked?", "H4": "Standard thresholds too trigger-happy with short histories?",
             "H5": "Can a cheap correction fix them?", "H6": "Does the masking index separate visible from masked crises?"}
    S.append(slide("""<div class="kicker">Pre-registered hypotheses</div><h2>The scoreboard</h2><div class="score">""" +
                   "".join(f'<div class="hrow" style="--d:{0.08 * i:.2f}s"><span class="hid">{h}</span><span class="hq">{plain[h]}</span>{chip(Hy.loc[h, "verdict"])}</div>'
                           for i, h in enumerate(Hy.index)) +
                   """</div><p class="foot">Inconclusive means the data needed do not exist yet (Jyoti's pre-FY14 quarterly figures; longer cohort histories), not that the test failed.</p>"""))

    # 8 H4
    S.append(slide(f"""<div class="kicker">H4 · the headline</div><h2>Promised {pct(0.10)}. Delivered {pct(c28.loc[0.10, 'empirical_fa'])}.</h2>
<div class="split"><div class="chartbox">{bars_h4(cal, 0.10)}</div>
<div class="side"><p>Simulated <b>healthy</b> firms, each thresholded from its own training history with the standard block bootstrap.</p>
<p>At n = {n28}, thresholds meant to flag one healthy firm in ten flagged about one in three. More history helps, but the gap never closes.</p>
<p>The same over-alarming appears on real US firms: the amber tier flagged <b>{pct(am.false_alarm)}</b> of healthy controls.</p>
<p class="dim small">A known effect for control charts with estimated parameters, here quantified for short financial histories.</p></div></div>"""))

    # 9 H5
    S.append(slide(f"""<div class="kicker">H5 · the fix</div><h2>Three cheap corrections, very different costs</h2>
<div class="split"><div class="chartbox">{bars_h5(fix, n28, tol)}</div>
<div class="side"><p>Fitted on one set of simulated firms, tested on a held-out set, at n = {n28}.</p>
<ul class="costs"><li><b>Inflation factor</b> raises every threshold: −{f28.loc['inflation_factor', 'detection_cost_points']:.1f} pts of detection.</li>
<li><b>Conformal</b> needs a pool of similar healthy firms: −{f28.loc['conformal', 'detection_cost_points']:.1f} pts.</li>
<li><b>Parametric bootstrap</b> re-estimates the model on each resample, targeting the real cause: −{f28.loc['parametric_bootstrap', 'detection_cost_points']:.1f} pts.</li></ul></div></div>"""))

    # 10 H3
    sc4 = ["S1", "S2", "S3", "S4"]; lab = {"S1": "cash drifts", "S2": "cash fine, others worsen", "S3": "small drift everywhere", "S4": "cash smoothed (concealed)"}
    S.append(slide("""<div class="kicker">H3 · one ratio or many</div><h2>Watch both: each wins where the other is blind</h2>
<table class="grid4"><thead><tr><th>Scenario</th><th>Cash-only (Kalman + CUSUM)</th><th>Multivariate</th><th>Winner</th></tr></thead><tbody>""" +
                   "".join(f'<tr><td><b>{s}</b> {lab[s]}</td><td>{pct(m[(s, "kalman")])}</td><td>{pct(m[(s, "mcusum")])}</td>'
                           f'<td class="{"teal" if m[(s, "mcusum")] > m[(s, "kalman")] else "coral"}">{"multivariate" if m[(s, "mcusum")] > m[(s, "kalman")] else "cash-only"}</td></tr>' for s in sc4) +
                   f"""</tbody></table><p class="foot">Detection at a matched {pct(0.10)} false-alarm rate. The S1 loss is not 'slight' as worded, and a plain ARIMA beat the
Kalman model on raw power ({pct(m[('S1', 'arima')])} in S1): the model was chosen for handling missing quarters, not for power.</p>"""))

    # 11 EDGAR
    S.append(slide(f"""<div class="kicker">Real data · US EDGAR panel</div><h2>{int(am.failed_firms)} bankruptcies, {int(am.controls)} healthy twins</h2>
<div class="split"><div class="chartbox"><div class="legend"><span><i class="dot teal"></i>healthy firms flagged</span><span><i class="dot coral"></i>failed firms flagged</span></div>{dumbbell(ed)}</div>
<div class="side"><p>Monitored over the {M} quarters before each bankruptcy filing; nothing tuned on this data. The further apart the dots, the better.</p>
<p><b>Simple rules</b> flagged {pct(rule.detection)} of failures for {pct(rule.false_alarm)} false alarms{', a lower expected cost than every statistical tier (paired CIs exclude zero)' if beat_stats else ''}.</p>
<p>{'Rules and Altman are <b>not distinguishable</b> (paired CIs include zero).' if alt_indist else 'Rules beat Altman at some cost ratio.'}</p>
<p>The <span class="amber">amber</span> tier: {pct(am.detection)} vs {pct(am.false_alarm)}; the CI of the difference ({100 * am.diff_ci_lo:+.1f} to {100 * am.diff_ci_hi:+.1f} pts) includes zero.</p></div></div>"""))

    # 12 why
    S.append(slide(f"""<div class="kicker">Diagnosis (post hoc)</div><h2>Why the statistical layer struggled</h2>
<div class="split"><div class="chartbox">{adaptive_anim()}<p class="note">Animation: an illustration, not data.</p></div>
<div class="side"><p><b>1 · The model forgives slow decline.</b> An adaptive level follows a steady slide, so each surprise stays small.
Threshold-free AUC: Kalman CUSUM {ci(a_k)}; same CUSUM against the fixed training mean {ci(a_s)}.</p>
<p><b>2 · Failing firms were weak all along.</b> Median cash ratio in the last training quarters: {lc.loc[(cash, 'failed'), 'training_level_median']:.2f} (failed) vs
{lc.loc[(cash, 'control'), 'training_level_median']:.2f} (healthy). Just looking at the lowest level scores {ci(a_l)}.</p>
<p class="tag">Combine "worse than its own past" with "weak in absolute terms".</p></div></div>"""))

    # 13 checks
    S.append(slide(f"""<div class="kicker">Did it survive scrutiny?</div><h2>Robustness and post-hoc checks</h2>
<div class="checks">
 <div class="ck"><b>CUSUM k, window shift, dropping ratios, earlier events</b><span>ranking of methods unchanged</span></div>
 <div class="ck"><b>Winsorising every ratio</b><span>no rate moves more than {100 * wmax:.1f} pts</span></div>
 <div class="ck"><b>Non-overlapping inputs instead of TTM</b><span>AUC {tt.auc.iloc[0]:.2f} with TTM; alternatives inside its CI</span></div>
 <div class="ck"><b>Lead times of three to four years</b><span>{pct(rl.of_those_already_on_before_window)} of early rule alarms were on before monitoring began: static weakness</span></div>
 <div class="ck"><b>Rules vs Altman, paired bootstrap</b><span>{'not distinguishable' if alt_indist else 'different'}</span></div>
 <div class="ck"><b>Jyoti FY2017 interpolated</b><span>would invent a distress signal; leaving it missing was right</span></div>
</div><p class="note">Every check is in the explainer (Sections 7–8); post-hoc checks are labelled exploratory and change no verdict.</p>"""))

    # 14 limits
    S.append(slide("""<div class="kicker">Limits</div><h2>What this does not show</h2>
<ul class="lim">
<li><b>The two deep cases were known in advance:</b> illustrations, not tests. The confirmatory cohort's annual data are too short for the detectors.</li>
<li><b>Controls are survivors.</b> Real loan books contain firms that weaken later, so separation here is easier than in practice.</li>
<li><b>Fraud beats ratios.</b> Debt kept off the accounts (NMC) is invisible to any ratio.</li>
<li><b>Gaussian simulation.</b> Real ratios have heavier tails, so the simulation-fitted correction transferred only partly.</li>
<li><b>The US diagnosis is post hoc:</b> a hypothesis to test on new data.</li></ul>"""))

    # 15 takeaways
    S.append(slide(f"""<div class="kicker">Takeaways</div><h2>What to remember</h2>
<div class="take">
 <div class="tk"><span class="tn coral">{pct(c28.loc[0.10, 'empirical_fa'])}</span><p>Standard alarm thresholds on short histories promise {pct(0.10)} false alarms and deliver several times that.</p></div>
 <div class="tk"><span class="tn teal">{pct(f28.loc['parametric_bootstrap', 'empirical_fa'], 1)}</span><p>A parametric bootstrap with refitting fixes it for {f28.loc['parametric_bootstrap', 'detection_cost_points']:.1f} pts of detection.</p></div>
 <div class="tk"><span class="tn">{pct(rule.detection)}</span><p>On real bankruptcies, simple cash-first rules did best; the statistical layer needs a level-aware, non-adaptive baseline.</p></div>
</div>
<p class="foot"><b>Next:</b> thresholds calibrated on real healthy firms · a one-sided multivariate detector · "worse than its past <i>and</i> weak in absolute terms" · longer histories for the cohort.</p>""", "end"))
    return S


CSS = """
/* Layout: one 1280x720 stage, scaled to fit the window; a "control room" for alarms. Dark-only by design. */
:root{
  --ink:#0f1e25; --panel:#152a33; --line:#27444f; --paper:#e7eee9; --muted:#93aba8;
  --teal:#3fa894; --coral:#e45a44; --amber:#e0a33a;
  --display:"Bricolage Grotesque","Avenir Next Condensed","Arial Narrow",sans-serif;
  --body:"Atkinson Hyperlegible","Segoe UI",Verdana,sans-serif;
  --mono:"IBM Plex Mono",Consolas,"Courier New",monospace;
  color-scheme:dark;
}
html,body{height:100%}
body{margin:0;background:var(--ink);color:var(--paper);font-family:var(--body);overflow:hidden;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));box-sizing:border-box}
.viewport{position:relative;height:100%;display:grid;place-items:center}
.stage{width:1280px;height:720px;position:relative;transform-origin:center center;flex:none}
.slide{position:absolute;inset:0;padding:54px 72px 64px;box-sizing:border-box;display:none;flex-direction:column;gap:14px;
  background:radial-gradient(1200px 600px at 85% -10%,rgba(63,168,148,.10),transparent 60%),var(--ink)}
.slide.active{display:flex}
.slide::after{content:"";position:absolute;inset:0;pointer-events:none;opacity:.35;
  background-image:linear-gradient(var(--line) 1px,transparent 1px),linear-gradient(90deg,var(--line) 1px,transparent 1px);
  background-size:40px 40px;mask-image:linear-gradient(180deg,transparent,rgba(0,0,0,.25) 70%,transparent)}
.slide>*{position:relative;z-index:1}
.kicker{font-family:var(--mono);font-size:14px;letter-spacing:.14em;text-transform:uppercase;color:var(--teal)}
h1,h2{font-family:var(--display);font-weight:750;letter-spacing:-.02em;margin:0;text-wrap:balance;line-height:1.02}
h1.title{font-size:76px;max-width:1000px;font-variation-settings:"wdth" 85}
h1 em{font-style:normal;color:var(--coral)}
h2{font-size:46px;max-width:1100px}
.lede{font-size:21px;max-width:820px;color:var(--muted);margin:4px 0 0;line-height:1.45}
.sub,.foot{font-size:18px;color:var(--muted);max-width:1080px;margin:0;line-height:1.45}
.foot b,.sub b{color:var(--paper)}
.meta{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:auto}
.note{font-family:var(--mono);font-size:12px;color:var(--muted);margin:auto 0 0}
.dim{color:var(--muted)} .small{font-size:14px}
.teal{color:var(--teal)} .coral{color:var(--coral)} .amber{color:var(--amber)}
.hero-anim{height:290px;margin-top:6px} .hero-anim svg{height:100%;width:auto}
.wide-anim{height:330px} .wide-anim svg{height:100%;width:100%}
.two{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:22px 26px;display:flex;flex-direction:column;gap:6px}
.card p{margin:0;font-size:17px;line-height:1.5}
.firm{font-family:var(--display);font-size:24px;font-weight:700}
.firm .dim{font-family:var(--mono);font-size:12px;font-weight:400;display:block;margin-top:2px}
.big{font-family:var(--display);font-size:64px;font-weight:800;line-height:1;margin:6px 0;font-variant-numeric:tabular-nums}
.tag{font-family:var(--mono);font-size:14px;color:var(--teal);margin-top:auto!important;padding-top:6px}
.stack{list-style:none;margin:6px 0 0;padding:0;display:flex;flex-direction:column;gap:7px}
.stack li{display:grid;grid-template-columns:62px 260px 1fr;align-items:center;gap:18px;background:var(--panel);border:1px solid var(--line);
  border-radius:4px;padding:10px 16px;font-size:17px}
.stack .ln{font-family:var(--mono);color:var(--ink);background:var(--teal);border-radius:3px;text-align:center;padding:3px 0;font-weight:500}
.stack b{font-family:var(--display);font-size:21px}
.stack .lx{color:var(--muted)}
.three{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;font-size:16px;line-height:1.5;color:var(--muted)}
.three b{color:var(--paper);display:block;font-family:var(--display);font-size:19px;margin-bottom:2px}
.eda{display:grid;grid-template-columns:430px 1fr;gap:26px;align-items:start}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:14px 16px}
.tv{white-space:nowrap;font-family:var(--display);font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;color:var(--paper)}
.tl{font-size:14px;color:var(--muted);line-height:1.4;margin-top:4px}
.chartbox{min-width:0}
.chartbox svg{width:100%;height:auto;display:block}
.ct{font-family:var(--mono);font-size:13px;color:var(--muted);margin-bottom:4px}
.cap{font-size:15px;color:var(--muted);margin:4px 0 0}
.split{display:grid;grid-template-columns:1.25fr 1fr;gap:34px;align-items:center;flex:1}
.side p{font-size:18px;line-height:1.5;margin:0 0 12px}
.side .tag{display:block}
.costs{margin:0;padding-left:20px;font-size:18px;line-height:1.5;display:flex;flex-direction:column;gap:10px}
table{border-collapse:collapse;font-size:18px}
.map{width:100%;margin-top:8px}
.map th{font-family:var(--mono);font-size:13px;font-weight:400;color:var(--muted);text-align:left;padding:0 10px 8px;letter-spacing:.06em;text-transform:uppercase}
.map td{padding:10px;border-top:1px solid var(--line)}
.map td:first-child{color:var(--paper)} .map td:last-child{color:var(--teal);font-family:var(--display);font-size:20px}
.map tr.warn td:last-child{color:var(--amber)}
.map .arrow{color:var(--muted);width:40px;text-align:center}
.grid4{width:100%;margin-top:10px}
.grid4 th{font-family:var(--mono);font-size:13px;font-weight:400;color:var(--muted);text-align:left;padding:0 12px 10px;text-transform:uppercase;letter-spacing:.06em}
.grid4 td{padding:14px 12px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:20px}
.grid4 td:last-child{font-family:var(--display);font-weight:700}
.score{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.hrow{display:grid;grid-template-columns:64px 1fr 170px;align-items:center;gap:18px;padding:12px 18px;background:var(--panel);border:1px solid var(--line);border-radius:4px}
.hid{font-family:var(--mono);font-size:20px;color:var(--teal)}
.hq{font-size:19px}
.chip{font-family:var(--mono);font-size:13px;letter-spacing:.08em;padding:5px 10px;border-radius:3px;text-align:center;border:1px solid currentColor}
.chip.supported{color:var(--teal)} .chip.inconclusive{color:var(--amber)} .chip.refuted{color:var(--coral)}
.chip::before{margin-right:6px} .chip.supported::before{content:"✓"} .chip.inconclusive::before{content:"…"} .chip.refuted::before{content:"✕"}
.legend{display:flex;gap:22px;font-size:14px;color:var(--muted);margin-bottom:4px}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;vertical-align:-1px}
.dot.teal{background:var(--teal)} .dot.coral{background:var(--coral)}
.checks{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}
.ck{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px 18px;display:flex;flex-direction:column;gap:4px}
.ck b{font-family:var(--display);font-size:20px} .ck span{color:var(--teal);font-size:16px}
.lim{margin:6px 0 0;padding-left:22px;font-size:21px;line-height:1.5;display:flex;flex-direction:column;gap:12px;max-width:1080px}
.lim b{font-family:var(--display)}
.take{display:grid;grid-template-columns:repeat(3,1fr);gap:22px;margin-top:10px}
.tk{border-top:3px solid var(--line);padding-top:12px}
.tn{font-family:var(--display);font-size:66px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.tk p{font-size:18px;line-height:1.45;color:var(--muted)}
/* svg marks: text wears text tokens, marks wear series tokens */
svg text{font-family:var(--mono);font-size:13px;fill:var(--muted)}
svg .val{fill:var(--paper);font-size:14px} svg .val.small{font-size:12px}
svg .rowlab{fill:var(--paper);font-family:var(--body);font-size:15px}
svg .lab-teal{fill:var(--teal)} svg .lab-coral{fill:var(--coral)} svg .lab-amber{fill:var(--amber)} svg .lab-paper{fill:var(--paper)}
.axis{stroke:var(--line);stroke-width:1.5} .grid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 4}
.ci{stroke:var(--paper);stroke-width:1.5;opacity:.6}
.promise{stroke:var(--teal);stroke-width:2;stroke-dasharray:6 5}
.thr{stroke:var(--amber);stroke-width:2;stroke-dasharray:6 5}
.band{fill:var(--teal);opacity:.12}
.m-coral{fill:var(--coral)} .m-teal{fill:var(--teal)} .m-amber{fill:var(--amber)} .m-paper{fill:var(--paper);opacity:.85} .m-muted{fill:var(--muted);opacity:.55}
.ring{stroke:var(--ink);stroke-width:2}
.link{stroke:var(--muted);stroke-width:2;opacity:.6}
polyline,.s-teal{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.s-teal{stroke:var(--teal)} .s-coral{stroke:var(--coral)} .s-paper{stroke:var(--paper)} .dash{stroke-dasharray:none}
.gapline{stroke:var(--teal);stroke-width:3} .gapline-c{stroke:var(--coral);stroke-width:3}
.pulse{fill:none;stroke:var(--amber);stroke-width:2}
/* motion: plays when a slide becomes active; everything is visible at rest */
@media (prefers-reduced-motion:no-preference){
  .slide.active .draw{stroke-dasharray:1;stroke-dashoffset:1;animation:draw 1.6s ease-out var(--d,0s) forwards}
  .slide.active .draw.slow{animation-duration:2.2s}
  .slide.active .grow{transform-box:fill-box;transform-origin:bottom;animation:grow .7s cubic-bezier(.2,.8,.2,1) var(--d,0s) backwards}
  .slide.active .growx{transform-box:fill-box;transform-origin:left;animation:growx .7s cubic-bezier(.2,.8,.2,1) var(--d,0s) backwards}
  .slide.active .pop{transform-box:fill-box;transform-origin:center;animation:pop .35s ease-out var(--d,0s) backwards}
  .slide.active .fade{animation:fade .6s ease-out var(--d,0s) backwards}
  .slide.active .alarm{animation:fade .3s ease-out var(--d,0s) backwards}
  .slide.active .pulse{transform-box:fill-box;transform-origin:center;animation:pulse 1.4s ease-out 2.4s infinite}
  .slide.active .stack li,.slide.active .hrow,.slide.active .map tr{animation:rise .5s ease-out var(--d,0s) backwards}
}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes grow{from{transform:scaleY(0)}}
@keyframes growx{from{transform:scaleX(0)}}
@keyframes pop{from{transform:scale(0)}}
@keyframes fade{from{opacity:0}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}}
@keyframes pulse{from{transform:scale(1);opacity:.9}to{transform:scale(3.2);opacity:0}}
/* chrome: a progress trace that climbs like a CUSUM toward h */
.chrome{position:absolute;left:72px;right:72px;bottom:22px;display:flex;align-items:center;gap:16px;z-index:2;font-family:var(--mono);font-size:12px;color:var(--muted)}
.track{flex:1;height:3px;background:var(--line);border-radius:2px;position:relative}
.fill{position:absolute;left:0;top:0;bottom:0;background:var(--teal);border-radius:2px;transition:width .4s ease}
.fill.hit{background:var(--amber)}
.nav{background:none;border:1px solid var(--line);color:var(--paper);font-family:var(--mono);font-size:13px;padding:4px 10px;border-radius:3px;cursor:pointer}
.nav:hover{border-color:var(--teal)} .nav:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
"""

JS = """
(function(){
  const slides=[...document.querySelectorAll('.slide')], stage=document.querySelector('.stage');
  const fill=document.querySelector('.fill'), count=document.querySelector('.count');
  let i=0;
  function fit(){const vw=document.querySelector('.viewport').clientWidth, vh=document.querySelector('.viewport').clientHeight;
    stage.style.transform='scale('+Math.min(vw/1280,vh/720)+')';}
  function show(n,push){i=Math.max(0,Math.min(slides.length-1,n));
    slides.forEach((s,k)=>{s.classList.toggle('active',k===i);s.setAttribute('aria-hidden',k===i?'false':'true');});
    const p=(i+1)/slides.length; fill.style.width=(100*p)+'%'; fill.classList.toggle('hit',i===slides.length-1);
    count.textContent=String(i+1).padStart(2,'0')+' / '+String(slides.length).padStart(2,'0');
    if(push!==false){try{history.replaceState(null,'','#s'+(i+1));}catch(_){location.hash='s'+(i+1);}}}
  function fromHash(){const m=/^#s(\\d+)$/.exec(location.hash); return m?parseInt(m[1],10)-1:0;}
  document.addEventListener('keydown',ev=>{
    if(['ArrowRight','PageDown',' ','Enter'].includes(ev.key)){ev.preventDefault();show(i+1);}
    else if(['ArrowLeft','PageUp','Backspace'].includes(ev.key)){ev.preventDefault();show(i-1);}
    else if(ev.key==='Home'){show(0);} else if(ev.key==='End'){show(slides.length-1);}});
  document.getElementById('prev').addEventListener('click',()=>show(i-1));
  document.getElementById('next').addEventListener('click',()=>show(i+1));
  stage.addEventListener('click',ev=>{if(ev.target.closest('button'))return; const r=stage.getBoundingClientRect();
    show(ev.clientX>r.left+r.width*0.35?i+1:i-1);});
  let x0=null; stage.addEventListener('touchstart',ev=>{x0=ev.touches[0].clientX;},{passive:true});
  stage.addEventListener('touchend',ev=>{if(x0===null)return; const dx=ev.changedTouches[0].clientX-x0; if(Math.abs(dx)>40)show(i+(dx<0?1:-1)); x0=null;});
  window.addEventListener('resize',fit); window.addEventListener('hashchange',()=>show(fromHash(),false));
  fit(); show(fromHash(),false);
})();
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700'
         '&family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&family=IBM+Plex+Mono:wght@400;500&display=swap">')


def fragment() -> str:
    S = build_slides()
    return (f'<title>Judging a Firm Against Its Own History</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n{FONTS}\n'
            f'<style>{CSS}</style>\n<div class="viewport"><div class="stage" role="region" aria-label="Slide deck">{"".join(S)}'
            '<div class="chrome"><button class="nav" id="prev" aria-label="Previous slide">←</button><div class="track"><div class="fill"></div></div>'
            '<span class="count"></span><button class="nav" id="next" aria-label="Next slide">→</button></div></div></div>'
            f'<script>{JS}</script>')


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fragment", help="also write a body fragment (for publishing) to this path")
    a = ap.parse_args(argv)
    frag = fragment()
    out = P.tables.parent / "slides.html"
    out.write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                   "<meta name='viewport' content='width=device-width, initial-scale=1, viewport-fit=cover'></head><body>"
                   + frag + "</body></html>", encoding="utf-8")
    print("wrote", out)
    if a.fragment:
        Path(a.fragment).write_text(frag, encoding="utf-8"); print("wrote", a.fragment)


if __name__ == "__main__":
    main()
