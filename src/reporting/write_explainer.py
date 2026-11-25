"""Plain-language, in-depth explainer of the whole project -> reports/explainer.html (+ explainer.pdf).

    python -m src.reporting.write_explainer [--no-pdf]

Like report.md, every number is read from reports/tables/*.csv or configs/config.yaml at build time; none
is typed (CLAUDE.md rule 3). Figures are embedded from reports/figures/. The PDF is printed by a local
headless Edge or Chrome if one is installed.
"""
from __future__ import annotations

import argparse
import base64
import html
import subprocess
from pathlib import Path

import pandas as pd

from src import evaluation as _ev
from src.config import Paths, load_config

cfg = load_config(); P = Paths.from_config(cfg)
ROOT = Path(__file__).resolve().parents[2]
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/usr/bin/google-chrome", "/usr/bin/chromium",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]


def T(name, **kw):
    return pd.read_csv(P.tables / f"{name}.csv", **kw)


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def e(x):
    return html.escape(str(x))


def table(header, rows, cls=""):
    h = "".join(f"<th>{e(c)}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(name, caption):
    f = P.figures / name
    if not f.exists():
        return f'<p class="missing">[figure {e(name)} not generated yet]</p>'
    b64 = base64.b64encode(f.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{e(caption)}"><figcaption>{caption}</figcaption></figure>'


def box(title, body, kind="note"):
    return f'<div class="box {kind}"><div class="box-title">{title}</div>{body}</div>'


def verdict_badge(v):
    return f'<span class="badge {v.lower()}">{e(v)}</span>'


POSTHOC = "post-hoc / exploratory"


def ph_h3(title):
    """Heading of a post-hoc analysis: always carries the label (src.sanity checks every one)."""
    return f'<h3 class="posthoc">{title} <span class="badge info">{POSTHOC}</span></h3>'


def aci(r):
    return f"{r.auc:.2f} (95% CI {r.auc_ci_lo:.2f}–{r.auc_ci_hi:.2f})"


def pts(x):
    return f"{100 * x:+.1f}"


CSS = """
:root{--ink:#1f2328;--muted:#57606a;--line:#d0d7de;--navy:#1d3557;--teal:#2a9d8f;--coral:#c62f3a;--amber:#b7791f;--bg:#ffffff;--soft:#f6f8fa}
@page{size:A4;margin:18mm 16mm 18mm 16mm}
*{box-sizing:border-box}
body{font-family:"Segoe UI",Calibri,Arial,sans-serif;color:var(--ink);background:var(--bg);font-size:10.5pt;line-height:1.5;max-width:900px;margin:0 auto;padding:16px}
h1{font-size:24pt;color:var(--navy);margin:0 0 4px}
h2{font-size:16pt;color:var(--navy);border-bottom:2px solid var(--navy);padding-bottom:3px;margin-top:28px;break-before:page}
h2.nobreak{break-before:auto}
h3{font-size:12.5pt;color:var(--navy);margin:18px 0 6px;break-after:avoid-page;page-break-after:avoid}
h3+p{break-before:avoid-page}
p{margin:6px 0 8px}
.sub{color:var(--muted);font-size:11.5pt;margin-bottom:14px}
.cover{border-left:6px solid var(--navy);padding:6px 0 6px 16px;margin:10px 0 18px}
table{border-collapse:collapse;width:100%;margin:8px 0 12px;font-size:9.3pt;break-inside:auto}
th,td{border:1px solid var(--line);padding:4px 6px;vertical-align:top;text-align:left}
th{background:var(--soft)}
tr{break-inside:avoid}
figure{margin:10px 0 14px;break-inside:avoid}
figure img{width:100%;border:1px solid var(--line)}
figcaption{font-size:9pt;color:var(--muted);margin-top:4px}
.box{border:1px solid var(--line);border-left:5px solid var(--navy);background:var(--soft);padding:8px 12px;margin:10px 0;break-inside:avoid}
.box.good{border-left-color:var(--teal)} .box.warn{border-left-color:var(--amber)} .box.bad{border-left-color:var(--coral)}
.box-title{font-weight:700;margin-bottom:3px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:8.5pt;font-weight:700;color:#fff;background:var(--muted)}
.badge.supported,.badge.pass{background:var(--teal)} .badge.inconclusive,.badge.warn{background:var(--amber)}
.badge.refuted,.badge.fail{background:var(--coral)} .badge.info{background:#6c757d}
.layers{display:flex;flex-direction:column;gap:5px;margin:10px 0}
.layer{display:flex;border:1px solid var(--line);break-inside:avoid}
.layer .n{background:var(--navy);color:#fff;font-weight:700;min-width:64px;display:flex;align-items:center;justify-content:center;font-size:10pt}
.layer .t{padding:5px 10px}
.layer .t b{color:var(--navy)}
.small{font-size:9pt;color:var(--muted)}
code{font-family:Consolas,monospace;font-size:9pt;background:var(--soft);padding:0 3px}
ul,ol{margin:4px 0 8px 18px;padding-left:4px} li{margin:2px 0}
.qa dt{font-weight:700;color:var(--navy);margin-top:8px} .qa dd{margin:2px 0 6px 14px}
.toc li{margin:1px 0}
.missing{color:var(--coral)}
.checks td:nth-child(3){white-space:nowrap}
.design td:nth-child(1){width:15%;font-weight:600} .design td:nth-child(5){width:13%}
@media screen{h2{margin-top:48px}}
"""


def _cell(det, fa):
    return f"{pct(det)} / {pct(fa)}"


def posthoc_section(cfg, PH, ed, okf, okc, fix, n28, pa, cms, stat_tiers):
    """Section 8: the post-hoc review checks (config `post_hoc`); every heading carries the post-hoc label."""
    ph = cfg["post_hoc"]; M = cfg["edgar"]["monitor_quarters"]; o = []
    o.append('<h2>8. Additional checks (post-hoc)</h2>')
    o.append(box("Status of this section",
                 f"<p>Every analysis here was added on {e(ph['date'])}, after all results had been seen, in response to a review of this "
                 f"document. Each one is <b>{POSTHOC}</b>: no hypothesis verdict uses it, its settings sit in new config blocks "
                 "(<code>evaluation</code>, <code>post_hoc</code>) logged as deviations, and no frozen setting changed. Scripts: "
                 "<code>src/post_hoc.py</code> (EDGAR) and <code>src/simulate/self_starting.py</code> (simulation).</p>", "warn"))

    # 8.1 paired
    pr = PH["paired_rules"]
    o.append(ph_h3("8.1 Are the rules really better? A paired comparison"))
    o.append("<p>The Section 6 table compares each method with chance, bootstrapping failed firms and controls independently. To "
             "compare <i>two methods</i> the design must be paired: resample matched units (a failed firm together with its analysed "
             f"control, {okf} units, {okc} with a control) and evaluate both methods on the same resample "
             f"({cfg['evaluation']['ci_resamples']:,} resamples). Rules minus Altman Z''-EM:</p>")
    lab = {"detection": "Detection (higher is better)", "false_alarm": "False alarms (lower is better)",
           **{f"expected_cost_cm{x}": f"Expected cost, C_M = {x} × C_FA (lower is better)" for x in cms}}
    rows = []
    for k, r in pa.iterrows():
        isr = not k.startswith("expected")
        f_ = (lambda v: pct(v)) if isr else (lambda v: f"{v:.2f}")
        d_ = (lambda v: f"{pts(v)} pts") if isr else (lambda v: f"{v:+.2f}")
        rows.append([lab[k], f_(r.rules), f_(r.other_value), f"{d_(r.difference)} ({d_(r.ci_lo)} to {d_(r.ci_hi)})", e(r.verdict)])
    o.append(table(["Metric", "Any Layer 2 rule", "Altman Z''-EM", "Rules minus Altman (95% paired CI)", "Reading"], rows))
    tier_cost = {m_: pr[(pr.other == m_) & pr.metric.str.startswith("expected_cost")] for m_ in stat_tiers}
    allb = [m_ for m_, g in tier_cost.items() if g.verdict.str.startswith("rules better").all()]
    worse_fa = [m_ for m_ in stat_tiers if pr[(pr.other == m_) & (pr.metric == "false_alarm")].verdict.iloc[0].startswith("rules worse")]
    o.append(f"<p>Against the statistical tiers the picture is clear: the rules have a lower expected cost at every cost ratio, with "
             f"paired CIs excluding zero, against {len(allb)} of {len(stat_tiers)} tiers ({', '.join(e(x) for x in allb)}). They do raise "
             f"more false alarms than {', '.join(e(x) for x in worse_fa) or 'none of them'}, but miss far fewer failures. "
             "<b>Conclusion: 'rules beat the statistical tiers' survives a paired test; 'rules beat Altman' does not, and is reworded "
             "as 'not distinguishable' throughout.</b> Full table: <code>posthoc_paired_rules.csv</code>.</p>")

    # 8.2 fixed horizons
    fh = PH["fixed_horizon"]; sw = PH["static_weakness"].set_index(["method", "role"])
    hs = [h for h in ph["fixed_horizons_quarters"] if h < M]; hpre = [h for h in ph["fixed_horizons_quarters"] if h >= M]
    o.append(ph_h3("8.2 Fixed horizons: is a long lead time early warning, or weakness from the start?"))
    o.append(f"<p>Each cell counts only alarms raised at least h quarters before the bankruptcy quarter, using information available "
             f"by then (detectors are causal and thresholds come from training). For h ≥ {M} there is no monitored quarter left, so the "
             f"rules and Altman are instead read in the last <i>training</i> quarter, before monitoring starts. Cells: failed flagged / "
             f"healthy flagged; † = 95% CI of the difference includes zero.</p>")
    rows = []
    for m_ in fh.method.unique():
        g = fh[fh.method == m_].set_index("horizon_quarters")
        cells = []
        for h in hs + hpre:
            if h in g.index:
                r = g.loc[h]; cells.append(_cell(r.detection, r.false_alarm) + (" †" if r.diff_ci_lo <= 0 <= r.diff_ci_hi else ""))
            else:
                cells.append("n/a (no statistic before monitoring)")
        rows.append([e(m_)] + cells)
    o.append(table(["Method"] + [f"h = {h} quarters" for h in hs] + [f"h = {h}: flagged before monitoring starts" for h in hpre], rows))
    rl, rc = sw.loc[("any Layer 2 rule", "failed")], sw.loc[("any Layer 2 rule", "control")]
    al, ac = sw.loc[("Altman Z''-EM distress", "failed")], sw.loc[("Altman Z''-EM distress", "control")]
    mostly = min(rl.of_those_already_on_before_window, al.of_those_already_on_before_window) >= 0.5
    o.append(f"<p><b>Static weakness.</b> Of the failed firms whose first rule alarm came with a lead of {M - 3}–{M - 1} quarters (the first "
             f"three monitored quarters: {pct(rl.share_of_alarms_in_first_3_quarters, 0)} of all rule alarms), {pct(rl.of_those_already_on_before_window, 0)} "
             f"already had a rule flag on in the last training quarter; for Altman the figures are {pct(al.share_of_alarms_in_first_3_quarters, 0)} "
             f"and {pct(al.of_those_already_on_before_window, 0)}. Before monitoring even starts, the rules flag {pct(rl.flagged_before_window_all_firms, 0)} "
             f"of failed firms vs {pct(rc.flagged_before_window_all_firms, 0)} of controls, and Altman {pct(al.flagged_before_window_all_firms, 0)} vs "
             f"{pct(ac.flagged_before_window_all_firms, 0)}. "
             + (f"So lead times of {M - 3}–{M - 1} quarters <b>mostly mean 'flagged from the start'</b>: they measure a firm that already looks weak "
                "(static weakness), not an early warning of change. " if mostly else
                "So long lead times are only partly explained by weakness present from the start. ")
             + "The statistical tiers' rates fall off quickly with h and stay close to their false-alarm rates.</p>")

    # 8.3 persistence
    pe = PH["persistence"]
    o.append(ph_h3("8.3 Persistence: requiring the flag on consecutive quarters"))
    o.append(table(["Method", "Consecutive quarters required", "Failed flagged (95% CI)", "Healthy flagged (95% CI)", "Median lead (quarters)",
                    f"Expected cost, C_M = {cms[0]}"],
                   [[e(r.method), str(int(r.consecutive_quarters)), f"{pct(r.detection)} ({pct(r.det_ci_lo)}–{pct(r.det_ci_hi)})",
                     f"{pct(r.false_alarm)} ({pct(r.fa_ci_lo)}–{pct(r.fa_ci_hi)})", f"{r.median_lead_quarters:g}", f"{getattr(r, f'expected_cost_cm{cms[0]}'):.2f}"]
                    for r in pe.itertuples()]))
    p1 = pe[pe.consecutive_quarters == 1].set_index("method"); pk = pe[pe.consecutive_quarters == pe.consecutive_quarters.max()].set_index("method")
    keep_share = pk.loc["any Layer 2 rule", "false_alarm"] / p1.loc["any Layer 2 rule", "false_alarm"]
    o.append(f"<p>Requiring {int(pe.consecutive_quarters.max())} consecutive quarters lowers healthy-firm false alarms only modestly "
             f"(rules {pct(p1.loc['any Layer 2 rule', 'false_alarm'])} → {pct(pk.loc['any Layer 2 rule', 'false_alarm'])}; Altman "
             f"{pct(p1.loc[chr(65) + 'ltman Z' + chr(39) * 2 + '-EM distress', 'false_alarm'])} → {pct(pk.loc['Altman Z' + chr(39) * 2 + '-EM distress', 'false_alarm'])}) and "
             f"costs detection (rules {pct(p1.loc['any Layer 2 rule', 'detection'])} → {pct(pk.loc['any Layer 2 rule', 'detection'])}); the CIs "
             f"overlap. {pct(keep_share, 0)} of the controls flagged by a rule stay flagged for {int(pe.consecutive_quarters.max())} "
             "consecutive quarters" + (": most control false alarms are persistent flags, consistent with static weakness rather than "
                                       "one-quarter noise.</p>" if keep_share >= 0.5 else ".</p>"))

    # 8.4 filing dates
    fl = PH["filing_lead"]
    o.append(ph_h3("8.4 Lead times from filing dates, not period ends"))
    o.append("<p>Each alarm quarter is dated by the SEC filing date of its first-filed figures (stored with every fact), and an alarm "
             "counts only if it was filed <i>before</i> the bankruptcy 8-K (controls: before their failed firm's 8-K).</p>")
    o.append(table(["Method", "Failed flagged: period-end → filed before event", "Healthy flagged: period-end → filed before event",
                    "Median lead (quarters): period-end → filing date", "Median reporting lag (days)", "Alarms filed only after the event"],
                   [[e(r.method), f"{pct(r.detection_period_end_basis)} → {pct(r.detection_filed_before_event)}",
                     f"{pct(r.false_alarm_period_end_basis)} → {pct(r.false_alarm_filed_before_event)}",
                     f"{r.median_lead_quarters_period_end:g} → {r.median_lead_quarters_filing_date:.1f}", f"{r.median_reporting_lag_days:g}",
                     str(int(r.alarms_filed_after_event))] for r in fl.itertuples()]))
    same_rank = list(fl.sort_values("detection_period_end_basis").method) == list(fl.sort_values("detection_filed_before_event").method)
    o.append(f"<p>Measured from filing dates, the median reporting lag is {fl.median_reporting_lag_days.min():g}–"
             f"{fl.median_reporting_lag_days.max():g} days, and {int(fl.alarms_filed_after_event.min())}–{int(fl.alarms_filed_after_event.max())} "
             "alarms per method were not public before the bankruptcy filing; they no longer count. Dropping them can raise a method's "
             "median lead (the remaining alarms are the earlier ones). The ranking of methods by detection "
             + ("does not change." if same_rank else "changes (see table).") + "</p>")

    # 8.5 calibration horizon
    ch = PH["calibration_horizon"].set_index("item").value
    simrows = [i for i in ch.index if i.startswith("simulation (")]
    o.append(ph_h3("8.5 Is h calibrated for the number of steps actually monitored?"))
    o.append(f"<p>The bootstrap sets h for paths of {int(ch['calibration horizon (detectors.monitor_horizon)'])} steps; EDGAR monitors "
             f"{int(ch['EDGAR monitored quarters (edgar.monitor_quarters)'])} quarters and the simulation "
             f"{int(ch['simulation monitored periods (simulation.monitor_length)'])} periods: <b>no mismatch in the design</b>. In practice "
             "missing quarters carry the CUSUM forward unchanged, so some firms have fewer effective steps: median "
             f"{ch['EDGAR cash-ratio steps actually monitored: median']:g}, 10th percentile {ch['EDGAR cash-ratio steps actually monitored: 10th percentile']:g}, "
             f"minimum {ch['EDGAR cash-ratio steps actually monitored: minimum']:g}; {pct(ch['EDGAR share of analysed firms with all monitored steps observed'], 0)} "
             "of firms have all steps. Fewer steps can only <i>lower</i> the false-alarm rate. In simulation: "
             + "; ".join(f"{e(i.split(': ')[1])} → {pct(ch[i])}" for i in simrows)
             + ". So the step count cannot explain the real-data over-alarming; if anything it slightly flatters it. (Check: the "
             f"recomputed cash threshold equals the main run's for {pct(ch['recomputed h_cash equals the main run' + chr(39) + 's h_cash (share of analysed firms)'], 0)} of firms.)</p>")

    # 8.6 TTM overlap
    tt = PH["ttm_overlap"]; ac_ = PH["kalman_acf"]
    o.append(ph_h3("8.6 Overlapping trailing-twelve-month ratios"))
    o.append("<p>Consecutive TTM values share three of their four quarters, which induces autocorrelation. Layer 3 (the cash detector "
             "alone, pre-registered settings) was re-run on two inputs without overlap. Firms enter only if they meet the same "
             "minimum-training rule on that input:</p>")
    o.append(table(["Input to the cash detector", "Failed / controls", "Failed flagged", "Healthy flagged", "AUC (95% CI)"],
                   [[e(r.input), f"{int(r.failed_firms)} / {int(r.controls)}", f"{pct(r.detection)} ({pct(r.det_ci_lo)}–{pct(r.det_ci_hi)})",
                     f"{pct(r.false_alarm)} ({pct(r.fa_ci_lo)}–{pct(r.fa_ci_hi)})", aci(r)] for r in tt.itertuples()]))
    a0 = ac_[(ac_["input"] == tt["input"].iloc[0]) & (ac_.lag_quarters == 1)].set_index("errors")
    base_auc = tt.iloc[0]
    overl = all((r.auc_ci_lo <= base_auc.auc <= r.auc_ci_hi) for r in tt.iloc[1:].itertuples())
    o.append(f"<p>Autocorrelation of the standardised Kalman errors on healthy controls (TTM input, lag 1): pooled "
             f"{a0.loc['training (in-sample)', 'pooled_acf']:.2f} in training and {a0.loc['monitored window', 'pooled_acf']:.2f} in the "
             f"monitored window (median per firm {a0.loc['training (in-sample)', 'median_per_firm_acf']:.2f} and "
             f"{a0.loc['monitored window', 'median_per_firm_acf']:.2f}); all lags are in <code>posthoc_kalman_acf.csv</code>. Within a firm "
             "the errors show little serial correlation (the fitted AR(1) part absorbs the overlap); the larger pooled value mostly "
             "reflects firms whose monitored errors sit persistently off zero, i.e. a shift in level that the CUSUM accumulates, not "
             "short-run dependence. Removing the overlap "
             + ("does not change discrimination detectably (every AUC CI contains the TTM estimate), so TTM overlap does not explain the "
                "weak real-data result." if overl else "changes discrimination (see table).") + "</p>")

    # 8.7 winsor
    w = PH["winsor"]
    capc = [c for c in w.columns if c.startswith("cap_")]
    o.append(ph_h3(f"8.7 Winsorising the ratios at the {pct(ph['winsor_quantiles'][0], 0)} / {pct(ph['winsor_quantiles'][1], 0)} quantiles"))
    o.append(table(["Method", "Failed flagged: base → winsorised", "Healthy flagged: base → winsorised"],
                   [[e(r.method), f"{pct(r.base_detection)} → {pct(r.detection)}", f"{pct(r.base_false_alarm)} → {pct(r.false_alarm)}"]
                    for r in w.itertuples()]))
    mx = float(max((w.detection - w.base_detection).abs().max(), (w.false_alarm - w.base_false_alarm).abs().max()))
    o.append("<p>Caps were set from every panel firm's training quarters pooled (" + "; ".join(
        f"<code>{c[4:]}</code> {e(w[c].iloc[0])}, {pct(w['share_training_capped_' + c[4:]].iloc[0])} of training quarters capped" for c in capc)
             + f"). Winsorising changes no rate by more than {100 * mx:.1f} points"
             + (", so outliers do not explain the main results.</p>" if mx <= 0.05 else ".</p>"))

    # 8.8 survivors + industry
    ind = PH["industry"]
    o.append(ph_h3("8.8 Controls are survivors; industry mix"))
    o.append("<p>EDGAR controls were required to have no Item 1.03 (bankruptcy) 8-K in 2012–2023 "
             "(<code>edgar.controls.no_item_103_2012_2023</code>); cohort controls were never insolvent. They are survivors, which makes separation "
             "easier than a lender's real problem, where the healthy book includes firms that will weaken later. Industry mix of the "
             "analysed samples:</p>")
    ed_ = ind[ind["sample"] == "EDGAR (analysed)"].pivot_table(index="group", columns="role", values="firms", fill_value=0)
    co_ = ind[ind["sample"] == "India cohort"].pivot_table(index="group", columns="role", values="firms", fill_value=0)
    o.append(table(["EDGAR: SIC division", "Failed", "Controls"],
                   [[e(g), str(int(r.get("failed", 0))), str(int(r.get("control", 0)))] for g, r in ed_.sort_values("failed", ascending=False).iterrows()]))
    o.append(table(["Cohort: Screener sector", "Failed", "Controls", "Negative control"],
                   [[e(g), str(int(r.get("failed", 0))), str(int(r.get("control", 0))), str(int(r.get("negative_control", 0)))] for g, r in co_.iterrows()]))

    # 8.9 MI
    mi = PH["mi_healthy"]; hp = [c for c in mi.columns if c.startswith("share_above_healthy_p")]
    simH = mi[(mi["sample"].str.startswith("simulation")) & (mi.group == "H")].iloc[0]
    o.append(ph_h3("8.9 The masking index relative to healthy firms"))
    o.append(f"<p>On simulated healthy firms the peak MI has median {simH.median_peak_mi:.2f} and is above zero for "
             f"{pct(simH.share_mi_above_0)} of them, because both detectors over-alarm with uncorrected thresholds and the structural side "
             "shares information with cash. <b>MI &gt; 0 does not by itself mean 'masked'.</b> A healthy-centred version subtracts the "
             f"healthy median, or flags MI above the healthy {', '.join(c.split('_p')[1] + 'th' for c in hp)} percentile:</p>")
    o.append(table(["Sample", "Group", "Firms", "Median peak MI", "Median MI minus healthy median"] + [f"Share above healthy {c.split('_p')[1]}th pct" for c in hp]
                   + ["AUC vs healthy"],
                   [[e(r["sample"]), e(r.group), str(int(r.firms)), f"{r.median_peak_mi:.2f}", f"{r.median_centred_mi:+.2f}"]
                    + [pct(r[c]) for c in hp] + ["—" if pd.isna(r.auc_vs_healthy) else f"{r.auc_vs_healthy:.2f}"] for _, r in mi.iterrows()]))
    simg = mi[mi["sample"].str.startswith("simulation") & (mi.group != "H")].set_index("group")
    edg = mi[mi["sample"].str.startswith("EDGAR") & mi.group.str.startswith("failed")]
    o.append(f"<p>Centred on healthy firms, the scenario that stands out most is {e(simg.auc_vs_healthy.idxmax())} (AUC vs healthy "
             f"{simg.auc_vs_healthy.max():.2f}); concealment (S4) reaches only {simg.loc['S4', 'auc_vs_healthy']:.2f}. On EDGAR "
             + ("no failed-firm archetype has a median MI above the healthy controls' median."
                if (edg.median_centred_mi <= 0).all() else "some failed-firm archetypes sit above the healthy median (table).") + "</p>")

    # 8.10 self-starting
    ss_ = PH["sim_self_starting"]; fx = fix
    o.append(ph_h3("8.10 A self-starting benchmark for short histories (simulation, H4/H5 setting)"))
    o.append("<p>Hawkins-style self-starting charts standardise each new value by the estimates from all earlier values and map the "
             "result to N(0, 1) through its exact distribution, so the threshold needs no estimated parameter. Two versions on the same "
             "held-out simulated firms as H4/H5 (regenerated from their seeds and checked against the stored runs): Q-statistics "
             "assuming iid values, and recursive residuals of an AR(1) regression. Cells: false alarms / detection in S1.</p>")
    labs = [("block_bootstrap", "Uncorrected"), ("inflation_factor", "Inflation factor"), ("conformal", "Conformal"), ("parametric_bootstrap", "Parametric")]
    sl = list(ss_.method.unique())
    rows = []
    for a in sorted(ss_.nominal.unique()):
        for n_ in sorted(ss_.n.unique()):
            f0 = fx[(fx.n == n_) & (fx.nominal == a)].set_index("method"); s0 = ss_[(ss_.n == n_) & (ss_.nominal == a)].set_index("method")
            rows.append([pct(a, 0), str(n_)] + [_cell(f0.loc[k, "empirical_fa"], f0.loc[k, "detection_S1"]) for k, _ in labs]
                        + [("<b>" if s0.loc[x, "within_tolerance"] else "") + _cell(s0.loc[x, "empirical_fa"], s0.loc[x, "detection_S1"])
                           + ("</b>" if s0.loc[x, "within_tolerance"] else "") for x in sl])
    o.append(table(["Nominal", "n"] + [l for _, l in labs] + [e(x) for x in sl], rows))
    r28 = ss_[(ss_.n == n28) & (ss_.nominal == 0.10)].set_index("method")
    ar_ = [x for x in sl if "AR(1)" in x][0]; iid_ = [x for x in sl if "iid" in x][0]
    best = max(f0_ := [(k, fx[(fx.n == n28) & (fx.nominal == 0.10) & (fx.method == k)].iloc[0]) for k, _ in labs[1:]],
               key=lambda kv: kv[1].detection_S1)
    o.append(f"<p>Bold (self-starting columns only) = within ±{cfg['simulation']['h5_tolerance_points']} points of nominal. The iid version over-alarms "
             f"({pct(r28.loc[iid_, 'empirical_fa'])} at n = {n28}, nominal 10%) because the simulated ratios are AR(1) with "
             f"φ = {cfg['simulation']['ar_phi']}, which violates its independence assumption. The AR(1) version reaches "
             f"{pct(r28.loc[ar_, 'empirical_fa'])} false alarms with S1 detection {pct(r28.loc[ar_, 'detection_S1'])} at n = {n28}; the "
             f"most powerful correction there ({e(dict(labs)[best[0]])}) has {pct(best[1].empirical_fa)} / {pct(best[1].detection_S1)}, "
             "so the self-starting chart detects more at a slightly different false-alarm rate (not a matched comparison). <b>Caution:</b> the simulator <i>is</i> a Gaussian AR(1) process, so this comparator is correctly "
             "specified for the simulated data by construction, while the pre-registered local-level model is not. Its advantage may "
             "not transfer to real ratios; it has not been run on EDGAR, and doing so would need its own pre-registration.</p>")
    return o


RETRO = "retrospective EDA"


def eda_section(cfg, X):
    """Section 3.1: what the data look like, i.e. the properties the design choices depend on (src/eda.py)."""
    E = cfg["eda"]; cash = cfg["masking"]["cash_ratio"]; need = cfg["detectors"]["min_periods_layer3"]
    n28 = cfg["simulation"]["h5_training_length"]; M = cfg["edgar"]["monitor_quarters"]
    hl, rp, se, dn, cv, et = (X[k] for k in ["history_length", "ratio_properties", "seasonality", "denominators", "covariance", "event_time"])
    o = [f'<h3>3.1 What the data look like: the exploratory analysis behind the design <span class="badge info">{RETRO}</span></h3>']
    o.append(box("What EDA existed, and what was added",
                 "<p><b>Before the pre-registration</b> the EDA covered only the two case studies (<code>src/run_nmc.py</code>, "
                 "<code>src/run_jyoti.py</code>, report Section 4): ratio panels, red-flag tables and correlations for NMC and Jyoti. That "
                 "work motivated the <i>architecture</i>: a cash-first detector (Jyoti's cash broke first) plus a balance-sheet detector "
                 "(NMC's cash stayed calm). It did <b>not</b> examine the statistical properties that the detector settings depend on "
                 "(history length, persistence, seasonality, tails, missing values, small denominators), and the US panel was never "
                 "explored before it was analysed.</p>"
                 f"<p><b>Added on {e(E['date'])}, after all results</b> (<code>src/eda.py</code>, config <code>eda</code>, tables "
                 "<code>eda_*.csv</code>): a descriptive EDA of exactly those properties. It uses each EDGAR firm's <i>training</i> "
                 "quarters only, so it never looks at the window in which alarms are counted (the event-time table below is the one "
                 "exception, and says so). It changes no setting and no verdict. It shows whether the data are consistent with each "
                 "choice; it cannot show that the choice was made for that reason.</p>", "warn"))

    # history length
    o.append("<p><b>(a) How much history is there?</b> Usable training periods of the cash ratio, by data source:</p>")
    o.append(table(["Sample", "Unit", "Firms", "Median (10th–90th pct)", "Range", f"Share with ≥ {need}", f"Share with ≥ {n28}"],
                   [[e(r["sample"]), e(r.unit), str(int(r.firms)), f"{r['median']:g} ({r.p10:g}–{r.p90:g})", f"{r['min']:g}–{r['max']:g}",
                     pct(r.share_at_least_min_periods, 0), pct(r.share_at_least_h5_n, 0)] for _, r in hl.iterrows()]))
    allf = hl.iloc[0]; anf = hl[hl["sample"] == "EDGAR: analysed failed firms"].iloc[0]
    o.append(f"<p>Every source is short. Even after requiring ≥ {need} usable quarters, the analysed US firms have a median of "
             f"{anf['median']:g} training quarters, and only {pct(anf.share_at_least_h5_n, 0)} reach the n = {n28} at which H4/H5 were "
             f"judged in simulation. Of all firms that passed Layer 0, {pct(allf.share_at_least_min_periods, 0)} have ≥ {need}. The annual "
             "sources (cohort, NMC, Jyoti) never reach the minimum. <b>This is the data fact behind the whole calibration study:</b> "
             "short histories are the normal case, not an edge case, so a threshold method must be judged at short n.</p>")

    # ratio properties
    a = rp[rp.role == "all analysed"].set_index("ratio")
    o.append("<p><b>(b) Which ratios are usable, and how do they behave?</b> All analysed US firms, training quarters only. "
             "Coverage = share of training quarters with a value. Persistence = median per-firm lag-1 autocorrelation of the level. "
             "Tails = median per-firm excess kurtosis of the quarter-on-quarter change (0 for a normal distribution).</p>")
    o.append(table(["Ratio", "Coverage", f"Firms with ≥ {need} values", "Persistence (lag-1 ACF of level)", "Lag-1 ACF of change",
                    "Tails: median excess kurtosis", "Share of firms with excess kurtosis > 3"],
                   [[f"<code>{e(k)}</code>", pct(r.share_quarters_observed, 0), pct(r.share_firms_with_min_periods, 0),
                     f"{r.median_acf1_level:.2f}", f"{r.median_acf1_change:.2f}", f"{r.median_excess_kurtosis_change:.2f}",
                     pct(r.share_firms_kurtosis_above_normal_x3, 0)] for k, r in a.iterrows()]))
    best_cov = a.share_quarters_observed.idxmax()
    o.append(f"<p>What this shows. <code>{e(cash)}</code> has the best coverage ({pct(a.loc[cash, 'share_quarters_observed'], 0)} of "
             f"quarters) and the lightest tails of the cash-flow ratios (excess kurtosis {a.loc[cash, 'median_excess_kurtosis_change']:.2f} "
             f"against {a.loc['ocf_to_total_debt', 'median_excess_kurtosis_change']:.2f} for OCF ÷ total debt). Caveat: the analysed sample "
             "was selected on cash-ratio coverage, so its coverage is flattered"
             + ("" if best_cov == cash else f" (and {e(best_cov)} covers more)") + ". Liquidity cover is almost never available "
             f"({pct(a.loc['liquid_to_st_debt', 'share_quarters_observed'], 0)}), because short-term debt is rarely tagged. Every ratio is "
             f"persistent (lag-1 ACF of the level {a.median_acf1_level.min():.2f}–{a.median_acf1_level.max():.2f}), so values cannot be treated "
             "as independent draws around a fixed mean. That is why the model has a moving level and an AR(1) term. Interest cover and "
             f"Z''-EM have the heaviest tails ({pct(a.loc['ebitda_interest_cover', 'share_firms_kurtosis_above_normal_x3'], 0)} and "
             f"{pct(a.loc['altman_z2_em', 'share_firms_kurtosis_above_normal_x3'], 0)} of firms above 3). The cash ratio itself has "
             f"{pct(a.loc[cash, 'share_firms_kurtosis_above_normal_x3'], 0)} of firms above 3: moderately non-normal, consistent with the "
             "partial transfer of the Gaussian-simulation correction (Section 6).</p>")

    # seasonality
    s4 = se[se.lag_quarters == 4].set_index("input"); s1 = se[se.lag_quarters == 1].set_index("input")
    ttm, sq = s4.index[0], s4.index[1]
    o.append(f"<p><b>(c) Seasonality: why trailing-twelve-month flows.</b> With single-quarter cash flow the quarter-on-quarter change "
             f"of the cash ratio has a lag-4 autocorrelation of {s4.loc[sq, 'median_acf_of_change']:+.2f} (positive for "
             f"{pct(s4.loc[sq, 'share_firms_positive'], 0)} of firms), the signature of a seasonal pattern, and a lag-1 value of "
             f"{s1.loc[sq, 'median_acf_of_change']:+.2f}. With TTM flows these are {s4.loc[ttm, 'median_acf_of_change']:+.2f} and "
             f"{s1.loc[ttm, 'median_acf_of_change']:+.2f}, and the typical size of a change falls from {s4.loc[sq, 'median_sd_of_change']:.2f} to "
             f"{s4.loc[ttm, 'median_sd_of_change']:.2f}. TTM removes the seasonality that a simple local-level model cannot represent, at the "
             "cost of overlap between consecutive values (checked in Section 8.6: no detectable effect on discrimination).</p>")

    # denominators
    d = dn.set_index(["item", "role"])
    o.append(f"<p><b>(d) Small denominators: why two ε values.</b> Among training quarters that report interest, interest is at or below "
             f"{pct(cfg['ratios']['denominator_eps_rel_assets']['activity'], 0)} of total assets in {pct(d.loc[('interest', 'control'), 'share_below_activity_eps'], 0)} "
             f"of control quarters ({pct(d.loc[('interest', 'failed'), 'share_below_activity_eps'], 0)} for failed firms), but at or below "
             f"{pct(cfg['ratios']['denominator_eps_rel_assets']['default'], 1)} in only {pct(d.loc[('interest', 'control'), 'share_below_default_eps'], 0)}. "
             "A single ε = 1% would therefore blank interest cover for about a quarter of quarters, mostly in healthy low-debt firms. "
             f"Revenue is that small in only {pct(d.loc[('revenue', 'failed'), 'share_below_activity_eps'], 0)}–"
             f"{pct(d.loc[('revenue', 'control'), 'share_below_activity_eps'], 0)} of quarters, so the stricter ε for sales-based ratios costs little.</p>")

    # covariance
    c = cv.set_index("dimension_p").loc["all"]
    o.append(f"<p><b>(e) Covariance for the multivariate detector.</b> For the {int(c.firms)} US firms where Layer 4 runs, the training "
             f"innovations give a median of {c.median_n_complete:g} complete rows (minimum {c.min_n_complete:g}) for 2–4 ratios: about "
             f"{c.median_n_over_p:g} observations per dimension. The sample covariance is usually well-conditioned (median condition number "
             f"{c.median_cond_sample:.1f}; 90th percentile {c.p90_cond_sample:.1f}). Ledoit–Wolf's estimated shrinkage intensity has median "
             f"{c.median_shrinkage:.2f}: for most firms it shrinks the covariance <b>all the way to a scaled identity</b>. In practice the "
             "multivariate detector therefore treats the ratios' innovations as uncorrelated. That is stable, but it discards the "
             "correlation information that a joint detector is meant to use. This was not known before this EDA.</p>")

    # event time
    o.append(f"<p><b>(f) When does the decline happen?</b> Median cash ratio of analysed firms by quarters before the event "
             f"(<i>this table uses the monitored window and outcomes</i>; descriptive only):</p>")
    o.append(table(["Quarters before event", "Failed firms: median (n)", "Controls: median (n)", "Gap", "Inside the monitored window?"],
                   [[f"{int(r.tau_quarters)}", f"{r.failed_median:.2f} ({int(r.failed_firms)})", f"{r.control_median:.2f} ({int(r.control_firms)})",
                     f"{r.gap_control_minus_failed:.2f}", "yes" if r.in_monitored_window else "no (training)"] for r in et.itertuples()]))
    first, last4 = et.iloc[0], et[et.tau_quarters == -4].iloc[0]
    o.append(f"<p>Failed firms already sit below controls {int(-first.tau_quarters)} quarters out (gap {first.gap_control_minus_failed:.2f}: "
             f"the 'weak all along' finding), and the gap then widens steadily to {last4.gap_control_minus_failed:.2f} four quarters before "
             f"the event. The decline is gradual and spread over several years, not a single jump, and most of it falls inside a "
             f"{M}-quarter window. Both facts support a cumulative detector (CUSUM) over a {M}-quarter window, and both also explain why "
             "a 'worse than its own past' detector misses part of the signal.</p>")
    return o


def design_table(cfg, X, sm, bl, short_f, mt_, M_, au_):
    """Design-choice table: value, rationale, evidence from the data, and the status of the justification."""
    dc, sc, rb = cfg["detectors"], cfg["simulation"], cfg["robustness"]; cms = cfg["cost"]["c_miss"]
    cash = cfg["masking"]["cash_ratio"]; need = dc["min_periods_layer3"]
    rp = X["ratio_properties"]; a = rp[rp.role == "all analysed"].set_index("ratio")
    hl = X["history_length"]; allf = hl.iloc[0]
    se = X["seasonality"]; s4 = se[se.lag_quarters == 4].set_index("input")
    dn = X["denominators"].set_index(["item", "role"]); cv = X["covariance"].set_index("dimension_p").loc["all"]
    mc = X["monte_carlo"]; et = X["event_time"]
    kacf = X["kalman_acf"]; k1 = kacf[(kacf.lag_quarters == 1) & (kacf["input"] == kacf["input"].iloc[0])].set_index("errors")
    mcB = mc[mc.use.str.startswith("bootstrap threshold") & (mc.tail_probability == 0.10)].iloc[0]
    mcP = mc[mc.use.str.startswith("parametric") & (mc.tail_probability == 0.10)].iloc[0]
    mcS = mc[mc.use.str.startswith("simulated false-alarm")].iloc[0]
    n1 = int(dc["training"]["train_fraction"] * need); pool = need - n1
    f16 = et[et.tau_quarters == -M_].iloc[0] if (et.tau_quarters == -M_).any() else et.iloc[0]
    f4 = et[et.tau_quarters == -4].iloc[0]
    pre, retro, pre_r = ("pre-registered with a reason", "pre-registered; reason added retrospectively", "logged decision")
    tag = lambda s: f'<span class="small"><i>{s}</i></span>'  # noqa: E731
    rows = [
        ["Change detector: CUSUM (not Shewhart or EWMA)", "one-sided CUSUM (Layer 3); Crosier MCUSUM (Layer 4)",
         "The target is a slow, persistent drift. A Shewhart chart looks at one period at a time and only sees large jumps. A CUSUM "
         "adds up small deviations and is the optimal detector of a sustained shift (Page 1954; Moustakides 1986) [verify before citing]. "
         "EWMA would be a reasonable alternative with similar power.",
         f"Failed firms' median cash ratio falls step by step ({f16.failed_median:.2f} → {f4.failed_median:.2f} from {M_} to 4 quarters out; "
         "table 3.1f): a gradual slide, not a jump.", tag(retro)],
        ["CUSUM reference value k", f"k = {dc['cusum']['k']} (SD units)",
         "k is half the shift the chart is tuned to detect fastest; k = 0.5 targets a one-SD shift, the textbook default (Montgomery) "
         "[verify before citing]. A smaller k reacts to smaller drifts but alarms more.",
         f"Robustness: k ∈ {{{', '.join(f'{x:g}' for x in rb['cusum_k_grid'])}}} leaves the ranking of methods unchanged (Section 7).", tag(retro)],
        ["Nominal false-alarm rates", ", ".join(pct(x, 0) for x in dc["nominal_false_alarm_rates"]),
         "Conventional test levels. Two levels feed two alert tiers: the looser one for an early, cheap 'amber' look and the stricter "
         "one for 'red'. The actual rate on real data is measured, not assumed.",
         "None needed: this is a policy choice. H4 shows the actual rate is far above either.", tag(retro)],
        ["Monitored window", f"{M_} quarters (EDGAR); {sc['monitor_length']} periods (simulation); h calibrated for {dc['monitor_horizon']} steps",
         "Long enough to cover the multi-year run-up to failure and short enough to leave training data. The same length everywhere, so "
         "simulation and real data are comparable.",
         f"The failed-vs-control gap widens from {f16.gap_control_minus_failed:.2f} at the window's start to {f4.gap_control_minus_failed:.2f} four quarters before the event "
         "(table 3.1f). Calibration horizon equals the window (Section 8.5).", tag("blueprint §10.1; reason added retrospectively")],
        ["Cash ratio", f"<code>{e(cash)}</code> (OCF ÷ current liabilities)",
         "Cash-first: operating cash is hard to manipulate. Current liabilities are the obligations cash must meet within a year, are "
         "reported by almost every firm and never zero. Debt-based denominators depend on debt tags that are often missing (decision #21).",
         f"Best coverage of the cash-flow ratios ({pct(a.loc[cash, 'share_quarters_observed'], 0)} of quarters vs "
         f"{pct(a.loc['ocf_to_total_debt', 'share_quarters_observed'], 0)} for OCF ÷ debt) and lighter tails (excess kurtosis "
         f"{a.loc[cash, 'median_excess_kurtosis_change']:.2f} vs {a.loc['ocf_to_total_debt', 'median_excess_kurtosis_change']:.2f}).",
         tag("blueprint; reason added retrospectively")],
        ["Flows as trailing-twelve-month sums (EDGAR)", f"<code>edgar.flows: {e(cfg['edgar']['flows'])}</code>",
         "Quarterly cash flow is seasonal; the local-level model has no seasonal term. TTM sums remove seasonality.",
         f"Lag-4 autocorrelation of the change: {s4.loc[s4.index[1], 'median_acf_of_change']:+.2f} with single quarters vs "
         f"{s4.loc[s4.index[0], 'median_acf_of_change']:+.2f} with TTM (table 3.1c).", tag(pre_r + " #20")],
        ["Model: local level + AR(1), parameters frozen", "statsmodels UnobservedComponents, fitted on training only",
         "A per-firm baseline must allow the 'normal' level to differ between firms and drift slowly, and must handle missing quarters "
         "without filling them in (the Kalman filter skips them).",
         f"Ratios are persistent (lag-1 ACF of the level {a.median_acf1_level.min():.2f}–{a.median_acf1_level.max():.2f}) and only "
         f"{pct(a.loc[cash, 'share_quarters_observed'], 0)} of cash-ratio training quarters are observed. Caveat: the adaptive level also "
         f"absorbs slow decline (Section 6, Finding 3).", tag("pre-registered; missing-data reason stated (§8)")],
        ["Bootstrap resamples", f"{dc['bootstrap']['n_resamples']:,} (thresholds); {cfg['evaluation']['ci_resamples']:,} (EDGAR CIs); {sc['ci_resamples']:,} (simulation CIs)",
         "Enough that resampling noise is small next to the sampling noise of the firms themselves.",
         f"Monte Carlo SE of a 10% tail with {int(mcB.resamples):,} resamples: {mcB.mc_se_points:.2f} points, against a sampling SE of "
         f"{mcS.mc_se_points:.2f} points from {int(mcS.resamples)} simulated firms. The {int(mcP.resamples)} parametric refits (decision #15) "
         f"carry {mcP.mc_se_points:.2f} points.", tag(retro)],
        ["Block bootstrap and block length", f"circular block bootstrap; b = max({dc['bootstrap']['min_block']}, ⌈n<sup>1/3</sup>⌉). "
         f"Simulation: {e(sm.block_length_by_n)}; EDGAR: {int(bl.min())}–{int(bl.max())} (median {bl.median():g})",
         "Resampling runs of consecutive errors keeps any serial dependence that the model left behind; the n<sup>1/3</sup> rate is the "
         "standard order for block length, and data-driven selectors are unstable at these lengths (decision #12).",
         f"Raw ratios are strongly autocorrelated, but the model's standardised errors on healthy firms are close to white "
         f"(median per-firm lag-1 ACF {k1.loc['training (in-sample)', 'median_per_firm_acf']:.2f}; Section 8.6). So blocks are cheap "
         "insurance here rather than essential.", tag("block length: " + pre_r + " #12; block vs iid: reason added retrospectively")],
        ["Covariance estimator", "Ledoit–Wolf shrinkage of the training innovations",
         "With few observations per ratio the sample covariance can be nearly singular, and its inverse (used by the MCUSUM) explodes. "
         "Shrinkage guarantees a stable inverse.",
         f"About {cv.median_n_over_p:g} observations per dimension; sample condition number median {cv.median_cond_sample:.1f} "
         f"(90th pct {cv.p90_cond_sample:.1f}). Median shrinkage {cv.median_shrinkage:.2f}: usually all the way to a scaled identity, so the "
         "correlations are effectively ignored (table 3.1e). The stability goal is met, at the cost of discarding correlation.",
         tag("pre-registered; reason added retrospectively, and qualified by the EDA")],
        ["Near-zero denominator rule", ", ".join(f"ε = {v:g} ({k})" for k, v in cfg["ratios"]["denominator_eps_rel_assets"].items()) + " × total assets",
         "Ratios over tiny denominators are meaningless and produce absurd values; the cut-off must not blank normal values.",
         f"ε = {cfg['ratios']['denominator_eps_rel_assets']['activity']:g} would blank interest cover in "
         f"{pct(dn.loc[('interest', 'control'), 'share_below_activity_eps'], 0)} of control quarters; ε = "
         f"{cfg['ratios']['denominator_eps_rel_assets']['default']:g} blanks {pct(dn.loc[('interest', 'control'), 'share_below_default_eps'], 0)} (table 3.1d).",
         tag(pre_r + " #4")],
        ["Training split for the calibrator", f"first {pct(dc['training']['train_fraction'], 0)} fits, the rest gives pseudo-out-of-sample errors",
         "In-sample errors are too small, because the model was fitted to them; errors on held-back data include estimation error, "
         "which is what a future quarter will face.",
         f"At the minimum of {need} periods this leaves {n1} to fit and {pool} errors to resample, which is why short histories are "
         "fragile (H4).", tag("pre-registered (blueprint §7); reason added retrospectively")],
        ["Minimum training history", f"≥ {need} usable periods (≥ {mt_} quarters on EDGAR)",
         f"The model has four variance/AR parameters to estimate, and the calibrator splits training again ({n1} + {pool} at the minimum). "
         "Fewer periods leave almost nothing to resample.",
         f"{pct(allf.share_at_least_min_periods, 0)} of US firms that passed Layer 0 meet it; {short_f} failed firms were dropped. A "
         "deliberate trade-off between model stability and sample size, and a selection toward established filers (Section 3).", tag(retro)],
        ["Cost of a miss vs a false alarm", f"C_FA = {cfg['cost']['c_fa']}, C_M ∈ {{{', '.join(map(str, cms))}}}",
         "A missed failure costs a lender the loss on the loan; a false alarm costs a review. Published estimates put the ratio well above "
         "one (Altman, Haldeman &amp; Narayanan 1977) [verify before citing], and the true ratio is lender-specific, so a range is reported "
         "rather than one value.",
         "None needed: a policy range. The conclusions do not change across it (Section 8.1).", tag(retro)],
    ]
    return table(["Choice", "Value used", "Why", "What the data show", "Status of the justification"], rows, "design")


def viva_extra(cfg, PH, ed, au, H, m, okf, okc):
    """Additional likely viva questions (post-hoc review); every number is read from a table."""
    a_k = au.loc["cash CUSUM on Kalman errors (pre-registered)"]; a_s = au.loc["cash CUSUM on deviations from the training mean (non-adaptive)"]
    w = PH["winsor"].set_index("method"); tt = PH["ttm_overlap"].set_index("input"); sw = PH["static_weakness"].set_index(["method", "role"])
    fl = PH["filing_lead"].set_index("method"); pa = PH["paired_rules"]; pa = pa[pa.other == "Altman Z''-EM distress"]
    mi = PH["mi_healthy"]; simH = mi[(mi["sample"].str.startswith("simulation")) & (mi.group == "H")].iloc[0]
    au_, ac_ = ed.loc["amber (uncorrected)"], ed.loc["amber (n-corrected)"]
    M = cfg["edgar"]["monitor_quarters"]; ttm0, sq = tt.index[0], tt.index[1]
    rl = sw.loc[("any Layer 2 rule", "failed")]
    return [
        ("Why use the Kalman model if ARIMA had more power in H3?",
         f"Honestly: not because of power. ARIMA beat it in simulation (S1 at a matched 10% false-alarm rate: {pct(m[('S1', 'arima')])} vs "
         f"{pct(m[('S1', 'kalman')])}). The local-level model was pre-registered because it handles missing periods natively, which real "
         "filings need. On real data its adaptive level is the leading suspect for the weak discrimination: a non-adaptive CUSUM scored "
         f"AUC {a_s.auc:.2f} vs {a_k.auc:.2f}, with overlapping CIs. A fair ARIMA / non-adaptive comparison on new data is future work."),
        ("Why did the correction only take real false alarms from about one half to one third?",
         f"The amber tier went from {pct(au_.false_alarm)} to {pct(ac_.false_alarm)}. The inflation factor was learned in a Gaussian AR(1) "
         "simulation and real ratios are rougher. Post-hoc checks rule out three simple explanations: the calibration horizon matches the "
         f"window; winsorising moved amber false alarms only from {pct(w.loc['amber (uncorrected)', 'base_false_alarm'])} to "
         f"{pct(w.loc['amber (uncorrected)', 'false_alarm'])}; and non-overlapping inputs give the cash detector "
         f"{pct(tt.loc[sq, 'false_alarm'])} false alarms vs {pct(tt.loc[ttm0, 'false_alarm'])} with TTM. What remains is model "
         "misspecification (level shifts, accounting events), which is why thresholds calibrated on real healthy firms are the next step."),
        ("Is the multivariate detector direction-blind? What would a one-sided version do?",
         "Yes. Crosier's MCUSUM accumulates the length of the deviation vector, so a run of improvements can alarm (decision log #16, "
         "accepted as a limitation). A one-sided version, accumulating only deterioration (e.g. only components in the bad direction, or a "
         "projection on a pre-set deterioration direction), would remove alarms caused by improvement and so should lower false alarms. It "
         "was not built or tested, so I cannot say by how much (future work)."),
        ("What would you do about heavy tails?",
         "The heavy-tail simulation scenario (S5) was cut for time. On real data the extreme values sit mainly in debtor days and Z''-EM. "
         f"Post hoc, winsorising every ratio moved the amber tier from {pct(w.loc['amber (uncorrected)', 'base_detection'])} / "
         f"{pct(w.loc['amber (uncorrected)', 'base_false_alarm'])} to {pct(w.loc['amber (uncorrected)', 'detection'])} / "
         f"{pct(w.loc['amber (uncorrected)', 'false_alarm'])} (failed / healthy flagged) and left the rules unchanged. So tails are not the "
         "main problem. The principled fix would be a Student-t state-space model or robust (Huberised) innovations."),
        ("Are lead times of three to four years real early warnings?",
         f"{'Mostly not' if rl.of_those_already_on_before_window >= 0.5 else 'Partly'}. Of the failed firms first flagged by a rule in the first three monitored quarters, {pct(rl.of_those_already_on_before_window, 0)} "
         "were already flagged before monitoring began: that is static weakness. Measured from filing dates, the median rule lead is "
         f"{fl.loc['any Layer 2 rule', 'median_lead_quarters_filing_date']:.1f} quarters rather than "
         f"{fl.loc['any Layer 2 rule', 'median_lead_quarters_period_end']:g}."),
        ("Do the rules beat Altman?",
         "Not distinguishably. The point estimates favour the rules at every cost ratio, but in a paired bootstrap over matched firms "
         f"every CI includes zero (e.g. detection {pts(pa[pa.metric == 'detection'].difference.iloc[0])} points, 95% CI "
         f"{pts(pa[pa.metric == 'detection'].ci_lo.iloc[0])} to {pts(pa[pa.metric == 'detection'].ci_hi.iloc[0])}). They do beat the "
         "statistical tiers on expected cost."),
        ("Isn't the masking index positive for almost everyone?",
         f"Yes, with uncorrected thresholds: healthy simulated firms have median peak MI {simH.median_peak_mi:.2f} and "
         f"{pct(simH.share_mi_above_0)} are above zero. Only a healthy-centred MI (minus the healthy median, or above a healthy percentile) is "
         "interpretable, and the structural side shares information with cash. Section 8 reports both."),
        ("Your healthy controls never went bankrupt. Doesn't that flatter the results?",
         f"Yes. Controls are survivors (no bankruptcy filing through 2023), so separating them from {okf} failed firms is easier than a "
         f"lender's real problem. With only {okc} controls the false-alarm CIs are also wide. This is stated as a limitation."),
    ]


def build() -> str:
    H = T("hypotheses").set_index("id")
    cal = T("sim_calibration_curve"); fix = T("sim_fixes"); infl = T("sim_inflation_fit"); mt = T("sim_matched_fa")
    sep = T("sim_masking_separation"); fp = T("sim_fingerprints").set_index("scenario"); meta = T("sim_meta").iloc[0]
    arch = T("real_archetypes").set_index("firm"); l2 = T("layer2_first_trip"); l3 = T("real_layer3_status").set_index("firm")
    ed = T("edgar_detection_summary").set_index("method"); ef = T("edgar_funnel").set_index("step")
    eam = T("edgar_archetype_mi").set_index("arch_archetype"); ems = T("edgar_mi_separation").iloc[0]
    au = T("robust_edgar_auc").set_index("score")
    V = T("robust_edgar_variants"); ru = T("robust_edgar_rules").set_index("rule_set"); rp = T("robust_edgar_reproduce").iloc[0]
    jf = T("robust_jyoti_fill").set_index(["fill_missing", "rule"]); ae = T("robust_case_alt_events").set_index(["firm", "definition"])
    hd = T("robust_nmc_hidden_debt").set_index("hidden_debt_multiple"); SC = T("sanity_checks")
    rn = pd.read_csv(P.tables / "ratios_nmc.csv", index_col=0); rj = pd.read_csv(P.tables / "ratios_jyoti.csv", index_col=0)
    sc, dc, rb = cfg["simulation"], cfg["detectors"], cfg["robustness"]
    n28 = sc["h5_training_length"]
    c28 = cal[(cal.n == n28) & (cal.detector == "kalman_cusum")].set_index("nominal")
    k28 = cal[cal.detector == "kalman_cusum"]
    f28 = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    f28_5 = fix[(fix.n == n28) & (fix.nominal == 0.05)].set_index("method")
    m = mt.set_index(["scenario", "detector"]).detection
    jfirst = l2[l2.firm == "jyoti"].set_index("rule"); nfirst = l2[l2.firm == "nmc"].set_index("rule")
    lcov = rn.liquid_to_st_debt
    s12, s14 = sep.iloc[0], sep.iloc[1]
    am_u, am_c, rd_u, alt, rule = (ed.loc[k] for k in ["amber (uncorrected)", "amber (n-corrected)", "red (uncorrected)",
                                                         "Altman Z''-EM distress", "any Layer 2 rule"])
    nfail = int(ef.loc["failed firms in panel", "count"]); nctrl = int(ef.loc["controls matched", "count"])
    okf, okc = int(am_u.failed_firms), int(am_u.controls)
    jy_ocf = jfirst.loc["OCF < 0"]; jy_alt = jfirst.loc["Altman Z''-EM < 4.35 (distress)"]
    nw_neg = int(rj.index[rj.equity_to_assets < 0][0])
    i28 = infl[(infl.n == n28) & (infl.nominal == 0.10)].iloc[0]
    cnt = SC.status.value_counts()
    co = T("cohort_archetypes"); cf, cc = co[co.role == "failed"], co[co.role == "control"]
    cev = pd.to_datetime(cf.event_date).dt.year
    cun = [r.name for r in cf.itertuples() if r.firm_id not in set(cc.matched_to)]
    cw = T("cohort_rule_detection").set_index("signal").loc["any Layer 2 rule, common years only"]
    a_k = au.loc["cash CUSUM on Kalman errors (pre-registered)"]; a_s = au.loc["cash CUSUM on deviations from the training mean (non-adaptive)"]
    a_l = au.loc["lowest cash ratio in the window (level)"]; a_z = au.loc["lowest Altman Z''-EM in the window (level)"]
    a_m = au.loc["multivariate MCUSUM (pre-registered)"]
    lvs = au[au.index.str.startswith("median share")].iloc[0]
    cr = "ocf_to_current_liabilities"
    masked_from = hd.index[hd.archetype == "Masked"].min()
    rv = rb["nmc_revealed_debt_usd_bn"]
    tol = sc["h5_tolerance_points"]
    lc = T("robust_edgar_level_vs_change"); lc = lc[lc["sample"] == "analysed"].set_index(["ratio", "role"])
    PH = {n: T(f"posthoc_{n}") for n in ["paired_rules", "fixed_horizon", "static_weakness", "persistence", "filing_lead",
                                          "ttm_overlap", "kalman_acf", "calibration_horizon", "winsor", "industry",
                                          "mi_healthy", "sim_self_starting"]}
    pr = PH["paired_rules"]; pa = pr[pr.other == "Altman Z''-EM distress"].set_index("metric")
    cms = cfg["cost"]["c_miss"]
    alt_indist = all(pa.loc[f"expected_cost_cm{x}", "verdict"].startswith("not") for x in cms)
    stat_tiers = [m for m in pr.other.unique() if m != "Altman Z''-EM distress"]
    rules_beat_stats = all(pr[(pr.other == m) & pr.metric.str.startswith("expected_cost")].verdict.str.startswith("rules better").all()
                           for m in stat_tiers)
    amber_chance = [k for k in ["amber (uncorrected)", "amber (n-corrected)"] if ed.loc[k, "diff_ci_lo"] <= 0]
    l2r, arc = cfg["layer2"]["rules"], cfg["archetype"]["masked"]["structural_flags"]; az = cfg["altman_z2_em"]
    mt_, M_ = cfg["edgar"]["min_training_quarters"], cfg["edgar"]["monitor_quarters"]
    fun = ef.reset_index()
    short_f = int(fun[fun.step.str.startswith("failed: too little training")]["count"].sum())
    short_c = int(fun[fun.step.str.startswith("control: too little training")]["count"].sum())
    L1 = T("edgar_layer1_panel"); efr = T("edgar_firm_results")
    fa_, ca_ = _ev.analysed(efr)
    n_all = L1.groupby("role").cik.nunique()
    X = {n: T(f"eda_{n}") for n in ["history_length", "ratio_properties", "seasonality", "denominators", "covariance",
                                    "monte_carlo", "event_time"]}
    X["kalman_acf"] = T("posthoc_kalman_acf")

    o = []
    # ------------------------------------------------------------------ cover
    o.append(f"""<div class="cover"><h1>Judging a firm against its own history</h1>
<div class="sub">An explainer of the early-warning project: what was done, why, what was found, how it was checked,
and what is still open</div>
<div class="small">Foundations of Data Science course project · pre-registration dated {e(cfg['prereg_date'])} · random seed
{cfg['seed']} · generated by <code>src/reporting/write_explainer.py</code>. Every number below is read from
<code>reports/tables/</code> or <code>configs/config.yaml</code> when this file is built, never typed by hand.</div></div>""")
    o.append(box("How to read this document",
                 "<p>Section 1 is the whole project on one page. Section 2 explains every technical idea in plain words. "
                 "Sections 3–5 walk through the data (Section 3.1: the exploratory analysis behind the design choices), the method "
                 "(every design choice and its justification) and the results hypothesis by hypothesis. Section 6 tests the "
                 "method on real US companies. Section 7 shows the robustness checks; Section 8 holds additional post-hoc checks "
                 "added after a review of this document (each one labelled post-hoc / exploratory); Section 9 lists the sanity "
                 "checks. Sections 10–13 cover limits, the remaining to-do list, how to reproduce everything, and likely viva "
                 "questions.</p>"))
    o.append('<ol class="toc"><li>The project on one page</li><li>Key ideas in plain words</li><li>The data</li>'
             '<li>The method, layer by layer</li><li>Results, hypothesis by hypothesis</li><li>Does it work on real firms? (US EDGAR panel)</li>'
             '<li>Robustness: do the conclusions survive changes?</li><li>Additional checks (post-hoc)</li>'
             '<li>How we know it is right: sanity and common-sense checks</li><li>Limitations</li><li>What is left to do</li><li>Reproducing everything</li><li>Viva preparation</li></ol>')

    # ------------------------------------------------------------------ 1
    o.append('<h2>1. The project on one page</h2>')
    o.append("<p><b>The problem.</b> Lenders and auditors want to know early when a company is heading for a cash-flow crisis. "
             "The classic tools (Altman's Z-score, logit and hazard models) compare a firm with many other firms and need large "
             "samples of past failures. For mid-sized firms in India or the Gulf there are often no good peers and only a few years "
             "of history.</p>")
    o.append("<p><b>The idea.</b> Instead of comparing a firm with others, compare it with <i>its own past</i>. Learn what "
             "'normal' looks like for this one firm, then raise an alarm when its financial ratios drift persistently away from "
             "normal in the bad direction. The project builds this as a stack of layers, from simple ratio checks to statistical "
             "change detectors, and asks two questions: <i>does a firm's own cash-flow history warn early?</i> and <i>when does "
             "cash flow hide the problem instead?</i></p>")
    o.append("<p><b>The two stories that motivated it.</b> At Jyoti Structures (India, insolvency admitted "
             f"{e(cfg['firms']['jyoti']['event_date'])}) operating cash flow turned negative in FY{int(jy_ocf.first_trip_fy)}, "
             f"{jy_ocf.lead_years:.2f} years before the event. At NMC Health (UAE/UK, administration {e(cfg['firms']['nmc']['event_date'])}) "
             "reported cash flow looked healthy right up to the collapse; the trouble was hidden in the balance sheet "
             "(and, as it turned out, in debt that was never reported).</p>")
    o.append("<p><b>What we found, in five lines.</b></p><ol>"
             f"<li><b>The standard way of setting alarm thresholds is badly wrong with short histories.</b> Thresholds meant to "
             f"give 10% false alarms gave {pct(c28.loc[0.10, 'empirical_fa'])} on simulated healthy firms with {n28} periods of history. "
             f"Over-alarming of control charts with estimated parameters is known in the quality-control literature; the project "
             f"confirms and quantifies it for short financial histories (Section 5).</li>"
             f"<li><b>It can be fixed cheaply.</b> Three corrections bring the false-alarm rate to {pct(f28.loc['inflation_factor', 'empirical_fa'])}, "
             f"{pct(f28.loc['conformal', 'empirical_fa'])} and {pct(f28.loc['parametric_bootstrap', 'empirical_fa'])}; the best one costs "
             f"only {f28.loc['parametric_bootstrap', 'detection_cost_points']:.1f} points of detection.</li>"
             f"<li><b>Watching several ratios beats watching cash alone when cash is quiet or massaged</b>, but loses when the "
             f"problem is in cash itself.</li>"
             f"<li><b>The case studies match the two stories:</b> cash warned early at Jyoti; at NMC only liquidity cover "
             f"(cash ÷ short-term debt) warned, from FY{int(lcov.index[lcov < 1][0])}.</li>"
             f"<li><b>On {okf} real US bankruptcies and {okc} healthy controls, the simple pre-set rules had "
             f"{'a lower expected cost than every statistical alarm tier at every cost ratio (paired 95% CIs exclude zero)' if rules_beat_stats else 'a lower expected cost than the statistical tiers, though not always with a CI that excludes zero'}"
             f"{', but they cannot be told apart from Altman Z' + chr(39) * 2 + '-EM (paired CIs include zero)' if alt_indist else ''}.</b> "
             f"The amber statistical alarms {'showed no detectable advantage over chance' if amber_chance else 'separated the groups'} "
             f"with {okf} failed vs {okc} healthy firms ({pct(am_u.detection)} vs {pct(am_u.false_alarm)} flagged; 95% CI of the difference "
             f"{pts(am_u.diff_ci_lo)} to {pts(am_u.diff_ci_hi)} points). A post-hoc diagnosis suggests why (Section 6).</li></ol>")
    rows = []
    plain = {"H1": "Does Jyoti's own cash history warn ≥ 2 years early, using the statistical detector?",
             "H2": "At NMC, did cash flags stay silent while balance-sheet flags fired?",
             "H3": "Is watching many ratios better than one when cash is masked?",
             "H4": "Are standard alarm thresholds too trigger-happy with short histories?",
             "H5": "Can a cheap correction fix the thresholds?",
             "H6": "Does the masking index separate 'visible' from 'masked' crises in a cohort?"}
    why = {"H1": "Not enough pre-crisis data to train the detector (needs BSE quarterly FY2010–13).",
           "H2": "Yes, but only one balance-sheet flag fired, so NMC is 'Other', not 'Masked', by our own rule.",
           "H3": "Code verdict supported; interpretation: partially (Section 5). Big win when cash is quiet (true by construction) "
                 "or smoothed, a large loss when cash itself drifts.",
           "H4": "Yes: several times more false alarms than promised, at every history length.",
           "H5": "Yes: all three fixes land within the tolerance; the parametric one costs least detection.",
           "H6": "Cohort collected (India), but annual data are too short for the masking index; rule-level patterns are only "
                 "illustrative."}
    for i in H.index:
        rows.append([f"<b>{i}</b>", e(plain[i]), verdict_badge(H.loc[i, "verdict"]), e(why[i])])
    o.append(table(["", "Question (plain words)", "Verdict", "In one line"], rows))

    # ------------------------------------------------------------------ 2
    o.append('<h2>2. Key ideas in plain words</h2>')
    ideas = [
        ("Financial ratio", "A number that compares two items in the accounts, e.g. operating cash flow ÷ current liabilities. "
         "Ratios make firms of different size comparable and make a firm comparable with itself over time."),
        ("Orientation: lower = worse", "Every ratio is flipped if needed so that a fall always means deterioration. Debtor days "
         "(how long customers take to pay) are bad when high, so the project uses <i>minus</i> debtor days (<code>neg_dso</code>). "
         "This lets one detector look for 'falls' in every ratio."),
        ("Operating cash flow (OCF)", "Cash actually generated by the business in a period, before investment and financing. Profits "
         "can be manipulated with accounting choices; cash is harder to fake. That is why the project is 'cash-first'."),
        ("Event and event time", "The event is the formal insolvency (court admission, administration, or a US bankruptcy 8-K filing). "
         "Event time counts years or quarters before it: τ = 0 is the period containing the event, τ = −1 the one before."),
        ("Rule flag (Layer 2)", f"A simple pre-set test such as 'OCF below zero' or the Layer 2 rule 'EBITDA covers interest less "
         f"than {l2r['ebitda_interest_cover_min']:g} times'. Cheap, transparent, and fixed in advance. (The archetype rule uses its "
         f"own, different interest-cover flag, &lt; {arc['interest_cover_max']:g}; see the table in Section 4.)"),
        ("Altman Z''-EM", "A classic distress score combining working capital, retained earnings, operating profit and equity, each "
         f"scaled by assets. Below {cfg['altman_z2_em']['distress']} is the 'distress zone'. Used as the benchmark everyone knows."),
        ("State-space / Kalman model (Layer 3)", "A model that keeps a running estimate of the 'normal level' of a ratio and predicts "
         "the next value. Its parameters are learned on the training years and then frozen. The <b>standardised forecast error</b> "
         "z = (actual − predicted) ÷ its usual size tells how surprising each new value is."),
        ("CUSUM", "A running total of bad surprises: S<sub>t</sub> = max(0, S<sub>t−1</sub> − z<sub>t</sub> − k). Small surprises are "
         f"absorbed by the allowance k = {dc['cusum']['k']}; a run of bad surprises makes S grow; an alarm sounds when S passes the "
         "threshold h. It catches slow, persistent drifts that no single quarter would reveal."),
        ("Multivariate CUSUM (Layer 4)", "The same idea on several ratios at once (Crosier's MCUSUM), using their joint covariance "
         "(estimated with Ledoit–Wolf shrinkage because the samples are small). It can notice that 'everything is a bit worse'. "
         "Its weakness: it reacts to change in any direction, including improvement."),
        ("False-alarm rate: nominal vs empirical", "The threshold h is chosen to promise, say, a 10% chance of a false alarm on a "
         "healthy firm over the monitored window (the <i>nominal</i> rate). The <i>empirical</i> rate is what actually happens when "
         "you run it on healthy firms. If empirical ≫ nominal, the thresholds are 'miscalibrated' (too trigger-happy)."),
        ("Bootstrap and block bootstrap (Layer 5)", "To set h for one firm, we resample that firm's own past forecast errors many "
         f"times ({dc['bootstrap']['n_resamples']:,} resamples), simulate the CUSUM on each fake future, and take the value exceeded "
         "only 10% of the time. 'Block' means we resample runs of consecutive quarters, not single ones, to keep their dependence."),
        ("Why short histories break it", "With few years of data the model's parameters are guessed with error, and the handful of "
         "past errors under-represents how wild the future can be. Both make h too low, so alarms fire too often. Hypothesis H4 "
         "measures by how much."),
        ("The three fixes (H5)", "<b>Inflation factor</b>: multiply h by a factor that depends on history length, learned in "
         "simulation. <b>Conformal threshold</b>: set h directly from the CUSUM maxima of many similar healthy simulated firms. "
         "<b>Parametric bootstrap with refitting</b>: simulate new histories from the fitted model and re-estimate the model on each, "
         "so the threshold includes the estimation error."),
        ("Masking index (Layer 6)", "MI = (structural detector ÷ its threshold) − (cash detector ÷ its threshold). Positive means "
         "the balance-sheet side is alarming while cash looks calm: the crisis is 'masked'. The <b>fingerprint</b> records which "
         "detector fired first. Two cautions: (1) the 'structural' side (liquidity cover, interest cover, Z''-EM) is not "
         "independent of cash: liquidity cover's numerator is cash and short-term investments and Z''-EM's working capital "
         f"includes cash (in simulation the ratios' innovations are correlated at {sc['innovation_corr']}), so MI compares "
         "overlapping information; (2) with uncorrected thresholds MI is above zero for almost every healthy firm too, so "
         "MI &gt; 0 alone does not mean 'masked' (Section 8)."),
        ("Archetypes", "A pre-set rule labels each failed firm: <b>Visible</b> (OCF negative for two consecutive years or more), "
         "<b>Masked</b> (OCF positive throughout but at least two of three balance-sheet flags trip), or <b>Other</b>."),
        ("Pre-registration", "All hypotheses, rules and thresholds were written down, dated and fingerprinted (SHA-256 hash) before "
         "any detector was run. This prevents tuning the method until it 'works' on the known cases (hindsight bias)."),
        ("Detection, false alarm, expected cost, AUC", "Detection = share of failed firms flagged before the event. False alarm = "
         "share of healthy matched firms flagged over the same window. Expected cost weighs a missed failure as several times worse "
         "than a false alarm. AUC = probability that a random failed firm looks worse than a random healthy one (0.5 = coin flip, "
         "1 = perfect), a threshold-free measure."),
    ]
    o.append(table(["Idea", "What it means here"], [[f"<b>{a}</b>", b] for a, b in ideas]))

    # ------------------------------------------------------------------ 3
    o.append('<h2>3. The data</h2>')
    o.append(table(["Source", "What", "Used for", "How it was checked"], [
        ["<b>NMC Health</b>", f"Annual accounts {int(rn.index.min())}–{int(rn.index.max())}, US$ thousands, as originally reported "
         f"({len(rn)} years)", "Case study for H2 (Layers 0–2)", "Accounting identities; units plausibility; event date verified (English High Court administration order)"],
        ["<b>Jyoti Structures</b>", f"Annual accounts FY{int(rj.index.min())}–FY{int(rj.index.max())}, INR crore, from Screener.in "
         "(FY2017 missing)", "Case study for H1 (Layers 0–2)", "Identities; near-zero denominators masked; FY2017 left missing; event date verified (NCLT order)"],
        ["<b>SEC EDGAR (US)</b>", f"Quarterly XBRL filings: {nfail} firms with a bankruptcy 8-K (Item 1.03) in "
         f"{cfg['edgar']['event_search']['start'][:4]}–{cfg['edgar']['event_search']['end'][:4]}, {nctrl} matched healthy controls",
         "External test of Layers 0–6 on real data", "Identities per firm; first-filed values only (no hindsight restatements); controls never filed for bankruptcy; financials excluded"],
        ["<b>India cohort</b>", f"{len(cf)} listed Indian firms admitted to insolvency {cev.min()}–{cev.max()} (RBI's 2017 large-defaulter "
         f"lists), {len(cc)} size- and industry-matched healthy controls, Jet Airways as negative control; annual, Screener.in",
         "H6 (confirmatory test) and rule-level checks", "Frame and rule fingerprinted before any figure was read; parser reproduces Jyoti exactly; identities; controls never insolvent"],
        ["<b>Simulation</b>", f"{int(meta.firms_per_cell)} synthetic firms per history length and scenario, {sc['n_ratios']} ratios each, "
         f"history lengths {', '.join(map(str, sc['training_lengths']))}", "H3, H4, H5 and the masking index", "Separate seeds for fitting corrections and for testing them; crash count; determinism"],
    ]))
    o.append(f"<p><b>Why the US panel?</b> Layers 3–5 need at least {dc['min_periods_layer3']} usable periods of history. NMC has "
             f"{len(rn)} annual points and Jyoti none before FY14, and the cohort is annual too. EDGAR is "
             "free, machine-readable and large, so it is the place where the models can actually be fitted and tested on real "
             f"companies. Of the {nfail} failed firms, {okf} (and {okc} of their controls) had enough clean quarterly history; the rest "
             "were dropped for documented reasons (too little history, identity failures, model failures) listed in "
             "<code>edgar_funnel.csv</code>. These rules were written into the config before the download.</p>")
    o.append(f"<p><b>What the minimum history means in practice.</b> 'At least {mt_} usable training quarters' sits <i>before</i> a "
             f"{M_}-quarter monitored window, so a firm needs at least {mt_ + M_} quarters ({(mt_ + M_) / 4:g} years) of quarterly "
             "filings, plus the quarters needed to form its first trailing-twelve-month value. This is more than the "
             f"'{mt_ // 4:g} years of history' the training minimum alone suggests. {short_f} failed firms and {short_c} controls were "
             "dropped for too little training data, so the analysed sample leans toward long-established filers.</p>")
    o.append(box("Data decisions worth remembering",
                 "<ul><li>Accounting identities (assets = liabilities + equity, the profit bridge, the cash-conversion cycle) are checked "
                 "before anything runs; a failure stops the run and names the firm, year and columns.</li>"
                 "<li>Missing years are never filled in. Section 7 shows why: interpolating Jyoti's FY2017 would invent a distress "
                 "signal from a number that was never published.</li>"
                 f"<li>Ratios with near-zero denominators are blanked (|denominator| ≤ {pct(cfg['ratios']['denominator_eps_rel_assets']['activity'], 0)} "
                 "of total assets for sales-based ratios) instead of producing absurd values.</li>"
                 "<li>US quarterly data: year-to-date figures were turned into quarters, flows summed over the trailing twelve months, "
                 "and only the first-filed value kept, so no later correction leaks back in time.</li></ul>", "good"))
    o.extend(eda_section(cfg, X))

    # ------------------------------------------------------------------ 4
    o.append('<h2>4. The method, layer by layer</h2>')
    layers = [("Layer 0", "Ratios and audit", "Build oriented ratios, mask impossible values, check identities. Output: <code>ratios_*.csv</code>."),
              ("Layer 1", "Event-time study", "Line firms up by years/quarters before the event and plot each ratio against healthy controls."),
              ("Layer 2", "Rule flags", f"Pre-set Layer 2 rules: OCF &lt; {cfg['layer2']['rules']['ocf_max']}; Layer 2 rule EBITDA/interest &lt; "
               f"{cfg['layer2']['rules']['ebitda_interest_cover_min']}; debtor days &gt; {cfg['layer2']['rules']['debtor_days_max']}; Altman Z''-EM &lt; {cfg['altman_z2_em']['distress']}. "
               "On EDGAR, 'any Layer 2 rule' means the first three; Altman is reported separately."),
              ("Layer 3", "Cash detector", "Kalman model on OCF ÷ current liabilities, standardised errors, one-sided CUSUM."),
              ("Layer 4", "Multivariate detector", "Kalman errors on cash plus structural ratios, Ledoit–Wolf covariance, Crosier MCUSUM."),
              ("Layer 5", "Thresholds and alert tiers", f"Block-bootstrap thresholds (plus the corrections). <b>Amber</b>: either detector over its "
               f"{pct(dc['tiers']['amber_alpha'], 0)} threshold. <b>Red</b>: multivariate over its {pct(dc['tiers']['red_alpha'], 0)} threshold <i>and</i> a Layer 2 rule agrees."),
              ("Layer 6", "Masking index and fingerprint", "Which side (cash or structure) alarms, and which first; archetype rule.")]
    o.append('<div class="layers">' + "".join(f'<div class="layer"><div class="n">{a}</div><div class="t"><b>{b}.</b> {c}</div></div>'
                                              for a, b, c in layers) + "</div>")
    o.append("<h3>Which threshold is used where</h3><p>Two interest-cover thresholds appear in this document. Both were "
             "pre-registered, for different purposes, and neither was changed. The code was checked: there is no mix-up (the H2 "
             "verdict calls the archetype flags, <code>src/run_real.py</code>; the red tier and every 'rule' figure call the "
             "Layer 2 rules, <code>src/detectors/layer2_rules.py</code>).</p>")
    jf_ = cfg["jyoti"]["flags"]; nz = cfg["nmc"]["altman_z2"]
    o.append(table(["Quantity", "Threshold", "Config key (pre-registered?)", "Used in"], [
        ["EBITDA / interest: <b>Layer 2 rule</b>", f"&lt; {l2r['ebitda_interest_cover_min']:g}", "<code>layer2.rules.ebitda_interest_cover_min</code> (yes)",
         "Layer 2 flags for every firm; EDGAR 'any Layer 2 rule'; red tier; cohort rule detection; Figures 5 and 7A"],
        ["EBITDA / interest: <b>archetype structural flag</b>", f"&lt; {arc['interest_cover_max']:g}", "<code>archetype.masked.structural_flags.interest_cover_max</code> (yes)",
         "Archetype rule (Visible / Masked / Other); H2 verdict for NMC; cohort 'first structural flag'; NMC hidden-debt test; Figure 7B"],
        ["EBITDA / interest: legacy EDA flag", f"&lt; {jf_['interest_cover_min']:g}", "<code>jyoti.flags.interest_cover_min</code> (no; legacy)",
         "Legacy Jyoti EDA tables only; no detector or verdict uses it"],
        ["Liquidity cover: archetype structural flag", f"&lt; {arc['liquidity_cover_max']:g}", "<code>archetype.masked.structural_flags.liquidity_cover_max</code> (yes)",
         "Archetype rule; H2; Figure 7B (not a Layer 2 rule)"],
        ["Altman Z''-EM distress (Layer 2 rule and archetype flag)", f"&lt; {az['distress']:g}", "<code>altman_z2_em.distress</code> (yes)",
         "Layer 2 Altman flag; archetype; EDGAR 'Altman Z''-EM distress' benchmark"],
        ["Altman Z'' without the EM constant (legacy figure only)", f"distress &lt; {nz['distress']:g}, safe &gt; {nz['safe']:g}", "<code>nmc.altman_z2</code> (no; legacy)",
         "Zone lines of the legacy NMC EDA figure only; not used in this document"],
    ]))
    shift_ok = abs(nz["distress"] + az["constant"] - az["distress"]) < 1e-9 and abs(nz["safe"] + az["constant"] - az["safe"]) < 1e-9
    o.append(box("Caveat: where the Altman Z''-EM cut-off comes from",
                 f"<p>The score is Z''-EM = {az['constant']} + {az['coef']['wc_ta']}·X1 + {az['coef']['re_ta']}·X2 + {az['coef']['ebit_ta']}·X3 + "
                 f"{az['coef']['bve_tl']}·X4 (Altman's emerging-market score; Altman 2005, building on Altman, Hartzell &amp; Peck 1995) "
                 "[verify before citing]. The project uses distress &lt; "
                 f"{az['distress']:g} and safe &gt; {az['safe']:g}. "
                 + (f"These equal the classic Z'' zone boundaries ({nz['distress']:g} and {nz['safe']:g}, on the score without the constant) "
                    f"plus the EM constant {az['constant']}. " if shift_ok else "")
                 + "That shifted pair is widely quoted for the EM score, but it is a convention: other sources quote different cut-offs "
                 "(some apply the constant-free zones directly, others map the EM score to bond-rating equivalents) "
                 "[verify before citing: check the exact table in Altman (2005)]. Altman's detection and false-alarm rates in Section 6 "
                 "depend on this choice; the cut-off was pre-registered and not tuned.</p>", "warn"))
    sm = T("sim_meta").iloc[0]; bl = fa_.block_length.dropna()
    o.append("<h3>Design choices and why</h3><p>Every fixed setting, the value actually used (read from the config or the code), why it "
             "was chosen, and what the data say about it (Section 3.1). The last column is honest about the record: the pre-registration "
             "fixed most values without stating a reason, so for those rows the reason was <b>written retrospectively</b> "
             f"({e(cfg['eda']['date'])}, after all results). The values themselves were never changed. Where the EDA qualifies a "
             "choice (covariance shrinkage, block bootstrap), the row says so.</p>")
    o.append(design_table(cfg, X, sm, bl, short_f, mt_, M_, au))
    o.append(box("Choices the data argue against, or that remain weak",
                 "<ul><li><b>The adaptive level.</b> It handles missing quarters and firm-specific levels, but it absorbs slow decline "
                 "(Section 6, Finding 3). The event-time table shows exactly the slow decline it forgives.</li>"
                 "<li><b>Ledoit–Wolf on these sample sizes</b> usually shrinks to a scaled identity, so the multivariate detector rarely "
                 "uses correlations. A correlation-aware alternative (fixed shrinkage, or a pooled correlation across firms) is future work.</li>"
                 "<li><b>Gaussian assumptions.</b> Interest cover and Z''-EM have heavy tails; winsorising made little difference (Section 8.7), "
                 "but a robust or Student-t model would be the principled fix.</li>"
                 "<li><b>The minimum history</b> trades sample size for stability; there is no data-driven optimum, only the trade-off shown.</li></ul>", "warn"))
    o.append("<p><b>Why layers?</b> Each layer is useful on its own and cheaper than the next. A lender can stop at Layer 2 "
             "with very little data; Layers 3–5 need a longer history but adapt to each firm; Layer 6 says <i>what kind</i> of "
             "trouble it is. Running them side by side also lets the project test whether the expensive layers add anything over "
             "the cheap ones. Section 6 shows that on real US firms they mostly did not.</p>")
    o.append("<p><b>Training and monitoring.</b> For each firm the history is split: a <i>training</i> window where the model "
             "learns 'normal', and a <i>monitored</i> window "
             f"({cfg['edgar']['monitor_quarters']} quarters ending at the event on EDGAR) where alarms count. Nothing from the "
             "monitored window is used to fit the model or set the threshold. Healthy controls are monitored over exactly the same "
             "calendar window as their matched failed firm, so a control alarm is a real-world false alarm.</p>")

    # ------------------------------------------------------------------ 5
    o.append('<h2>5. Results, hypothesis by hypothesis</h2>')
    o.append("<p>Each hypothesis was written with explicit 'supported if' and 'refuted if' criteria before any detector ran "
             "(<code>docs/PREREGISTRATION.md</code>). The verdicts below are produced by code from those criteria, not chosen by hand.</p>")
    # H4 first: it is the headline
    o.append(f"<h3>H4: Are the standard thresholds too trigger-happy? {verdict_badge(H.loc['H4', 'verdict'])}</h3>")
    o.append("<p><b>Test.</b> Simulate healthy firms (no deterioration at all), set each firm's threshold with the standard block "
             "bootstrap from its own training history, then count how many raise an alarm anyway.</p>")
    rows = []
    for n_, g in k28.groupby("n"):
        g = g.set_index("nominal")
        rows.append([str(n_), f"{pct(g.loc[0.05, 'empirical_fa'])} ({pct(g.loc[0.05, 'ci_lo'])}–{pct(g.loc[0.05, 'ci_hi'])})",
                     f"{pct(g.loc[0.10, 'empirical_fa'])} ({pct(g.loc[0.10, 'ci_lo'])}–{pct(g.loc[0.10, 'ci_hi'])})"])
    o.append(table(["History length n (periods)", "Promised 5% → actual (95% CI)", "Promised 10% → actual (95% CI)"], rows))
    o.append(f"<p><b>Meaning.</b> A lender told 'one in ten healthy firms will be flagged' would actually see about one in three "
             f"at n = {n28}, and about one in two at n = {int(k28.n.min())}. More history helps but never closes the gap within the "
             "range a real firm has. This is the project's most solid result: it is large, it holds at every n, and it shows up "
             "again on real US firms (Section 6).</p>")
    o.append(box("Related work: this is a known effect, quantified here",
                 "<p>Control charts whose in-control parameters are estimated from a short 'Phase I' sample are known to give false-alarm "
                 "rates far from nominal in 'Phase II' monitoring. See the review by Jensen, Jones-Farmer, Champ &amp; Woodall (2006), and "
                 "Jones, Champ &amp; Rigdon (2004) on the CUSUM with estimated parameters [verify before citing]. Two classical remedies are "
                 "Hawkins' self-starting CUSUM (Hawkins 1987) and Quesenberry's Q-charts (Quesenberry 1991) [verify before citing]. "
                 "H4 therefore <b>confirms and quantifies</b> this effect for a financial early-warning setting (short, autocorrelated "
                 "ratio histories, a block-bootstrap calibrator, a fixed monitoring window); it is not a new discovery. Section 8 adds "
                 "a self-starting comparator.</p>"))
    o.append(fig("sim_calibration_curve.png", "Figure 1. Actual vs promised false-alarm rates. Every point sits far above the diagonal (perfect calibration)."))

    o.append(f"<h3>H5: Can it be fixed cheaply? {verdict_badge(H.loc['H5', 'verdict'])}</h3>")
    o.append(f"<p><b>Test.</b> Fit each correction on one set of simulated firms, then test it on a <i>different</i> set "
             f"(held out). Pre-registered success: within ±{tol} points of the promised 10% at n = {n28}.</p>")
    labs = {"block_bootstrap": "No correction (standard)", "inflation_factor": "Inflation factor",
            "conformal": "Conformal threshold", "parametric_bootstrap": "Parametric bootstrap (refit)"}
    o.append(table(["Method", "False alarms, promised 10% (95% CI)", "False alarms, promised 5%", "Detection of a real drift (S1)", "Detection lost vs no correction"],
                   [[labs[k], f"{pct(f28.loc[k, 'empirical_fa'])} ({pct(f28.loc[k, 'ci_lo'])}–{pct(f28.loc[k, 'ci_hi'])})",
                     pct(f28_5.loc[k, "empirical_fa"]), pct(f28.loc[k, "detection_S1"]), f"{f28.loc[k, 'detection_cost_points']:.1f} points"]
                    for k in labs]))
    o.append(f"<p><b>Meaning.</b> All three fixes work, but they are not equal. The inflation factor (h × {i28.c_required:.2f} at "
             f"n = {n28}) also raises the threshold of firms whose threshold was already fine, so it throws away "
             f"{f28.loc['inflation_factor', 'detection_cost_points']:.1f} points of detection. The parametric bootstrap targets the actual "
             "source of the error (uncertain parameters) and loses almost nothing. The conformal threshold is nearly as good but "
             "needs a pool of similar healthy firms, which a single-firm monitor does not have.</p>")
    o.append(fig("sim_calibration_fixes.png", "Figure 2. After correction, false-alarm rates sit in or near the green ±3-point band at every history length."))

    o.append(f"<h3>H3: Many ratios or one? {verdict_badge(H.loc['H3', 'verdict'])}</h3>")
    o.append("<p><b>Test.</b> Four simulated scenarios: <b>S1</b> the cash ratio drifts down; <b>S2</b> cash is fine but the other "
             "ratios deteriorate; <b>S3</b> everything drifts a little; <b>S4</b> everything deteriorates but reported cash is smoothed "
             "(concealment). All detectors are first tuned to the <i>same</i> 10% false-alarm rate, so the comparison is fair.</p>")
    o.append(table(["Scenario", "ARIMA + CUSUM", "Kalman + CUSUM", "Random walk + CUSUM", "Multivariate"],
                   [[sc_] + [pct(m[(sc_, d)]) for d in ["arima", "kalman", "rw", "mcusum"]] for sc_ in ["S1", "S2", "S3", "S4"]]))
    o.append(f"<p><b>Meaning.</b> When the problem is in cash (S1), the cash-only detectors win and the multivariate one is diluted. "
             f"When cash is quiet (S2) or massaged (S4), only the multivariate detector sees it. Hence the design choice to run both "
             f"side by side. Honest caveat: the S1 loss ({100 * (m[('S1', 'mcusum')] - m[('S1', 'kalman')]):+.1f} points) is not "
             "'slight' as the hypothesis worded it, and a plain ARIMA model beat the Kalman model on raw power here.</p>")
    mk = lambda sc_, d: m[(sc_, d)]  # noqa: E731
    uni4 = [mk("S4", d) for d in ["arima", "kalman", "rw"]]; uni2 = [mk("S2", d) for d in ["arima", "kalman", "rw"]]
    ar_wins = [sc_ for sc_ in ["S1", "S2", "S3", "S4"] if mk(sc_, "arima") > mk(sc_, "kalman")]
    o.append(box(f"H3 verdict and interpretation",
                 f"<p><b>Code-computed verdict (pre-registered criterion):</b> {verdict_badge(H.loc['H3', 'verdict'])} "
                 f"({e(H.loc['H3', 'evidence'])}).</p>"
                 "<p><b>Interpretation (author's reading, not a verdict): partially supported.</b></p><ul>"
                 f"<li>The loss when the drift is in cash is not 'slight': multivariate minus Kalman is {pts(mk('S1', 'mcusum') - mk('S1', 'kalman'))} "
                 f"points in S1. In S3 (small drift everywhere) it is {pts(mk('S3', 'mcusum') - mk('S3', 'kalman'))} points against Kalman "
                 f"and {pts(mk('S3', 'mcusum') - mk('S3', 'arima'))} against ARIMA.</li>"
                 f"<li>ARIMA beat Kalman on power in {', '.join(ar_wins)}, so the univariate model chosen is not the most powerful one here.</li>"
                 f"<li>The S2 win is true by construction: the cash ratio does not change in S2, so cash-only detectors can only fire at "
                 f"about their false-alarm rate ({pct(min(uni2))}–{pct(max(uni2))} 'detection').</li>"
                 f"<li>S4 conceals less than its name suggests: cash-only detectors still detect {pct(min(uni4))}–{pct(max(uni4))} of S4 "
                 f"firms (Kalman: {pct(mk('S4', 'kalman'))} in S4 vs {pct(mk('S1', 'kalman'))} in S1).</li></ul>", "warn"))
    o.append(fig("sim_power_curves.png", "Figure 3. Detection at a matched 10% false-alarm rate: by drift size (left two) and by history length (right)."))

    o.append(f"<h3>H2: NMC, silent cash and a warning balance sheet {verdict_badge(H.loc['H2', 'verdict'])}</h3>")
    o.append(f"<p><b>Result.</b> {e(H.loc['H2', 'evidence'])}. Liquidity cover (cash and deposits ÷ short-term debt) was below 1 in "
             f"{', '.join('FY' + str(y) for y in lcov.index[lcov < 1])}, falling to {lcov.loc[lcov.index.max()]:.2f} in "
             f"{int(lcov.index.max())}.</p>")
    o.append(f"<p><b>Why 'H2 SUPPORTED' and 'archetype Other' are not in conflict.</b> H2's pre-registered criterion needs the cash flag "
             f"(OCF &lt; {l2r['ocf_max']}) to stay silent and <i>any one</i> structural flag (liquidity cover &lt; {arc['liquidity_cover_max']:g}, "
             f"EBITDA / interest &lt; {arc['interest_cover_max']:g}, Z''-EM &lt; {az['distress']:g}) to trip. The archetype rule needs positive OCF "
             f"<i>and at least {cfg['archetype']['masked']['min_structural_flags']} of the 3</i> structural flags. At NMC exactly "
             f"{int(arch.loc['nmc', 'n_structural_flags_tripped'])} flag tripped, which satisfies the first and not the second.</p>")
    o.append(box("Common-sense caveat we kept",
                 f"<p>Only <b>one</b> of the three balance-sheet flags ever tripped, so the pre-registered archetype rule labels NMC "
                 f"<b>{e(arch.loc['nmc', 'archetype'])}</b>, not 'Masked'. We report this rather than bend the rule. Section 7 shows why: "
                 f"NMC's published accounts left out billions of debt. Add hidden debt of {masked_from:g}× the reported amount or more "
                 "and the same rule calls NMC Masked, years before the collapse.</p>"
                 f"<p><b>This is a {POSTHOC} sensitivity illustration, not evidence:</b> NMC never published restated accounts. What the "
                 "stress test does (<code>src/robustness.py</code>): in every year, hidden debt D = multiple × reported total debt is added "
                 "as short-term debt, so short-term debt, total debt, current liabilities and total liabilities each rise by D; book equity "
                 "and retained earnings fall by D while total assets, cash and deposits stay unchanged (the borrowed money is assumed "
                 "spent, not held as cash); interest rises by D × the reported interest ÷ total debt rate; EBITDA, EBIT, revenue and "
                 "operating cash flow are unchanged. So liquidity cover falls (bigger denominator), EBITDA / interest falls, and Z''-EM "
                 "falls through X1 (working capital), X2 (retained earnings) and X4 (equity ÷ liabilities) while X3 (EBIT ÷ assets) is "
                 "unchanged. The same multiple is used in every year (hidden debt proportional to reported debt throughout, a "
                 "simplification), and the accounting identities are re-checked for every multiple.</p>", "warn"))
    o.append(fig("layer1_event_time_nmc.png", "Figure 4. NMC in event time (lower = worse). Cash ratios improve into the event; liquidity and interest cover deteriorate."))

    o.append(f"<h3>H1: Jyoti, does the statistical detector warn two years early? {verdict_badge(H.loc['H1', 'verdict'])}</h3>")
    o.append(f"<p><b>Why it could not be tested.</b> {e(l3.loc['jyoti', 'reason'])}. The simpler layers do tell the story: OCF &lt; 0, "
             f"weak interest cover and high debtor days all first trip in FY{int(jfirst.loc['Any rule', 'first_trip_fy'])} "
             f"(lead {jfirst.loc['Any rule', 'lead_years']:.2f} years), net worth only turns negative in FY{nw_neg}, and Altman Z''-EM "
             f"reaches distress only in FY{int(jy_alt.first_trip_fy)}, after the event. So cash warned first, but H1 is about the "
             "statistical detector, and that needs older, higher-frequency data.</p>")
    o.append(fig("layer2_flag_heatmap_jyoti.png", "Figure 5. Jyoti: three rules trip from FY14; Altman only after the event."))

    o.append(f"<h3>H6: Does the masking index separate crisis types? {verdict_badge(H.loc['H6', 'verdict'])}</h3>")
    o.append(f"<p>In simulation the index separates 'cash drifts' (S1) from 'structure drifts' (S2) very well (AUC {s12.auc:.2f}) "
             f"and from concealment (S4) more weakly (AUC {s14.auc:.2f}).</p>")
    o.append(f"<p><b>The cohort.</b> {len(cf)} failed Indian firms were selected by the pre-set rule from RBI's 2017 lists, before "
             f"any of their figures were read, with {len(cc)} matched controls ({', '.join(e(n) for n in cun) or 'none'} could not be "
             f"matched). <b>Why H6 is still inconclusive:</b> {e(H.loc['H6', 'evidence'])}.</p>")
    o.append(table(["Failed firm", "Event FY", "Archetype", "First cash flag", "First structural flag", "Order"],
                   [[e(r.name), f"FY{int(r.event_fy)}", e(r.archetype), f"FY{int(r.first_cash_flag_fy)}" if pd.notna(r.first_cash_flag_fy) else "none",
                     f"FY{int(r.first_structural_flag_fy)}" if pd.notna(r.first_structural_flag_fy) else "none", e(r.flag_order)] for r in cf.itertuples()]))
    oth = cf[cf.archetype == "Other"]; s_first = oth[oth.flag_order.isin(["structural first", "structural only"])]
    o.append(f"<p><b>What the cohort does show (exploratory).</b> No firm is 'Masked' by the rule: Screener has no split of cash "
             f"against short-term debt, so liquidity cover can never be checked and firms can trip at most two of three structural "
             f"flags. The order of flags says nothing for the Visible firms: 'Visible' is <i>defined</i> by OCF &lt; 0 in at least "
             f"{cfg['archetype']['visible']['ocf_negative_consecutive_years']} consecutive pre-event years, so an early cash flag is built "
             f"into the label (definitional, not evidence). Among the {len(oth)} 'Other' firms, a balance-sheet flag came first or alone "
             f"in {len(s_first)} ({', '.join(e(n) for n in s_first.name) or 'none'}). Over the years both members of a pair have data, the "
             f"simple rules flagged {int(cw.failed_flagged)} of {int(cw.failed_firms)} failed firms and {int(cw.controls_flagged)} of "
             f"{int(cw.controls)} controls. With {len(cf)} firms this is an illustration, not proof.</p>")
    o.append(f"<p>On US firms, an exploratory analogue points the right way (Masked minus Visible: AUC {ems.auc:.2f}, "
             f"95% CI {ems.auc_ci_lo:.2f}–{ems.auc_ci_hi:.2f}) but the interval includes 'no effect', with only "
             f"{int(eam.loc['Masked', 'firms'])} Masked firms.</p>")
    o.append(fig("sim_masking_index.png", "Figure 6. Masking index and which detector fires first, by simulated scenario."))

    # ------------------------------------------------------------------ 6
    o.append('<h2>6. Does it work on real firms? (US EDGAR panel)</h2>')
    o.append(f"<p>{okf} US firms that filed for bankruptcy and {okc} matched healthy firms (same industry, similar size), monitored "
             f"over the {cfg['edgar']['monitor_quarters']} quarters before each bankruptcy filing. Nothing was tuned on this data.</p>")
    o.append(table(["Method", "Failed firms flagged", "Healthy firms flagged", "Difference (95% CI)", "Median warning"],
                   [[e(k), pct(r.detection), pct(r.false_alarm),
                     f"{100 * r.det_minus_fa:+.1f} pts ({100 * r.diff_ci_lo:+.1f} to {100 * r.diff_ci_hi:+.1f})",
                     f"{r.median_lead_quarters:.1f} quarters"] for k, r in ed.iterrows()]))
    o.append(box("Finding 1: the calibration problem is real, not a simulation artefact",
                 f"<p>The uncorrected amber tier flagged {pct(am_u.false_alarm)} of healthy firms against a promised 10%, just as H4 "
                 f"predicted. The simulation-fitted correction lowered this to {pct(am_c.false_alarm)}, which is better but still far off: "
                 "real ratios have jumps, seasonality and accounting events that the simple simulated world does not.</p>", "warn"))
    o.append(box("Finding 2: simple rules did best, but not distinguishably better than Altman",
                 f"<p>The pre-set rules flagged {pct(rule.detection)} of failed firms with {pct(rule.false_alarm)} false alarms. A paired "
                 f"bootstrap (resampling failed firms together with their matched controls, the same resample for both methods; Section 8) "
                 + ("shows a lower expected cost than every statistical tier at every cost ratio, with 95% CIs that exclude zero. "
                    if rules_beat_stats else "does not show a lower expected cost than every statistical tier with CIs excluding zero. ")
                 + (f"Against Altman Z''-EM the rules' point estimates are better at every cost ratio, but the paired CIs include zero "
                    f"(cost difference at C_M = {cms[0]}: {pa.loc[f'expected_cost_cm{cms[0]}', 'difference']:+.2f}, 95% CI "
                    f"{pa.loc[f'expected_cost_cm{cms[0]}', 'ci_lo']:+.2f} to {pa.loc[f'expected_cost_cm{cms[0]}', 'ci_hi']:+.2f}): with this "
                    "sample, <b>rules and Altman are not distinguishable</b>. "
                    if alt_indist else "Against Altman Z''-EM the paired CIs exclude zero for at least one cost ratio (Section 8). ")
                 + f"Most of the rules' power comes from 'OCF &lt; 0' alone ({pct(ru.loc['only OCF < 0', 'detection'])} / "
                 f"{pct(ru.loc['only OCF < 0', 'false_alarm'])}): a cash-first result, just not from the statistical layer.</p>", "good"))
    o.append(box("Finding 3: no detectable advantage over chance for the amber alarms, and why",
                 f"<p>With {okf} failed vs {okc} healthy firms, the 95% interval of 'failed flagged minus healthy flagged' includes zero for "
                 f"{' and '.join(amber_chance) or 'no amber tier'}: no detectable advantage over chance on this sample (which is not proof "
                 "of no effect; the intervals are wide). A threshold-free check (AUC; bootstrap over firms, "
                 f"failed and control firms resampled separately) gives {aci(a_k)} for the pre-registered cash CUSUM and {aci(a_m)} for "
                 f"the multivariate one, against {aci(a_l)} for simply looking at the <i>lowest level</i> of the cash ratio and {aci(a_z)} "
                 "for the lowest Altman score. Two reasons, found after seeing the data (so they are hypotheses, not results):</p><ol>"
                 f"<li><b>The model forgives slow decline.</b> The Kalman model treats part of every change as a new 'normal level' "
                 f"(median firm: {pct(lvs.auc, 0)} of its variance on the moving level; a quarter of firms above {pct(lvs.auc_ci_hi, 0)}). "
                 f"A steady slide is absorbed, so the surprises stay small. The same CUSUM measured against the fixed training average "
                 f"scores AUC {aci(a_s)}.</li>"
                 f"<li><b>Failing firms were weak all along.</b> In their last training quarters, failed firms' median OCF ÷ current "
                 f"liabilities was already {lc.loc[(cr, 'failed'), 'training_level_median']:.2f} versus "
                 f"{lc.loc[(cr, 'control'), 'training_level_median']:.2f} for healthy firms. A method that only asks 'is this firm worse "
                 "than its own past?' cannot see a firm whose past was already bad.</li></ol>"
                 "<p>The practical lesson: combine 'worse than its own past' with 'weak in absolute terms', and prefer a "
                 "non-adaptive baseline for slow deterioration.</p>", "bad"))
    o.append(fig("robust_edgar_tradeoff_nmc_hidden_debt.png", "Figure 7. (A) Every threshold setting of the statistical detectors lies "
                 "close to the chance line; rules and Altman sit far above it. (B) NMC's 2018 ratios if hidden debt is added "
                 f"({POSTHOC} stress test, Section 5); the dotted lines are the <i>archetype structural flags</i> (EBITDA / interest "
                 f"&lt; {arc['interest_cover_max']:g}, not the Layer 2 rule's {l2r['ebitda_interest_cover_min']:g})."))
    o.append(fig("edgar_layer1_event_time.png", f"Figure 8. Analysed US firms only ({fa_.cik.nunique()} failed, {ca_.cik.nunique()} controls: "
                 "the same firms as every rate in this section) in event time (lower = worse); 'n &lt;=' in the legend is the most firms "
                 f"with a value in any one quarter. <code>edgar_layer1_panel.csv</code> holds {int(n_all.get('failed', 0))} failed firms and "
                 f"{int(n_all.get('control', 0))} controls that reached the ratio stage; {int(n_all.get('failed', 0)) - fa_.cik.nunique()} failed "
                 f"firms and {int(n_all.get('control', 0)) - ca_.cik.nunique()} controls of those were not analysed (mostly too little training "
                 "data, or a control whose failed firm was dropped); an earlier version of this figure plotted all of them. Failed firms start lower and fall further."))

    # ------------------------------------------------------------------ 7
    o.append('<h2>7. Robustness: do the conclusions survive changes?</h2>')
    o.append("<p>These checks were added after the main results (logged as a deviation) and change one thing at a time. "
             "No verdict depends on them.</p>")
    amb = V[V.method == "amber (uncorrected)"]; rul = V[V.method == "any Layer 2 rule"]
    o.append(table(["What was changed", "What happened"], [
        ["Re-running everything with the original settings", f"{int(rp.identical)} of {int(rp.firm_runs)} firm runs identical: the pipeline is deterministic."],
        [f"CUSUM allowance k ∈ {{{', '.join(f'{x:g}' for x in rb['cusum_k_grid'])}}}; training window ±{rb['training_window_shift']} quarters; "
         "dropping each structural ratio; event moved a year earlier",
         f"Amber false alarms stay between {pct(amb.false_alarm.min())} and {pct(amb.false_alarm.max())} (always far above 10%); rules "
         f"keep detecting {pct(rul.detection.min())}–{pct(rul.detection.max())}. The ranking of methods does not change."],
        ["Raising or lowering the threshold (× " + ", ".join(f"{x:g}" for x in rb["h_multiplier_grid"]) + ")",
         "Detection and false alarms move together; no setting separates failed from healthy firms well (Figure 7A)."],
        ["Each rule alone / leaving each rule out", "; ".join(f"{e(k)}: {pct(r.detection)} / {pct(r.false_alarm)}" for k, r in ru.iterrows()
                                                              if k.startswith("only")) + " (flagged failed / healthy). The debtor-days rule adds nothing on US firms."],
        ["Jyoti FY2017: leave missing vs fill in by interpolation",
         f"Cash rules unchanged (first trip FY{int(jf.loc[('none', 'Any rule'), 'first_trip_fy'])}), but interpolation moves Altman into distress a "
         f"year early (FY{int(jf.loc[('linear', 'Altman Z' + chr(39) * 2 + '-EM < 4.35 (distress)'), 'first_trip_fy'])}) using an invented value. Leaving it missing was right."],
        ["Alternative event dates (one year earlier; for NMC the short-seller report of " + e(rb['alt_event_dates']['nmc']['date']) + ")",
         f"Jyoti's rule lead goes from {ae.loc[('jyoti', 'pre-registered'), 'lead_years_any_rule']:.2f} to "
         f"{ae.loc[('jyoti', ae.loc['jyoti'].index[1]), 'lead_years_any_rule']:.2f} years; NMC's picture is unchanged (no rule trips, archetype "
         f"{e(ae.loc[('nmc', 'pre-registered'), 'archetype'])})."],
        [f"NMC with hidden debt added (0×, " + ", ".join(f"{x:g}×" for x in hd.index[1:]) + f" reported debt; {POSTHOC} stress test, "
         "assumptions in Section 5)",
         f"Archetype becomes Masked from {masked_from:g}×; Z''-EM 2018 falls from {hd.loc[0.0, 'z2_em_2018']:.2f} to "
         f"{hd.loc[hd.index.max(), 'z2_em_2018']:.2f}. For scale: debt was later put at about US${rv['revealed_total']}bn against "
         f"US${rv['reported']}bn reported."],
    ]))

    # ------------------------------------------------------------------ 8 (post-hoc review checks)
    o.extend(posthoc_section(cfg, PH, ed, okf, okc, fix, n28, pa, cms, stat_tiers))

    # ------------------------------------------------------------------ 9
    o.append('<h2>9. How we know it is right: sanity and common-sense checks</h2>')
    o.append(f"<p><code>python -m src.sanity</code> runs {len(SC)} automated checks: "
             f"{int(cnt.get('PASS', 0))} pass, {int(cnt.get('WARN', 0))} warnings, {int(cnt.get('INFO', 0))} recorded facts, "
             f"{int(cnt.get('FAIL', 0))} failures. A <b>warning</b> means 'not a bug, but a reader must know this'. These checks test "
             "traceability, internal consistency and plumbing (frozen settings, look-ahead, determinism, numbers tracing to tables); "
             "they do <b>not</b> test scientific validity: a check can pass while a conclusion is still wrong. The full list:</p>")
    o.append(table(["Area", "Check", "Status", "Detail"],
                   [[e(r.area), e(r.check), verdict_badge(r.status), e(r.detail if len(str(r.detail)) < 260 else str(r.detail)[:257] + "…")]
                    for r in SC.itertuples()], "checks"))
    o.append("<h3>Common-sense catches during the project</h3><p>Things that looked fine to the code but wrong to a human, and what "
             "was done about them:</p><ul>"
             "<li><b>The pre-registration hash.</b> The config and pre-registration files were legitimately extended later (logged "
             "deviations), so their hashes no longer matched. Exact copies of the frozen versions were rebuilt from the edit history; "
             "their hashes match the recorded fingerprint, and every frozen setting is shown to be unchanged.</li>"
             "<li><b>Two identical numbers in the H5 table.</b> The detection rate of one fix equalled the false-alarm rate of "
             "another. It looked like a copy-paste bug; it is a coincidence (same count, different firms), now a permanent check.</li>"
             "<li><b>A hidden source of randomness.</b> One script seeded a model with Python's <code>hash()</code>, which changes "
             "every run. It never affected a result, but it would have broken reproducibility; replaced.</li>"
             "<li><b>A wrong court.</b> The NMC event was recorded as an ADGM court order; primary records show the English High "
             "Court made the 2020 order. Date unchanged, text corrected.</li>"
             "<li><b>Two ratios move 'the wrong way' at bankruptcy.</b> Failed US firms have <i>fewer</i> debtor days and "
             "<i>better</i> accruals than controls near the end. Not a sign error: big write-downs push profit below cash flow, and "
             "debtor days depend on industry. These two ratios measure earnings quality in normal times, not distress at the brink.</li>"
             "<li><b>Liquidity cover is almost never available on US quarterly data</b> because short-term debt is rarely tagged, so "
             "the structural detector usually runs without it. Recorded, not hidden.</li>"
             "<li><b>An over-claim in our own text.</b> A draft sentence said rules beat amber on false alarms 'in every variant'; "
             "the table showed one exception and one tie. The sentence is now computed from the table.</li>"
             "<li><b>Earlier in the build</b> (decision log): an identity that failed on Jyoti because of other income; interest "
             "cover blanked for healthy firms by a too-strict near-zero rule; a simulation crash from empty bootstrap pools; zero "
             "thresholds from degenerate fits, now excluded and counted; and a run that ran out of memory with too many workers.</li></ul>")
    o.append(box("Why the core machinery is trustworthy",
                 f"<p>If the CUSUM or threshold code were wrong, even thresholds learned from the right distribution would miss. They do "
                 f"not: the conformal threshold hits {pct(fix[(fix.n == n28) & (fix.method == 'conformal') & (fix.nominal == 0.05)].empirical_fa.iloc[0])} "
                 f"for a promised 5% and {pct(f28.loc['conformal', 'empirical_fa'])} for 10%. False alarms fall as history grows, power rises "
                 "with drift size, every alarm lies inside the monitored window, and the legacy scripts are reproduced exactly "
                 "(numbers and pixel-identical figures).</p>", "good"))

    # ------------------------------------------------------------------ 9
    o.append('<h2>10. Limitations</h2><ul>'
             "<li><b>Two deep cases were known before the pre-registration</b>, so they are illustrations, not tests. The cohort is "
             "the confirmatory test, but its annual data are too short for the detectors, so H6 could not be judged.</li>"
             "<li><b>The cohort is small and narrow:</b> India only (no machine-readable GCC source), events in 2017–2018, mostly "
             "steel, infrastructure and manufacturing; controls have only a few years of history on Screener's public pages.</li>"
             f"<li><b>Tiny samples.</b> {len(rn)} annual points for NMC; no pre-FY14 history for Jyoti.</li>"
             "<li><b>Fraud defeats ratio analysis.</b> If debt is kept off the accounts (NMC), no ratio can see it; the hidden-debt "
             "test shows how much difference that makes.</li>"
             "<li><b>Simulation realism.</b> Gaussian AR(1) dynamics; heavy tails and training-period breaks were cut for time. "
             "The real-data results show the cost of this simplification.</li>"
             "<li><b>Vendor and XBRL data.</b> Screener proxies for Jyoti's working capital; untagged debt on EDGAR.</li>"
             "<li><b>The premise tension.</b> Failed US firms were weak all along (Section 6, Finding 3; Section 8, fixed horizons), and a "
             "detector that asks only 'is this firm worse than its own past?' cannot see a firm whose past was already bad. The "
             "contribution is therefore the <b>calibration methodology plus the layered framework</b>, not predictive superiority over "
             "simple level rules or Altman.</li>"
             "<li><b>Controls are survivors.</b> EDGAR controls are firms with no bankruptcy filing through 2023 (and the cohort's "
             "controls were never insolvent). A real lender's healthy book also contains firms that weaken and recover or fail later, "
             "so separation here is easier than in practice. Industry mix is in Section 8.</li>"
             "<li><b>Lead times</b> in the main tables are measured from period ends; Section 8 re-measures them from SEC filing dates "
             "(shorter), and shows that much of the long lead of rules and Altman is weakness already present before monitoring.</li>"
             "<li><b>The masking index</b> compares overlapping information (the structural ratios contain cash balances) and is "
             "positive for most healthy firms under uncorrected thresholds; only a healthy-centred version is interpretable (Section 8).</li>"
             f"<li><b>The Altman cut-off</b> ({az['distress']:g}) is a convention; other sources quote other cut-offs (Section 4).</li>"
             "<li><b>The multivariate detector</b> reacts to change in any direction and can alarm on improvement.</li>"
             "<li><b>The US diagnosis (Section 6) is post hoc</b>; it should be tested on new data before being believed.</li></ul>")

    # ------------------------------------------------------------------ 10
    o.append('<h2 class="nobreak">11. What is left to do</h2>')
    o.append(table(["#", "Task", "Why it matters", "Who"], [
        ["1", "Commit the repository (include <code>docs/PREREG_FINGERPRINT.txt</code> and <code>docs/prereg_snapshot/</code>)", "Timestamped proof that the plan came before the results", "You"],
        ["2", "Optional: quarterly or longer histories for the cohort (CMIE Prowess via your library) would let the detectors run and make H6 testable", "H6 is inconclusive only for lack of periods", "You"],
        ["3", "Optional: Jyoti's BSE quarterly results FY2010–13", "Makes H1 testable", "You, then <code>make real</code>"],
        ["—", "Delete the stray folder <code>C:\\c\\</code> (two validation files created by mistake)", "Housekeeping", "You"],
        ["4", "Check NMC's figures against the original annual reports (the one remaining [verify] item)", "The case study rests on them", "You"],
        ["5", "Read and edit <code>reports/report.md</code> into your own voice", "It is a generated draft", "You"],
        ["6", "Slides, if the course needs a presentation", "Not built yet", "Ask, and it can be generated from the same tables"],
        ["7", "Read the design-choices table (Section 4) and check that every retrospective rationale is one you agree with and can defend", "The reasons were written after the results; a viva examiner will ask why each value was chosen", "You"],
        ["F1", "Conformal thresholds calibrated on a large pool of real healthy EDGAR firms (not on simulated firms)", "The simulation-fitted correction transferred only partly", "Future work"],
        ["F2", "A one-sided (directional) MCUSUM that accumulates only deterioration", "The current multivariate detector can alarm on improvement", "Future work"],
        ["F3", "Level-aware and non-adaptive detectors (e.g. 'worse than its own past AND weak in absolute terms')", "The adaptive level model absorbs slow decline; failed firms are weak for years", "Future work"],
        ["F4", "A logit / Altman head-to-head re-estimated in a small-sample regime", "Altman's published coefficients were not fitted to these firms", "Future work"],
        ["F5", "Test the post-hoc AR(1) self-starting comparator (Section 8) on real data, pre-registered first", "It is correctly specified for the simulator, so its simulated advantage may not transfer", "Future work"],
    ]))

    # ------------------------------------------------------------------ 11
    o.append('<h2 class="nobreak">12. Reproducing everything</h2>')
    o.append(table(["Command", "What it does"], [
        ["<code>python -m pytest</code>", "All tests, including exact reproduction of the original scripts"],
        ["<code>python -m src.run_layers --tables --figures</code>", "Layers 0–2 for NMC and Jyoti"],
        ["<code>python -m src.simulate.run</code> then <code>python -m src.simulate.report</code>", "Simulation (about 1–2 hours on 9 workers) and its tables/figures"],
        ["<code>python -m src.run_real</code>", "Archetypes, Layer 3 status, H1–H6 verdicts"],
        ["<code>python -m src.ingest.edgar</code> then <code>python -m src.run_edgar</code>", "Download and analyse the US panel"],
        ["<code>python -m src.robustness</code>", "All robustness analyses"],
        ["<code>python -m src.post_hoc</code> and <code>python -m src.simulate.self_starting</code>", f"The {POSTHOC} checks of Section 8"],
        ["<code>python -m src.sanity</code>", "All sanity checks (also runs the tests)"],
        ["<code>python -m src.reporting.write_report</code> / <code>write_explainer</code>", "Rebuild report.md / this document"],
    ]))
    o.append("<p class='small'>Key files: <code>configs/config.yaml</code> (every setting), <code>docs/PREREGISTRATION.md</code> (hypotheses and "
             "deviations log), <code>docs/decision_log.md</code> (every judgement call), <code>reports/tables/</code> (every number), "
             "<code>reports/figures/</code>.</p>")

    # ------------------------------------------------------------------ 12
    qa = [
        ("Why not just use Altman's Z-score?", f"It needs no history and did well on US firms ({pct(alt.detection)} flagged), but it also flagged "
         f"{pct(alt.false_alarm)} of healthy firms, and in a paired comparison the pre-set rules and Altman "
         f"{'cannot be told apart' if alt_indist else 'differ'} on expected cost. That Altman missed NMC on its published accounts and "
         "entered distress at Jyoti only after the event are two known anecdotes: illustrations, not evidence. The project's point is a "
         "per-firm monitor with a known false-alarm rate, plus a way to tell visible from masked crises."),
        ("Isn't the simulation just showing what you built into it?", "The simulation checks the statistics (does a promised 10% stay 10%?) "
         "under known conditions. Its main finding, over-alarming with short histories, was then confirmed on real US firms that the "
         "simulation never saw."),
        ("How do you know you didn't tune the method to fit NMC and Jyoti?", "All rules and thresholds were fixed and hashed before any "
         "detector ran; the frozen settings are shown to be unchanged; deviations are logged; and NMC came out 'Other' against the "
         "story we expected, which we reported."),
        ("Your statistical detectors show no detectable advantage over chance on US data. Is the project a failure?", "No: that is a finding. The calibration problem and "
         "its fixes are confirmed, the rule layer works well, and the diagnosis (adaptive models absorb slow decline; failing firms "
         "are weak for years) says how to build the next version. Reporting it honestly is the point of pre-registration."),
        ("Why is NMC not 'Masked'?", "Because only one of three balance-sheet flags fired on the published numbers. With the debt that "
         "was later found, it would be Masked. Ratio analysis cannot see falsified inputs."),
        ("Why block bootstrap and not ordinary bootstrap?", "To keep any serial dependence the model leaves in its errors. Honestly, the EDA "
         "shows the raw ratios are strongly autocorrelated but the standardised errors of healthy firms are close to white "
         "(Section 4 table), so on these data the blocks are cheap insurance rather than essential."),
        ("Was there EDA behind the design choices?", "Before the pre-registration, only for the two case studies; it motivated the "
         "cash-plus-balance-sheet architecture. A descriptive EDA of the properties the settings depend on (history length, "
         "persistence, seasonality, tails, small denominators, covariance) was added afterwards, on training data only (Section 3.1). "
         "It supports most choices, and it showed one that was not expected: Ledoit–Wolf shrinks to the identity for most firms."),
        ("What would you do with more time?", "Get longer or quarterly histories for the cohort and Jyoti; add a non-adaptive and a level-aware "
         "detector; model heavy tails; carry the filing-date lead times (Section 8) into the main tables; test thresholds calibrated on "
         "real healthy firms."),
    ]
    qa += viva_extra(cfg, PH, ed, au, H, m, okf, okc)
    o.append('<h2>13. Viva preparation</h2><dl class="qa">' + "".join(f"<dt>{e(q)}</dt><dd>{a}</dd>" for q, a in qa) + "</dl>")
    o.append(f"<p class='small'>Built from the project tables; the sanity checks also scan this document and confirm every number in it "
             "traces to a table or the config.</p>")
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>" \
           f"<title>Early-Warning Project Explainer</title><style>{CSS}</style></head><body>{''.join(o)}</body></html>"


def to_pdf(src: Path, out: Path) -> bool:
    exe = next((b for b in BROWSERS if Path(b).exists()), None)
    if exe is None:
        print("no headless Edge/Chrome found; open explainer.html and print to PDF"); return False
    r = subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                        src.resolve().as_uri()], capture_output=True, text=True, timeout=180)
    ok = out.exists() and out.stat().st_size > 0
    print("wrote", out if ok else f"PDF failed: {r.stderr[-300:]}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    out = P.tables.parent / "explainer.html"
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
    if not a.no_pdf:
        to_pdf(out, out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
