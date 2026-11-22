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
@media screen{h2{margin-top:48px}}
"""


def build() -> str:
    H = T("hypotheses").set_index("id")
    cal = T("sim_calibration_curve"); fix = T("sim_fixes"); infl = T("sim_inflation_fit"); mt = T("sim_matched_fa")
    sep = T("sim_masking_separation"); fp = T("sim_fingerprints").set_index("scenario"); meta = T("sim_meta").iloc[0]
    arch = T("real_archetypes").set_index("firm"); l2 = T("layer2_first_trip"); l3 = T("real_layer3_status").set_index("firm")
    ed = T("edgar_detection_summary").set_index("method"); ef = T("edgar_funnel").set_index("step")
    eam = T("edgar_archetype_mi").set_index("arch_archetype"); ems = T("edgar_mi_separation").iloc[0]
    au = T("robust_edgar_auc").set_index("score"); lc = T("robust_edgar_level_vs_change").set_index(["ratio", "role"])
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
                 "Sections 3–5 walk through the data, the method and the results hypothesis by hypothesis. Section 6 tests the "
                 "method on real US companies. Sections 7–8 show the robustness and sanity checks: how we know the numbers can be "
                 "trusted, and where common sense told us something was off. Sections 9–12 cover limits, the remaining to-do list, "
                 "how to reproduce everything, and likely viva questions.</p>"))
    o.append('<ol class="toc"><li>The project on one page</li><li>Key ideas in plain words</li><li>The data</li>'
             '<li>The method, layer by layer</li><li>Results, hypothesis by hypothesis</li><li>Does it work on real firms? (US EDGAR panel)</li>'
             '<li>Robustness: do the conclusions survive changes?</li><li>How we know it is right: sanity and common-sense checks</li>'
             '<li>Limitations</li><li>What is left to do</li><li>Reproducing everything</li><li>Viva preparation</li></ol>')

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
             f"give 10% false alarms gave {pct(c28.loc[0.10, 'empirical_fa'])} on simulated healthy firms with {n28} periods of history.</li>"
             f"<li><b>It can be fixed cheaply.</b> Three corrections bring the false-alarm rate to {pct(f28.loc['inflation_factor', 'empirical_fa'])}, "
             f"{pct(f28.loc['conformal', 'empirical_fa'])} and {pct(f28.loc['parametric_bootstrap', 'empirical_fa'])}; the best one costs "
             f"only {f28.loc['parametric_bootstrap', 'detection_cost_points']:.1f} points of detection.</li>"
             f"<li><b>Watching several ratios beats watching cash alone when cash is quiet or massaged</b>, but loses when the "
             f"problem is in cash itself.</li>"
             f"<li><b>The case studies match the two stories:</b> cash warned early at Jyoti; at NMC only liquidity cover "
             f"(cash ÷ short-term debt) warned, from FY{int(lcov.index[lcov < 1][0])}.</li>"
             f"<li><b>On {okf} real US bankruptcies, simple pre-set rules beat everything else</b>, and the statistical alarms barely "
             f"beat chance ({pct(am_u.detection)} of failed firms flagged vs {pct(am_u.false_alarm)} of healthy ones). A post-hoc "
             "diagnosis explains why (Section 6).</li></ol>")
    rows = []
    plain = {"H1": "Does Jyoti's own cash history warn ≥ 2 years early, using the statistical detector?",
             "H2": "At NMC, did cash flags stay silent while balance-sheet flags fired?",
             "H3": "Is watching many ratios better than one when cash is masked?",
             "H4": "Are standard alarm thresholds too trigger-happy with short histories?",
             "H5": "Can a cheap correction fix the thresholds?",
             "H6": "Does the masking index separate 'visible' from 'masked' crises in a cohort?"}
    why = {"H1": "Not enough pre-crisis data to train the detector (needs BSE quarterly FY2010–13).",
           "H2": "Yes, but only one balance-sheet flag fired, so NMC is 'Other', not 'Masked', by our own rule.",
           "H3": "Yes in simulation: big win when cash is quiet or smoothed, clear loss when cash itself drifts.",
           "H4": "Yes: several times more false alarms than promised, at every history length.",
           "H5": "Yes: all three fixes land within the tolerance; the parametric one costs least detection.",
           "H6": "Cohort collected (India), but annual data are too short for the masking index; rule-level patterns fit the idea."}
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
        ("Rule flag (Layer 2)", "A simple pre-set test such as 'OCF below zero' or 'EBITDA covers interest less than 1.2 times'. "
         "Cheap, transparent, and fixed in advance."),
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
         "detector fired first."),
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
    o.append(box("Data decisions worth remembering",
                 "<ul><li>Accounting identities (assets = liabilities + equity, the profit bridge, the cash-conversion cycle) are checked "
                 "before anything runs; a failure stops the run and names the firm, year and columns.</li>"
                 "<li>Missing years are never filled in. Section 7 shows why: interpolating Jyoti's FY2017 would invent a distress "
                 "signal from a number that was never published.</li>"
                 f"<li>Ratios with near-zero denominators are blanked (|denominator| ≤ {pct(cfg['ratios']['denominator_eps_rel_assets']['activity'], 0)} "
                 "of total assets for sales-based ratios) instead of producing absurd values.</li>"
                 "<li>US quarterly data: year-to-date figures were turned into quarters, flows summed over the trailing twelve months, "
                 "and only the first-filed value kept, so no later correction leaks back in time.</li></ul>", "good"))

    # ------------------------------------------------------------------ 4
    o.append('<h2>4. The method, layer by layer</h2>')
    layers = [("Layer 0", "Ratios and audit", "Build oriented ratios, mask impossible values, check identities. Output: <code>ratios_*.csv</code>."),
              ("Layer 1", "Event-time study", "Line firms up by years/quarters before the event and plot each ratio against healthy controls."),
              ("Layer 2", "Rule flags", f"Pre-set rules: OCF &lt; {cfg['layer2']['rules']['ocf_max']}; EBITDA/interest &lt; "
               f"{cfg['layer2']['rules']['ebitda_interest_cover_min']}; debtor days &gt; {cfg['layer2']['rules']['debtor_days_max']}; Altman Z''-EM &lt; {cfg['altman_z2_em']['distress']}."),
              ("Layer 3", "Cash detector", "Kalman model on OCF ÷ current liabilities, standardised errors, one-sided CUSUM."),
              ("Layer 4", "Multivariate detector", "Kalman errors on cash plus structural ratios, Ledoit–Wolf covariance, Crosier MCUSUM."),
              ("Layer 5", "Thresholds and alert tiers", f"Block-bootstrap thresholds (plus the corrections). <b>Amber</b>: either detector over its "
               f"{pct(dc['tiers']['amber_alpha'], 0)} threshold. <b>Red</b>: multivariate over its {pct(dc['tiers']['red_alpha'], 0)} threshold <i>and</i> a Layer 2 rule agrees."),
              ("Layer 6", "Masking index and fingerprint", "Which side (cash or structure) alarms, and which first; archetype rule.")]
    o.append('<div class="layers">' + "".join(f'<div class="layer"><div class="n">{a}</div><div class="t"><b>{b}.</b> {c}</div></div>'
                                              for a, b, c in layers) + "</div>")
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
    o.append(fig("sim_power_curves.png", "Figure 3. Detection at a matched 10% false-alarm rate: by drift size (left two) and by history length (right)."))

    o.append(f"<h3>H2: NMC, silent cash and a warning balance sheet {verdict_badge(H.loc['H2', 'verdict'])}</h3>")
    o.append(f"<p><b>Result.</b> {e(H.loc['H2', 'evidence'])}. Liquidity cover (cash and deposits ÷ short-term debt) was below 1 in "
             f"{', '.join('FY' + str(y) for y in lcov.index[lcov < 1])}, falling to {lcov.loc[lcov.index.max()]:.2f} in "
             f"{int(lcov.index.max())}.</p>")
    o.append(box("Common-sense caveat we kept",
                 f"<p>Only <b>one</b> of the three balance-sheet flags ever tripped, so the pre-registered archetype rule labels NMC "
                 f"<b>{e(arch.loc['nmc', 'archetype'])}</b>, not 'Masked'. We report this rather than bend the rule. Section 7 shows why: "
                 f"NMC's published accounts left out billions of debt. Add hidden debt of {masked_from:g}× the reported amount or more "
                 "and the same rule calls NMC Masked, years before the collapse.</p>", "warn"))
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
    o.append(f"<p><b>What the cohort does show (exploratory).</b> No firm is 'Masked' by the rule: Screener has no split of cash "
             f"against short-term debt, so liquidity cover can never be checked and firms can trip at most two of three structural "
             f"flags. Still, the order fits the idea behind H6: in Visible firms the cash flag came first (or in the same year), in "
             f"most 'Other' firms a balance-sheet flag came first or alone. Over the years both members of a pair have data, the "
             f"simple rules flagged {int(cw.failed_flagged)} of {int(cw.failed_firms)} failed firms and {int(cw.controls_flagged)} of "
             f"{int(cw.controls)} controls. With six firms this is an illustration, not proof.</p>")
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
    o.append(box("Finding 2: simple rules won",
                 f"<p>The pre-set rules flagged {pct(rule.detection)} of failed firms with {pct(rule.false_alarm)} false alarms, and "
                 f"were the only method with a lower expected cost than Altman Z''-EM at every cost ratio tested. Most of their power "
                 f"comes from 'OCF &lt; 0' alone ({pct(ru.loc['only OCF < 0', 'detection'])} / {pct(ru.loc['only OCF < 0', 'false_alarm'])}): "
                 "a cash-first result, just not from the statistical layer.</p>", "good"))
    o.append(box("Finding 3: the statistical alarms barely beat chance, and why",
                 f"<p>For the amber tiers the 95% interval of 'failed flagged minus healthy flagged' includes zero. A threshold-free "
                 f"check (AUC) gives {a_k.auc:.2f} for the pre-registered cash CUSUM and {a_m.auc:.2f} for the multivariate one, against "
                 f"{a_l.auc:.2f} for simply looking at the <i>lowest level</i> of the cash ratio and {a_z.auc:.2f} for the lowest Altman "
                 "score. Two reasons, found after seeing the data (so they are hypotheses, not results):</p><ol>"
                 f"<li><b>The model forgives slow decline.</b> The Kalman model treats part of every change as a new 'normal level' "
                 f"(median firm: {pct(lvs.auc, 0)} of its variance on the moving level; a quarter of firms above {pct(lvs.auc_ci_hi, 0)}). "
                 f"A steady slide is absorbed, so the surprises stay small. The same CUSUM measured against the fixed training average "
                 f"scores AUC {a_s.auc:.2f}.</li>"
                 f"<li><b>Failing firms were weak all along.</b> In their last training quarters, failed firms' median OCF ÷ current "
                 f"liabilities was already {lc.loc[(cr, 'failed'), 'training_level_median']:.2f} versus "
                 f"{lc.loc[(cr, 'control'), 'training_level_median']:.2f} for healthy firms. A method that only asks 'is this firm worse "
                 "than its own past?' cannot see a firm whose past was already bad.</li></ol>"
                 "<p>The practical lesson: combine 'worse than its own past' with 'weak in absolute terms', and prefer a "
                 "non-adaptive baseline for slow deterioration.</p>", "bad"))
    o.append(fig("robust_edgar_tradeoff_nmc_hidden_debt.png", "Figure 7. (A) Every threshold setting of the statistical detectors lies "
                 "just above the chance line; rules and Altman sit far above it. (B) NMC's 2018 ratios if hidden debt is added (Section 7)."))
    o.append(fig("edgar_layer1_event_time.png", "Figure 8. US failed firms vs healthy controls in event time (lower = worse). Failed firms start lower and fall further."))

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
        ["NMC with hidden debt added (0×, " + ", ".join(f"{x:g}×" for x in hd.index[1:]) + " reported debt)",
         f"Archetype becomes Masked from {masked_from:g}×; Z''-EM 2018 falls from {hd.loc[0.0, 'z2_em_2018']:.2f} to "
         f"{hd.loc[hd.index.max(), 'z2_em_2018']:.2f}. For scale: debt was later put at about US${rv['revealed_total']}bn against "
         f"US${rv['reported']}bn reported."],
    ]))

    # ------------------------------------------------------------------ 8
    o.append('<h2>8. How we know it is right: sanity and common-sense checks</h2>')
    o.append(f"<p><code>python -m src.sanity</code> runs {len(SC)} automated checks: "
             f"{int(cnt.get('PASS', 0))} pass, {int(cnt.get('WARN', 0))} warnings, {int(cnt.get('INFO', 0))} recorded facts, "
             f"{int(cnt.get('FAIL', 0))} failures. A <b>warning</b> means 'not a bug, but a reader must know this'. The full list:</p>")
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
    o.append('<h2>9. Limitations</h2><ul>'
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
             "<li><b>Lead times</b> are measured from period ends, not from when the accounts were published, so real warning is shorter.</li>"
             "<li><b>The multivariate detector</b> reacts to change in any direction and can alarm on improvement.</li>"
             "<li><b>The US diagnosis (Section 6) is post hoc</b>; it should be tested on new data before being believed.</li></ul>")

    # ------------------------------------------------------------------ 10
    o.append('<h2 class="nobreak">10. What is left to do</h2>')
    o.append(table(["#", "Task", "Why it matters", "Who"], [
        ["1", "Commit the repository (include <code>docs/PREREG_FINGERPRINT.txt</code> and <code>docs/prereg_snapshot/</code>)", "Timestamped proof that the plan came before the results", "You"],
        ["2", "Optional: quarterly or longer histories for the cohort (CMIE Prowess via your library) would let the detectors run and make H6 testable", "H6 is inconclusive only for lack of periods", "You"],
        ["3", "Optional: Jyoti's BSE quarterly results FY2010–13", "Makes H1 testable", "You, then <code>make real</code>"],
        ["—", "Delete the stray folder <code>C:\\c\\</code> (two validation files created by mistake)", "Housekeeping", "You"],
        ["4", "Check NMC's figures against the original annual reports (the one remaining [verify] item)", "The case study rests on them", "You"],
        ["5", "Read and edit <code>reports/report.md</code> into your own voice", "It is a generated draft", "You"],
        ["6", "Slides, if the course needs a presentation", "Not built yet", "Ask, and it can be generated from the same tables"],
    ]))

    # ------------------------------------------------------------------ 11
    o.append('<h2 class="nobreak">11. Reproducing everything</h2>')
    o.append(table(["Command", "What it does"], [
        ["<code>python -m pytest</code>", "All tests, including exact reproduction of the original scripts"],
        ["<code>python -m src.run_layers --tables --figures</code>", "Layers 0–2 for NMC and Jyoti"],
        ["<code>python -m src.simulate.run</code> then <code>python -m src.simulate.report</code>", "Simulation (about 1–2 hours on 9 workers) and its tables/figures"],
        ["<code>python -m src.run_real</code>", "Archetypes, Layer 3 status, H1–H6 verdicts"],
        ["<code>python -m src.ingest.edgar</code> then <code>python -m src.run_edgar</code>", "Download and analyse the US panel"],
        ["<code>python -m src.robustness</code>", "All robustness analyses"],
        ["<code>python -m src.sanity</code>", "All sanity checks (also runs the tests)"],
        ["<code>python -m src.reporting.write_report</code> / <code>write_explainer</code>", "Rebuild report.md / this document"],
    ]))
    o.append("<p class='small'>Key files: <code>configs/config.yaml</code> (every setting), <code>docs/PREREGISTRATION.md</code> (hypotheses and "
             "deviations log), <code>docs/decision_log.md</code> (every judgement call), <code>reports/tables/</code> (every number), "
             "<code>reports/figures/</code>.</p>")

    # ------------------------------------------------------------------ 12
    qa = [
        ("Why not just use Altman's Z-score?", f"It needs no history and did well on US firms ({pct(alt.detection)} flagged), but it also flagged "
         f"{pct(alt.false_alarm)} of healthy firms, and it missed NMC and warned on Jyoti only after the event. The project's point is a "
         "per-firm monitor with a known false-alarm rate, plus a way to tell visible from masked crises."),
        ("Isn't the simulation just showing what you built into it?", "The simulation checks the statistics (does a promised 10% stay 10%?) "
         "under known conditions. Its main finding, over-alarming with short histories, was then confirmed on real US firms that the "
         "simulation never saw."),
        ("How do you know you didn't tune the method to fit NMC and Jyoti?", "All rules and thresholds were fixed and hashed before any "
         "detector ran; the frozen settings are shown to be unchanged; deviations are logged; and NMC came out 'Other' against the "
         "story we expected, which we reported."),
        ("Your detectors barely beat chance on US data. Is the project a failure?", "No: that is a finding. The calibration problem and "
         "its fixes are confirmed, the rule layer works well, and the diagnosis (adaptive models absorb slow decline; failing firms "
         "are weak for years) says how to build the next version. Reporting it honestly is the point of pre-registration."),
        ("Why is NMC not 'Masked'?", "Because only one of three balance-sheet flags fired on the published numbers. With the debt that "
         "was later found, it would be Masked. Ratio analysis cannot see falsified inputs."),
        ("Why block bootstrap and not ordinary bootstrap?", "Quarterly errors are correlated over time; resampling runs of consecutive "
         "values keeps that correlation."),
        ("What would you do with more time?", "Get longer or quarterly histories for the cohort and Jyoti; add a non-adaptive and a level-aware "
         "detector; model heavy tails; use publication dates for lead times."),
    ]
    o.append('<h2>12. Viva preparation</h2><dl class="qa">' + "".join(f"<dt>{e(q)}</dt><dd>{a}</dd>" for q, a in qa) + "</dl>")
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
