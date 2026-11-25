"""Short report -> reports/report_short.md, report_short.html and report_short.pdf.

    python -m src.reporting.write_short_report [--no-pdf]

A concise version of report.md for readers who want the whole project in a few pages. Like every report here, each
number is read from reports/tables/*.csv or configs/config.yaml at build time; none is typed (src.sanity checks it).
"""
from __future__ import annotations

import argparse
import base64
import html
import re

import pandas as pd

from src.config import Paths, load_config
from src.reporting.write_explainer import to_pdf

cfg = load_config(); P = Paths.from_config(cfg)


def T(name):
    return pd.read_csv(P.tables / f"{name}.csv")


def pct(x, d=0):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


# ------------------------------------------------------------------ tiny document model -> markdown + html
def H(t, lvl=2): return ("h", lvl, t)
def Pg(t): return ("p", t)
def L(*items): return ("ul", list(items))
def Tb(head, rows): return ("table", head, rows)
def F(name, cap): return ("fig", name, cap)


def _inline_html(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def to_md(doc):
    out = []
    for b in doc:
        if b[0] == "h": out.append("#" * b[1] + " " + b[2])
        elif b[0] == "p": out.append(b[1])
        elif b[0] == "ul": out.append("\n".join(f"- {x}" for x in b[1]))
        elif b[0] == "table":
            out.append("| " + " | ".join(b[1]) + " |\n|" + "---|" * len(b[1]) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in b[2]))
        elif b[0] == "fig": out.append(f"![{b[2]}](figures/{b[1]})\n\n*{b[2]}*")
    return "\n\n".join(out) + "\n"


def to_html(doc, title):
    out = []
    for b in doc:
        if b[0] == "h": out.append(f"<h{b[1]}>{_inline_html(b[2])}</h{b[1]}>")
        elif b[0] == "p": out.append(f"<p>{_inline_html(b[1])}</p>")
        elif b[0] == "ul": out.append("<ul>" + "".join(f"<li>{_inline_html(x)}</li>" for x in b[1]) + "</ul>")
        elif b[0] == "table":
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline_html(h)}</th>" for h in b[1]) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline_html(str(c))}</td>" for c in r) + "</tr>" for r in b[2]) + "</tbody></table>")
        elif b[0] == "fig":
            f = P.figures / b[1]
            src = "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode() if f.exists() else ""
            out.append(f'<figure><img src="{src}" alt="{html.escape(b[2])}"><figcaption>{_inline_html(b[2])}</figcaption></figure>')
    css = ("@page{size:A4;margin:16mm}body{font-family:'Segoe UI',Calibri,Arial,sans-serif;font-size:10.5pt;line-height:1.5;color:#1f2328;"
           "max-width:780px;margin:0 auto;padding:16px}h1{font-size:20pt;color:#1d3557;margin:0 0 4px}h2{font-size:13.5pt;color:#1d3557;"
           "border-bottom:1.5px solid #1d3557;padding-bottom:2px;margin:20px 0 6px;break-after:avoid}h3{font-size:11pt;margin:12px 0 4px}"
           "table{border-collapse:collapse;width:100%;font-size:9.3pt;margin:6px 0 10px}th,td{border:1px solid #d0d7de;padding:3px 6px;"
           "text-align:left;vertical-align:top}th{background:#f6f8fa}tr{break-inside:avoid}figure{margin:8px 0;break-inside:avoid}"
           "figure img{width:100%;border:1px solid #d0d7de}figcaption{font-size:9pt;color:#57606a}code{font-family:Consolas,monospace;"
           "font-size:9pt;background:#f6f8fa;padding:0 3px}ul{margin:4px 0 8px 18px;padding-left:4px}p{margin:5px 0 8px}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(out)}</body></html>")


# ------------------------------------------------------------------ content
def build():
    Hy = T("hypotheses").set_index("id"); cal = T("sim_calibration_curve"); fix = T("sim_fixes"); mt = T("sim_matched_fa")
    ed = T("edgar_detection_summary").set_index("method"); ef = T("edgar_funnel").set_index("step")
    au = T("robust_edgar_auc").set_index("score"); l2 = T("layer2_first_trip"); pr = T("posthoc_paired_rules")
    rn = pd.read_csv(P.tables / "ratios_nmc.csv", index_col=0)
    sw = T("posthoc_static_weakness").set_index(["method", "role"]); w_ = T("posthoc_winsor"); fl = T("posthoc_filing_lead").set_index("method")
    lc = T("robust_edgar_level_vs_change"); lc = lc[lc["sample"] == "analysed"].set_index(["ratio", "role"])
    co = T("cohort_archetypes"); cf = co[co.role == "failed"]
    hl, rp, se, dn, cv, et = (T(f"eda_{k}") for k in ["history_length", "ratio_properties", "seasonality", "denominators", "covariance", "event_time"])
    sc, dc = cfg["simulation"], cfg["detectors"]; n28 = sc["h5_training_length"]; tol = sc["h5_tolerance_points"]
    M = cfg["edgar"]["monitor_quarters"]; cash = cfg["masking"]["cash_ratio"]; l2r = cfg["layer2"]["rules"]; az = cfg["altman_z2_em"]
    c28 = cal[(cal.n == n28) & (cal.detector == "kalman_cusum")].set_index("nominal")
    k10 = cal[(cal.detector == "kalman_cusum") & (cal.nominal == 0.10)].sort_values("n")
    f28 = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    m = mt.set_index(["scenario", "detector"]).detection
    jf = l2[l2.firm == "jyoti"].set_index("rule"); jo = jf.loc["OCF < 0"]; ja = jf.loc["Altman Z''-EM < 4.35 (distress)"]
    lcov = rn.liquid_to_st_debt
    am, amc, rule, alt = (ed.loc[x] for x in ["amber (uncorrected)", "amber (n-corrected)", "any Layer 2 rule", "Altman Z''-EM distress"])
    okf, okc = int(am.failed_firms), int(am.controls); nfail = int(ef.loc["failed firms in panel", "count"])
    pa = pr[pr.other == "Altman Z''-EM distress"].set_index("metric")
    stat = [x for x in pr.other.unique() if x != "Altman Z''-EM distress"]
    beat_stats = all(pr[(pr.other == x) & pr.metric.str.startswith("expected_cost")].verdict.str.startswith("rules better").all() for x in stat)
    alt_indist = all(pa.loc[f"expected_cost_cm{x}", "verdict"].startswith("not") for x in cfg["cost"]["c_miss"])
    a_k, a_s, a_l = (au.loc[x] for x in ["cash CUSUM on Kalman errors (pre-registered)",
                                          "cash CUSUM on deviations from the training mean (non-adaptive)",
                                          "lowest cash ratio in the window (level)"])
    a = rp[rp.role == "all analysed"].set_index("ratio"); s4 = se[se.lag_quarters == 4].set_index("input")
    d = dn.set_index(["item", "role"]); c = cv.set_index("dimension_p").loc["all"]
    anf = hl[hl["sample"] == "EDGAR: analysed failed firms"].iloc[0]; f0, f4 = et.iloc[0], et[et.tau_quarters == -4].iloc[0]
    rl = sw.loc[("any Layer 2 rule", "failed")]
    wmax = float(max((w_.detection - w_.base_detection).abs().max(), (w_.false_alarm - w_.base_false_alarm).abs().max()))
    ac = lambda r: f"{r.auc:.2f} ({r.auc_ci_lo:.2f}–{r.auc_ci_hi:.2f})"  # noqa: E731
    cev = pd.to_datetime(cf.event_date).dt.year
    tiers = [x for x in ed.index if x.startswith(("amber", "red"))]
    chance = [x for x in tiers if ed.loc[x, "diff_ci_lo"] <= 0]; sep = [x for x in tiers if ed.loc[x, "diff_ci_lo"] > 0]

    doc = [("h", 1, "Judging a Firm Against Its Own History: Early Warning of Cash-Flow Crises"),
           Pg(f"*Foundations of Data Science course project. Pre-registered {cfg['prereg_date']}. Every number is generated from "
              "`reports/tables/`; full detail in `report.md` and `explainer.html`.*"),
           H("Summary"),
           L(f"**Question.** Can a firm's own ratio history warn of a cash-flow crisis, and how far can alarm thresholds be trusted when that history is short?",
             f"**Main finding.** Standard alarm thresholds are far too trigger-happy on short histories: set for {pct(0.10)} false alarms, they gave "
             f"{pct(c28.loc[0.10, 'empirical_fa'], 1)} at n = {n28}. A parametric bootstrap with refitting brings this to "
             f"{pct(f28.loc['parametric_bootstrap', 'empirical_fa'], 1)} at a cost of {f28.loc['parametric_bootstrap', 'detection_cost_points']:.1f} points of detection.",
             f"**Real data.** On {okf} US bankruptcies and {okc} matched healthy firms, simple cash-first rules worked best "
             f"({pct(rule.detection)} of failures flagged, {pct(rule.false_alarm)} false alarms). Of the statistical alarm tiers, "
             + (f"{', '.join(chance)} showed no detectable advantage over chance" if chance else "every tier separated failed from healthy firms")
             + (f"; only {', '.join(sep)} did, and weakly." if sep else "."),
             "**Verdicts.** " + "; ".join(f"{i} {Hy.loc[i, 'verdict'].lower()}" for i in Hy.index) + "."),

           H("1. Problem and idea"),
           Pg("Distress models such as Altman's Z-score compare a firm with many peers and need large samples of past failures. Mid-sized "
              "firms in India or the Gulf often have neither. This project instead compares each firm **with its own past**: learn what "
              "'normal' looks like for that firm, then alarm when its ratios drift persistently in the bad direction."),
           Pg(f"Two cases motivated it. At **Jyoti Structures** operating cash flow (OCF) turned negative in FY{int(jo.first_trip_fy)}, "
              f"{jo.lead_years:.1f} years before insolvency; Altman Z''-EM reached distress only in FY{int(ja.first_trip_fy)}, after the event. At **NMC Health** "
              f"reported cash flow stayed healthy; only liquidity cover (cash ÷ short-term debt) fell below 1, to {lcov.loc[lcov.index.max()]:.2f} "
              f"in {int(lcov.index.max())}. So the method watches cash and the balance sheet separately."),

           H("2. Data"),
           Tb(["Source", "What", "Used for"], [
               ["NMC Health", f"{len(rn)} annual reports, as originally published", "Case study (H2)"],
               ["Jyoti Structures", "Annual accounts from Screener.in", "Case study (H1)"],
               ["India cohort", f"{len(cf)} firms admitted to insolvency {cev.min()}–{cev.max()} + matched controls", "Confirmatory test (H6)"],
               ["US SEC EDGAR", f"Quarterly filings: {nfail} firms with a bankruptcy filing, matched healthy controls; {okf} + {okc} had enough history", "Real-data test of the full method"],
               ["Simulation", f"{sc['firms_per_cell']} synthetic firms per setting, histories n = {', '.join(map(str, sc['training_lengths']))}", "H3, H4, H5"]]),
           Pg("Accounting identities are checked before any analysis, missing years are never filled in, and only first-published figures are used (no hindsight)."),

           H("3. Method"),
           L("**Layer 0 – Ratios.** All ratios oriented so that lower = worse; near-zero denominators blanked.",
             "**Layer 1 – Event time.** Firms lined up by quarters (or years) before the event.",
             f"**Layer 2 – Rules.** OCF < {l2r['ocf_max']}; EBITDA/interest < {l2r['ebitda_interest_cover_min']:g}; debtor days > {l2r['debtor_days_max']}; Altman Z''-EM < {az['distress']:g}.",
             f"**Layer 3 – Cash detector.** A state-space (Kalman) model of OCF ÷ current liabilities learns the firm's normal level; a CUSUM adds up bad surprises beyond k = {dc['cusum']['k']}.",
             "**Layer 4 – Multivariate detector.** The same on cash plus balance-sheet ratios together (Crosier MCUSUM).",
             f"**Layer 5 – Thresholds.** Set by block-bootstrapping the firm's own forecast errors; amber alarm at {pct(dc['tiers']['amber_alpha'])}, red at {pct(dc['tiers']['red_alpha'])} plus a rule.",
             "**Layer 6 – Masking index.** Is the balance sheet alarming while cash looks calm?"),
           Pg(f"Each firm's history is split into a training window (learn normal) and a {M}-quarter monitored window (count alarms). Healthy "
              "controls are monitored over the same dates, so any control alarm is a real false alarm. All hypotheses, rules and thresholds were fixed before any detector ran."),

           H("4. What the data look like (EDA) and the design choices it supports"),
           Tb(["Data fact (US training quarters)", "Design choice"], [
               [f"Median {anf['median']:g} training quarters; only {pct(anf.share_at_least_h5_n)} reach n = {n28}", "Test thresholds at short n (H4, H5)"],
               [f"Ratios are persistent (lag-1 autocorrelation {a.median_acf1_level.min():.2f}–{a.median_acf1_level.max():.2f})", "Moving-level + AR(1) model, not independent draws"],
               [f"Quarterly cash is seasonal (lag-4 autocorrelation of changes {s4.iloc[1].median_acf_of_change:+.2f}; {s4.iloc[0].median_acf_of_change:+.2f} with 12-month sums)", "Trailing-twelve-month cash flows"],
               [f"OCF ÷ current liabilities has the best coverage ({pct(a.loc[cash, 'share_quarters_observed'])}) and light tails", "Cash ratio for Layer 3"],
               [f"Interest ≤ {pct(cfg['ratios']['denominator_eps_rel_assets']['activity'])} of assets in {pct(d.loc[('interest', 'control'), 'share_below_activity_eps'])} of healthy quarters", "A smaller near-zero cut-off for interest"],
               [f"Failed firms sit {f0.gap_control_minus_failed:.2f} below controls {int(-f0.tau_quarters)} quarters out; gap grows to {f4.gap_control_minus_failed:.2f}", f"Cumulative CUSUM over {M} quarters"],
               [f"Covariance shrinkage is usually total (median {c.median_shrinkage:.2f})", "Weakness: the multivariate detector ignores correlations"]]),
           Pg("*This EDA was run after the results (training data only); it checks that the choices fit the data, and changes nothing.*"),

           H("5. Results"),
           H("H4 – Thresholds on short histories are too trigger-happy: " + Hy.loc["H4", "verdict"].lower(), 3),
           Tb(["History n", "Promised", "Actual false alarms (95% CI)"],
              [[r.n, pct(0.10), f"{pct(r.empirical_fa, 1)} ({pct(r.ci_lo, 1)}–{pct(r.ci_hi, 1)})"] for r in k10.itertuples()]),
           Pg("A lender promised one false alarm in ten would get about one in three. The effect is known for control charts with estimated parameters; this quantifies it for short financial histories."),
           F("sim_calibration_curve.png", "Figure 1. Actual vs promised false-alarm rates in simulation. Every point is far above the diagonal."),
           H("H5 – Cheap corrections fix it: " + Hy.loc["H5", "verdict"].lower(), 3),
           Tb(["Method", f"False alarms (promised {pct(0.10)})", "Detection of a real drift", "Detection lost"],
              [[lab, pct(f28.loc[k, "empirical_fa"], 1), pct(f28.loc[k, "detection_S1"], 1), f"{f28.loc[k, 'detection_cost_points']:.1f} pts"]
               for k, lab in [("block_bootstrap", "No correction"), ("inflation_factor", "Inflation factor"),
                              ("conformal", "Conformal threshold"), ("parametric_bootstrap", "Parametric bootstrap")]]),
           Pg(f"All three land within ±{tol} points of the target on held-out firms. The parametric bootstrap targets the real cause (uncertain model parameters) and loses least."),
           H("H3 – One ratio or many: " + Hy.loc["H3", "verdict"].lower(), 3),
           Tb(["Scenario", "Cash-only detector", "Multivariate detector"],
              [[f"{s} – {lab}", pct(m[(s, "kalman")]), pct(m[(s, "mcusum")])] for s, lab in
               [("S1", "cash drifts"), ("S2", "cash fine, others worsen"), ("S3", "small drift everywhere"), ("S4", "cash smoothed (concealed)")]]),
           Pg(f"Each detector wins where the other is blind, so both are run. The loss in S1 is larger than 'slight', and plain ARIMA had more power ({pct(m[('S1', 'arima')])} in S1)."),
           H("H1, H2, H6 – Case studies and cohort", 3),
           L(f"**H1 ({Hy.loc['H1', 'verdict'].lower()}).** Jyoti has no usable pre-FY14 history for the detector; the simple rules did warn {jo.lead_years:.1f} years early.",
             f"**H2 ({Hy.loc['H2', 'verdict'].lower()}).** NMC's cash flag stayed silent while liquidity cover fell below 1. Only one of three balance-sheet flags tripped, so by the pre-set rule NMC is not 'Masked'.",
             f"**H6 ({Hy.loc['H6', 'verdict'].lower()}).** The India cohort ({len(cf)} firms) has only annual data, too short for the detectors."),

           H("6. Real-data test: US bankruptcies"),
           Tb(["Method", "Failed firms flagged", "Healthy firms flagged", "Difference (95% CI)"],
              [[k, pct(r.detection), pct(r.false_alarm), f"{100 * r.det_minus_fa:+.1f} pts ({100 * r.diff_ci_lo:+.1f} to {100 * r.diff_ci_hi:+.1f})"] for k, r in ed.iterrows()]),
           L(f"**Over-alarming is real.** Uncorrected amber flagged {pct(am.false_alarm)} of healthy firms (promised {pct(0.10)}); the correction lowered it to {pct(amc.false_alarm)}, not enough.",
             f"**Rules did best.** " + ("Lower expected cost than every statistical tier (paired CIs exclude zero). " if beat_stats else "")
             + ("Not distinguishable from Altman Z''-EM (paired CIs include zero)." if alt_indist else "Better than Altman at some cost ratios."),
             f"**Why the statistical layer struggled** (found after the results): the adaptive model absorbs slow decline (AUC {ac(a_k)} vs {ac(a_s)} against a fixed baseline), "
             f"and failed firms were weak all along (training cash ratio {lc.loc[(cash, 'failed'), 'training_level_median']:.2f} vs {lc.loc[(cash, 'control'), 'training_level_median']:.2f}; "
             f"the lowest level alone scores AUC {ac(a_l)})."),
           F("robust_edgar_tradeoff_nmc_hidden_debt.png", "Figure 2. (A) Every threshold setting of the statistical detectors lies near the chance line; rules and Altman sit well above it. (B) NMC stress test with hidden debt added."),

           H("7. Robustness"),
           L("Changing the CUSUM allowance, the training window, the ratio set or the event date leaves the ranking of methods unchanged.",
             f"Winsorising every ratio moves no rate by more than {100 * wmax:.1f} points.",
             f"Long lead times are mostly static weakness: {pct(rl.of_those_already_on_before_window)} of early rule alarms were already on before monitoring began. "
             f"Measured from filing dates, the median rule lead is {fl.loc['any Layer 2 rule', 'median_lead_quarters_filing_date']:.1f} quarters, not {fl.loc['any Layer 2 rule', 'median_lead_quarters_period_end']:g}.",
             "Interpolating Jyoti's missing FY2017 would have invented a distress signal; leaving it missing was right."),

           H("8. Limitations"),
           L("The two case studies were known in advance, so they illustrate rather than test.",
             "Controls are survivors; a real loan book is harder to separate.",
             "No ratio can see debt kept off the accounts (NMC).",
             "The simulation is Gaussian; real ratios have heavier tails, so the correction transferred only partly.",
             "The US diagnosis was found after the results and must be tested on new data."),

           H("9. Conclusion"),
           L("Alarm thresholds learned from a short history over-alarm badly; this must be corrected before any per-firm monitor is trusted.",
             "A parametric bootstrap with refitting is a cheap, effective correction.",
             "In practice, cash-first rules beat the statistical detectors. The next version should combine 'worse than its own past' with 'weak in absolute terms', use a non-adaptive baseline, and calibrate thresholds on real healthy firms."),
           ]
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args(argv)
    doc = build(); out = P.tables.parent
    (out / "report_short.md").write_text(to_md(doc), encoding="utf-8")
    h = out / "report_short.html"; h.write_text(to_html(doc, "Early-Warning Project: Short Report"), encoding="utf-8")
    print("wrote", out / "report_short.md"); print("wrote", h)
    if not a.no_pdf:
        to_pdf(h, h.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
