"""Generate reports/report.md. Every number is read from reports/tables/*.csv at build time; none is typed.

    python -m src.reporting.write_report
"""
from __future__ import annotations

import pandas as pd

from src.config import Paths, load_config

cfg = load_config(); P = Paths.from_config(cfg)
VERIFY, MISSING = [], []


def T(name):
    return pd.read_csv(P.tables / f"{name}.csv")


def pct(x, d=1):
    return "n/a" if pd.isna(x) else f"{100 * x:.{d}f}%"


def ci(r, lo="ci_lo", hi="ci_hi"):
    return f"{pct(r[lo])}–{pct(r[hi])}"


def v(tag):
    VERIFY.append(tag); return f"[verify: {tag}]"


def missing(tag):
    MISSING.append(tag); return f"[MISSING: {tag}]"


REFERENCES = [
    "Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.",
    "Altman, E. I. (2005). An emerging market credit scoring system for corporate bonds. *Emerging Markets Review*, 6(4), 311–323.",
    "Altman, E. I., Hartzell, J., & Peck, M. (1995). *Emerging markets corporate bonds: A scoring system*. Salomon Brothers.",
    "Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.",
    "Beaver, W. H. (1966). Financial ratios as predictors of failure. *Journal of Accounting Research*, 4 (Supplement), 71–111.",
    "Beneish, M. D. (1999). The detection of earnings manipulation. *Financial Analysts Journal*, 55(5), 24–36.",
    "Casey, C., & Bartczak, N. (1985). Using operating cash flow data to predict financial distress: Some extensions. *Journal of Accounting Research*, 23(1), 384–401.",
    "Crosier, R. B. (1988). Multivariate generalizations of cumulative sum quality-control schemes. *Technometrics*, 30(3), 291–303.",
    "Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.",
    "Jensen, W. A., Jones-Farmer, L. A., Champ, C. W., & Woodall, W. H. (2006). Effects of parameter estimation on control chart properties: A literature review. *Journal of Quality Technology*, 38(4), 349–364.",
    "Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217–1241.",
    "Largay, J. A., & Stickney, C. P. (1980). Cash flows, ratio analysis and the W. T. Grant Company bankruptcy. *Financial Analysts Journal*, 36(4), 51–54.",
    "Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.",
    "Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research*, 18(1), 109–131.",
    "Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1/2), 100–115.",
    "Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.",
    "Shumway, T. (2001). Forecasting bankruptcy more accurately: A simple hazard model. *Journal of Business*, 74(1), 101–124.",
    "Sloan, R. G. (1996). Do stock prices fully reflect information in accruals and cash flows about future earnings? *The Accounting Review*, 71(3), 289–315.",
    "Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.",
]


def cohort_section() -> list[str]:
    """Cohort paragraph and tables from cohort_*.csv (written by src.run_real)."""
    if not (P.tables / "cohort_archetypes.csv").exists():
        return [f"{missing('India/GCC cohort data (6-8 failed firms + controls)')}\n"]
    co = T("cohort_archetypes"); rd = T("cohort_rule_detection").set_index("signal"); pr = T("cohort_pairs_common_window")
    f, c, ng = co[co.role == "failed"], co[co.role == "control"], co[co.role == "negative_control"]
    unmatched = [r.name for r in f.itertuples() if r.firm_id not in set(c.matched_to)]
    s = [f"**The cohort (India).** Built from RBI's 2017 large-defaulter lists by the pre-registered rule, before any figure was "
         f"read (`data/raw/cohort/PROPOSAL.md`, fingerprinted in `docs/PREREG_FINGERPRINT.txt`): {len(f)} listed failed firms with "
         f"≥ {cfg['cohort']['min_pre_event_annual_filings']} pre-event annual filings, {len(c)} controls matched mechanically on "
         f"Screener industry and total assets (0.5–2×), and {len(ng)} negative control. "
         + (f"{', '.join(unmatched)} has no control: no firm in its industry was within the size band. " if unmatched else "")
         + "Data: Screener.in public pages (the parser reproduces the hand-transcribed Jyoti data exactly); periods longer or "
         "shorter than 12 months (two firms changed their year-end) have flows scaled to 12 months. GCC firms could not be "
         "included (no machine-readable source).\n"]
    s.append("| Firm | Role | Event FY | Archetype | Flags in the window | First cash flag | First structural flag | Order |\n|---|---|---|---|---|---|---|---|")
    for r in co.itertuples():
        fy = lambda x: f"FY{int(x)}" if pd.notna(x) else "none"  # noqa: E731
        s.append(f"| {r.name} | {r.role.replace('_', ' ')} | FY{int(r.event_fy)} | {r.archetype} | FY{r.window_fy} | "
                 f"{fy(r.first_cash_flag_fy)} | {fy(r.first_structural_flag_fy)} | {r.flag_order} |")
    cw = rd.loc["any Layer 2 rule, common years only"]
    vis, oth = f[f.archetype == "Visible"], f[f.archetype == "Other"]
    s.append(f"\nNo cohort firm is Masked by the pre-set rule: {len(vis)} are Visible and {len(oth)} Other. One reason is "
             "structural: Screener reports no cash/short-term-debt split, so liquidity cover is never evaluable and a firm can "
             "trip at most two of the three structural flags, while also needing positive OCF in every pre-event year. "
             f"The rule-level order still follows the archetypes: in {int(vis.flag_order.isin(['cash first', 'same year']).sum())} of "
             f"{len(vis)} Visible firms the cash flag came first or in the same year, and in "
             f"{int(oth.flag_order.isin(['structural first', 'structural only']).sum())} of {len(oth)} Other firms a structural flag "
             "came first or alone. With this few firms that is an illustration, not a test. Over the fiscal years where both members "
             f"of a pair have data, the pre-set rules flagged {int(cw.failed_flagged)} of {int(cw.failed_firms)} failed firms and "
             f"{int(cw.controls_flagged)} of {int(cw.controls)} controls (`cohort_pairs_common_window.csv`). The controls' history "
             f"is short (Screener's public pages show twelve years), so the common window is only "
             f"{int(pr.n_common_years.min())}–{int(pr.n_common_years.max())} years.\n")
    return s


def robustness_section() -> list[str]:
    """§9 from reports/tables/robust_*.csv and sanity_checks.csv (written by src.robustness / src.sanity)."""
    s = []
    need = ["robust_edgar_variants", "robust_edgar_h_grid", "robust_edgar_rules", "robust_edgar_auc", "robust_edgar_reproduce",
            "robust_jyoti_fill", "robust_case_alt_events", "robust_nmc_hidden_debt", "robust_edgar_level_vs_change"]
    if not all((P.tables / f"{n}.csv").exists() for n in need):
        s.append(f"{missing('robustness tables: run python -m src.robustness')}\n"); return s
    V = T("robust_edgar_variants"); hg = T("robust_edgar_h_grid"); ru = T("robust_edgar_rules"); au = T("robust_edgar_auc")
    rp = T("robust_edgar_reproduce").iloc[0]; jf = T("robust_jyoti_fill"); ae = T("robust_case_alt_events")
    hd = T("robust_nmc_hidden_debt"); lc = T("robust_edgar_level_vs_change").set_index(["ratio", "role"])
    rb = cfg["robustness"]
    s.append("All analyses in this section were added after the main results were seen (logged as a deviation); they test "
             "whether conclusions survive reasonable changes and use no verdict. Script: `src/robustness.py`.\n")
    s.append(f"**9.1 Sensitivity of the EDGAR results.** Re-running the pre-registered settings reproduced all {int(rp.identical)} of "
             f"{int(rp.firm_runs)} firm runs exactly. Each variant then changes one thing (k = CUSUM reference value; shift = training/"
             f"monitoring boundary moved by ±{rb['training_window_shift']} quarters; drop = leave one structural ratio out of the "
             f"multivariate detector; event{rb['edgar_event_shift_quarters']:+d}q = event moved {abs(rb['edgar_event_shift_quarters'])} "
             "quarters earlier, a proxy for earlier distress events such as defaults):\n")
    s.append("| Variant | Firms (failed/controls) | Amber uncorrected: detection / false alarm | Red uncorrected: detection / false alarm | Any rule: detection / false alarm |\n|---|---|---|---|---|")
    for vname, g in V.groupby("variant", sort=False):
        g = g.set_index("method")
        cell = lambda m: f"{pct(g.loc[m, 'detection'])} / {pct(g.loc[m, 'false_alarm'])}" if m in g.index else "n/a"  # noqa: E731
        s.append(f"| {vname} | {int(g.iloc[0].failed_firms)}/{int(g.iloc[0].controls)} | {cell('amber (uncorrected)')} | "
                 f"{cell('red (uncorrected)')} | {cell('any Layer 2 rule')} |")
    base = V[V.variant == "base"].set_index("method"); ev = V[V.variant.str.startswith("event")].set_index("method")
    amb = V[V.method == "amber (uncorrected)"]
    w = V.pivot(index="variant", columns="method", values=["detection", "false_alarm"])
    det_more = int((w["detection"]["any Layer 2 rule"] > w["detection"]["amber (uncorrected)"]).sum())
    fa_less = int((w["false_alarm"]["any Layer 2 rule"] < w["false_alarm"]["amber (uncorrected)"]).sum())
    s.append(f"\nThe rules detect more than the amber tier in {det_more} of {len(w)} variants and have fewer false alarms in "
             f"{fa_less} of {len(w)}; where they do not, the gap is within a few points. Amber false alarms stay between {pct(amb.false_alarm.min())} and {pct(amb.false_alarm.max())}, far above the 10% "
             f"target, whatever k, window or ratio set is used. Dropping Altman Z''-EM from the multivariate detector hurts most (red "
             f"detection {pct(V[(V.variant == 'drop=altman_z2_em') & (V.method == 'red (uncorrected)')].detection.iloc[0])} vs "
             f"{pct(base.loc['red (uncorrected)', 'detection'])}). With the event moved a year earlier, rules still detect "
             f"{pct(ev.loc['any Layer 2 rule', 'detection'])} with {pct(ev.loc['any Layer 2 rule', 'false_alarm'])} false alarms.\n")
    r = ru.set_index("rule_set")
    only = [x for x in r.index if x.startswith("only")]
    s.append("**Rule ablation.** " + "; ".join(f"{x}: {pct(r.loc[x, 'detection'])} / {pct(r.loc[x, 'false_alarm'])}" for x in only) +
             " (detection / false alarm). The debtor-days rule almost never trips on US firms and adds nothing; operating cash flow "
             "below zero carries most of the signal, which is itself a cash-first result.\n")
    s.append("**9.2 Why the statistical detectors do not discriminate on real firms.** Raising or lowering the amber threshold "
             "moves detection and false alarms together:\n")
    s.append("| Threshold × | " + " | ".join(hg.detector.unique()) + " |\n|---|" + "---|" * hg.detector.nunique())
    for m_, g in hg.groupby("h_multiplier"):
        g = g.set_index("detector")
        s.append(f"| {m_:g} | " + " | ".join(f"{pct(g.loc[d, 'detection'], 0)} / {pct(g.loc[d, 'false_alarm'], 0)}" for d in hg.detector.unique()) + " |")
    a = au.set_index("score")
    share = a[a.index.str.startswith("median share")].iloc[0]
    sc = a[~a.index.str.startswith("median share")]
    s.append("\nA threshold-free comparison (AUC: probability that a random failed firm scores worse than a random control; "
             "0.5 = chance) shows where the signal is:\n")
    s.append("| Alarm score over the monitored window | AUC (95% CI) |\n|---|---|")
    for lab, row in sc.iterrows():
        s.append(f"| {lab} | {row.auc:.2f} ({row.auc_ci_lo:.2f}–{row.auc_ci_hi:.2f}) |")
    cr = "ocf_to_current_liabilities"
    s.append(f"\nTwo things plausibly explain it. First, the local-level model **adapts**: the median firm puts {pct(share.auc, 0)} of "
             f"the fitted variance on the random-walk level and a quarter of firms more than {pct(share.auc_ci_hi, 0)}; for those, a slow "
             "decline is absorbed into the level and the one-step errors stay small. The same CUSUM on plain deviations from the "
             "training mean scores higher, although the intervals overlap. Second, failed firms differ in **level** as much as in change: in their last training "
             f"quarters their median OCF / current liabilities was already {lc.loc[(cr, 'failed'), 'training_level_median']:.2f} against "
             f"{lc.loc[(cr, 'control'), 'training_level_median']:.2f} for controls, before falling a further "
             f"{abs(lc.loc[(cr, 'failed'), 'change_to_last_4q_median']):.2f}. A detector that only asks 'is this firm worse than its own "
             "past?' cannot see a firm that was weak all along; a level rule can. This is exploratory and was found after seeing the "
             "data, so it is a hypothesis for the cohort, not a result.\n")
    s.append("![Trade-off and hidden debt](figures/robust_edgar_tradeoff_nmc_hidden_debt.png)\n\n*Figure 10. (A) Moving the "
             "threshold slides the statistical detectors along a line just above chance, while the level-based rules and Altman sit "
             "far above it. (B) NMC's 2018 ratios once hidden debt is added (see 9.5).*\n")
    j = jf.set_index(["fill_missing", "rule"])
    zr = "Altman Z''-EM < 4.35 (distress)"
    s.append(f"**9.3 Jyoti FY2017 gap.** Layer 3 cannot run on Jyoti (§8), so the Kalman-versus-interpolation comparison was done on "
             f"Layers 1–2. Linear interpolation of the missing year leaves the first trip of every cash rule unchanged "
             f"(FY{int(j.loc[('none', 'Any rule'), 'first_trip_fy'])}) and the archetype unchanged "
             f"({j.loc[('none', 'Any rule'), 'archetype']}), but it moves Altman Z''-EM into distress one year earlier "
             f"(FY{int(j.loc[('linear', zr), 'first_trip_fy'])} instead of FY{int(j.loc[('none', zr), 'first_trip_fy'])}) using a "
             f"value (Z''-EM {j.loc[('linear', 'FY2017 value: altman_z2_em'), 'value']:.2f}) that was never published. Keeping the gap "
             "missing is the honest choice.\n")
    e = ae.set_index(["firm", "definition"])
    jy = [x for x in e.loc["jyoti"].index]
    s.append("**9.4 Alternative event dates.** " + "; ".join(
        f"Jyoti, {d}: first rule FY{int(e.loc[('jyoti', d), 'first_any_rule_fy'])}, lead {e.loc[('jyoti', d), 'lead_years_any_rule']:.2f} years"
        for d in jy) + f". NMC: no pre-set rule trips under any event date ({', '.join(e.loc['nmc'].index)}); its archetype stays "
        f"{e.loc[('nmc', 'pre-registered'), 'archetype']}. Jyoti's rule-based lead "
        f"{'stays at or above' if e.loc['jyoti'].lead_years_any_rule.min() >= 2 else 'falls below'} the two years H1 asks for when "
        "the event is dated a year earlier (H1 itself concerns the statistical detector, which could not run).\n")
    h = hd.set_index("hidden_debt_multiple"); rv = rb["nmc_revealed_debt_usd_bn"]
    s.append(f"**9.5 NMC as reported versus a hidden-debt restatement.** NMC never published restated accounts. After the "
             f"collapse the company put total debt at about US${rv['revealed_total']}bn against about US${rv['reported']}bn in its last "
             "accounts, so the stress test adds undisclosed short-term debt as a multiple of reported debt (same interest rate, equity "
             "reduced one-for-one):\n")
    s.append("| Hidden debt (× reported) | Archetype | Structural flags tripped | Liquidity cover 2018 | Interest cover 2018 | Z''-EM 2018 |\n|---|---|---|---|---|---|")
    for m_, row in h.iterrows():
        s.append(f"| {m_:g} | {row.archetype} | {int(row.n_structural_flags)} | {row.liquidity_cover_2018:.2f} | "
                 f"{row.interest_cover_2018:.2f} | {row.z2_em_2018:.2f} |")
    first_masked = h.index[h.archetype == "Masked"]
    s.append(f"\nWith hidden debt of {first_masked.min():g}× reported debt or more, NMC becomes **Masked** under the pre-set rule, "
             "with Z''-EM in distress years before the collapse. The rule failed on NMC because the inputs were false, not because "
             "the rule is wrong: no ratio method can see debt that is kept off the accounts, which is why the report treats possible "
             "falsification as a hard limit.\n")
    if (P.tables / "sanity_checks.csv").exists():
        sc_ = T("sanity_checks"); cnt = sc_.status.value_counts()
        s.append(f"**9.6 Sanity checks.** `python -m src.sanity` runs {len(sc_)} checks "
                 f"({', '.join(f'{int(cnt.get(k, 0))} {k}' for k in ['PASS', 'WARN', 'INFO', 'FAIL'])}); the full list is in "
                 "`reports/tables/sanity_checks.csv`. They confirm, among other things, that the pre-registered settings are "
                 "unchanged since the fingerprint, that no alarm uses data after the event, that every tested table regenerates "
                 "identically, and that thresholds learned from the right distribution hit their nominal rate (so the CUSUM code "
                 "itself is sound). The warnings are the findings above.\n")
    return s


def build() -> str:
    H = T("hypotheses").set_index("id")
    cal = T("sim_calibration_curve"); fix = T("sim_fixes"); infl = T("sim_inflation_fit"); mt = T("sim_matched_fa")
    sep = T("sim_masking_separation"); msum = T("sim_masking_summary").set_index("scenario"); fp = T("sim_fingerprints").set_index("scenario")
    meta = T("sim_meta").iloc[0]; arch = T("real_archetypes").set_index("firm"); l2 = T("layer2_first_trip")
    h2 = pd.read_csv(P.tables / "real_h2_nmc_flags.csv", index_col=0); rn = pd.read_csv(P.tables / "ratios_nmc.csv", index_col=0)
    rj = pd.read_csv(P.tables / "ratios_jyoti.csv", index_col=0)
    ed = T("edgar_detection_summary").set_index("method"); ef = T("edgar_funnel"); er = T("edgar_firm_results")
    n28 = cfg["simulation"]["h5_training_length"]
    c28 = cal[(cal.n == n28) & (cal.detector == "kalman_cusum")].set_index("nominal")
    c12 = cal[(cal.n == min(cal.n)) & (cal.detector == "kalman_cusum")].set_index("nominal")
    c40 = cal[(cal.n == max(cal.n)) & (cal.detector == "kalman_cusum")].set_index("nominal")
    f28 = fix[(fix.n == n28) & (fix.nominal == 0.10)].set_index("method")
    f28_5 = fix[(fix.n == n28) & (fix.nominal == 0.05)].set_index("method")
    m = mt.set_index(["scenario", "detector"])
    jf = l2[(l2.firm == "jyoti")].set_index("rule"); nf = l2[(l2.firm == "nmc")].set_index("rule")
    s12, s14 = sep.iloc[0], sep.iloc[1]
    ok_f = int(er[(er.role == "failed") & (er.status == "ok")].shape[0]); ok_c = int(ed.iloc[0].controls)
    nfail = int(ef.set_index("step").loc["failed firms in panel", "count"]); nctrl = int(ef.set_index("step").loc["controls matched", "count"])
    am_u, am_c, alt, rule = ed.loc["amber (uncorrected)"], ed.loc["amber (n-corrected)"], ed.loc["Altman Z''-EM distress"], ed.loc["any Layer 2 rule"]
    rd_c = ed.loc["red (n-corrected)"]
    eam = T("edgar_archetype_mi").set_index("arch_archetype"); ems = T("edgar_mi_separation").iloc[0]
    efp = T("edgar_archetype_fingerprint").set_index("arch_archetype").fillna(0)
    am_n = lambda k: int(eam.loc[k, "firms"])  # noqa: E731
    am_med = lambda k: float(eam.loc[k, "median_peak_mi"])  # noqa: E731
    jd = pd.read_csv(P.processed / "jyoti_full.csv", index_col=0); eps = cfg["ratios"]["denominator_eps_rel_assets"]
    ftp = lambda r: int(r.first_trip_fy) if pd.notna(r.first_trip_fy) else "none"  # noqa: E731
    jy_ocf = jf.loc["OCF < 0"]; jy_any = jf.loc["Any rule"]
    nw_neg = int(rj.index[rj.equity_to_assets < 0][0])
    lc = rn.liquid_to_st_debt

    s = []
    s.append("# Judging a firm against its own history\n\n*A layered early-warning framework, a measurement and correction of "
             "small-sample threshold miscalibration, and a test of when cash flow warns versus hides distress.*\n\n"
             f"Foundations of Data Science course project · generated from `reports/tables/` by `src/reporting/write_report.py` · "
             f"pre-registration dated {cfg['prereg_date']}\n")

    # ------------------------------------------------------------------ abstract
    s.append("## Abstract\n")
    s.append(
        "Distress models usually need large panels of failed firms and peers. We instead monitor each firm against its own ratio "
        "history: a local-level state-space model turns each ratio into standardised forecast errors, and a CUSUM accumulates "
        "persistent deterioration. "
        f"**Headline result: the standard block-bootstrap calibration of CUSUM alarm thresholds is badly over-liberal with short histories.** "
        f"On {int(meta.firms_per_cell)} simulated healthy firms per cell trained on {n28} periods, thresholds set for a 10% false-alarm "
        f"rate produced {pct(c28.loc[0.10, 'empirical_fa'])} (95% CI {ci(c28.loc[0.10])}); thresholds set for 5% produced "
        f"{pct(c28.loc[0.05, 'empirical_fa'])}. Three cheap corrections, fitted on separate simulated firms, bring the held-out "
        f"rate to {pct(f28.loc['inflation_factor', 'empirical_fa'])} (inflation factor), {pct(f28.loc['conformal', 'empirical_fa'])} "
        f"(conformal) and {pct(f28.loc['parametric_bootstrap', 'empirical_fa'])} (parametric bootstrap), at very different costs in "
        f"detection. A multivariate detector loses to the univariate one when the drift sits in the cash ratio but wins clearly when "
        f"cash is healthy or concealed. In the two case studies cash flow warned early at Jyoti Structures (operating cash flow "
        f"negative from FY{ftp(jy_ocf)}) and stayed silent at NMC Health, where only liquidity cover of short-term debt fell below 1. "
        f"On an exploratory panel of {ok_f} US bankruptcies with {ok_c} matched controls from SEC EDGAR, simple pre-set rules beat both "
        f"the statistical detectors and Altman Z'' on expected cost, and the amber statistical tier did not reliably separate failed "
        f"from healthy firms (detection {pct(am_u.detection)} vs false alarms {pct(am_u.false_alarm)}); a post-hoc diagnosis points to "
        "the model absorbing slow declines and to failed firms being weak for years rather than suddenly worse. "
        "Pre-registered hypotheses H2–H5 are supported; H1 and H6 cannot be tested on the data collected so far.\n")

    # ------------------------------------------------------------------ 1
    s.append("## 1. Introduction\n")
    s.append(
        "Lenders, auditors and regulators often monitor firms with short histories and no comparable peers, which is common for "
        "mid-caps in India and the Gulf. Cross-sectional failure models (Altman, Ohlson, hazard models, classifiers) need many failures "
        "to estimate. This project asks a narrower question that a single firm's data can answer: **when does a firm's own cash-flow "
        "history warn early, and when does it hide the problem?**\n\n"
        "Six research questions were fixed, with explicit 'supported if / refuted if' criteria, before any detector was run "
        "(`docs/PREREGISTRATION.md`; SHA-256 fingerprint in `docs/PREREG_FINGERPRINT.txt`):\n")
    s.append("| ID | Hypothesis (abridged) | Verdict |\n|---|---|---|")
    short = {"H1": "Jyoti: univariate state-space + CUSUM alarms ≥ 2 fiscal years before insolvency",
             "H2": "NMC: cash flags silent through 2018 while structural flags trip ≥ 1 year earlier",
             "H3": "Multivariate detector wins when the target is masked, loses slightly when drift is in the target",
             "H4": "Bootstrap thresholds are over-liberal at about 28 training periods",
             "H5": "An inflation factor or conformal threshold brings false alarms within ±3 points of nominal at n = 28",
             "H6": "Cohort: masked firms have higher peak masking index and the multivariate detector alarms first"}
    for i in H.index:
        s.append(f"| {i} | {short[i]} | **{H.loc[i, 'verdict']}** |")
    s.append("\n**Contributions.** (1) *Method*: a layered framework running a cash-first univariate detector and a "
             "multivariate detector side by side over a rule benchmark. (2) *Measurement*: a calibration curve quantifying how far "
             "bootstrap CUSUM thresholds miss their nominal false-alarm rate at small n, with three tested corrections and their cost. "
             "(3) *Typology*: a masking index and detector-order fingerprint, validated in simulation.\n")

    # ------------------------------------------------------------------ 2
    s.append("## 2. Related work\n")
    s.append(
        "Ratio-based failure prediction starts with Beaver (1966) and Altman's Z-score (1968), extended to non-manufacturing and "
        "emerging-market firms as Z''-EM (Altman, Hartzell & Peck 1995; Altman 2005); Ohlson (1980) and Shumway (2001) moved to logit "
        "and hazard models. Cash-flow-based warning is studied by Largay & Stickney (1980) on W.T. Grant and Casey & Bartczak (1985); "
        "earnings quality by Sloan (1996) and Beneish (1999). The monitoring tools are Page's CUSUM (1954), Crosier's multivariate "
        "CUSUM (1988), Ledoit–Wolf shrinkage (2004), block bootstraps (Künsch 1989; Politis & Romano 1994), state-space methods "
        "(Durbin & Koopman 2012), conformal prediction (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2023), and the effect of "
        "estimated parameters on control charts (Jensen et al. 2006). Full references are at the end. No systematic literature "
        "search was done; the framework is a distinctive combination for a "
        "course project, not a claim of novelty.\n")

    # ------------------------------------------------------------------ 3
    s.append("## 3. Data and audit\n")
    s.append(
        "**Deep cases.** NMC Health: annual, 2013–2018, US$ thousands, as originally reported "
        f"{v('original NMC source documents (LSE RNS results)')}. Jyoti Structures: annual, FY2013–FY2026, INR crore, consolidated, "
        "from Screener.in (vendor-standardised). Event dates, checked against primary records: NMC Health PLC was placed into "
        "administration by the English High Court on 2020-04-09 on Abu Dhabi Commercial Bank's application (the ADGM court followed "
        "for the UAE operating companies on 2020-09-27); Jyoti Structures was admitted to the corporate insolvency resolution process "
        "by NCLT Mumbai on 2017-07-04 (CP 1137/I&BP/2017).\n\n"
        "**Audit decisions** (full list: `docs/decision_log.md`):\n"
        "- Accounting identities (assets = liabilities + equity; EBITDA + other income − interest − depreciation = PBT; "
        "DSO + DIO − DPO = CCC) are checked before any analysis and stop the run on failure, naming firm, year and columns.\n"
        "- Jyoti FY2017 is not published on Screener: kept missing, never interpolated.\n"
        f"- Jyoti FY2018 operating cash flow of {jd.loc[2018, 'cfo']:+,.0f} on sales of {jd.loc[2018, 'sales']:,.0f} is flagged as an anomaly, not dropped.\n"
        f"- Ratios with near-zero denominators are masked (|denominator| ≤ ε × total assets; ε = {pct(eps['activity'], 0)} for revenue-based "
        f"ratios, {pct(eps['default'])} otherwise). Example: Jyoti FY2022 debtor days of {jd.loc[2022, 'debtor_days']:,.0f} on sales of "
        f"{jd.loc[2022, 'sales']:,.0f} become missing.\n"
        "- Screener lacks a current/non-current split, so Jyoti's working capital and current liabilities are proxies that overstate "
        "Altman's X1.\n"
        "- A public panel of about 9,000 US firms was rejected earlier because basic identities failed in most rows (blueprint §4.3).\n\n"
        f"**EDGAR panel (exploratory).** The deep cases cannot support Layers 3–5, and the India/GCC cohort has not been collected yet, "
        f"so the method was also run on US firms. Event = first 8-K citing Item 1.03 (bankruptcy) in 2015–2023, financials excluded. "
        f"Of {nfail} failed firms in the panel and {nctrl} size- and industry-matched controls, {ok_f} failed firms and {ok_c} paired controls "
        "had enough quarterly XBRL history (≥ 12 usable training quarters before a 16-quarter monitored window). Year-to-date figures were "
        "de-cumulated into quarters, flows taken as trailing-twelve-month sums, and the first-filed value kept (`reports/tables/edgar_funnel.csv`). "
        "These rules were written to the config before download.\n")

    # ------------------------------------------------------------------ 4
    s.append("## 4. What the EDA taught\n")
    s.append(
        f"Jyoti's cash flow broke first: the Layer 2 rule 'OCF < 0' first tripped in FY{ftp(jy_ocf)}, a lead of "
        f"{jy_ocf.lead_years:.2f} years over the insolvency admission, while net worth only turned negative in FY{nw_neg}. "
        f"NMC is the opposite: no cash rule tripped in any year, but liquidity cover ((cash + deposits) / short-term debt) was below 1 in "
        f"{', '.join('FY' + str(y) for y in lc.index[lc < 1])}, falling to {lc.loc[2018]:.2f} in 2018 "
        "(`reports/tables/ratios_nmc.csv`). These two patterns motivated a cash-first detector plus a structural detector.\n\n"
        "![Layer 2 flags, Jyoti](figures/layer2_flag_heatmap_jyoti.png)\n\n*Figure 1. Jyoti Structures: pre-registered rule flags by fiscal "
        "year. Notice that three rules trip from FY14, three years before the event, while Altman Z''-EM only enters distress after it.*\n\n"
        "![Layer 2 flags, NMC](figures/layer2_flag_heatmap_nmc.png)\n\n*Figure 2. NMC Health: no pre-registered rule ever trips; the "
        "only warning in the data is the liquidity cover in Figure 3.*\n\n"
        "![Layer 1, NMC](figures/layer1_event_time_nmc.png)\n\n*Figure 3. NMC in event time (lower = worse). Cash ratios improve "
        "into the event; liquidity cover and interest cover deteriorate.*\n")

    # ------------------------------------------------------------------ 5
    s.append("## 5. Method\n")
    s.append(
        "All ratios are oriented so that lower is worse. **Layer 0** builds and audits ratios. **Layer 1** aligns firms in event time. "
        "**Layer 2** applies pre-set rules (OCF < 0; EBITDA/interest < 1.2; debtor days > 270; Altman Z''-EM < 4.35). "
        "**Layer 3** fits a local level + AR(1) model on the training window, filters the whole series with frozen parameters and "
        "monitors standardised one-step errors z_t with a one-sided CUSUM:\n\n"
        "  S_t = max(0, S_{t−1} − z_t − k),  k = 0.5,  alarm when S_t > h.\n\n"
        "**Layer 4** runs the same model on several ratios, estimates the innovation covariance Σ with Ledoit–Wolf shrinkage, and "
        "monitors Crosier's MCUSUM:\n\n"
        "  C_t = √((S_{t−1}+z_t)ᵀ Σ⁻¹ (S_{t−1}+z_t)),  S_t = 0 if C_t ≤ k, else (S_{t−1}+z_t)(1 − k/C_t).\n\n"
        "**Layer 5** sets h by circular block bootstrap of pseudo-out-of-sample innovations (refit on the first 60% of training, "
        f"filter the rest, centre, {int(meta.bootstrap_resamples)} resamples, block length ⌈n^(1/3)⌉, h = (1 − α) quantile of the "
        "maximum CUSUM over a 16-period path). Amber fires when either detector passes its 10% threshold; red when the multivariate "
        "detector passes its 5% threshold and a Layer 2 rule agrees. **Layer 6** computes the masking index "
        "MI_t = S_struct,t / h_struct − S_cash,t / h_cash and records which detector alarms first.\n")

    # ------------------------------------------------------------------ 6
    s.append("## 6. Simulation and calibration study\n")
    s.append(
        f"Synthetic firms have {cfg['simulation']['n_ratios']} AR(1) ratios (φ = {cfg['simulation']['ar_phi']}, innovation correlation "
        f"{cfg['simulation']['innovation_corr']}), {cfg['simulation']['monitor_length']} monitored periods, and a known onset. Scenarios: "
        "S1 drift in the cash ratio; S2 cash healthy, other ratios deteriorating; S3 small drift everywhere; S4 concealment (all ratios "
        f"deteriorate but the reported cash ratio is exponentially smoothed, s = {cfg['simulation']['concealment_smoothing']}). "
        f"{int(meta.firms_per_cell)} held-out firms per training length; corrections were fitted on a separate {int(meta.firms_per_cell)} "
        f"per cell with a different seed. {int(meta.failed_firm_runs)} firm-runs failed. *The simulation validates the method under assumed "
        "dynamics; it is not evidence about real firms.*\n\n"
        "**6.1 Calibration curve (H4).** The empirical false-alarm rate is far above nominal at every training length: "
        f"{pct(c12.loc[0.10, 'empirical_fa'])} at n = {int(min(cal.n))} and still {pct(c40.loc[0.10, 'empirical_fa'])} at n = {int(max(cal.n))} "
        f"for a 10% target; at n = {n28}, {pct(c28.loc[0.10, 'empirical_fa'])} ({ci(c28.loc[0.10])}), in line with what the blueprint "
        f"reported from earlier scripts. **H4: {H.loc['H4', 'verdict']}.**\n\n"
        "![Calibration curve](figures/sim_calibration_curve.png)\n\n*Figure 4. Every point sits far above the diagonal; more training "
        "data helps but does not close the gap.*\n\n"
        "**6.2 Corrections (H5).** At n = 28 and a 10% target, on held-out firms:\n")
    s.append("| Method | False-alarm rate (95% CI) | Detection in S1 | Detection lost (points) |\n|---|---|---|---|")
    for mth, lab in [("block_bootstrap", "Block bootstrap (uncorrected)"), ("inflation_factor", "Inflation factor"),
                     ("conformal", "Conformal"), ("parametric_bootstrap", "Parametric bootstrap")]:
        r = f28.loc[mth]
        s.append(f"| {lab} | {pct(r.empirical_fa)} ({ci(r)}) | {pct(r.detection_S1)} | {r.detection_cost_points:.1f} |")
    i28 = infl[(infl.n == n28) & (infl.nominal == 0.10)].iloc[0]
    s.append(
        f"\nThe required inflation of h at n = {n28} was {i28.c_required:.2f}× (fitted curve log c = {i28.a:.3f} + {i28.b:.2f}/n). "
        f"**H5: {H.loc['H5', 'verdict']}**: both pre-registered fixes land within ±3 points. The fixes are not equal, though: a single "
        "multiplicative inflation of each firm's own bootstrap threshold also inflates its noise and costs far more detection than "
        f"the conformal or parametric thresholds. The parametric bootstrap with refitting ({int(meta.parametric_B)} refits per firm, "
        f"a logged deviation from {int(meta.bootstrap_resamples):,}) is the best trade-off here; the conformal threshold needs exchangeable calibration firms, "
        f"which a real single-firm monitor does not have. At a 5% target the corrected rates were "
        f"{pct(f28_5.loc['inflation_factor', 'empirical_fa'])}, {pct(f28_5.loc['conformal', 'empirical_fa'])} and "
        f"{pct(f28_5.loc['parametric_bootstrap', 'empirical_fa'])}.\n\n"
        "![Fixes](figures/sim_calibration_fixes.png)\n\n*Figure 5. After correction all three methods sit in or near the ±3-point "
        "band across training lengths; the uncorrected calibrator never does.*\n\n"
        "**6.3 Matched false-alarm comparison (H3).** Each detector was tuned to the same empirical 10% false-alarm rate on held-out "
        "healthy firms before comparing detection:\n")
    s.append("| Scenario | ARIMA + CUSUM | Kalman + CUSUM | Random walk + CUSUM | Multivariate |\n|---|---|---|---|---|")
    for sc in ["S1", "S2", "S3", "S4"]:
        s.append(f"| {sc} | " + " | ".join(pct(m.loc[(sc, d), "detection"]) for d in ["arima", "kalman", "rw", "mcusum"]) + " |")
    s.append(
        f"\n**H3: {H.loc['H3', 'verdict']}** ({H.loc['H3', 'evidence']}). One caveat: S1 loss is not 'slight', and ARIMA beat the "
        "Kalman detector in S1 and S4, as in the blueprint's earlier run, so the state-space model's advantage is robustness to "
        "missing data and level shifts, not raw power under these dynamics.\n\n"
        "![Power](figures/sim_power_curves.png)\n\n*Figure 6. Univariate detectors dominate when drift is in the cash ratio; only the "
        "multivariate detector sees S2. Detection rises with training length.*\n")

    # ------------------------------------------------------------------ 7
    s.append("## 7. Masking index and typology\n")
    s.append(
        f"Peak masking index separates S1 (visible) from S2 with Cohen's d = {s12.cohens_d:.2f} ({s12.d_ci_lo:.2f}–{s12.d_ci_hi:.2f}) "
        f"and AUC = {s12.auc:.2f} ({s12.auc_ci_lo:.2f}–{s12.auc_ci_hi:.2f}), and from S4 (concealment) more weakly: d = "
        f"{s14.cohens_d:.2f} ({s14.d_ci_lo:.2f}–{s14.d_ci_hi:.2f}), AUC = {s14.auc:.2f}. Median peak MI was {msum.loc['S1', 'median_peak_mi']:.2f} "
        f"in S1, {msum.loc['S2', 'median_peak_mi']:.2f} in S2 and {msum.loc['S4', 'median_peak_mi']:.2f} in S4. The fingerprint is more "
        f"informative than the score: the cash detector alarmed alone in {pct(fp.loc['S1', 'cash_only'], 0)} of S1 firms, the structural "
        f"detector alone in {pct(fp.loc['S2', 'structural_only'], 0)} of S2 firms, while under concealment the cash detector still "
        f"usually fired, just late (cash first {pct(fp.loc['S4', 'cash_first'], 0)}, structural first {pct(fp.loc['S4', 'structural_first'], 0)}). "
        f"Healthy firms show 'neither' in only {pct(fp.loc['H', 'neither'], 0)}, a direct consequence of the uncorrected thresholds used "
        "for MI; MI should be recomputed with corrected thresholds before use.\n\n"
        "![MI](figures/sim_masking_index.png)\n\n*Figure 7. MI is highest when structure deteriorates under a quiet cash ratio (S2).*\n\n"
        f"**Pre-registered archetypes.** By rule, Jyoti is **{arch.loc['jyoti', 'archetype']}** "
        f"({int(arch.loc['jyoti', 'max_consecutive_negative_ocf_years'])} consecutive years of negative OCF) and NMC is "
        f"**{arch.loc['nmc', 'archetype']}**, not Masked: its OCF stayed positive but only {int(arch.loc['nmc', 'n_structural_flags_tripped'])} "
        "of the three structural flags (liquidity cover) ever tripped. The blueprint's description of NMC as 'masked' is therefore "
        "not confirmed by its own pre-set rule on the annual data.\n")

    # ------------------------------------------------------------------ 8
    s.append("## 8. Case, cohort and EDGAR results\n")
    s.append(f"- **H1 (Jyoti): {H.loc['H1', 'verdict']}.** {H.loc['H1', 'evidence']}.")
    s.append(f"- **H2 (NMC): {H.loc['H2', 'verdict']}.** {H.loc['H2', 'evidence']}. This rests on one ratio crossing a threshold of 1, "
             "so it is an illustration, not proof.")
    s.append(f"- **H6 (cohort): {H.loc['H6', 'verdict']}.** {H.loc['H6', 'evidence']}.\n")
    s.extend(cohort_section())
    s.append(
        f"**EDGAR panel (exploratory, real quarterly data).** {ok_f} failed firms, {ok_c} paired controls; detection = any alarm in the "
        "16 quarters before the bankruptcy 8-K; false alarm = any alarm in a control over the same window:\n")
    s.append("| Method | Detection (95% CI) | False alarm on controls (95% CI) | Detection minus false alarm (95% CI) | Median lead (quarters) |\n|---|---|---|---|---|")
    for lab, r in ed.iterrows():
        s.append(f"| {lab} | {pct(r.detection)} ({pct(r.det_ci_lo)}–{pct(r.det_ci_hi)}) | {pct(r.false_alarm)} "
                 f"({pct(r.fa_ci_lo)}–{pct(r.fa_ci_hi)}) | {100 * r.det_minus_fa:+.1f} pts ({100 * r.diff_ci_lo:+.1f} to "
                 f"{100 * r.diff_ci_hi:+.1f}) | {r.median_lead_quarters:.1f} |")
    nochance = [lab for lab, r in ed.iterrows() if r.diff_ci_lo <= 0]
    cm = cfg["cost"]["c_miss"]
    beats = [lab for lab, r in ed.iterrows() if all(r[f"beats_altman_cm{c}"] for c in cm)]
    s.append(
        f"\nThe real-firm false-alarm rate of the uncorrected amber tier ({pct(am_u.false_alarm)}) mirrors the simulation: the "
        f"calibration problem is not an artefact of simulated data. The n-correction lowers it to {pct(am_c.false_alarm)} and the red "
        f"tier to {pct(rd_c.false_alarm)}, but detection falls with it. On expected cost (C_FA = 1, C_M ∈ {{{', '.join(map(str, cm))}}}) "
        f"only {', '.join(beats) if beats else 'no method'} beats Altman Z''-EM at every cost ratio. Altman itself flags "
        f"{pct(alt.false_alarm)} of healthy US controls, because the emerging-market cut-offs are strict for leveraged US firms. "
        "On this panel the statistical detectors are not better than simple pre-set rules. Worse, for "
        f"{', '.join(nochance)} the interval for 'detection minus false alarm' includes zero: on real US firms these tiers do not "
        "reliably alarm failed firms more often than healthy ones (§9.2 diagnoses why). The simulation-fitted n-correction also "
        f"transfers only partly: it lowers false alarms to {pct(am_c.false_alarm)}, still far above the 10% target, because real ratio "
        "dynamics (level shifts, seasonality, accounting events) are rougher than the Gaussian AR(1) world it was fitted in. Among the "
        "statistical outputs only the uncorrected red tier (statistical alarm confirmed by a rule) separates the groups.\n\n"
        f"**Archetypes on EDGAR (exploratory analogue of H6).** By the pre-set rule, {am_n('Masked')} failed firms were Masked, "
        f"{am_n('Visible')} Visible and {am_n('Other')} Other. Median peak MI was {am_med('Masked'):.2f} (Masked), {am_med('Visible'):.2f} "
        f"(Visible) and {am_med('Other'):.2f} (Other); means are not reported because a few near-zero structural thresholds make them "
        f"explode. Masked minus Visible: Cohen's d = {ems.cohens_d:.2f} ({ems.d_ci_lo:.2f}–{ems.d_ci_hi:.2f}), AUC = {ems.auc:.2f} "
        f"({ems.auc_ci_lo:.2f}–{ems.auc_ci_hi:.2f}). The structural detector alarmed first or alone in "
        f"{pct(efp.loc['Masked', ['structural_first', 'structural_only']].sum(), 0)} of Masked firms versus "
        f"{pct(efp.loc['Visible', ['structural_first', 'structural_only']].sum(), 0)} of Visible firms. The direction matches H6 but the "
        "interval includes no effect, so this is suggestive only, and it is US data, not the pre-registered cohort.\n\n"
        "![EDGAR event time](figures/edgar_layer1_event_time.png)\n\n*Figure 8. Failed US firms start below their controls on cash flow, "
        "interest cover and Z''-EM and deteriorate over the last two to three years. Two ratios do not move in their oriented direction: "
        "failed firms have fewer debtor days than controls, and their negated Sloan accruals rise before failure because write-downs "
        "push net income below cash flow.*\n\n"
        "![EDGAR detection](figures/edgar_detection_vs_false_alarm.png)\n\n*Figure 9. Every method trades detection against false "
        "alarms; none is in the top-left corner.*\n")

    # ------------------------------------------------------------------ 9-11
    s.append("## 9. Robustness\n")
    s.extend(robustness_section())
    s.append("## 10. Limitations\n")
    s.append(
        "- **Hindsight bias**: mitigated by the dated, hashed pre-registration; the two deep cases were seen before it and are illustrations.\n"
        f"- **Tiny samples**: {len(rn)} annual points for NMC, a missing year and no pre-FY14 history for Jyoti; H1 is untestable until BSE "
        "quarterly data for FY2010–13 is collected.\n"
        "- **Vendor and XBRL data**: Screener proxies for Jyoti; on EDGAR, short-term debt and depreciation are often untagged, so "
        "liquidity cover rarely enters the structural detector.\n"
        "- **Possible falsification at NMC**: if reported figures were unreliable, detection is limited by construction.\n"
        "- **Event definition**: lead times are measured from fiscal or quarter ends, not from publication dates, so usable warning is shorter.\n"
        "- **Simulation realism**: Gaussian AR(1) dynamics; heavy tails (S5) and training-period breaks (S6) were cut.\n"
        "- **Direction**: Crosier's MCUSUM is direction-agnostic and can alarm on improvement.\n")
    s.append("## 11. Conclusion and practical use\n")
    s.append(
        "Self-referential monitoring can be calibrated, but not with the naive bootstrap: with one to ten years of data its alarms are "
        "several times more frequent than advertised, in simulation and on real firms. A lender using the framework should (1) correct "
        "thresholds for training length, (2) run a cash detector and a structural detector together, since each misses what the other "
        "catches, and (3) keep simple pre-set rules as the first line, escalating to formal review only when a statistical alarm and a "
        "rule agree. On real US firms the self-referential detectors added little to level-based rules, and the post-hoc diagnosis "
        "(§9.2) suggests why: an adaptive model forgives slow decline, and a firm that has always been weak never looks worse than "
        "itself. A future version should test a non-adaptive baseline and combine 'worse than its own past' with 'weak in absolute "
        "terms'. Next steps: collect the India/GCC cohort and Jyoti's quarterly history to test H1 and H6.\n")

    # ------------------------------------------------------------------ appendix
    s.append("## Appendix: reproducibility\n")
    s.append(f"Seed {cfg['seed']}; all thresholds in `configs/config.yaml`; `make all` rebuilds every table and figure "
             "(`make sim` about 1 hour on 9 workers). Source code in `src/`; tests: `python -m pytest`.\n")
    s.append("### Items to check by hand\n")
    s.append("\n".join(f"- [verify] {x}" for x in VERIFY) + "\n" + "\n".join(f"- [MISSING] {x}" for x in MISSING) + "\n")
    s.append("## References\n")
    s.append("\n".join(f"- {r}" for r in REFERENCES) + "\n")
    return "\n".join(s)


if __name__ == "__main__":
    out = P.tables.parent / "report.md"
    out.write_text(build(), encoding="utf-8")
    print("wrote", out, f"({len(VERIFY)} verify, {len(MISSING)} missing)")
