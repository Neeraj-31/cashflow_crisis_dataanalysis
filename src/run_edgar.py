"""EDGAR external-transfer panel: Layers 0-6 on real quarterly data (exploratory; config `edgar`).

    python -m src.run_edgar [--tables] [--figures]

Per failed firm: data truncated at the event quarter (first Item 1.03 8-K); monitored window = the 16
quarters ending at the event quarter; training = all earlier quarters (>= 12 usable cash-ratio values).
Each matched control is monitored over its failed firm's window (pseudo-event); any alarm there is a
real-firm false alarm. Nothing is tuned on this data.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Paths, load_config, rng  # noqa: E402
from src.detectors import layer2_rules  # noqa: E402
from src.detectors.monitor import load_inflation, monitor_firm, usable_training_periods  # noqa: E402
from src.ingest import edgar  # noqa: E402
from src.masking import archetype  # noqa: E402
from src.ratios import IdentityError, check_identities, compute_ratios  # noqa: E402
from src.reporting.style import C, EVENT, STYLE  # noqa: E402

L1_RATIOS = ["ocf_to_current_liabilities", "liquid_to_st_debt", "ebitda_interest_cover", "altman_z2_em", "neg_dso", "neg_sloan_accruals"]


def firm_run(cik, role, event_q, matched_to, cfg, paths, inflation, shift=0, keep=None):
    """shift: move the training/monitoring boundary by `shift` quarters (robustness; 0 = pre-registered).
    keep: optional dict that receives the monitor output ('out'), the rule flags ('F') and the monitored index."""
    f = paths.raw / "edgar" / "facts" / f"CIK{int(cik):010d}.csv.gz"
    base = {"cik": int(cik), "role": role, "event_q": int(event_q), "matched_to": matched_to}
    if not f.exists():
        return {**base, "status": "no facts file"}, None
    q = edgar.quarterly_canonical(pd.read_csv(f))
    gap = cfg["edgar"]["max_gap_to_event_quarters"]
    q = q.loc[:event_q] if not q.empty else q
    if q.empty or q.index.max() < event_q - gap:
        return {**base, "status": f"last filing more than {gap} quarters before event"}, None
    q = q.reindex(range(q.index.min(), event_q + 1))          # quarters after the last filing stay NaN
    try:
        check_identities(q, cfg["identities"], f"CIK{cik}")
    except IdentityError as e:
        return {**base, "status": f"identity failure: {str(e).splitlines()[0][:150]}"}, None
    r = compute_ratios(q, cfg)
    M = cfg["edgar"]["monitor_quarters"]; cash = cfg["masking"]["cash_ratio"]
    start = event_q - M + 1 + shift                          # first monitored quarter
    n_train = int((r.index < start).sum())
    usable = usable_training_periods(r, cash, n_train)
    F = layer2_rules.rule_flags(q, r, cfg)
    arch = archetype.classify(q, r, event_q + 1, cfg, periods_per_year=4)   # all quarters up to and incl. the event quarter
    row = {**base, "status": "ok", "n_train": n_train, "usable_training": usable, **{f"arch_{k}": v for k, v in arch.items()}}
    mon = r.index[r.index >= start]
    alt = F.iloc[:, -1].reindex(mon); rules = (F.iloc[:, :-1].reindex(mon) == 1).any(axis=1)
    row["altman_alarm_q"] = int(alt.index[alt == 1][0]) if (alt == 1).any() else None
    row["rule_alarm_q"] = int(rules.index[rules][0]) if rules.any() else None
    panel = r.reindex(range(event_q - 20, event_q + 1))[L1_RATIOS].assign(tau=lambda d: d.index - event_q)
    if usable < cfg["edgar"]["min_training_quarters"] or n_train < cfg["detectors"]["min_periods_layer3"]:
        row["status"] = f"too little training data ({usable} usable)"
        return row, panel
    try:
        out, summ = monitor_firm(r, F, n_train, cfg, rng(int(cik) % 100000), inflation)
    except Exception as e:  # numerical failure in a fit: report, never hide
        row["status"] = f"model failure: {type(e).__name__}"
        return row, panel
    row.update({k: v for k, v in summ.items()})
    if keep is not None:
        keep.update(out=out, F=F, mon=mon, r=r, n_train=n_train)
    return row, panel


def tables(cfg, paths):
    panel = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv")
    inflation = load_inflation(paths.tables)
    rows, l1 = [], []
    for p in panel.itertuples():
        for cik, role in [(p.cik, "failed"), (p.control_cik, "control")]:
            if pd.isna(cik):
                continue
            row, pan = firm_run(int(cik), role, int(p.event_q), int(p.cik), cfg, paths, inflation)
            row.update(name=p.name if role == "failed" else "", sic=p.sic, event_date=p.file_date)
            rows.append(row)
            if pan is not None:
                l1.append(pan.assign(cik=int(cik), role=role))
    R = pd.DataFrame(rows)
    R.to_csv(paths.tables / "edgar_firm_results.csv", index=False)
    pd.concat(l1).to_csv(paths.tables / "edgar_layer1_panel.csv", index=False)
    summarise(R, cfg, paths)


def _rate(x, rg, B=2000):
    x = np.asarray(x, float); m = len(x)
    if m == 0:
        return np.nan, np.nan, np.nan
    bs = x[rg.integers(0, m, (B, m))].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def summarise(R, cfg, paths):
    rg = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 4242]))
    rg_d = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 4243]))
    ok = R[R.status == "ok"].copy()
    # keep only pairs where both members ran, so detection and false-alarm rates use the same windows
    pairs = ok[ok.role == "failed"].cik
    ctrl_ok = ok[(ok.role == "control") & ok.matched_to.isin(pairs)]
    fail_ok = ok[(ok.role == "failed")]
    funnel = pd.DataFrame([{"step": "failed firms in panel", "count": int((R.role == "failed").sum())},
                           {"step": "controls matched", "count": int((R.role == "control").sum())},
                           *[{"step": f"{role}: {s}", "count": int(n)} for (role, s), n in
                             R.assign(s=R.status.str.split(":").str[0]).groupby(["role", "s"]).size().items()]])
    funnel.to_csv(paths.tables / "edgar_funnel.csv", index=False)
    methods = {"amber (uncorrected)": "first_amber_uncorrected", "red (uncorrected)": "first_red_uncorrected",
               "amber (n-corrected)": "first_amber_corrected", "red (n-corrected)": "first_red_corrected",
               "Altman Z''-EM distress": "altman_alarm_q", "any Layer 2 rule": "rule_alarm_q"}
    if fail_ok.get("inflation_0.10", pd.Series(dtype=float)).isna().all():   # simulation not run: no n-correction
        methods = {k: v for k, v in methods.items() if "corrected)" not in k or "uncorrected" in k}
    rows = []
    for lab, col in methods.items():
        det = fail_ok[col].notna(); fa = ctrl_ok[col].notna()
        lead = (fail_ok.event_q - fail_ok[col]).dropna()
        d, dlo, dhi = _rate(det, rg); f, flo, fhi = _rate(fa, rg)
        row = {"method": lab, "failed_firms": len(fail_ok), "controls": len(ctrl_ok), "detection": d, "det_ci_lo": dlo,
               "det_ci_hi": dhi, "false_alarm": f, "fa_ci_lo": flo, "fa_ci_hi": fhi,
               "median_lead_quarters": float(lead.median()) if len(lead) else np.nan,
               "share_lead_ge_8q": float((lead >= 8).mean()) if len(lead) else np.nan}
        for cm in cfg["cost"]["c_miss"]:
            row[f"expected_cost_cm{cm}"] = cfg["cost"]["c_fa"] * f + cm * (1 - d)
        # discrimination: detection minus false alarms (independent bootstrap of the two groups, separate stream)
        a_, b_ = det.to_numpy(float), fa.to_numpy(float)
        diff = [a_[rg_d.integers(0, len(a_), len(a_))].mean() - b_[rg_d.integers(0, len(b_), len(b_))].mean() for _ in range(2000)]
        row.update(det_minus_fa=d - f, diff_ci_lo=float(np.quantile(diff, .025)), diff_ci_hi=float(np.quantile(diff, .975)))
        rows.append(row)
    S = pd.DataFrame(rows)
    alt = S.set_index("method").loc["Altman Z''-EM distress"]
    for cm in cfg["cost"]["c_miss"]:
        S[f"beats_altman_cm{cm}"] = S[f"expected_cost_cm{cm}"] < alt[f"expected_cost_cm{cm}"]
    S.to_csv(paths.tables / "edgar_detection_summary.csv", index=False)
    red_np = {t: float(fail_ok[f"red_without_prior_amber_{t}"].mean()) for t in ("uncorrected", "corrected")}
    pd.DataFrame([{"tier_check": f"red without prior amber ({t}), failed firms", "share": v} for t, v in red_np.items()]
                 ).to_csv(paths.tables / "edgar_tier_check.csv", index=False)
    # archetypes, MI and fingerprints (H6 analogue, exploratory)
    fa_ = fail_ok.copy()
    arch = fa_.groupby("arch_archetype").agg(firms=("cik", "size"), mean_peak_mi=("peak_masking_index", "mean"),
                                             median_peak_mi=("peak_masking_index", "median")).reset_index()
    arch.to_csv(paths.tables / "edgar_archetype_mi.csv", index=False)
    fp = pd.crosstab(fa_.arch_archetype, fa_.fingerprint, normalize="index").reset_index()
    fp.to_csv(paths.tables / "edgar_archetype_fingerprint.csv", index=False)
    m, v = fa_[fa_.arch_archetype == "Masked"].peak_masking_index.dropna().to_numpy(), fa_[fa_.arch_archetype == "Visible"].peak_masking_index.dropna().to_numpy()
    if len(m) >= 3 and len(v) >= 3:
        def eff(a, b):
            sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            ranks = pd.Series(np.r_[b, a]).rank().to_numpy()
            return (a.mean() - b.mean()) / sp, (ranks[len(b):].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        d, auc = eff(m, v)
        bs = np.array([eff(m[rg.integers(0, len(m), len(m))], v[rg.integers(0, len(v), len(v))]) for _ in range(2000)])
        pd.DataFrame([{"comparison": "peak MI, Masked minus Visible", "masked_n": len(m), "visible_n": len(v), "cohens_d": d,
                       "d_ci_lo": np.nanquantile(bs[:, 0], .025), "d_ci_hi": np.nanquantile(bs[:, 0], .975), "auc": auc,
                       "auc_ci_lo": np.nanquantile(bs[:, 1], .025), "auc_ci_hi": np.nanquantile(bs[:, 1], .975)}]
                     ).to_csv(paths.tables / "edgar_mi_separation.csv", index=False)
    print(funnel.to_string(index=False)); print(S.round(3).to_string(index=False)); print(arch.to_string(index=False))


def figures(cfg, paths):
    L = pd.read_csv(paths.tables / "edgar_layer1_panel.csv")
    labels = {"ocf_to_current_liabilities": "OCF (TTM) / current liabilities", "liquid_to_st_debt": "(cash + ST investments) / ST debt",
              "ebitda_interest_cover": "EBITDA / interest (TTM)", "altman_z2_em": "Altman Z''-EM", "neg_dso": "debtor days (negated)",
              "neg_sloan_accruals": "Sloan accruals (negated)"}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8)); axes = axes.ravel()
        for ax, c in zip(axes, L1_RATIOS):
            for role, col in [("control", C["teal"]), ("failed", C["coral"])]:
                g = L[L.role == role].groupby("tau")[c]
                q1, med, q3, n = g.quantile(.25), g.median(), g.quantile(.75), g.count()
                ax.fill_between(med.index, q1, q3, color=col, alpha=.18, lw=0)
                ax.plot(med.index, med, "-", color=col, lw=2, label=f"{role} median (IQR band), n<={int(n.max())}")
            ax.axvline(0, **EVENT); ax.set_title(labels[c], fontsize=10); ax.set_xlabel("quarters relative to event (0 = Item 1.03 8-K)")
        axes[0].legend(fontsize=8)
        fig.suptitle("Layer 1 on the EDGAR panel: failed firms vs matched controls in event time (lower = worse)", fontsize=12.5, weight="bold", x=.01, ha="left")
        fig.tight_layout(); fig.savefig(paths.figures / "edgar_layer1_event_time.png"); plt.close(fig)
        S = pd.read_csv(paths.tables / "edgar_detection_summary.csv")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for r in S.itertuples():
            ax.errorbar(r.false_alarm * 100, r.detection * 100, xerr=[[100 * (r.false_alarm - r.fa_ci_lo)], [100 * (r.fa_ci_hi - r.false_alarm)]],
                        yerr=[[100 * (r.detection - r.det_ci_lo)], [100 * (r.det_ci_hi - r.detection)]], fmt="o", capsize=3, ms=7, label=r.method)
        ax.set_xlabel("false-alarm rate on matched healthy controls (%)"); ax.set_ylabel("detection within 16 quarters before event (%)")
        ax.set_title("Detection vs false alarms on real US firms (95% bootstrap CIs)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(paths.figures / "edgar_detection_vs_false_alarm.png"); plt.close(fig)
    print("saved EDGAR figures")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", action="store_true"); ap.add_argument("--figures", action="store_true")
    ap.add_argument("--summary", action="store_true", help="re-summarise edgar_firm_results.csv without re-running firms")
    a = ap.parse_args(argv); cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    if a.summary:
        summarise(pd.read_csv(paths.tables / "edgar_firm_results.csv"), cfg, paths); return
    both = not (a.tables or a.figures)
    if a.tables or both: tables(cfg, paths)
    if a.figures or both: figures(cfg, paths)


if __name__ == "__main__":
    main()
