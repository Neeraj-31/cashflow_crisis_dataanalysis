"""Sanity and common-sense checks across the whole project -> reports/tables/sanity_checks.csv.

    python -m src.sanity [--skip-tests]

Each row: area, check, status (PASS / FAIL / WARN / INFO), detail. WARN = not wrong, but a reader should
know; INFO = a recorded fact. Plausibility bands come from config `sanity`; none is used by any analysis.
"""
from __future__ import annotations

import argparse
import filecmp
import gzip
import hashlib
import html
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import Paths, load_config
from src.ingest import canonical
from src.ratios import check_identities

ROOT = Path(__file__).resolve().parents[1]
PREREG_BLOCKS = ["seed", "cohort", "archetype", "altman_z2_em", "layer2", "detectors", "simulation", "masking",
                 "firms", "ratios", "identities", "cost", "negative_controls"]
rows: list[dict] = []


def add(area, check, ok, detail="", level=None):
    status = level or ("PASS" if ok else "FAIL")
    rows.append(dict(area=area, check=check, status=status, detail=str(detail)))


def T(paths, name):
    return pd.read_csv(paths.tables / f"{name}.csv")


# ------------------------------------------------------------------ provenance
def provenance(cfg):
    fp = (ROOT / "docs" / "PREREG_FINGERPRINT.txt").read_text(encoding="utf-8")
    want = dict((m.group(2), m.group(1)) for m in re.finditer(r"^([0-9a-f]{64}) \*(\S+)", fp, re.M))
    for rel, snap in [("configs/config.yaml", "config.yaml"), ("docs/PREREGISTRATION.md", "PREREGISTRATION.md")]:
        h = hashlib.sha256((ROOT / "docs" / "prereg_snapshot" / snap).read_bytes()).hexdigest()
        add("provenance", f"snapshot of {rel} matches the recorded pre-registration hash", h == want.get(rel), h[:16])
    snap = yaml.safe_load((ROOT / "docs" / "prereg_snapshot" / "config.yaml").read_text(encoding="utf-8"))
    changed = [k for k in PREREG_BLOCKS if snap.get(k) != cfg.get(k)]
    add("provenance", "pre-registered config blocks unchanged since the fingerprint", not changed,
        "changed: " + ", ".join(changed) if changed else "unchanged: " + ", ".join(PREREG_BLOCKS))
    added = [k for k in cfg if k not in snap]
    add("provenance", "config blocks added after the fingerprint (must be logged as deviations)", True,
        ", ".join(added) or "none", level="INFO")
    robust_new = [k for k in cfg.get("robustness", {}) if k not in snap.get("robustness", {})]
    add("provenance", "robustness keys added after the fingerprint", True, ", ".join(robust_new) or "none", level="INFO")
    add("provenance", "global seed is 20261121", cfg["seed"] == 20261121, cfg["seed"])
    bad = []
    for f in (ROOT / "src").rglob("*.py"):
        txt = f.read_text(encoding="utf-8")
        if re.search(r"np\.random\.seed\(|default_rng\(\s*\)|random\.seed\(", txt):
            bad.append(f.relative_to(ROOT).as_posix())
        if re.search(r"\bhash\(\w+\)\s*%", txt):
            bad.append(f.relative_to(ROOT).as_posix() + " (salted hash() as seed)")
    add("provenance", "no unseeded or salted random number generators in src/", not bad, "; ".join(bad) or "none found")


# ------------------------------------------------------------------ data
def data(cfg, paths):
    sb = cfg["sanity"]
    for firm, loader in [("nmc", canonical.nmc), ("jyoti", None)]:
        canon = canonical.LOADERS[firm](paths, cfg["firms"][firm]["fy_end_month"]) if loader is None else loader(paths)
        try:
            check_identities(canon, cfg["identities"], firm); ok, det = True, f"{len(canon)} firm-years"
        except Exception as e:  # noqa: BLE001
            ok, det = False, str(e).splitlines()[0]
        add("data", f"{firm}: accounting identities hold", ok, det)
        lo, hi = sb[f"{firm}_total_assets_band"]
        ta = canon.total_assets.dropna()
        add("data", f"{firm}: total assets in a plausible unit range", bool(((ta >= lo) & (ta <= hi)).all()),
            f"min {ta.min():,.0f}, max {ta.max():,.0f} (band {lo:,.0f}-{hi:,.0f})")
        if firm == "nmc":
            add("data", "nmc: total assets grow every year (as reported 2013-2018)", bool(ta.is_monotonic_increasing),
                "consistent with the debt-funded expansion story", level=None)
    P = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv")
    ev = pd.read_csv(paths.raw / "edgar" / "item103_filings_2012_2023.csv")
    add("data", "EDGAR: each failed firm appears once", P.cik.is_unique, f"{len(P)} failed firms")
    ctrl = P.control_cik.dropna().astype(int)
    add("data", "EDGAR: no control is also a failed firm in the panel", not ctrl.isin(P.cik).any(), f"{len(ctrl)} controls")
    hit = ctrl[ctrl.isin(ev.cik)]
    add("data", "EDGAR: no control has any Item 1.03 (bankruptcy) 8-K in 2012-2023", hit.empty,
        f"{len(hit)} controls with an Item 1.03 filing" if len(hit) else "none")
    dup = ctrl[ctrl.duplicated()]
    add("data", "EDGAR: control re-use across failed firms", True,
        f"{dup.nunique()} control firm(s) matched to more than one failed firm" if len(dup) else "each control used once",
        level="INFO" if len(dup) else None)
    sic_fin = P.sic.between(*cfg["edgar"]["exclude_sic_range"]).sum()
    add("data", "EDGAR: financials (SIC 6000-6799) excluded", sic_fin == 0, f"{sic_fin} financial firms in panel")
    L = T(paths, "edgar_layer1_panel")
    ext = {c: float((L[c].abs() > sb["extreme_ratio_abs"]).mean()) for c in L.columns if c not in ("tau", "cik", "role")}
    worst = max(ext, key=ext.get)
    add("data", f"EDGAR: share of firm-quarters with |ratio| > {sb['extreme_ratio_abs']}", True,
        "; ".join(f"{k} {v:.1%}" for k, v in ext.items()) + f" (largest: {worst})",
        level="WARN" if ext[worst] > 0.05 else "INFO")


# ------------------------------------------------------------------ look-ahead and windows
def windows(cfg, paths):
    R = T(paths, "edgar_firm_results"); ok = R[R.status == "ok"]
    M = cfg["edgar"]["monitor_quarters"]; bad = 0
    for col in ["first_amber_uncorrected", "first_amber_corrected", "first_red_uncorrected", "first_red_corrected",
                "altman_alarm_q", "rule_alarm_q"]:
        x = ok[[col, "event_q"]].dropna()
        bad += int(((x[col] > x.event_q) | (x[col] < x.event_q - M + 1)).sum())
    add("look-ahead", "EDGAR: every alarm lies inside the 16 monitored quarters ending at the event", bad == 0, f"{bad} alarms outside")
    add("look-ahead", "EDGAR: training ends before monitoring (n_train >= 12 usable, monitor = 16 quarters)",
        bool((ok.usable_training >= cfg["edgar"]["min_training_quarters"]).all()),
        f"min usable training quarters among analysed firms: {int(ok.usable_training.min())}")
    c = R[R.role == "control"]; P = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv").set_index("cik").event_q
    same = (c.event_q.to_numpy() == P.reindex(c.matched_to).to_numpy())
    add("look-ahead", "EDGAR: each control is monitored over its failed firm's window", bool(same.all()), f"{int(same.sum())}/{len(c)} controls")
    sc = cfg["simulation"]
    add("look-ahead", "simulation: correction-fitting firms and held-out firms use different seed streams",
        sc["seed_offset_train"] != sc["seed_offset_heldout"], f"offsets {sc['seed_offset_train']} vs {sc['seed_offset_heldout']}")


# ------------------------------------------------------------------ orientation (lower = worse)
CORE = ["ocf_to_current_liabilities", "liquid_to_st_debt", "ebitda_interest_cover", "altman_z2_em"]


def orientation(paths):
    L = T(paths, "edgar_layer1_panel")
    ratios = [c for c in L.columns if c not in ("tau", "cik", "role")]
    last = L[L.tau.between(-3, 0)]                      # last 4 quarters (sparse ratios need pooling)
    at0 = last.groupby("role")[ratios].median(); cnt = last.groupby("role")[ratios].count()
    worse = [c for c in CORE if at0.loc["failed", c] < at0.loc["control", c]]
    add("orientation", "EDGAR, last 4 quarters: failed-firm median below control median on the 4 distress ratios (lower = worse)",
        len(worse) == len(CORE), f"{len(worse)}/{len(CORE)}: " +
        "; ".join(f"{c} {at0.loc['failed', c]:.2f} vs {at0.loc['control', c]:.2f}" for c in CORE))
    other = [c for c in ratios if c not in CORE]
    add("orientation", "EDGAR, last 4 quarters: debtor days and accruals (not distress ratios)", True,
        "; ".join(f"{c} failed {at0.loc['failed', c]:.2f} vs control {at0.loc['control', c]:.2f}" for c in other) +
        ". Failed firms do NOT look worse here: debtor days depend on industry mix, and big write-downs before "
        "bankruptcy push net income far below OCF, so measured accruals turn negative (negated: high). "
        "These two ratios measure earnings quality in normal times, not distress at the brink.", level="WARN")
    f = L[L.role == "failed"].groupby("tau")[CORE].median()
    early, late = f.loc[f.index <= -12].median(), f.loc[f.index >= -3].median()
    down = [c for c in CORE if late[c] < early[c]]
    add("orientation", "EDGAR failed firms: distress-ratio medians fall from 3+ years out to the last 4 quarters",
        len(down) == len(CORE), f"{len(down)}/{len(CORE)} fall: " + ", ".join(f"{c} {early[c]:.2f} -> {late[c]:.2f}" for c in CORE))
    cov = L[L.tau.between(-15, 0)].groupby("role")[ratios].apply(lambda d: d.notna().mean())
    add("data", "EDGAR: share of monitored firm-quarters where each ratio is available", True,
        "; ".join(f"{c} {cov[c].mean():.0%}" for c in ratios) +
        ". Short-term debt is rarely tagged in quarterly XBRL, so liquidity cover is sparse and the structural "
        "detector usually runs without it.", level="WARN" if cov.mean().min() < 0.5 else "INFO")


# ------------------------------------------------------------------ simulation
def simulation(cfg, paths):
    tol = cfg["sanity"]["calibration_tolerance_points"] / 100; n28 = cfg["simulation"]["h5_training_length"]
    fx = T(paths, "sim_fixes"); c = fx[(fx.n == n28) & (fx.method == "conformal")]
    ok = bool((abs(c.empirical_fa - c.nominal) <= tol).all())
    add("simulation", "machinery check: thresholds learned from the right distribution (conformal) hit nominal at n=28",
        ok, "; ".join(f"nominal {r.nominal:.0%} -> {r.empirical_fa:.1%}" for r in c.itertuples())
        + " (if this failed, the CUSUM/max-statistic code itself would be suspect)")
    cal = T(paths, "sim_calibration_curve")
    for det in ("kalman_cusum", "mcusum"):
        for a in (0.05, 0.10):
            s = cal[(cal.detector == det) & (cal.nominal == a)].sort_values("n")
            rho = pd.Series(s.empirical_fa.to_numpy()).corr(pd.Series(s.n.to_numpy()), method="spearman")
            add("simulation", f"{det}, nominal {a:.0%}: uncorrected false alarms shrink as training grows", rho < 0,
                f"Spearman rho {rho:.2f}; " + ", ".join(f"n={r.n}: {r.empirical_fa:.1%}" for r in s.itertuples()))
    pw = T(paths, "sim_power_drift"); nonmono = []
    for (scen, det), g in pw.groupby(["scenario", "detector"]):
        d = g.sort_values("drift").detection.to_numpy()
        if (np.diff(d) < -0.03).any():
            nonmono.append(f"{scen}/{det}")
    add("simulation", "power rises with drift size (allowing 3-point noise)", not nonmono, ", ".join(nonmono) or "all monotone")
    meta = T(paths, "sim_meta").iloc[0]
    add("simulation", "no simulated firm run crashed", meta.failed_firm_runs == 0, f"{meta.failed_firm_runs} failed runs")
    deg = int(fx.degenerate_h0_firms.sum())
    add("simulation", "degenerate (h <= 0) thresholds are excluded and counted, not silently used", True,
        f"{deg} firm-cells across all n and alphas", level="INFO")
    f = fx[(fx.n == n28) & (fx.nominal == 0.05)].set_index("method")
    if abs(f.loc["inflation_factor", "detection_S1"] - f.loc["block_bootstrap", "empirical_fa"]) < 1e-12:
        try:
            from src.calibration import thresholds as th
            from src.simulate import analysis as A
            with gzip.open(paths.processed / "sim_runs.pkl.gz", "rb") as fh:
                R = A.Runs(pickle.load(fh)["runs"], cfg)
            inf = T(paths, "sim_inflation_fit"); r = inf[inf.nominal == 0.05].iloc[0]
            hb = R.h("heldout", n28, "h_kalman_0.05")
            v1 = R.maxs("heldout", n28, "H", "kalman") > hb
            v2 = R.detected("heldout", n28, "S1", "kalman", hb * th.inflation_factor(n28, (r.a, r.b)))
            add("simulation", "coincidence check: two equal numbers in sim_fixes (n=28, 5%) come from different firms",
                not np.array_equal(v1, v2), f"both {v1.sum()}/{len(v1)}; vectors agree on {np.mean(v1 == v2):.0%} of firms")
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------ cohort
def cohort_checks(cfg, paths):
    meta_f = paths.raw / "cohort" / "cohort.csv"
    if not meta_f.exists():
        add("cohort", "cohort collected", True, "data/raw/cohort/cohort.csv not present", level="WARN"); return
    meta = pd.read_csv(meta_f); co = T(paths, "cohort_archetypes")
    fp = (ROOT / "docs" / "PREREG_FINGERPRINT.txt").read_text(encoding="utf-8")
    recorded = re.findall(r"^([0-9a-f]{64}) \*data/raw/cohort/PROPOSAL.md", fp, re.M)
    h = hashlib.sha256((paths.raw / "cohort" / "PROPOSAL.md").read_bytes()).hexdigest()
    add("cohort", "PROPOSAL.md is unchanged since its last recorded fingerprint (frame and rule fixed before data)",
        bool(recorded) and recorded[-1] == h, f"{len(recorded)} fingerprints recorded; current {h[:16]}")
    f = co[co.role == "failed"]
    lo, hi = cfg["cohort"]["event_window"]
    yrs = pd.to_datetime(f.event_date).dt.year
    add("cohort", f"failed-firm events fall in {lo}-{hi}", bool(yrs.between(lo, hi).all()), ", ".join(map(str, sorted(yrs))))
    need = cfg["cohort"]["min_pre_event_annual_filings"]
    add("cohort", f"every failed firm has >= {need} pre-event annual periods", bool((f.pre_event_annual_periods >= need).all()),
        ", ".join(f"{r.firm_id} {r.pre_event_annual_periods}" for r in f.itertuples()))
    add("cohort", "accounting identities hold for every cohort firm (checked by run_real before analysis)", len(co) == len(meta),
        f"{len(co)} of {len(meta)} firms analysed")
    c = meta[meta.role == "control"]
    ev = meta.set_index("firm_id").event_date
    add("cohort", "each control is aligned at its failed firm's event date", bool((c.event_date.to_numpy() == ev.reindex(c.matched_to).to_numpy()).all()))
    add("cohort", "no control is itself a cohort failed firm", not c.screener_path.isin(meta[meta.role != "control"].screener_path).any())
    cm = pd.read_csv(paths.raw / "cohort" / "control_matching.csv"); chosen = cm[cm.reason == "CHOSEN"]
    band = cfg["cohort_ingest"]["assets_band"]
    add("cohort", f"chosen controls lie inside the {band[0]:g}-{band[1]:g}x total-assets band",
        bool(chosen.assets_ratio.between(*band).all()), ", ".join(f"{r.failed} {r.assets_ratio:.2f}x" for r in chosen.itertuples()))
    sc = co[co.scaled_periods > 0]
    add("cohort", "non-12-month fiscal periods are scaled and flagged, not silently mixed", True,
        ", ".join(f"{r.firm_id}: {r.scaled_periods}" for r in sc.itertuples()) or "none", level="INFO")


# ------------------------------------------------------------------ EDGAR results
def edgar_checks(paths):
    """Reads the CIs from edgar_detection_summary.csv (the one source, src/evaluation.py) instead of re-drawing
    them: an earlier version re-bootstrapped here with another seed stream and quoted different CIs (decision #34)."""
    S = T(paths, "edgar_detection_summary")
    for r in S.itertuples():
        lo, hi = r.diff_ci_lo, r.diff_ci_hi
        add("EDGAR", f"{r.method}: alarms failed firms more often than healthy controls (detection minus false alarms > 0)",
            lo > 0, f"{100 * r.det_minus_fa:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]; detection {r.detection:.1%}, false alarms {r.false_alarm:.1%}",
            level=None if lo > 0 else "WARN")
    unc = S.set_index("method").loc["amber (uncorrected)"]
    add("EDGAR", "real-data echo of H4: uncorrected amber false-alarm rate on healthy controls far above 10% nominal",
        unc.fa_ci_lo > 0.10, f"{unc.false_alarm:.1%} [{unc.fa_ci_lo:.1%}, {unc.fa_ci_hi:.1%}]", level=None)
    f = paths.tables / "robust_edgar_reproduce.csv"
    if f.exists():
        r = pd.read_csv(f).iloc[0]
        add("reproducibility", "EDGAR rerun (robustness base variant) gives identical per-firm results", r.different == 0,
            f"{r.identical}/{r.firm_runs} firm runs identical")
    else:
        add("reproducibility", "EDGAR rerun", True, "robustness not run yet", level="WARN")


# ------------------------------------------------------------------ determinism of the case-study tables
def determinism(paths):
    sim = sorted(paths.tables.glob("sim_*.csv"))
    if (paths.processed / "sim_runs.pkl.gz").exists() and sim:
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sim}
        subprocess.run([sys.executable, "-m", "src.simulate.report", "--tables"], cwd=ROOT, check=True, capture_output=True)
        diff = [n for n, h in before.items() if hashlib.sha256((paths.tables / n).read_bytes()).hexdigest() != h]
        add("reproducibility", "simulation tables regenerate byte-identically from the stored runs", not diff,
            f"{len(before)} tables; different: {', '.join(diff) or 'none'}")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-m", "src.run_layers", "--tables", "--out-root", d], cwd=ROOT, check=True,
                       capture_output=True)
        new = sorted((Path(d) / "reports" / "tables").glob("*.csv")) or sorted(Path(d).rglob("*.csv"))
        diff = [p.name for p in new if (paths.tables / p.name).exists() and not filecmp.cmp(p, paths.tables / p.name, shallow=False)]
        add("reproducibility", "Layer 1/2 tables regenerate byte-identically", not diff and bool(new),
            f"{len(new)} tables compared; different: {', '.join(diff) or 'none'}")


# ------------------------------------------------------------------ report traceability
def _candidates(paths, cfg):
    vals, raw = set(), set()
    for f in paths.tables.glob("*.csv"):
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        for v in pd.to_numeric(d.select_dtypes(exclude="number").stack(), errors="coerce").dropna().tolist() + \
                d.select_dtypes("number").stack().tolist():
            vals.add(float(v))
        for s in d.select_dtypes(exclude="number").stack().astype(str):
            vals.update(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(s)))
            raw.update(re.findall(r"\d[\d,]*(?:\.\d+)?", str(s)))       # numbers quoted with thousands separators

    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            vals.add(float(o))
        elif isinstance(o, str):
            vals.update(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", o))
    walk(cfg)
    out = set()
    for v in vals:
        for x in (v, 100 * v, abs(v), abs(100 * v)):
            for k in range(0, 4):
                out.add(f"{x:.{k}f}")
            out.add(f"{x:,.0f}"); out.add(f"{x:g}")
    return out | raw | {t.rstrip(",") for t in raw}


def report_trace(cfg, paths, report="reports/report.md", label="report.md"):
    p = ROOT / report
    if not p.exists():
        add("report", f"{label} exists", False); return
    txt = p.read_text(encoding="utf-8")
    if p.suffix == ".html":                                     # visible text only
        txt = re.sub(r"<style.*?</style>|<img[^>]*>", " ", txt, flags=re.S)
        txt = re.sub(r"<sub>.*?</sub>", " ", txt)                # S_t subscripts
        txt = html.unescape(re.sub(r"<[^>]+>", " ", txt))
    txt = re.sub(r"\[verify[^\]]*\]|\(\w[^)]*\d{4}[a-z]?\)", " ", txt)     # citations
    body = re.sub(r"`[^`]*`|!\[[^\]]*\]\([^)]*\)|\([^)]*\.(png|csv|md)\)", " ", txt)
    toks = [t.rstrip(",") for t in re.findall(r"(?<![\w.,\-/§])[-+−]?\d[\d,]*(?:\.\d+)?(?![\w/])", body)]
    cand = _candidates(paths, cfg)
    small = {str(i) for i in range(0, 21)}
    miss = []
    for t in toks:
        t0 = t.replace("−", "-").lstrip("+")
        if t0 in small or re.fullmatch(r"(19|20)\d\d", t0.lstrip("-")) or t0 in cand or t0.lstrip("-") in cand:
            continue
        miss.append(t)
    share = 1 - len(miss) / max(1, len(toks))
    add("report", f"{label}: numbers traceable to a table or the config", len(miss) == 0,
        f"{len(toks)} numbers, {share:.1%} traced; untraced: {', '.join(sorted(set(miss))[:25]) or 'none'}",
        level=None if len(miss) == 0 else "WARN")


# ------------------------------------------------------------------ post-hoc review checks (added 2026-11-22)
def posthoc_checks(cfg, paths):
    S = T(paths, "edgar_detection_summary")
    exp = (ROOT / "reports" / "explainer.html").read_text(encoding="utf-8") if (ROOT / "reports" / "explainer.html").exists() else ""
    rep = (ROOT / "reports" / "report.md").read_text(encoding="utf-8") if (ROOT / "reports" / "report.md").exists() else ""
    san = {r["check"]: r["detail"] for r in rows if r["area"] == "EDGAR"}
    miss = []
    for r in S.itertuples():
        lo, hi = f"{100 * r.diff_ci_lo:+.1f}", f"{100 * r.diff_ci_hi:+.1f}"
        if f"({lo} to {hi})" not in exp: miss.append(f"explainer:{r.method}")
        if f"({lo} to {hi})" not in rep: miss.append(f"report:{r.method}")
        if not any(r.method in k and f"[{lo}, {hi}]" in v for k, v in san.items()): miss.append(f"sanity:{r.method}")
    add("post-hoc", "EDGAR detection-minus-false-alarm CIs are identical in the explainer, report and sanity table (one source: "
        "edgar_detection_summary.csv)", not miss, "; ".join(miss) or f"{len(S)} methods agree in all three places")
    f = paths.tables / "posthoc_paired_rules.csv"
    if f.exists():
        P = pd.read_csv(f)
        cons = ((P.verdict.str.startswith("not")) == ((P.ci_lo <= 0) & (P.ci_hi >= 0))).all()
        add("post-hoc", "paired-bootstrap table (rules minus each method) exists and its verdicts match its CIs", bool(cons),
            f"{len(P)} rows, {P.resamples.iloc[0]} resamples")
    else:
        add("post-hoc", "paired-bootstrap table exists", False, "posthoc_paired_rules.csv missing")
    f = paths.tables / "posthoc_fixed_horizon.csv"
    hz = [h for h in cfg["post_hoc"]["fixed_horizons_quarters"] if h < cfg["edgar"]["monitor_quarters"]]
    if f.exists():
        Fh = pd.read_csv(f); ok = all(len(Fh[Fh.horizon_quarters == h]) == len(S) for h in hz)
        add("post-hoc", "fixed-horizon table exists with every method at every in-window horizon", ok,
            f"horizons {cfg['post_hoc']['fixed_horizons_quarters']}, {len(Fh)} rows")
    else:
        add("post-hoc", "fixed-horizon table exists", False, "posthoc_fixed_horizon.csv missing")
    sec = re.search(r"<h2>8\. Additional checks \(post-hoc\)</h2>(.*?)<h2", exp, re.S)
    h3 = re.findall(r"<h3[^>]*>(.*?)</h3>", sec.group(1), re.S) if sec else []
    unl = [re.sub("<[^>]+>", "", h)[:40] for h in h3 if "post-hoc" not in h]
    unl += [re.sub("<[^>]+>", "", h)[:40] for h in re.findall(r'<h3 class="posthoc">(.*?)</h3>', exp, re.S) if "post-hoc" not in h]
    rsec = re.search(r"\*\*9\.7 Additional checks \(post-hoc / exploratory\)\.\*\*(.*?)\n## ", rep, re.S)
    bullets = re.findall(r"^- \*([^*]+)\*", rsec.group(1), re.M) if rsec else []
    unl += [b for b in bullets if "post-hoc" not in b]
    add("post-hoc", "every post-hoc section carries a 'post-hoc' label (explainer Section 8 headings, report 9.7 items)",
        bool(sec) and bool(h3) and bool(rsec) and not unl,
        f"{len(h3)} explainer headings, {len(bullets)} report items; unlabelled: {', '.join(unl) or 'none'}")
    snap = yaml.safe_load((ROOT / "docs" / "prereg_snapshot" / "config.yaml").read_text(encoding="utf-8"))
    dev = (ROOT / "docs" / "PREREGISTRATION.md").read_text(encoding="utf-8").split("## 10. Deviations log")[-1]
    unlogged = [k for k in cfg if k not in snap and f"`{k}`" not in dev]
    add("post-hoc", "every config block added after the fingerprint is named in the PREREGISTRATION deviations log", not unlogged,
        "unlogged: " + ", ".join(unlogged) if unlogged else "all logged")


def tests():
    r = subprocess.run([sys.executable, "-m", "pytest"], cwd=ROOT, capture_output=True, text=True)
    found = re.findall(r"(\d+) (passed|failed|error)", r.stdout)
    last = ", ".join(f"{n} {w}" for n, w in found) if found else f"{r.stdout.count('.')} passed (dots counted)"
    add("tests", "full test suite (incl. legacy reproduction: CSV-equal, pixel-identical figures)", r.returncode == 0, last)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    provenance(cfg); data(cfg, paths); windows(cfg, paths); orientation(paths); simulation(cfg, paths)
    cohort_checks(cfg, paths); edgar_checks(paths); determinism(paths); report_trace(cfg, paths)
    if (ROOT / "reports" / "explainer.html").exists():
        report_trace(cfg, paths, "reports/explainer.html", "explainer.html")
    if (ROOT / "reports" / "report_short.md").exists():
        report_trace(cfg, paths, "reports/report_short.md", "report_short.md")
    if (ROOT / "reports" / "slides.html").exists():
        report_trace(cfg, paths, "reports/slides.html", "slides.html")
    posthoc_checks(cfg, paths)
    if not a.skip_tests:
        tests()
    out = pd.DataFrame(rows)
    out.to_csv(paths.tables / "sanity_checks.csv", index=False)
    pd.set_option("display.max_colwidth", 140)
    print(out[["status", "area", "check", "detail"]].to_string(index=False))
    print("\n" + out.status.value_counts().to_string())


if __name__ == "__main__":
    main()
