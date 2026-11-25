"""Robustness / sensitivity analyses (exploratory; config `robustness`, added after the main runs).

    python -m src.robustness [--workers N] [--skip-edgar]

EDGAR panel (re-runs Layers 0-5 per firm under each variant, in parallel):
  base            pre-registered settings (must reproduce edgar_firm_results.csv exactly)
  k=<k>           CUSUM reference value k from robustness.cusum_k_grid (h re-bootstrapped with the same k)
  shift=+/-s      training/monitoring boundary moved by robustness.training_window_shift quarters
  drop=<ratio>    leave-one-structural-ratio-out for the multivariate and structural detectors
  event-4q        alternative event: robustness.edgar_event_shift_quarters before the Item 1.03 8-K
  plus, on the base run: amber threshold x multiplier (robustness.h_multiplier_grid) and
  leave-one-rule-out / single-rule versions of "any Layer 2 rule".
Case studies (annual data):
  Jyoti FY2017 gap: ratios.fill_missing none vs linear
  alternative event dates: one fiscal year earlier; NMC short-seller report date
  NMC hidden-debt stress test (robustness.nmc_hidden_debt_multiples)

Tables -> reports/tables/robust_*.csv. No verdict (H1-H6) uses these numbers.
"""
from __future__ import annotations

import argparse
import copy
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.config import Paths, load_config
from src.detectors import event_time, layer2_rules
from src.detectors.monitor import load_inflation
from src.masking import archetype
from src.ratios import check_identities, compute_ratios

from src import evaluation as ev  # noqa: E402

KEEP_COLS = ["cik", "role", "matched_to", "event_q", "status", "first_amber_uncorrected", "first_amber_corrected",
             "first_red_uncorrected", "first_red_corrected", "altman_alarm_q", "rule_alarm_q", "h_cash_0.10", "h_multi_0.10"]
METHODS = {k: ev.METHODS[k] for k in ["amber (uncorrected)", "amber (n-corrected)", "red (uncorrected)", "red (n-corrected)",
                                       "Altman Z''-EM distress", "any Layer 2 rule"]}   # same registry, robustness order


# ------------------------------------------------------------------ EDGAR variants
def variants(cfg):
    rb = cfg["robustness"]; out = [("base", cfg, 0, 0)]
    for k in rb["cusum_k_grid"]:
        if k != cfg["detectors"]["cusum"]["k"]:
            c = copy.deepcopy(cfg); c["detectors"]["cusum"]["k"] = k; out.append((f"k={k:g}", c, 0, 0))
    s = rb["training_window_shift"]
    out += [(f"shift={-s:+d}q", cfg, -s, 0), (f"shift={s:+d}q", cfg, s, 0)]
    for r in cfg["masking"]["structural_ratios"]:
        c = copy.deepcopy(cfg); c["masking"]["structural_ratios"] = [x for x in cfg["masking"]["structural_ratios"] if x != r]
        out.append((f"drop={r}", c, 0, 0))
    e = rb["edgar_event_shift_quarters"]
    out.append((f"event{e:+d}q", cfg, 0, e))
    return out


def _first(idx, mask):
    hits = np.flatnonzero(np.asarray(mask, bool))
    return int(idx[hits[0]]) if len(hits) else None


def _work(task):
    warnings.filterwarnings("ignore")
    from src.run_edgar import firm_run
    name, cfg, shift, eshift, cik, role, event_q, matched_to, paths, inflation = task
    keep = {} if name == "base" else None
    row, _ = firm_run(cik, role, event_q + eshift, matched_to, cfg, paths, inflation, shift=shift, keep=keep)
    out = {c: row.get(c) for c in KEEP_COLS}
    out.update(variant=name, event_q=event_q + eshift)
    if keep and "out" in keep:
        o = keep["out"]
        for m in cfg["robustness"]["h_multiplier_grid"]:
            hc, hm = row.get("h_cash_0.10", np.nan), row.get("h_multi_0.10", np.nan)
            out[f"hmult_{m:g}"] = _first(o.index, (o.S_cash > m * hc).to_numpy() | (o.S_multi > m * hm).to_numpy())
            out[f"cash_hmult_{m:g}"] = _first(o.index, (o.S_cash > m * hc).to_numpy())
            out[f"multi_hmult_{m:g}"] = _first(o.index, (o.S_multi > m * hm).to_numpy())
        out.update(_scores(keep, cfg, row))
        F, mon = keep["F"], keep["mon"]
        rules = F.iloc[:, :-1].reindex(mon) == 1
        for c in rules.columns:
            out[f"only::{c}"] = _first(mon, rules[c].to_numpy())
            out[f"drop::{c}"] = _first(mon, rules.drop(columns=c).any(axis=1).to_numpy())
    return out


def _scores(keep, cfg, row):
    """Threshold-free alarm scores over the monitored window (higher = more alarming), for AUC diagnostics."""
    from src.detectors import cusum as cs
    from src.detectors import statespace as ss
    o, r, n = keep["out"], keep["r"], keep["n_train"]
    cash = cfg["masking"]["cash_ratio"]; k = cfg["detectors"]["cusum"]["k"]
    x = r[cash].to_numpy(float); tr = x[:n][~np.isnan(x[:n])]
    sc = {"score_kalman_cusum": float(np.nanmax(o.S_cash)) if o.S_cash.notna().any() else np.nan,
          "score_mcusum": float(np.nanmax(o.S_multi)) if o.S_multi.notna().any() else np.nan}
    # static (non-adaptive) CUSUM: deviations from the training mean in training-SD units
    z = (x[n:] - tr.mean()) / tr.std(ddof=1) if len(tr) > 2 and tr.std(ddof=1) > 0 else np.full(len(x) - n, np.nan)
    sc["score_static_cusum"] = float(np.nanmax(cs.cusum(z, k))) if np.isfinite(z).any() else np.nan
    sc["score_cash_level"] = float(-np.nanmin(x[n:])) if np.isfinite(x[n:]).any() else np.nan
    zt = r["altman_z2_em"].to_numpy(float)[n:]
    sc["score_altman_level"] = float(-np.nanmin(zt)) if np.isfinite(zt).any() else np.nan
    # share of the fitted variance assigned to the random-walk level: high => the filter adapts to drifts
    try:
        mod = ss._uc(tr); p = ss.fit_uc(x[:n]); names = list(mod.param_names)
        var = {nm: p[i] for i, nm in enumerate(names) if nm.startswith("sigma2")}
        sc["level_variance_share"] = float(var.get("sigma2.level", np.nan) / sum(var.values()))
    except Exception:  # noqa: BLE001
        sc["level_variance_share"] = np.nan
    return sc


_auc = ev.auc
_pairs = ev.analysed


def _summ(f, c, col, rg, cfg, **lab):
    B = ev.ci_resamples(cfg)
    d, dlo, dhi = ev.rate_ci(f[col].notna(), rg, B); a, alo, ahi = ev.rate_ci(c[col].notna(), rg, B)
    lead = (f.event_q - f[col]).dropna()
    row = dict(**lab, failed_firms=len(f), controls=len(c), detection=d, det_ci_lo=dlo, det_ci_hi=dhi,
               false_alarm=a, fa_ci_lo=alo, fa_ci_hi=ahi, median_lead_quarters=float(lead.median()) if len(lead) else np.nan)
    for cm in cfg["cost"]["c_miss"]:
        row[f"expected_cost_cm{cm}"] = cfg["cost"]["c_fa"] * a + cm * (1 - d)
    return row


def edgar(cfg, paths, workers, only=None):
    panel = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv")
    inflation = load_inflation(paths.tables)
    tasks = []
    for name, c, shift, eshift in variants(cfg):
        if only and name not in only:
            continue
        for p in panel.itertuples():
            for cik, role in [(p.cik, "failed"), (p.control_cik, "control")]:
                if not pd.isna(cik):
                    tasks.append((name, c, shift, eshift, int(cik), role, int(p.event_q), int(p.cik), paths, inflation))
    print(f"{len(tasks)} firm runs on {workers} workers")
    with Pool(workers) as pool:
        rows = pool.map(_work, tasks, chunksize=8)
    R = pd.DataFrame(rows)
    prev = paths.processed / "robust_edgar_runs.csv"
    if only and prev.exists():                      # re-run a subset of variants and merge with the stored runs
        old = pd.read_csv(prev)
        R = pd.concat([old[~old.variant.isin(only)], R], ignore_index=True)
        order = [v[0] for v in variants(cfg)]
        R = R.sort_values("variant", key=lambda x: x.map(order.index), kind="stable")
    R.to_csv(prev, index=False)
    rg = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 5151]))

    # 1. reproducibility: base variant vs the main run
    main = pd.read_csv(paths.tables / "edgar_firm_results.csv")
    b = R[R.variant == "base"]
    cols = ["status", "first_amber_uncorrected", "first_amber_corrected", "first_red_uncorrected", "first_red_corrected",
            "altman_alarm_q", "rule_alarm_q"]
    m = main.set_index(["cik", "role", "matched_to"])[cols]; bb = b.set_index(["cik", "role", "matched_to"])[cols].reindex(m.index)
    same = ((m.astype(str) == bb.astype(str)) | (m.isna() & bb.isna())).all(axis=1)
    pd.DataFrame([{"check": "robustness base run reproduces edgar_firm_results.csv", "firm_runs": len(m),
                   "identical": int(same.sum()), "different": int((~same).sum())}]).to_csv(paths.tables / "robust_edgar_reproduce.csv", index=False)

    # 2. variant table
    rows = []
    for v, g in R.groupby("variant", sort=False):
        f, c = _pairs(g)
        for lab, col in METHODS.items():
            if v.startswith("k=") and "corrected)" in lab and "uncorrected" not in lab:
                continue  # inflation factor was fitted at the pre-registered k only
            rows.append(_summ(f, c, col, rg, cfg, variant=v, method=lab))
    pd.DataFrame(rows).to_csv(paths.tables / "robust_edgar_variants.csv", index=False)

    # 3. threshold-multiplier curve (amber, uncorrected) and rule ablation, base sample
    f, c = _pairs(b)
    hm = [_summ(f, c, f"{pre}hmult_{m:g}", rg, cfg, detector=lab, h_multiplier=m)
          for pre, lab in [("", "amber: cash OR multivariate"), ("cash_", "cash CUSUM only (one-sided)"),
                           ("multi_", "multivariate MCUSUM only (direction-free)")]
          for m in cfg["robustness"]["h_multiplier_grid"] if f"{pre}hmult_{m:g}" in b]
    pd.DataFrame(hm).to_csv(paths.tables / "robust_edgar_h_grid.csv", index=False)
    rr = [_summ(f, c, "rule_alarm_q", rg, cfg, rule_set="all three rules (pre-registered)")]
    for col in [x for x in b.columns if x.startswith("only::")]:
        rr.append(_summ(f, c, col, rg, cfg, rule_set=f"only {col[6:]}"))
    for col in [x for x in b.columns if x.startswith("drop::")]:
        rr.append(_summ(f, c, col, rg, cfg, rule_set=f"drop {col[6:]}"))
    pd.DataFrame(rr).to_csv(paths.tables / "robust_edgar_rules.csv", index=False)
    # threshold-free discrimination (AUC, failed vs paired controls) of each alarm score
    labels = {"score_kalman_cusum": "cash CUSUM on Kalman errors (pre-registered)", "score_mcusum": "multivariate MCUSUM (pre-registered)",
              "score_static_cusum": "cash CUSUM on deviations from the training mean (non-adaptive)",
              "score_cash_level": "lowest cash ratio in the window (level)", "score_altman_level": "lowest Altman Z''-EM in the window (level)"}
    au = []
    for col, lab in labels.items():
        if col not in f:
            continue
        a, c_ = f[col].to_numpy(float), c[col].to_numpy(float)
        bs = [_auc(a[rg.integers(0, len(a), len(a))], c_[rg.integers(0, len(c_), len(c_))]) for _ in range(1000)]
        au.append(dict(score=lab, failed_firms=int(np.isfinite(a).sum()), controls=int(np.isfinite(c_).sum()), auc=_auc(a, c_),
                       auc_ci_lo=float(np.nanquantile(bs, .025)), auc_ci_hi=float(np.nanquantile(bs, .975))))
    if "level_variance_share" in b:
        lv = b[b.status == "ok"].level_variance_share
        au.append(dict(score="median share of fitted variance on the adaptive level (all analysed firms)", failed_firms=None,
                       controls=None, auc=float(lv.median()), auc_ci_lo=float(lv.quantile(.25)), auc_ci_hi=float(lv.quantile(.75))))
    pd.DataFrame(au).to_csv(paths.tables / "robust_edgar_auc.csv", index=False)
    print(pd.DataFrame(au).round(3).to_string(index=False))
    print(pd.DataFrame(rows).query("method == 'amber (uncorrected)'")[["variant", "failed_firms", "controls", "detection", "false_alarm"]].round(3).to_string(index=False))


# ------------------------------------------------------------------ case studies
def _case(firm, cfg, paths, event_date=None):
    from src.run_layers import firm_results
    c = copy.deepcopy(cfg)
    if event_date is not None:
        c["firms"][firm]["event_date"] = event_date
    res = firm_results(firm, c, paths)
    arch = archetype.classify(res["canon"], res["ratios"], res["efy"], c)
    return res, arch


def cases(cfg, paths):
    rb = cfg["robustness"]
    # (a) Jyoti FY2017 gap: none vs linear interpolation
    rows = []
    for fill in ("none", "linear"):
        c = copy.deepcopy(cfg); c["ratios"]["fill_missing"] = fill
        res, arch = _case("jyoti", c, paths)
        for rule, r in res["first"].iterrows():
            rows.append(dict(fill_missing=fill, rule=rule, first_trip_fy=r.first_trip_fy, lead_years=r.lead_years,
                             pre_event_years_tripped=r.pre_event_years_tripped, archetype=arch["archetype"]))
        for col in ("ocf_to_current_liabilities", "altman_z2_em"):
            rows.append(dict(fill_missing=fill, rule=f"FY2017 value: {col}", value=res["ratios"][col].get(2017, np.nan),
                             archetype=arch["archetype"]))
    pd.DataFrame(rows).to_csv(paths.tables / "robust_jyoti_fill.csv", index=False)

    # (b) alternative event definitions
    rows = []
    for firm in ("nmc", "jyoti"):
        fc = cfg["firms"][firm]
        defs = [("pre-registered", fc["event_date"])]
        shifted = (pd.Timestamp(fc["event_date"]) + pd.DateOffset(years=rb["case_event_shift_fy"])).strftime("%Y-%m-%d")
        defs.append((f"{rb['case_event_shift_fy']:+d} fiscal year", shifted))
        if firm in rb.get("alt_event_dates", {}):
            a = rb["alt_event_dates"][firm]; defs.append((a["label"], a["date"]))
        for lab, d in defs:
            res, arch = _case(firm, cfg, paths, d)
            anyr = res["first"].loc["Any rule"]
            rows.append(dict(firm=firm, definition=lab, event_date=d, event_fy=res["efy"],
                             first_any_rule_fy=anyr.first_trip_fy, lead_years_any_rule=anyr.lead_years,
                             archetype=arch["archetype"], structural_flags_tripped=arch["structural_flags_tripped"]))
    pd.DataFrame(rows).to_csv(paths.tables / "robust_case_alt_events.csv", index=False)

    # (c) NMC hidden-debt stress test
    from src.run_layers import firm_results
    base = firm_results("nmc", cfg, paths)
    canon0, efy = base["canon"], base["efy"]
    rows = []
    for m in rb["nmc_hidden_debt_multiples"]:
        cn = canon0.copy()
        D = m * cn.total_debt
        rate = cn.interest / cn.total_debt
        for col in ("st_debt", "total_debt", "current_liabilities", "total_liabilities"):
            cn[col] = cn[col] + D
        cn["equity"] = cn.equity - D; cn["retained_earnings"] = cn.retained_earnings - D
        cn["interest"] = cn.interest + rate * D
        check_identities(cn, cfg["identities"], f"nmc hidden debt x{m:g}")
        r = compute_ratios(cn, cfg)
        S = archetype.structural_flags(cn, r, cfg)
        arch = archetype.classify(cn, r, efy, cfg)
        row = dict(hidden_debt_multiple=m, archetype=arch["archetype"], n_structural_flags=arch["n_structural_flags_tripped"])
        for col in S.columns:
            hit = S.index[S[col] == 1]
            row[f"first FY: {col}"] = int(hit[0]) if len(hit) else None
        row["liquidity_cover_2018"] = float(r.liquid_to_st_debt.get(2018, np.nan))
        row["interest_cover_2018"] = float((cn.ebitda / cn.interest).get(2018, np.nan))
        row["z2_em_2018"] = float(r.altman_z2_em.get(2018, np.nan))
        rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "robust_nmc_hidden_debt.csv", index=False)
    print(t.to_string(index=False))


def level_vs_change(cfg, paths):
    """Is the failed-control gap a difference in LEVEL (already there in the training years) or in CHANGE
    (deterioration during the monitored window)? Self-referential detectors only see change.
    Reported for the analysed sample (the firms behind every EDGAR rate) and, as before, for every firm with ratios."""
    from src.run_edgar import analysed_panel
    La, L = analysed_panel(paths)
    M = cfg["edgar"]["monitor_quarters"]
    ratios = ["ocf_to_current_liabilities", "ebitda_interest_cover", "altman_z2_em"]
    rows = []
    for sample, D in [("analysed", La), ("all firms with ratios", L)]:
        train = D[D.tau < -M + 1].groupby(["role", "cik"])[ratios].median()
        mon = D[D.tau.between(-3, 0)].groupby(["role", "cik"])[ratios].median()
        chg = (mon - train)
        for c in ratios:
            for role in ("failed", "control"):
                t, d = train.loc[role, c].dropna(), chg.loc[role, c].dropna()
                rows.append(dict(sample=sample, ratio=c, role=role, firms=len(t), training_level_median=float(t.median()),
                                 change_to_last_4q_median=float(d.median()), share_falling=float((d < 0).mean())))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "robust_edgar_level_vs_change.csv", index=False)
    print(t.round(3).to_string(index=False))


def figures(cfg, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.reporting.style import C, STYLE
    hg = pd.read_csv(paths.tables / "robust_edgar_h_grid.csv")
    S = pd.read_csv(paths.tables / "edgar_detection_summary.csv").set_index("method")
    hd = pd.read_csv(paths.tables / "robust_nmc_hidden_debt.csv")
    cols = {"amber: cash OR multivariate": C["coral"], "cash CUSUM only (one-sided)": C["navy"],
            "multivariate MCUSUM only (direction-free)": C["purple"]}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        ax[0].plot([0, 100], [0, 100], color=C["slate"], ls=":", lw=1, label="no better than chance")
        for d, g in hg.groupby("detector", sort=False):
            g = g.sort_values("h_multiplier")
            ax[0].plot(g.false_alarm * 100, g.detection * 100, "-o", ms=4, color=cols.get(d, C["slate"]), label=f"{d} (threshold x 0.5 to 4)")
        for m, mk, col in [("Altman Z''-EM distress", "s", C["teal"]), ("any Layer 2 rule", "D", C["green"])]:
            r = S.loc[m]
            ax[0].plot(r.false_alarm * 100, r.detection * 100, mk, ms=9, color=col, label=m)
        ax[0].set_xlim(0, 100); ax[0].set_ylim(0, 100)
        ax[0].set_xlabel("false alarms on matched healthy controls (%)"); ax[0].set_ylabel("detection of failed firms (%)")
        ax[0].set_title("A. EDGAR: statistical detectors sit just above chance"); ax[0].legend(fontsize=7.5, loc="lower right")
        x = np.arange(len(hd))
        for i, (col, lab, c) in enumerate([("liquidity_cover_2018", "liquidity cover", C["navy"]), ("interest_cover_2018", "EBITDA / interest", C["orange"]),
                                           ("z2_em_2018", "Altman Z''-EM", C["purple"])]):
            ax[1].bar(x + (i - 1) * .27, hd[col], .27, color=c, label=lab)
        sf = cfg["archetype"]["masked"]["structural_flags"]
        ax[1].axhline(sf["liquidity_cover_max"], color=C["navy"], ls=":", lw=1, label=f"archetype structural flag: liquidity cover < {sf['liquidity_cover_max']:g}")
        ax[1].axhline(sf["interest_cover_max"], color=C["orange"], ls=":", lw=1, label=f"archetype structural flag: EBITDA / interest < {sf['interest_cover_max']:g} "
                      f"(Layer 2 rule: < {cfg['layer2']['rules']['ebitda_interest_cover_min']:g})")
        ax[1].axhline(cfg["altman_z2_em"]["distress"], color=C["purple"], ls=":", lw=1, label=f"archetype structural flag: Z''-EM < {cfg['altman_z2_em']['distress']:g}")
        ax[1].axhline(0, color=C["ink"], lw=.8)
        ax[1].set_xticks(x, [f"{m:g}x\n{a}" for m, a in zip(hd.hidden_debt_multiple, hd.archetype)])
        ax[1].set_xlabel("hidden debt added (multiple of reported debt) / resulting archetype")
        ax[1].set_title("B. NMC 2018 ratios with hidden debt added (post-hoc stress test)"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(paths.figures / "robust_edgar_tradeoff_nmc_hidden_debt.png"); plt.close(fig)
    print("saved robustness figure")


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=min(9, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--skip-edgar", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None, help="re-run only these EDGAR variants (e.g. base)")
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    cases(cfg, paths)
    level_vs_change(cfg, paths)
    if not a.skip_edgar:
        edgar(cfg, paths, a.workers, a.variants)
    figures(cfg, paths)


if __name__ == "__main__":
    main()
