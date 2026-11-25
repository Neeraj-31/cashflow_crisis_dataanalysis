"""Post-hoc review checks on the EDGAR panel and the simulation outputs (config `post_hoc`, added 2026-11-22 AFTER
every result had been seen, in response to a review of reports/explainer.html). Exploratory only: no hypothesis
verdict uses any number written here.

    python -m src.post_hoc [--workers N] [--only paired horizon persistence filing ttm winsor industry mi]

Tables -> reports/tables/posthoc_*.csv
  paired       posthoc_paired_rules.csv        any Layer 2 rule minus each other method: detection, false alarms and
                                               expected cost, matched-unit bootstrap, same resample for both methods
  horizon      posthoc_fixed_horizon.csv       detection / false alarms counting only alarms raised >= h quarters
               posthoc_static_weakness.csv     were rule / Altman flags already on before the monitored window began?
  persistence  posthoc_persistence.csv         rules / Altman must be on in >= p consecutive monitored quarters
  filing       posthoc_filing_lead.csv         lead times measured from the SEC filing date of the alarm quarter
  ttm          posthoc_ttm_overlap.csv         Layer 3 (cash detector) on TTM vs single-quarter vs year-on-year inputs
               posthoc_kalman_acf.csv          autocorrelation of standardised Kalman errors, healthy controls
               posthoc_calibration_horizon.csv calibrated vs effectively monitored steps (+ simulated effect)
  winsor       posthoc_winsor.csv              Layers 2-5 with ratios winsorised at the post_hoc.winsor_quantiles
  industry     posthoc_industry.csv            analysed failed firms / controls per industry (EDGAR and cohort)
  mi           posthoc_mi_healthy.csv          masking index relative to healthy firms (simulation and EDGAR)
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import evaluation as ev  # noqa: E402
from src.config import Paths, load_config, rng  # noqa: E402

DAYS_PER_QUARTER = 365.25 / 4
TTM_VARIANTS = ["TTM ratio (pre-registered input)", "single-quarter ratio (no overlap)",
                "year-on-year change of the single-quarter ratio"]
TIME_METHODS = ["amber (uncorrected)", "red (uncorrected)", "amber (n-corrected)", "red (n-corrected)",
                "Altman Z''-EM distress", "any Layer 2 rule"]


def _panel(paths):
    P = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv")
    tasks = []
    for p in P.itertuples():
        for cik, role in [(p.cik, "failed"), (p.control_cik, "control")]:
            if not pd.isna(cik):
                tasks.append((int(cik), role, int(p.event_q), int(p.cik), p.file_date))
    return P, tasks


def _results(paths):
    R = pd.read_csv(paths.tables / "edgar_firm_results.csv")
    return R, *ev.analysed(R)


# ------------------------------------------------------------------ per-firm worker (Layers 0-3, no full monitor)
def _first_run(on: np.ndarray, p: int):
    """Index of the first position where `on` has been True for p consecutive positions (None if never)."""
    run = 0
    for i, v in enumerate(on):
        run = run + 1 if v else 0
        if run >= p:
            return i
    return None


def _layer3_cash(x: np.ndarray, n_train: int, cfg: dict, rg):
    """Pre-registered Layer 3 cash detector on one input series x (Kalman z, one-sided CUSUM, block-bootstrap h at
    the amber alpha). Called with rng(cik % 100000) first, it reproduces the h_cash of src.detectors.monitor."""
    from src.calibration import thresholds as th
    from src.detectors import cusum as cs
    from src.detectors import statespace as ss
    dc = cfg["detectors"]; k, H, B = dc["cusum"]["k"], dc["monitor_horizon"], dc["bootstrap"]["n_resamples"]
    b = th.block_length(n_train, dc["bootstrap"]["min_block"]); a = dc["tiers"]["amber_alpha"]
    z, _ = ss.kalman_z(x, n_train)
    h = th.bootstrap_h(th.pool_univariate(x[:n_train], dc["training"]["train_fraction"]), (dc["tiers"]["red_alpha"], a),
                       k, H, B, b, rg)[a]
    h = h if np.isfinite(h) and h > 0 else np.nan
    S = cs.cusum(z[n_train:], k)
    hit = np.flatnonzero(S > h) if np.isfinite(h) else np.array([], int)
    return dict(h=h, first=int(hit[0]) if len(hit) else None, score=float(np.nanmax(S)) if len(S) else np.nan,
                steps=int(np.isfinite(z[n_train:]).sum()), z_train=z[:n_train], z_mon=z[n_train:])


def _acf(z: np.ndarray, lags):
    out = {}
    for L in lags:
        a, b = z[L:], z[:-L]
        ok = np.isfinite(a) & np.isfinite(b)
        out[L] = (a[ok], b[ok])
    return out


def _firm(task):
    warnings.filterwarnings("ignore")
    from src.detectors import layer2_rules
    from src.ingest import edgar
    from src.run_edgar import prepare
    cik, role, event_q, matched_to, event_date, cfg, paths, parts = task
    M = cfg["edgar"]["monitor_quarters"]; start = event_q - M + 1; cash = cfg["masking"]["cash_ratio"]
    need = cfg["edgar"]["min_training_quarters"]
    out = {"cik": cik, "role": role, "event_q": event_q, "matched_to": matched_to}
    status, q, r = prepare(cik, event_q, cfg, paths)
    out["status0"] = status or "ok"
    if status is not None:
        return out, None
    mon = list(range(start, event_q + 1))
    if "flags" in parts:
        F = layer2_rules.rule_flags(q, r, cfg)
        rules = (F.iloc[:, :-1] == 1).any(axis=1); alt = F.iloc[:, -1] == 1
        out["rules_on"] = rules.reindex(mon, fill_value=False).to_numpy(bool)
        out["altman_on"] = alt.reindex(mon, fill_value=False).to_numpy(bool)
        out["rules_prewindow"] = bool(rules.get(start - 1, False)); out["altman_prewindow"] = bool(alt.get(start - 1, False))
        out["rules_prewindow_known"] = bool(F.iloc[:, :-1].reindex([start - 1]).notna().any(axis=None))
        out["altman_prewindow_known"] = bool(pd.notna(F.iloc[:, -1].get(start - 1, np.nan)))
    if "filing" in parts:
        f = pd.read_csv(paths.raw / "edgar" / "facts" / f"CIK{int(cik):010d}.csv.gz", usecols=["end", "filed"])
        f["qid"] = [edgar.period_id(e) for e in f.end]
        out["filed"] = pd.to_datetime(f.groupby("qid").filed.min()).reindex(mon).to_dict()
    if "ttm" in parts:
        n_train = int((r.index < start).sum())
        variants = {TTM_VARIANTS[0]: r[cash].to_numpy(float)}
        st, q1, r1 = prepare(cik, event_q, cfg, paths, flows="quarter")
        if st is None:
            r1 = r1.reindex(r.index)
            xq = r1[cash]
            variants[TTM_VARIANTS[1]] = xq.to_numpy(float)
            variants[TTM_VARIANTS[2]] = (xq - xq.shift(cfg["post_hoc"]["seasonal_lag_quarters"])).to_numpy(float)
        res = {}
        for name, x in variants.items():
            usable = int(np.isfinite(x[:n_train]).sum())
            if usable < need or n_train < cfg["detectors"]["min_periods_layer3"]:
                res[name] = dict(eligible=False, usable=usable); continue
            try:
                d = _layer3_cash(x, n_train, cfg, rng(int(cik) % 100000))
            except Exception as e:  # noqa: BLE001
                res[name] = dict(eligible=False, usable=usable, error=type(e).__name__); continue
            d["first_q"] = mon[d["first"]] if d["first"] is not None else None
            res[name] = dict(eligible=True, usable=usable, **d)
        out["ttm"] = res
    return out, None


def _collect(cfg, paths, workers, parts):
    _, tasks = _panel(paths)
    T = [(*t, cfg, paths, parts) for t in tasks]
    with Pool(workers) as pool:
        res = [o for o, _ in pool.map(_firm, T, chunksize=4)]
    return {(o["cik"], o["role"], o["matched_to"]): o for o in res}


def _key(row):
    return (int(row.cik), row.role, int(row.matched_to))


# ------------------------------------------------------------------ 2c: paired comparison, rules vs every method
def paired(cfg, paths):
    R, f, c = _results(paths)
    B = ev.ci_resamples(cfg); Fi, Ci = ev.cluster_indices(f, c, rng(cfg["post_hoc"]["seed_offsets"]["paired"]), B)
    cm = cfg["cost"]["c_miss"]; cfa = cfg["cost"]["c_fa"]
    stat = {m: ev.paired_stats(f[col].notna(), c[col].notna(), Fi, Ci, cfa, cm) for m, col in ev.METHODS.items()}
    point = {m: {"detection": f[col].notna().mean(), "false_alarm": c[col].notna().mean()} for m, col in ev.METHODS.items()}
    for m in point:
        for x in cm:
            point[m][f"expected_cost_cm{x}"] = cfa * point[m]["false_alarm"] + x * (1 - point[m]["detection"])
    base = "any Layer 2 rule"; better_if = {"detection": "higher", "false_alarm": "lower"}
    rows = []
    for other in [m for m in ev.METHODS if m != base]:
        for met in ["detection", "false_alarm"] + [f"expected_cost_cm{x}" for x in cm]:
            d = stat[base][met] - stat[other][met]
            lo, hi = np.nanquantile(d, [.025, .975]); est = point[base][met] - point[other][met]
            good = better_if.get(met, "lower")
            if lo > 0 or hi < 0:
                rules_better = (est > 0) if good == "higher" else (est < 0)
                verdict = f"rules {'better' if rules_better else 'worse'} (CI excludes 0)"
            else:
                verdict = "not distinguishable (CI includes 0)"
            rows.append(dict(comparison=f"{base} minus {other}", other=other, metric=met,
                             cost_ratio=int(met.split("cm")[1]) if "cm" in met else None,
                             rules=point[base][met], other_value=point[other][met], difference=est, ci_lo=lo, ci_hi=hi,
                             better_if=good, verdict=verdict, failed_firms=len(f), controls=len(c), resamples=B))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_paired_rules.csv", index=False)
    print(t[t.other == "Altman Z''-EM distress"].round(3).to_string(index=False))


# ------------------------------------------------------------------ 3a: fixed horizons and static weakness
def horizon(cfg, paths, F):
    R, f, c = _results(paths)
    M = cfg["edgar"]["monitor_quarters"]; B = ev.ci_resamples(cfg)
    rg = rng(cfg["post_hoc"]["seed_offsets"]["horizon"]); rows = []
    for h in cfg["post_hoc"]["fixed_horizons_quarters"]:
        for m in TIME_METHODS:
            col = ev.METHODS[m]
            if h < M:
                det = (f[col] <= f.event_q - h).to_numpy(float); fa = (c[col] <= c.event_q - h).to_numpy(float)
                basis = f"alarm in the monitored window at or before event quarter - {h}"
            elif m in ("Altman Z''-EM distress", "any Layer 2 rule"):
                k = "altman_prewindow" if m.startswith("Altman") else "rules_prewindow"
                det = np.array([F[_key(r)][k] for r in f.itertuples()], float)
                fa = np.array([F[_key(r)][k] for r in c.itertuples()], float)
                basis = f"flag already on in the last training quarter (event quarter - {M}), before monitoring starts"
            else:
                continue
            d, dlo, dhi = ev.rate_ci(det, rg, B); a, alo, ahi = ev.rate_ci(fa, rg, B)
            _, lo, hi = ev.diff_ci(det, fa, rg, B)
            rows.append(dict(horizon_quarters=h, method=m, basis=basis, failed_firms=len(f), controls=len(c),
                             detection=d, det_ci_lo=dlo, det_ci_hi=dhi, false_alarm=a, fa_ci_lo=alo, fa_ci_hi=ahi,
                             det_minus_fa=d - a, diff_ci_lo=lo, diff_ci_hi=hi))
    pd.DataFrame(rows).to_csv(paths.tables / "posthoc_fixed_horizon.csv", index=False)
    # static weakness: of the firms whose first alarm has a lead of 13-15 quarters (first 3 monitored quarters), how many were already flagged before the window?
    s = []
    for m, k in [("any Layer 2 rule", "rules_prewindow"), ("Altman Z''-EM distress", "altman_prewindow")]:
        col = ev.METHODS[m]
        for role, g in [("failed", f), ("control", c)]:
            lead = g.event_q - g[col]; long_ = g[lead >= M - 3]
            pre = np.array([F[_key(r)][k] for r in g.itertuples()], bool)
            pre_long = np.array([F[_key(r)][k] for r in long_.itertuples()], bool)
            s.append(dict(method=m, role=role, firms=len(g), alarmed=int(g[col].notna().sum()),
                          alarmed_in_first_3_monitored_quarters=len(long_),
                          share_of_alarms_in_first_3_quarters=len(long_) / max(1, int(g[col].notna().sum())),
                          of_those_already_on_before_window=float(pre_long.mean()) if len(long_) else np.nan,
                          flagged_before_window_all_firms=float(pre.mean())))
    t = pd.DataFrame(s); t.to_csv(paths.tables / "posthoc_static_weakness.csv", index=False)
    print(t.round(3).to_string(index=False))


# ------------------------------------------------------------------ 3b: persistence
def persistence(cfg, paths, F):
    R, f, c = _results(paths); M = cfg["edgar"]["monitor_quarters"]; B = ev.ci_resamples(cfg)
    rg = rng(cfg["post_hoc"]["seed_offsets"]["persistence"]); rows = []
    for m, k in [("any Layer 2 rule", "rules_on"), ("Altman Z''-EM distress", "altman_on")]:
        col = ev.METHODS[m]
        for p in [1] + list(cfg["post_hoc"]["persistence_quarters"]):
            first = {g: np.array([_first_run(F[_key(r)][k], p) is not None for r in d.itertuples()], float)
                     for g, d in [("f", f), ("c", c)]}
            lead = [M - 1 - _first_run(F[_key(r)][k], p) for r in f.itertuples() if _first_run(F[_key(r)][k], p) is not None]
            if p == 1:   # consistency: must reproduce the main table
                assert np.array_equal(first["f"], f[col].notna().to_numpy(float)), m
            d, dlo, dhi = ev.rate_ci(first["f"], rg, B); a, alo, ahi = ev.rate_ci(first["c"], rg, B)
            rows.append(dict(method=m, consecutive_quarters=p, failed_firms=len(f), controls=len(c), detection=d, det_ci_lo=dlo,
                             det_ci_hi=dhi, false_alarm=a, fa_ci_lo=alo, fa_ci_hi=ahi,
                             median_lead_quarters=float(np.median(lead)) if lead else np.nan,
                             **{f"expected_cost_cm{x}": cfg["cost"]["c_fa"] * a + x * (1 - d) for x in cfg["cost"]["c_miss"]}))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_persistence.csv", index=False)
    print(t.round(3).to_string(index=False))


# ------------------------------------------------------------------ 3h: lead time from the filing date
def filing(cfg, paths, F):
    R, f, c = _results(paths); P = pd.read_csv(paths.raw / "edgar" / "panel_failed.csv").set_index("cik")
    B = ev.ci_resamples(cfg); rg = rng(cfg["post_hoc"]["seed_offsets"]["filing"]); rows = []
    ev_date = lambda r: pd.Timestamp(P.loc[int(r.matched_to), "file_date"])  # noqa: E731  (controls: their failed firm's date)
    for m in TIME_METHODS:
        col = ev.METHODS[m]
        out = {}
        for g, d in [("f", f), ("c", c)]:
            avail, lead_q, lag_d = [], [], []
            for r in d.itertuples():
                a = getattr(r, col)
                if pd.isna(a):
                    avail.append(0.0); continue
                fd = F[_key(r)]["filed"].get(int(a), pd.NaT); e = ev_date(r)
                pe = pd.Period(year=int(a) // 4, quarter=int(a) % 4 + 1, freq="Q").end_time.normalize()
                ok = pd.notna(fd) and fd < e
                avail.append(float(ok))
                if pd.notna(fd):
                    lag_d.append((fd - pe).days)
                if ok:
                    lead_q.append((e - fd).days / DAYS_PER_QUARTER)
            out[g] = (np.array(avail), lead_q, lag_d)
        d, dlo, dhi = ev.rate_ci(out["f"][0], rg, B); a, alo, ahi = ev.rate_ci(out["c"][0], rg, B)
        main_lead = (f.event_q - f[col]).dropna()
        rows.append(dict(method=m, failed_firms=len(f), controls=len(c),
                         detection_period_end_basis=float(f[col].notna().mean()), false_alarm_period_end_basis=float(c[col].notna().mean()),
                         detection_filed_before_event=d, det_ci_lo=dlo, det_ci_hi=dhi,
                         false_alarm_filed_before_event=a, fa_ci_lo=alo, fa_ci_hi=ahi,
                         alarms_filed_after_event=int(f[col].notna().sum() - out["f"][0].sum()),
                         median_lead_quarters_period_end=float(main_lead.median()) if len(main_lead) else np.nan,
                         median_lead_quarters_filing_date=float(np.median(out["f"][1])) if out["f"][1] else np.nan,
                         median_reporting_lag_days=float(np.median(out["f"][2])) if out["f"][2] else np.nan))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_filing_lead.csv", index=False)
    print(t.round(3).to_string(index=False))


# ------------------------------------------------------------------ 3d + 3c: TTM overlap, error autocorrelation, horizon
def ttm(cfg, paths, F):
    R, f, c = _results(paths); B = ev.ci_resamples(cfg); rg = rng(cfg["post_hoc"]["seed_offsets"]["ttm"])
    names = TTM_VARIANTS
    rows, acf_rows = [], []
    for name in names:
        res = {g: [F[_key(r)]["ttm"].get(name, {"eligible": False}) for r in d.itertuples()] for g, d in [("f", f), ("c", c)]}
        ef = [x for x in res["f"] if x["eligible"]]
        # controls count only where their failed firm is also eligible under this variant
        ok_f = {int(r.cik) for r, x in zip(f.itertuples(), res["f"]) if x["eligible"]}
        ec = [x for r, x in zip(c.itertuples(), res["c"]) if x["eligible"] and int(r.matched_to) in ok_f]
        det = np.array([x["first"] is not None for x in ef], float); fa = np.array([x["first"] is not None for x in ec], float)
        d, dlo, dhi = ev.rate_ci(det, rg, B); a, alo, ahi = ev.rate_ci(fa, rg, B)
        sf, sc = np.array([x["score"] for x in ef]), np.array([x["score"] for x in ec])
        A = ev.auc(sf, sc)
        bs = [ev.auc(sf[rg.integers(0, len(sf), len(sf))], sc[rg.integers(0, len(sc), len(sc))]) for _ in range(B)]
        steps = np.array([x["steps"] for x in ef + ec])
        rows.append(dict(input=name, failed_firms=len(ef), controls=len(ec), detection=d, det_ci_lo=dlo, det_ci_hi=dhi,
                         false_alarm=a, fa_ci_lo=alo, fa_ci_hi=ahi, auc=A, auc_ci_lo=float(np.nanquantile(bs, .025)),
                         auc_ci_hi=float(np.nanquantile(bs, .975)), median_monitored_steps=float(np.median(steps)),
                         min_monitored_steps=int(steps.min()), share_full_16_steps=float((steps == cfg["detectors"]["monitor_horizon"]).mean())))
        for part in ("z_train", "z_mon"):
            for L in cfg["post_hoc"]["acf_lags"]:
                per, A_, B_ = [], [], []
                for x in ec:
                    a_, b_ = _acf(x[part], [L])[L]
                    if len(a_) >= 6:
                        per.append(np.corrcoef(a_, b_)[0, 1]); A_.append(a_); B_.append(b_)
                pooled = np.corrcoef(np.concatenate(A_), np.concatenate(B_))[0, 1] if A_ else np.nan
                acf_rows.append(dict(input=name, errors="training (in-sample)" if part == "z_train" else "monitored window",
                                     lag_quarters=L, controls=len(per), median_per_firm_acf=float(np.nanmedian(per)) if per else np.nan,
                                     pooled_acf=float(pooled)))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_ttm_overlap.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(paths.tables / "posthoc_kalman_acf.csv", index=False)
    print(t.round(3).to_string(index=False)); print(pd.DataFrame(acf_rows).round(3).to_string(index=False))
    # 3c: the bootstrap calibrates h for `monitor_horizon` steps; how many cash-ratio steps are really monitored?
    base = names[0]
    st = np.array([F[_key(r)]["ttm"][base]["steps"] for d in (f, c) for r in d.itertuples() if F[_key(r)]["ttm"][base]["eligible"]])
    hh = np.array([F[_key(r)]["ttm"][base]["h"] for d in (f, c) for r in d.itertuples() if F[_key(r)]["ttm"][base]["eligible"]])
    hm = pd.concat([f, c])["h_cash_0.10"].to_numpy(float)
    rows = [dict(item="calibration horizon (detectors.monitor_horizon)", value=cfg["detectors"]["monitor_horizon"]),
            dict(item="EDGAR monitored quarters (edgar.monitor_quarters)", value=cfg["edgar"]["monitor_quarters"]),
            dict(item="simulation monitored periods (simulation.monitor_length)", value=cfg["simulation"]["monitor_length"]),
            dict(item="EDGAR cash-ratio steps actually monitored: median", value=float(np.median(st))),
            dict(item="EDGAR cash-ratio steps actually monitored: 10th percentile", value=float(np.quantile(st, .10))),
            dict(item="EDGAR cash-ratio steps actually monitored: minimum", value=int(st.min())),
            dict(item="EDGAR share of analysed firms with all monitored steps observed", value=float((st == cfg["detectors"]["monitor_horizon"]).mean())),
            dict(item="recomputed h_cash equals the main run's h_cash (share of analysed firms)",
                 value=float(np.mean(np.isclose(hh, hm, equal_nan=True))))]
    sim = paths.processed / "sim_runs.pkl.gz"
    if sim.exists():
        from src.simulate import analysis as A
        with gzip.open(sim, "rb") as fh:
            Rs = A.Runs(pickle.load(fh)["runs"], cfg)
        n = cfg["simulation"]["h5_training_length"]; a = cfg["detectors"]["tiers"]["amber_alpha"]
        P_ = np.vstack([r["paths"][("H", "kalman")] for r in Rs.firms("heldout", n)]); h_ = Rs.h("heldout", n, f"h_kalman_{a:.2f}")
        okh = np.isfinite(h_)
        for m in sorted({int(np.quantile(st, .10)), int(np.median(st)), cfg["detectors"]["monitor_horizon"]}):
            rows.append(dict(item=f"simulation (n={n}, nominal {a:.0%}, uncorrected): false alarms when only the first {m} of "
                                  f"{cfg['detectors']['monitor_horizon']} calibrated steps are monitored",
                             value=float((P_[okh, :m].max(axis=1) > h_[okh]).mean())))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_calibration_horizon.csv", index=False)
    print(t.to_string(index=False))


# ------------------------------------------------------------------ 3e: winsorisation
def _w(task):
    warnings.filterwarnings("ignore")
    from src.run_edgar import firm_run
    cik, role, event_q, matched_to, cfg, paths, inflation, caps = task
    row, _ = firm_run(cik, role, event_q, matched_to, cfg, paths, inflation, winsor=caps)
    return {k: row.get(k) for k in ["cik", "role", "matched_to", "event_q", "status", *ev.METHODS.values()]}


def _training_rows(task):
    warnings.filterwarnings("ignore")
    from src.run_edgar import prepare
    cik, role, event_q, cols, cfg, paths = task
    st, q, r = prepare(cik, event_q, cfg, paths)
    if st is not None:
        return None
    return r.loc[r.index < event_q - cfg["edgar"]["monitor_quarters"] + 1, cols]


def winsor(cfg, paths, workers):
    from src.detectors.monitor import load_inflation
    _, tasks = _panel(paths)
    cols = list(dict.fromkeys([cfg["masking"]["cash_ratio"], *cfg["masking"]["structural_ratios"], "neg_dso"]))
    with Pool(workers) as pool:
        tr = pool.map(_training_rows, [(t[0], t[1], t[2], cols, cfg, paths) for t in tasks], chunksize=4)
    pool_ = pd.concat([x for x in tr if x is not None])
    lo_q, hi_q = cfg["post_hoc"]["winsor_quantiles"]
    caps = {c: (float(pool_[c].quantile(lo_q)), float(pool_[c].quantile(hi_q))) for c in cols}
    inflation = load_inflation(paths.tables)
    with Pool(workers) as pool:
        W = pd.DataFrame(pool.map(_w, [(t[0], t[1], t[2], t[3], cfg, paths, inflation, caps) for t in tasks], chunksize=4))
    W.to_csv(paths.processed / "posthoc_winsor_runs.csv", index=False)
    fW, cW = ev.analysed(W); _, f, c = _results(paths)
    B = ev.ci_resamples(cfg); rg = rng(cfg["post_hoc"]["seed_offsets"]["winsor"]); rows = []
    for m, col in ev.METHODS.items():
        d, dlo, dhi = ev.rate_ci(fW[col].notna(), rg, B); a, alo, ahi = ev.rate_ci(cW[col].notna(), rg, B)
        rows.append(dict(method=m, failed_firms=len(fW), controls=len(cW), detection=d, det_ci_lo=dlo, det_ci_hi=dhi,
                         false_alarm=a, fa_ci_lo=alo, fa_ci_hi=ahi, base_detection=float(f[col].notna().mean()),
                         base_false_alarm=float(c[col].notna().mean()), base_failed_firms=len(f), base_controls=len(c)))
    t = pd.DataFrame(rows)
    for cname, (lo, hi) in caps.items():
        t[f"cap_{cname}"] = f"[{lo:.3g}, {hi:.3g}]"
        t[f"share_training_capped_{cname}"] = float(((pool_[cname] < lo) | (pool_[cname] > hi)).mean())
    t.to_csv(paths.tables / "posthoc_winsor.csv", index=False)
    print(t.iloc[:, :12].round(3).to_string(index=False))


# ------------------------------------------------------------------ 3f: industries
def industry(cfg, paths):
    R, f, c = _results(paths)
    div = cfg["post_hoc"]["sic_divisions"]
    name = lambda s: next((d["name"] for d in div if d["from"] <= int(s) // 100 <= d["to"]), "unknown")  # noqa: E731
    rows = []
    for role, g in [("failed", f), ("control", c)]:
        for k, n in g.sic.map(name).value_counts().items():
            rows.append(dict(sample="EDGAR (analysed)", group=k, role=role, firms=int(n)))
    co = pd.read_csv(paths.raw / "cohort" / "cohort.csv")
    for (s, role), n in co.groupby(["sector", "role"]).size().items():
        rows.append(dict(sample="India cohort", group=s, role=role, firms=int(n)))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_industry.csv", index=False)
    print(t.to_string(index=False))


# ------------------------------------------------------------------ 3g: healthy-centred masking index
def mi(cfg, paths):
    pk = pd.read_csv(paths.tables / "sim_peak_mi_by_firm.csv", index_col=0)
    Hh = pk["H"].dropna(); med = float(Hh.median()); pq = cfg["post_hoc"]["mi_healthy_percentiles"]
    cut = {p: float(Hh.quantile(p)) for p in pq}
    rows = []
    for s in pk.columns:
        x = pk[s].dropna()
        rows.append(dict(sample="simulation, n = 28, held-out", group=s, firms=len(x), median_peak_mi=float(x.median()),
                         healthy_median=med, median_centred_mi=float(x.median() - med), share_mi_above_0=float((x > 0).mean()),
                         **{f"share_above_healthy_p{int(100 * p)}": float((x > cut[p]).mean()) for p in pq},
                         **{f"healthy_p{int(100 * p)}": cut[p] for p in pq},
                         auc_vs_healthy=ev.auc(x, Hh) if s != "H" else np.nan))
    R, f, c = _results(paths)
    cm = c.peak_masking_index.dropna(); cmed = float(cm.median()); ccut = {p: float(cm.quantile(p)) for p in pq}
    groups = [("healthy controls", cm)] + [(f"failed: {a}", g.peak_masking_index.dropna()) for a, g in f.groupby("arch_archetype")]
    for gname, x in groups:
        rows.append(dict(sample="EDGAR (analysed)", group=gname, firms=len(x), median_peak_mi=float(x.median()), healthy_median=cmed,
                         median_centred_mi=float(x.median() - cmed), share_mi_above_0=float((x > 0).mean()),
                         **{f"share_above_healthy_p{int(100 * p)}": float((x > ccut[p]).mean()) for p in pq},
                         **{f"healthy_p{int(100 * p)}": ccut[p] for p in pq},
                         auc_vs_healthy=ev.auc(x, cm) if gname != "healthy controls" else np.nan))
    t = pd.DataFrame(rows); t.to_csv(paths.tables / "posthoc_mi_healthy.csv", index=False)
    print(t.round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(9, max(1, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    want = set(a.only or ["paired", "horizon", "persistence", "filing", "ttm", "winsor", "industry", "mi"])
    if "paired" in want: paired(cfg, paths)
    if "industry" in want: industry(cfg, paths)
    if "mi" in want: mi(cfg, paths)
    parts = {p for p, need in [("flags", {"horizon", "persistence"}), ("filing", {"filing"}), ("ttm", {"ttm"})] if need & want}
    if parts:
        F = _collect(cfg, paths, a.workers, parts)
        if "horizon" in want: horizon(cfg, paths, F)
        if "persistence" in want: persistence(cfg, paths, F)
        if "filing" in want: filing(cfg, paths, F)
        if "ttm" in want: ttm(cfg, paths, F)
    if "winsor" in want: winsor(cfg, paths, a.workers)


if __name__ == "__main__":
    main()
