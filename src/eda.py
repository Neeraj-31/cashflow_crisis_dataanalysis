"""Design-choice EDA: the data properties behind each design choice (config `eda`, added 2026-11-25 AFTER every
result had been seen). Descriptive and retrospective: no detector, threshold or verdict reads anything written here.

    python -m src.eda

EDGAR statistics use each firm's TRAINING quarters only (before the 16-quarter monitored window), so nothing here
looks at the period in which alarms are counted (except eda_event_time.csv, which is labelled as such).

Tables -> reports/tables/eda_*.csv
  eda_history_length.csv    how much history each data source gives (usable training periods of the cash ratio)
  eda_ratio_properties.csv  per candidate ratio: coverage, persistence (ACF), tails, outliers, extreme values
  eda_seasonality.csv       cash ratio: single-quarter vs trailing-twelve-month (TTM) flows, ACF of changes by lag
  eda_denominators.csv      how often interest / revenue are small relative to total assets (near-zero masking rule)
  eda_covariance.csv        Layer 4 training innovations: sample size vs dimension, condition numbers, shrinkage
  eda_monte_carlo.csv       Monte Carlo standard error implied by each resample count
  eda_event_time.csv        median cash ratio of failed firms vs controls by quarter before the event (analysed sample)
"""
from __future__ import annotations

import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src import evaluation as ev
from src.config import Paths, load_config
from src.detectors import statespace as ss

MAD_TO_SD = 1.4826   # consistency constant: 1.4826 x MAD estimates the SD of a normal distribution (not a threshold)


# ------------------------------------------------------------------ small helpers
def acf(x: pd.Series, lag: int, min_values: int) -> float:
    """Autocorrelation at `lag` over pairs where both values exist (gaps are never filled)."""
    x = pd.Series(np.asarray(x, float))
    pairs = (x.notna() & x.shift(lag).notna()).sum()
    return float(x.autocorr(lag)) if pairs >= min_values else np.nan


def outlier_share(d: pd.Series, zmax: float, min_values: int) -> float:
    d = pd.Series(d).dropna()
    if len(d) < min_values:
        return np.nan
    mad = MAD_TO_SD * (d - d.median()).abs().median()
    if not mad > 0:
        return np.nan
    return float(((d - d.median()).abs() / mad > zmax).mean())


def excess_kurtosis(d: pd.Series, min_values: int) -> float:
    d = pd.Series(d).dropna()
    return float(d.kurt()) if len(d) >= min_values else np.nan


def _q(s, p):
    s = pd.Series(s).dropna()
    return float(s.quantile(p)) if len(s) else np.nan


# ------------------------------------------------------------------ per-firm extraction
def firm_training(cik, event_q, cfg, paths, flows="ttm"):
    """(canonical quarters, oriented ratios) restricted to the training quarters, or (None, None)."""
    from src.run_edgar import prepare
    status, q, r = prepare(cik, event_q, cfg, paths, flows=flows)
    if status is not None:
        return None, None
    start = event_q - cfg["edgar"]["monitor_quarters"] + 1
    return q[q.index < start], r[r.index < start]


def _one(args):
    cik, role, event_q = args
    cfg = load_config(); paths = Paths.from_config(cfg)
    q, r = firm_training(cik, event_q, cfg, paths)
    _, r1 = firm_training(cik, event_q, cfg, paths, flows="quarter")
    return (cik, role), (None if q is None else dict(q=q, r=r, r_quarter=r1, event_q=event_q))


def collect(cfg, paths, workers=None):
    R = pd.read_csv(paths.tables / "edgar_firm_results.csv")
    fa, ca = ev.analysed(R)
    an = pd.concat([fa, ca])[["cik", "role", "event_q"]]
    tasks = [(int(p.cik), p.role, int(p.event_q)) for p in an.itertuples()]
    with Pool(workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(_one, tasks)
    return R, fa, ca, {k: v for k, v in res if v is not None}


# ------------------------------------------------------------------ tables
def history_length(cfg, paths, R, fa, ca):
    rows = []
    reached = R[R.status.eq("ok") | R.status.str.startswith("too little training")]

    def add(sample, unit, s, note=""):
        s = pd.Series(s, dtype=float).dropna()
        rows.append(dict(sample=sample, unit=unit, firms=len(s), median=s.median(), p10=_q(s, .1), p90=_q(s, .9),
                         min=s.min(), max=s.max(),
                         share_at_least_min_periods=float((s >= cfg["detectors"]["min_periods_layer3"]).mean()),
                         share_at_least_h5_n=float((s >= cfg["simulation"]["h5_training_length"]).mean()), note=note))
    add("EDGAR: every firm that passed Layer 0", "quarters", reached.usable_training, "usable cash-ratio training quarters")
    add("EDGAR: analysed failed firms", "quarters", fa.usable_training)
    add("EDGAR: analysed controls", "quarters", ca.usable_training)
    co = pd.read_csv(paths.tables / "cohort_archetypes.csv")
    for role in ["failed", "control"]:
        add(f"India cohort: {role}", "years", co[co.role == role].usable_training_before_window,
            "usable annual training values before the monitored window")
    rn = pd.read_csv(paths.tables / "ratios_nmc.csv", index_col=0)
    add("NMC Health", "years", [rn[cfg["masking"]["cash_ratio"]].notna().sum()], "all annual values, no training split")
    rj = pd.read_csv(paths.tables / "ratios_jyoti.csv", index_col=0)
    tr = rj[rj.index <= cfg["firms"]["jyoti"]["training_end_fy"]]
    add("Jyoti Structures", "years", [tr[cfg["masking"]["cash_ratio"]].notna().sum()],
        f"annual values up to the training end FY{cfg['firms']['jyoti']['training_end_fy']}")
    return pd.DataFrame(rows)


def ratio_properties(cfg, data):
    E = cfg["eda"]; mv = E["min_values"]; need = cfg["detectors"]["min_periods_layer3"]
    big = cfg["sanity"]["extreme_ratio_abs"]
    rows = []
    for ratio in E["ratios"]:
        for role in ["failed", "control", "all analysed"]:
            per = []
            for (cik, rl), d in data.items():
                if role != "all analysed" and rl != role:
                    continue
                x = d["r"][ratio] if ratio in d["r"] else pd.Series(dtype=float)
                n = len(d["r"]); obs = int(x.notna().sum()); dx = x.diff()
                per.append(dict(n=n, obs=obs, usable=obs >= need, acf1=acf(x, 1, mv), dacf1=acf(dx, 1, mv),
                                kurt_change=excess_kurtosis(dx, mv), out=outlier_share(dx, E["outlier_robust_z"], mv),
                                big=int((x.abs() > big).sum())))
            P = pd.DataFrame(per)
            rows.append(dict(ratio=ratio, role=role, firms=len(P), training_quarters=int(P.n.sum()),
                             share_quarters_observed=P.obs.sum() / max(1, P.n.sum()),
                             share_firms_with_min_periods=float(P.usable.mean()),
                             firms_with_acf=int(P.acf1.notna().sum()),
                             median_acf1_level=P.acf1.median(), median_acf1_change=P.dacf1.median(),
                             median_excess_kurtosis_change=P.kurt_change.median(),
                             share_firms_kurtosis_above_normal_x3=float((P.kurt_change.dropna() > 3).mean()) if P.kurt_change.notna().any() else np.nan,
                             median_outlier_share_change=P.out.median(),
                             share_quarters_abs_above_extreme=P.big.sum() / max(1, P.obs.sum())))
    return pd.DataFrame(rows)


def seasonality(cfg, data):
    E = cfg["eda"]; mv = E["min_values"]; cash = cfg["masking"]["cash_ratio"]
    rows = []
    for label, key in [("TTM flows (pre-registered input)", "r"), ("single-quarter flows x 4 (no smoothing)", "r_quarter")]:
        acfs = {l: [] for l in E["acf_lags"]}; sd = []
        for d in data.values():
            r = d[key]
            if r is None or cash not in r:
                continue
            dx = r[cash].diff()
            for l in E["acf_lags"]:
                acfs[l].append(acf(dx, l, mv))
            if dx.notna().sum() >= mv:
                sd.append(float(dx.std()))
        for l in E["acf_lags"]:
            a = pd.Series(acfs[l]).dropna()
            rows.append(dict(input=label, lag_quarters=l, firms=len(a), median_acf_of_change=a.median(),
                             share_firms_positive=float((a > 0).mean()), median_sd_of_change=float(np.median(sd)) if sd else np.nan))
    return pd.DataFrame(rows)


def denominators(cfg, data):
    eps = cfg["ratios"]["denominator_eps_rel_assets"]
    rows = []
    for item in ["interest", "revenue"]:
        for role in ["failed", "control"]:
            num = tot = 0; cnt = {k: 0 for k in eps}
            for (cik, rl), d in data.items():
                if rl != role:
                    continue
                q = d["q"]
                if item not in q:
                    continue
                v, ta = q[item].abs(), q.total_assets.abs()
                ok = v.notna() & ta.notna() & (ta > 0)
                tot += int(ok.sum())
                for k, e_ in eps.items():
                    cnt[k] += int((v[ok] <= e_ * ta[ok]).sum())
            rows.append(dict(item=item, role=role, firm_quarters=tot,
                             **{f"share_below_{k}_eps": cnt[k] / max(1, tot) for k in eps},
                             **{f"eps_{k}": e_ for k, e_ in eps.items()}))
    return pd.DataFrame(rows)


def covariance(cfg, data, fa, ca):
    """Layer 4 training innovations, built exactly as src.detectors.monitor does (cash + structural ratios with
    >= min_periods usable training values; complete rows only)."""
    cash, struct = cfg["masking"]["cash_ratio"], cfg["masking"]["structural_ratios"]
    need = cfg["detectors"]["min_periods_layer3"]
    rows = []
    for (cik, role), d in data.items():
        r = d["r"]
        avail = [c for c in struct if c != cash and c in r and r[c].notna().sum() >= need]
        if not avail or r[cash].notna().sum() < need:
            continue
        Y = r[[cash] + avail].to_numpy(float); n = len(r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = np.column_stack([ss.kalman_z(Y[:, j], n)[0] for j in range(Y.shape[1])])
        Zt = Z[~np.isnan(Z).any(axis=1)]
        if len(Zt) < 4:
            continue
        S = np.cov(Zt, rowvar=False); lw = LedoitWolf().fit(Zt)
        rows.append(dict(cik=cik, role=role, p=Zt.shape[1], n_complete=len(Zt), n_over_p=len(Zt) / Zt.shape[1],
                         cond_sample=float(np.linalg.cond(S)), cond_ledoit_wolf=float(np.linalg.cond(lw.covariance_)),
                         shrinkage=float(lw.shrinkage_)))
    F = pd.DataFrame(rows)
    out = []
    for p_, g in [("all", F)] + [(str(int(k)), g) for k, g in F.groupby("p")]:
        out.append(dict(dimension_p=p_, firms=len(g), median_n_complete=g.n_complete.median(), min_n_complete=g.n_complete.min(),
                        median_n_over_p=g.n_over_p.median(), median_cond_sample=g.cond_sample.median(),
                        median_cond_ledoit_wolf=g.cond_ledoit_wolf.median(), median_shrinkage=g.shrinkage.median(),
                        p90_cond_sample=_q(g.cond_sample, .9), p90_cond_ledoit_wolf=_q(g.cond_ledoit_wolf, .9)))
    return pd.DataFrame(out)


def monte_carlo(cfg, paths):
    dc, sc = cfg["detectors"], cfg["simulation"]
    meta = pd.read_csv(paths.tables / "sim_meta.csv").iloc[0]
    rows = []
    for what, B in [("bootstrap threshold h (detectors.bootstrap.n_resamples)", dc["bootstrap"]["n_resamples"]),
                    ("EDGAR CIs (evaluation.ci_resamples)", cfg["evaluation"]["ci_resamples"]),
                    ("simulation CIs (simulation.ci_resamples)", sc["ci_resamples"]),
                    ("parametric-bootstrap refits (decision #15)", int(meta.parametric_B))]:
        for a in dc["nominal_false_alarm_rates"]:
            rows.append(dict(use=what, resamples=int(B), tail_probability=a,
                             mc_se_points=100 * np.sqrt(a * (1 - a) / B)))
    rows.append(dict(use=f"simulated false-alarm rate, {sc['firms_per_cell']} firms per cell (sampling SE, not Monte Carlo)",
                     resamples=sc["firms_per_cell"], tail_probability=dc["nominal_false_alarm_rates"][-1],
                     mc_se_points=100 * np.sqrt(dc["nominal_false_alarm_rates"][-1] * (1 - dc["nominal_false_alarm_rates"][-1]) / sc["firms_per_cell"])))
    return pd.DataFrame(rows)


def event_time(cfg, paths, fa, ca):
    L1 = pd.read_csv(paths.tables / "edgar_layer1_panel.csv"); cash = cfg["masking"]["cash_ratio"]
    L1 = pd.concat([L1[(L1.role == "failed") & L1.cik.isin(fa.cik)], L1[(L1.role == "control") & L1.cik.isin(ca.cik)]])
    rows = []
    for t in cfg["eda"]["event_time_taus"]:
        g = L1[L1.tau == t]
        f, c = g[g.role == "failed"][cash].dropna(), g[g.role == "control"][cash].dropna()
        rows.append(dict(tau_quarters=t, failed_firms=len(f), control_firms=len(c), failed_median=f.median(),
                         control_median=c.median(), gap_control_minus_failed=c.median() - f.median(),
                         in_monitored_window=t > -cfg["edgar"]["monitor_quarters"]))
    return pd.DataFrame(rows)


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    R, fa, ca, data = collect(cfg, paths)
    out = {"eda_history_length": history_length(cfg, paths, R, fa, ca),
           "eda_ratio_properties": ratio_properties(cfg, data),
           "eda_seasonality": seasonality(cfg, data),
           "eda_denominators": denominators(cfg, data),
           "eda_covariance": covariance(cfg, data, fa, ca),
           "eda_monte_carlo": monte_carlo(cfg, paths),
           "eda_event_time": event_time(cfg, paths, fa, ca)}
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for k, v in out.items():
        v.to_csv(paths.tables / f"{k}.csv", index=False)
        print(f"\n=== {k} ===\n{v.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
