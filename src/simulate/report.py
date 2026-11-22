"""Write simulation tables (reports/tables/sim_*.csv) and figures (reports/figures/sim_*.png).

    python -m src.simulate.report [--tables] [--figures] [--runs PATH]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Paths, load_config  # noqa: E402
from src.reporting.style import C, STYLE  # noqa: E402
from src.simulate import analysis as A  # noqa: E402

DET_COL = {"kalman": C["navy"], "arima": C["orange"], "rw": C["slate"], "mcusum": C["coral"], "kalman_cusum": C["navy"]}
DET_LAB = {"kalman": "Kalman + CUSUM", "arima": "ARIMA(1,0,0) + CUSUM", "rw": "Random walk + CUSUM",
           "mcusum": "Multivariate (Crosier)", "kalman_cusum": "Kalman + CUSUM"}
METH = {"block_bootstrap": ("Block bootstrap (uncorrected)", C["coral"]), "inflation_factor": ("Inflation factor", C["teal"]),
        "parametric_bootstrap": ("Parametric bootstrap", C["purple"]), "conformal": ("Conformal", C["navy"])}


def tables(cfg, paths, runs_path):
    d = A.load(runs_path)
    R = A.Runs(d["runs"], cfg)
    t = {}
    t["sim_calibration_curve"] = A.calibration_curve(R)
    t["sim_fixes"], t["sim_inflation_fit"] = A.fixes(R)
    t["sim_matched_fa"] = A.matched_comparison(R)
    t["sim_power_drift"], t["sim_power_n"] = A.power_curves(R)
    t["sim_masking_summary"], t["sim_fingerprints"], t["sim_masking_separation"], peak = A.masking(R)
    t["sim_hypotheses"] = A.verdicts(t["sim_calibration_curve"], t["sim_fixes"], t["sim_matched_fa"], cfg)
    blocks = {n: R.firms("heldout", n)[0]["block"] for n in cfg["simulation"]["training_lengths"]}
    t["sim_meta"] = pd.DataFrame([{**d["meta"], "training_lengths": str(cfg["simulation"]["training_lengths"]),
                                   "block_length_by_n": str(blocks), "monitor_length": cfg["simulation"]["monitor_length"],
                                   "bootstrap_resamples": cfg["detectors"]["bootstrap"]["n_resamples"],
                                   "failed_firm_runs": len(R.errors)}])
    pd.DataFrame({k: pd.Series(v) for k, v in peak.items()}).to_csv(paths.tables / "sim_peak_mi_by_firm.csv", index_label="firm")
    for k, v in t.items():
        v.to_csv(paths.tables / f"{k}.csv", index=False)
    print(t["sim_hypotheses"].to_string())


def figures(cfg, paths):
    T = lambda k: pd.read_csv(paths.tables / f"{k}.csv")  # noqa: E731
    cal, fix, pdr, pn = T("sim_calibration_curve"), T("sim_fixes"), T("sim_power_drift"), T("sim_power_n")
    ms, fp = T("sim_masking_summary"), T("sim_fingerprints")
    peak = pd.read_csv(paths.tables / "sim_peak_mi_by_firm.csv", index_col=0)
    with plt.rc_context(STYLE):
        # ---- calibration curve (headline)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        k = cal[cal.detector == "kalman_cusum"]
        cmap = plt.get_cmap("viridis")
        ns = sorted(k.n.unique())
        for i, n in enumerate(ns):
            s = k[k.n == n].sort_values("nominal")
            ax[0].errorbar(s.nominal * 100, s.empirical_fa * 100, yerr=[(s.empirical_fa - s.ci_lo) * 100, (s.ci_hi - s.empirical_fa) * 100],
                           marker="o", capsize=3, color=cmap(i / max(1, len(ns) - 1)), label=f"n = {n}")
        lim = [0, max(45, k.ci_hi.max() * 105)]
        ax[0].plot(lim, lim, color=C["ink"], ls=":", lw=1, label="perfect calibration")
        ax[0].set_xlim(3, 12); ax[0].set_ylim(lim); ax[0].set_xlabel("nominal false-alarm rate (%)"); ax[0].set_ylabel("empirical false-alarm rate (%)")
        ax[0].set_title("A. Empirical vs nominal (Kalman + CUSUM, block bootstrap)"); ax[0].legend(fontsize=8)
        for det, ls in [("kalman_cusum", "-"), ("mcusum", "--")]:
            for a, col in [(0.05, C["orange"]), (0.10, C["coral"])]:
                s = cal[(cal.detector == det) & (cal.nominal == a)].sort_values("n")
                ax[1].plot(s.n, s.empirical_fa * 100, ls, marker="o", color=col, label=f"{DET_LAB[det] if det != 'kalman_cusum' else 'Kalman + CUSUM'}, nominal {a:.0%}")
                ax[1].fill_between(s.n, s.ci_lo * 100, s.ci_hi * 100, color=col, alpha=.12, lw=0)
            ax[1].axhline(5, color=C["orange"], lw=.8, ls=":"); ax[1].axhline(10, color=C["coral"], lw=.8, ls=":")
        ax[1].set_xlabel("training length n (periods)"); ax[1].set_ylabel("empirical false-alarm rate (%)"); ax[1].set_xticks(ns)
        ax[1].set_title("B. False-alarm rate by training length"); ax[1].legend(fontsize=7.5)
        fig.suptitle("Calibration of bootstrap CUSUM thresholds on simulated healthy firms (held-out, 95% bootstrap CIs)", fontsize=12.5, weight="bold", x=.01, ha="left")
        fig.tight_layout(); fig.savefig(paths.figures / "sim_calibration_curve.png"); plt.close(fig)

        # ---- calibration before/after correction
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
        for a_i, a in enumerate((0.05, 0.10)):
            x = ax[a_i]
            x.axhspan((a - .03) * 100, (a + .03) * 100, color=C["green"], alpha=.15, lw=0, label="nominal +/- 3 points")
            x.axhline(a * 100, color=C["ink"], ls=":", lw=1)
            for m, (lab, col) in METH.items():
                s = fix[(fix.nominal == a) & (fix.method == m)].sort_values("n")
                if s.empty: continue
                x.plot(s.n, s.empirical_fa * 100, "-o", color=col, label=lab)
                x.fill_between(s.n, s.ci_lo * 100, s.ci_hi * 100, color=col, alpha=.12, lw=0)
            x.set_title(f"Nominal {a:.0%}"); x.set_xlabel("training length n"); x.set_xticks(sorted(fix.n.unique()))
        ax[0].set_ylabel("empirical false-alarm rate on held-out firms (%)"); ax[1].legend(fontsize=8)
        fig.suptitle("Calibration before and after correction (Kalman + CUSUM)", fontsize=12.5, weight="bold", x=.01, ha="left")
        fig.tight_layout(); fig.savefig(paths.figures / "sim_calibration_fixes.png"); plt.close(fig)

        # ---- power curves
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        for i, scen in enumerate(["S1", "S2"]):
            for det in ["arima", "kalman", "rw", "mcusum"]:
                s = pdr[(pdr.scenario == scen) & (pdr.detector == det)].sort_values("drift")
                ax[i].plot(s.drift, s.detection * 100, "-o", color=DET_COL[det], label=DET_LAB[det])
                ax[i].fill_between(s.drift, s.ci_lo * 100, s.ci_hi * 100, color=DET_COL[det], alpha=.1, lw=0)
            ax[i].set_xlabel("drift per period (innovation SD)")
            ax[i].set_title({"S1": "S1: drift in the target (cash) ratio", "S2": "S2: drift in the other ratios"}[scen])
        ax[0].set_ylabel("detection rate at matched 10% FA (%)"); ax[0].legend(fontsize=8)
        for scen, ls in [("S1", "-"), ("S2", "--"), ("S4", ":")]:
            for det in ["kalman", "mcusum"]:
                s = pn[(pn.scenario == scen) & (pn.detector == det)].sort_values("n")
                ax[2].plot(s.n, s.detection * 100, ls, marker="o", color=DET_COL[det], label=f"{scen}, {DET_LAB[det]}")
        ax[2].set_xlabel("training length n"); ax[2].set_title("Detection by training length"); ax[2].legend(fontsize=7.5)
        fig.suptitle("Power curves at a matched 10% false-alarm rate (held-out simulated firms)", fontsize=12.5, weight="bold", x=.01, ha="left")
        fig.tight_layout(); fig.savefig(paths.figures / "sim_power_curves.png"); plt.close(fig)

        # ---- masking index
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        order = ["H", "S1", "S2", "S3", "S4"]
        bp = ax[0].boxplot([peak[s].dropna() for s in order], labels=order, showfliers=False, patch_artist=True)
        for patch, s in zip(bp["boxes"], order):
            patch.set_facecolor({"S1": C["navy"], "S2": C["coral"], "S4": C["purple"]}.get(s, C["sky"])); patch.set_alpha(.6)
        ax[0].axhline(0, color=C["ink"], lw=.8); ax[0].set_ylabel("peak masking index"); ax[0].set_title("Peak MI by scenario (positive = structure leads)")
        f = fp.set_index("scenario").reindex(order)
        bottom = np.zeros(len(order))
        cols = {"cash_only": C["navy"], "cash_first": C["blue"], "both_together": C["amber"], "structural_first": C["orange"],
                "structural_only": C["coral"], "neither": C["nodata"]}
        for c in f.columns:
            ax[1].bar(order, f[c] * 100, bottom=bottom, color=cols.get(c, C["slate"]), label=c.replace("_", " "))
            bottom += f[c].to_numpy() * 100
        ax[1].set_ylabel("% of firms"); ax[1].set_title("Which detector alarms first"); ax[1].legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.suptitle("Masking index and detector-order fingerprint in simulation (n = 28, amber thresholds)", fontsize=12.5, weight="bold", x=.01, ha="left")
        fig.tight_layout(); fig.savefig(paths.figures / "sim_masking_index.png"); plt.close(fig)
    print("saved simulation figures")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true"); ap.add_argument("--figures", action="store_true")
    ap.add_argument("--runs", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    both = not (a.tables or a.figures)
    if a.tables or both: tables(cfg, paths, a.runs or paths.processed / "sim_runs.pkl.gz")
    if a.figures or both: figures(cfg, paths)


if __name__ == "__main__":
    main()
