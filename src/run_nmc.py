"""NMC Health annual EDA (2013-2018). Refactored from nmc_eda.py.

    python -m src.run_nmc [--tables] [--figures] [--out-root DIR]

With neither flag, does both. Tables -> reports/tables, figures -> reports/figures.
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from src.config import Paths, load_config
from src.ratios import nmc


def load(paths: Paths):
    raw = pd.read_csv(paths.raw / "nmc_raw.csv", parse_dates=["period_end"]).set_index("period_end")
    r = pd.read_csv(paths.raw / "nmc_ratios.csv", index_col=0, parse_dates=True)
    return raw, r


def tables(paths: Paths, cfg: dict) -> None:
    raw, r = load(paths)
    x = nmc.redflag_metrics(raw)
    summ = nmc.ratio_summary(r)
    z = nmc.baseline_zscores(r, cfg["baseline_end"])
    corr = nmc.correlations_with(r, "ocf_to_cl")

    x.to_csv(paths.tables / "nmc_redflags.csv")
    summ.to_csv(paths.tables / "nmc_ratio_summary.csv")
    z.to_csv(paths.tables / "nmc_ratio_zscores_vs_baseline.csv")
    corr.rename("corr_with_ocf_to_cl").to_csv(paths.tables / "nmc_corr_ocf_to_cl.csv")

    pd.set_option("display.width", 200)
    print("=== MISSING DATA ===\n", raw.isna().sum().sum(), "missing cells | n =", len(raw))
    print("\n=== SUMMARY (ratios) ===\n", summ.round(3))
    print("\n=== YoY / RED-FLAG METRICS ===\n", x.round(3).T)
    print(f"\n=== Z-SCORE OF EACH RATIO vs BASELINE (to {cfg['baseline_end']}) ===")
    print(z.round(2))
    print(f"\n=== TOP CORRELATIONS WITH ocf_to_cl (n={len(r)}, indicative only) ===")
    print(corr.round(2).to_string())


def figures(paths: Paths, cfg: dict) -> None:
    from src.reporting import nmc_figures
    raw, r = load(paths)
    nmc_figures.make_all(raw, r, nmc.redflag_metrics(raw), cfg, paths.figures)
    print("Saved NMC figures to", paths.figures)


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", action="store_true"); ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out-root", default=None, help="redirect outputs (tests)")
    a = ap.parse_args(argv)
    full = load_config(); cfg = full["nmc"]
    paths = Paths.from_config(full, out_root=a.out_root).ensure()
    both = not (a.tables or a.figures)
    if a.tables or both: tables(paths, cfg)
    if a.figures or both: figures(paths, cfg)


if __name__ == "__main__":
    main()
