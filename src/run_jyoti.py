"""Jyoti Structures detailed EDA (consolidated, INR crore, FY2013-FY2026).
Refactored from jyoti_build_full.py + jyoti_eda_full.py.

    python -m src.run_jyoti [--tables] [--figures] [--out-root DIR]

With neither flag, does both. Builds data/processed/jyoti_full.csv first if missing (or with --tables).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import Paths, load_config
from src.detectors import rule_flags
from src.ingest import jyoti_screener
from src.ratios import jyoti


def _compute(paths: Paths, cfg: dict, rebuild: bool):
    f = paths.processed / jyoti_screener.OUT_FILE
    if rebuild or not f.exists():
        jyoti_screener.main(paths)
    d = pd.read_csv(f, index_col=0)
    m = jyoti.metrics(d)
    F = rule_flags.jyoti_flags(d, m, cfg["flags"])
    return d, m, F


def tables(paths: Paths, cfg: dict) -> None:
    d, m, F = _compute(paths, cfg, rebuild=True)
    first = rule_flags.first_trip_year(F)
    cnt = rule_flags.flag_counts(F)

    m.round(3).to_csv(paths.tables / "jyoti_metrics_full.csv")
    F.to_csv(paths.tables / "jyoti_flags.csv")
    first.to_csv(paths.tables / "jyoti_flag_first_year.csv", index_label="flag")
    cnt.to_csv(paths.tables / "jyoti_flag_counts.csv")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("=== key metrics, pre-failure (FY2013-16) ===")
    print(m.loc[2013:2016, jyoti.PREFAILURE_COLS].round(2).T.to_string())
    print("\n=== first year each flag trips ===")
    for c, y in first.items(): print(f"  {c:36s}", "-" if pd.isna(y) else int(y))
    print("\nflags per year:", cnt.dropna().astype(int).to_dict())
    print("\n=== recurrence FY2022-26 ===")
    print(m.loc[2022:2026, jyoti.RECURRENCE_COLS].round(2).to_string())


def figures(paths: Paths, cfg: dict) -> None:
    from src.reporting import jyoti_figures
    d, m, F = _compute(paths, cfg, rebuild=False)
    jyoti_figures.make_all(d, m, F, cfg, paths.figures)
    print("Saved Jyoti figures to", paths.figures)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", action="store_true"); ap.add_argument("--figures", action="store_true")
    ap.add_argument("--out-root", default=None, help="redirect outputs (tests)")
    a = ap.parse_args(argv)
    full = load_config(); cfg = full["jyoti"]
    paths = Paths.from_config(full, out_root=a.out_root).ensure()
    both = not (a.tables or a.figures)
    if a.tables or both: tables(paths, cfg)
    if a.figures or both: figures(paths, cfg)


if __name__ == "__main__":
    main()
