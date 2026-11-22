"""The refactored pipeline must reproduce the outputs of the original root-level scripts.

Three layers of evidence:
 1. Legacy scripts (frozen verbatim in tests/legacy/) still reproduce the golden CSVs that shipped with
    the project (tests/golden/) -- proves the goldens are what the original code produces.
 2. The refactored pipeline (src/) reproduces those golden CSVs exactly.
 3. The refactored figures are pixel-identical to figures rendered by the legacy scripts on *this*
    machine (the golden PNGs were rendered elsewhere; font rasterisation differs across machines,
    so they are kept in tests/golden/original_figures/ for reference only).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests" / "legacy"
GOLDEN = ROOT / "tests" / "golden"
ENV = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}

# golden CSV -> (where the legacy scripts write it, where the refactor writes it, read_csv kwargs)
CSVS = {
    "jyoti_full.csv": ("jyoti_full.csv", "data/processed/jyoti_full.csv", {"index_col": 0}),
    "jyoti_metrics_full.csv": ("jyoti_metrics_full.csv", "reports/tables/jyoti_metrics_full.csv", {"index_col": 0}),
    "jyoti_flags.csv": ("jyoti_flags.csv", "reports/tables/jyoti_flags.csv", {"index_col": 0}),
    "nmc_redflags.csv": ("nmc_redflags.csv", "reports/tables/nmc_redflags.csv", {"index_col": 0}),
}
FIGS = {  # legacy path -> refactored path
    "fig1_lifecycle.png": "reports/figures/fig1_lifecycle.png",
    "fig2_crunch_anatomy.png": "reports/figures/fig2_crunch_anatomy.png",
    "fig3_balance_sheet.png": "reports/figures/fig3_balance_sheet.png",
    "fig4_warning_flags.png": "reports/figures/fig4_warning_flags.png",
    "fig5_recurrence.png": "reports/figures/fig5_recurrence.png",
    "plots_nmc/01_ratio_panels.png": "reports/figures/01_ratio_panels.png",
    "plots_nmc/02_growth_redflags.png": "reports/figures/02_growth_redflags.png",
    "plots_nmc/03_corr.png": "reports/figures/03_corr.png",
}


def _run(args, cwd):
    p = subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, capture_output=True, text=True)
    assert p.returncode == 0, f"{args} failed:\n{p.stdout}\n{p.stderr}"
    return p.stdout


@pytest.fixture(scope="module")
def legacy_out(tmp_path_factory):
    d = tmp_path_factory.mktemp("legacy")
    for f in LEGACY.glob("*.py"):
        shutil.copy(f, d)
    for f in ("nmc_raw.csv", "nmc_ratios.csv"):
        shutil.copy(ROOT / "data" / "raw" / f, d)
    for script in ("jyoti_build_full.py", "jyoti_eda_full.py", "nmc_eda.py"):
        _run([script], cwd=d)
    return d


@pytest.fixture(scope="module")
def new_out(tmp_path_factory):
    d = tmp_path_factory.mktemp("refactor")
    for mod in ("src.run_jyoti", "src.run_nmc"):
        _run(["-m", mod, "--tables", "--out-root", str(d)], cwd=ROOT)
        _run(["-m", mod, "--figures", "--out-root", str(d)], cwd=ROOT)
    return d


def _frame(path, kw):
    return pd.read_csv(path, **kw)


@pytest.mark.parametrize("golden", sorted(CSVS))
def test_legacy_scripts_reproduce_golden(legacy_out, golden):
    legacy_rel, _, kw = CSVS[golden]
    pd.testing.assert_frame_equal(_frame(legacy_out / legacy_rel, kw), _frame(GOLDEN / golden, kw), check_exact=True)


@pytest.mark.parametrize("golden", sorted(CSVS))
def test_refactor_reproduces_golden(new_out, golden):
    _, new_rel, kw = CSVS[golden]
    pd.testing.assert_frame_equal(_frame(new_out / new_rel, kw), _frame(GOLDEN / golden, kw), check_exact=True)


@pytest.mark.parametrize("legacy_rel", sorted(FIGS))
def test_refactor_figures_pixel_identical(legacy_out, new_out, legacy_rel):
    a = mpimg.imread(legacy_out / legacy_rel)
    b = mpimg.imread(new_out / FIGS[legacy_rel])
    assert a.shape == b.shape, f"{legacy_rel}: shape {a.shape} vs {b.shape}"
    assert np.array_equal(a, b), f"{legacy_rel}: {(a != b).any(axis=-1).mean():.4%} of pixels differ"


def test_config_seed():
    from src.config import load_config, rng
    assert load_config()["seed"] == 20261121
    assert rng().integers(1 << 30) == np.random.default_rng(20261121).integers(1 << 30)
