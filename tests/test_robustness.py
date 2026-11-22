"""Small checks for the robustness helpers."""
import numpy as np

from src.robustness import _auc, variants
from src.config import load_config


def test_auc_basic():
    assert _auc([3, 4], [1, 2]) == 1.0
    assert _auc([1, 2], [3, 4]) == 0.0
    assert _auc([1, 2], [1, 2]) == 0.5
    assert np.isnan(_auc([], [1]))


def test_variants_start_with_preregistered_base():
    cfg = load_config()
    v = variants(cfg)
    assert v[0][0] == "base" and v[0][1] is cfg and v[0][2] == 0 and v[0][3] == 0
    # every variant changes exactly one thing; the base config object itself is never mutated
    assert cfg["detectors"]["cusum"]["k"] == 0.5
    assert len(cfg["masking"]["structural_ratios"]) == 3
