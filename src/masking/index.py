"""Masking index and detector-order fingerprint (blueprint s.9).

MI_t = S_struct,t / h_struct - S_cash,t / h_cash
Large positive: structural evidence accumulates while cash flow stays quiet (masked).
Negative: cash flow leads (visible).
"""
from __future__ import annotations

import numpy as np

FINGERPRINTS = ["cash_only", "cash_first", "structural_only", "structural_first", "both_together", "neither"]


def masking_index(s_struct, h_struct, s_cash, h_cash) -> np.ndarray:
    return np.asarray(s_struct, float) / h_struct - np.asarray(s_cash, float) / h_cash


def fingerprint(t_cash, t_struct) -> str:
    """First-alarm order. The blueprint lists cash only / structural first / both together / neither;
    'cash_first' and 'structural_only' are kept separate so no case is silently merged."""
    if t_cash is None and t_struct is None:
        return "neither"
    if t_struct is None:
        return "cash_only"
    if t_cash is None:
        return "structural_only"
    if t_cash == t_struct:
        return "both_together"
    return "structural_first" if t_struct < t_cash else "cash_first"
