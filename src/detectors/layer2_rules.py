"""Layer 2: pre-registered rule-based flags (config `layer2.rules`) + Altman Z''-EM distress zone.

Each flag is 1.0 tripped, 0.0 not tripped, NaN not evaluable (inputs missing or masked).
Labels are generated from the config thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _fl(cond, valid) -> pd.Series:
    return pd.Series(np.where(valid, pd.Series(cond).fillna(False).astype(float), np.nan), index=valid.index)


def rule_flags(canon: pd.DataFrame, ratios: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """canon: canonical firm-year frame; ratios: output of src.ratios.compute_ratios (same fy index)."""
    t, az = cfg["layer2"]["rules"], cfg["altman_z2_em"]
    c = canon.reindex(index=ratios.index, columns=sorted(set(canon.columns) | {"ocf", "ebitda", "interest"}))
    ocf, ebitda, interest = c.ocf, c.ebitda, c.interest
    dso = -ratios.neg_dso
    z = ratios.altman_z2_em
    F = pd.DataFrame(index=ratios.index)
    F[f"OCF < {t['ocf_max']:g}"] = _fl(ocf < t["ocf_max"], ocf.notna())
    # cross-multiplied so a near-zero interest bill never divides; interest <= 0 => not tripped
    F[f"EBITDA / interest < {t['ebitda_interest_cover_min']:g}"] = _fl(
        (interest > 0) & (ebitda < t["ebitda_interest_cover_min"] * interest), ebitda.notna() & interest.notna())
    F[f"Debtor days > {t['debtor_days_max']:g}"] = _fl(dso > t["debtor_days_max"], dso.notna())
    if t.get("altman_z2_em_distress", False):
        F[f"Altman Z''-EM < {az['distress']:g} (distress)"] = _fl(z < az["distress"], z.notna())
    return F


def first_trips(F: pd.DataFrame, efy: int, event_date, fy_end: pd.Series) -> pd.DataFrame:
    """First pre-event (fy <= event fy) trip per rule, plus an 'Any rule' row.
    lead_fy = event fy - first-trip fy; lead_years = (event date - first-trip FY end) in years."""
    e = pd.Timestamp(event_date)
    pre = F.loc[F.index <= efy]
    rows = {}
    for col in list(F.columns) + ["Any rule"]:
        s = pre.max(axis=1, skipna=True) if col == "Any rule" else pre[col]
        hit = s.index[s == 1]
        fy = int(hit[0]) if len(hit) else None
        rows[col] = {
            "first_trip_fy": fy,
            "first_trip_tau": fy - efy if fy is not None else None,
            "lead_fy": efy - fy if fy is not None else None,
            "lead_years": round((e - fy_end[fy]).days / 365.25, 2) if fy is not None else None,
            "pre_event_years_evaluable": int((pre.notna().any(axis=1) if col == "Any rule" else pre[col].notna()).sum()),
            "pre_event_years_tripped": int((s == 1).sum()),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "rule"
    return out.astype({"first_trip_fy": "Int64", "first_trip_tau": "Int64", "lead_fy": "Int64"})
