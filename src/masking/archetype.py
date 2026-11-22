"""Pre-registered archetype rule (config `archetype`), applied from data, not reputation.

Visible: OCF < 0 in >= N consecutive pre-event fiscal years.
Masked:  OCF positive in every evaluable pre-event year AND >= M distinct structural flags trip at some
         point in the pre-event window (liquidity cover < 1, EBITDA / interest < 1.5, Z''-EM distress).
Other:   otherwise. Flags that cannot be evaluated (missing inputs) count as not tripped and are listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def structural_flags(canon: pd.DataFrame, ratios: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["archetype"]["masked"]["structural_flags"]
    c = canon.reindex(index=ratios.index, columns=sorted(set(canon.columns) | {"ocf", "ebitda", "interest"}))
    out = pd.DataFrame(index=ratios.index)

    def fl(cond, valid):
        return pd.Series(np.where(valid, cond.fillna(False).astype(float), np.nan), index=ratios.index)

    lc = ratios.liquid_to_st_debt
    out[f"liquidity cover < {s['liquidity_cover_max']:g}"] = fl(lc < s["liquidity_cover_max"], lc.notna())
    out[f"EBITDA / interest < {s['interest_cover_max']:g}"] = fl((c.interest > 0) & (c.ebitda < s["interest_cover_max"] * c.interest),
                                                                c.ebitda.notna() & c.interest.notna())
    z = ratios.altman_z2_em
    out[f"Z''-EM < {cfg['altman_z2_em']['distress']:g}"] = fl(z < cfg["altman_z2_em"]["distress"], z.notna())
    return out


def classify(canon: pd.DataFrame, ratios: pd.DataFrame, efy: int, cfg: dict, periods_per_year: int = 1) -> dict:
    """periods_per_year=4 for quarterly TTM data: '>= 2 consecutive years' becomes >= 8 consecutive quarters."""
    a = cfg["archetype"]
    pre = ratios.index[ratios.index < efy]          # strictly before the event fiscal year
    ocf = canon.reindex(index=ratios.index, columns=["ocf"]).ocf.loc[pre]
    neg = (ocf < 0).astype(int).where(ocf.notna())
    run = best = 0
    for v in neg:
        run = run + 1 if v == 1 else 0                # a missing year breaks the run (never assumed)
        best = max(best, run)
    S = structural_flags(canon, ratios, cfg).loc[pre]
    tripped = [c for c in S.columns if (S[c] == 1).any()]
    unevaluable = [c for c in S.columns if S[c].isna().all()]
    if best >= a["visible"]["ocf_negative_consecutive_years"] * periods_per_year:
        arch = "Visible"
    elif (ocf.dropna() > 0).all() and len(ocf.dropna()) and len(tripped) >= a["masked"]["min_structural_flags"]:
        arch = "Masked"
    else:
        arch = "Other"
    return {"archetype": arch, "max_consecutive_negative_ocf_years": best,
            "ocf_positive_all_pre_event_years": bool((ocf.dropna() > 0).all()) if len(ocf.dropna()) else None,
            "structural_flags_tripped": "; ".join(tripped) or "none", "n_structural_flags_tripped": len(tripped),
            "structural_flags_not_evaluable": "; ".join(unevaluable) or "none",
            "pre_event_years": f"{pre.min()}-{pre.max()}" if len(pre) else "none"}
