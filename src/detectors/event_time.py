"""Layer 1: event-time study.

Event time tau is measured in fiscal years: tau = 0 is the fiscal year that CONTAINS the event date,
tau = -1 the year before, etc. A firm-year with no filing stays NaN (never interpolated).
An optional control/peer band is the per-tau quantile range across control firms aligned on the
same tau (each control at its matched failed firm's event date).
"""
from __future__ import annotations

import pandas as pd


def event_fy(event_date: str | pd.Timestamp, fy_end_month: int) -> int:
    """Fiscal year containing the event. FY y covers the 12 months ending in month `fy_end_month` of year y."""
    e = pd.Timestamp(event_date)
    return e.year if e.month <= fy_end_month else e.year + 1


def event_fraction(event_date, fy_end_month: int) -> float:
    """Share of the event fiscal year elapsed at the event date (0 = FY start, 1 = FY end)."""
    e = pd.Timestamp(event_date)
    end = pd.Timestamp(f"{event_fy(e, fy_end_month)}-{fy_end_month:02d}-01") + pd.offsets.MonthEnd(0)
    start = end - pd.DateOffset(years=1)
    return (e - start) / (end - start)


def event_time_panel(ratios: pd.DataFrame, efy: int, window: tuple[int, int], cols: list[str]) -> pd.DataFrame:
    lo, hi = window
    taus = pd.RangeIndex(lo, hi + 1, name="tau")
    out = ratios.reindex([efy + t for t in taus])[cols]
    out.index = taus
    out.insert(0, "fy", [efy + t for t in taus])
    return out


def control_band(panels: list[pd.DataFrame], cols: list[str], q: tuple[float, float]) -> dict[str, pd.DataFrame] | None:
    """Per-ratio DataFrame indexed by tau with columns lo, median, hi, n. None if there are no controls."""
    if not panels:
        return None
    out = {}
    for c in cols:
        wide = pd.concat([p[c] for p in panels], axis=1)
        out[c] = pd.DataFrame({"lo": wide.quantile(q[0], axis=1), "median": wide.median(axis=1),
                               "hi": wide.quantile(q[1], axis=1), "n": wide.notna().sum(axis=1)})
    return out
