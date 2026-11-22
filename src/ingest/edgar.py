"""SEC EDGAR external-transfer panel (config `edgar`; exploratory, not the pre-registered cohort).

    python -m src.ingest.edgar            # pull events, frames, facts, controls -> data/raw/edgar/

Pipeline
1. Events: EDGAR full-text search for 8-Ks citing Item 1.03 (Bankruptcy or Receivership), 2012-2023.
   First such filing per company = event date. Financials (SIC 6000-6799) excluded.
2. Depth pre-filter and size: XBRL frames (us-gaap:Assets, every calendar quarter).
3. Controls: same 2-digit SIC, total assets within the band at the start of the monitored window, never
   filed an Item 1.03 8-K in 2012-2023. Closest in log assets wins.
4. Facts: companyfacts JSON per company; only the tags below are kept (first-filed value per period).
5. Quarterly canonical frame: YTD cash-flow/income items de-cumulated (Q2 = H1 - Q1, ...), flows as
   trailing-twelve-month sums, balance-sheet items at quarter end.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Paths, load_config

TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "ebit": ["OperatingIncomeLoss"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "interest": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "cfi": ["NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "cff": ["NetCashProvidedByUsedInFinancingActivities", "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"],
    "bank_deposits": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "payables": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_assets": ["Assets"],
    "equity_total": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "ltd_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations", "ConvertibleDebtNoncurrent",
                       "LongTermNotesPayable", "SeniorLongTermNotes", "ConvertibleNotesPayable"],
    "ltd_current": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent", "ConvertibleNotesPayableCurrent"],
    "st_borrowings": ["ShortTermBorrowings", "CommercialPaper"],
    "debt_current": ["DebtCurrent"],
    "ltd_total": ["LongTermDebt"],
}
FLOWS = ["revenue", "cogs", "ebit", "depreciation", "interest", "net_income", "ocf", "capex", "cfi", "cff"]


class Http:
    def __init__(self, ua: str, rps: float):
        self.ua, self.gap, self.last = ua, 1.0 / rps, 0.0

    def get_json(self, url: str, retries: int = 4):
        for i in range(retries):
            wait = self.gap - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                    if r.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                time.sleep(2 ** i)
            except Exception:
                time.sleep(2 ** i)
        raise RuntimeError(f"failed: {url}")


# ------------------------------------------------------------------ 1. events
def fetch_item103(http: Http) -> pd.DataFrame:
    rows, frm = [], 0
    q = urllib.parse.quote('"Item 1.03"')
    while True:
        url = (f"https://efts.sec.gov/LATEST/search-index?q={q}&forms=8-K&dateRange=custom"
               f"&startdt=2012-01-01&enddt=2023-12-31&from={frm}")
        d = http.get_json(url)
        hits = d["hits"]["hits"] if d else []
        if not hits:
            break
        for h in hits:
            s = h["_source"]
            if "1.03" not in (s.get("items") or []):
                continue
            for i, cik in enumerate(s.get("ciks") or []):
                rows.append({"cik": int(cik), "name": (s.get("display_names") or [""])[i].split("  (CIK")[0],
                             "sic": int(s["sics"][i]) if s.get("sics") and i < len(s["sics"]) and s["sics"][i] else None,
                             "file_date": s["file_date"], "adsh": s.get("adsh")})
        frm += len(hits)
        if frm >= min(d["hits"]["total"]["value"], 10000):
            break
    return pd.DataFrame(rows).drop_duplicates(["cik", "adsh"])


# ------------------------------------------------------------------ 2. frames
def fetch_assets_frames(http: Http, years=range(2009, 2024)) -> pd.DataFrame:
    rows = []
    for y in years:
        for q in range(1, 5):
            d = http.get_json(f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/CY{y}Q{q}I.json")
            for r in (d or {}).get("data", []):
                rows.append({"cik": int(r["cik"]), "quarter": f"{y}Q{q}", "assets": r["val"]})
    return pd.DataFrame(rows)


def sic_of(http: Http, cik: int, cache: dict) -> int | None:
    if cik not in cache:
        d = http.get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        cache[cik] = int(d["sic"]) if d and str(d.get("sic", "")).isdigit() else None
    return cache[cik]


# ------------------------------------------------------------------ 4. facts
def fetch_facts(http: Http, cik: int) -> pd.DataFrame:
    d = http.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    rows = []
    gaap = (d or {}).get("facts", {}).get("us-gaap", {})
    wanted = {t for ts in TAGS.values() for t in ts}
    for tag in wanted & set(gaap):
        for unit, facts in gaap[tag]["units"].items():
            if unit != "USD":
                continue
            for f in facts:
                rows.append({"tag": tag, "start": f.get("start"), "end": f["end"], "val": f["val"],
                             "filed": f["filed"], "form": f.get("form"), "fp": f.get("fp")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. quarterly canonical
def _quarterly_flow(f: pd.DataFrame) -> pd.Series:
    """De-cumulate duration facts into 3-month values keyed by calendar quarter of the period end."""
    f = f.dropna(subset=["start"]).copy()
    f["start"], f["end"] = pd.to_datetime(f.start), pd.to_datetime(f.end)
    f = f.sort_values("filed").drop_duplicates(["start", "end"], keep="first")      # as first reported
    f["days"] = (f.end - f.start).dt.days
    out = {}
    for _, r in f[(f.days >= 80) & (f.days <= 100)].iterrows():
        out.setdefault(r.end.to_period("Q"), r.val)
    for _, g in f[f.days > 100].groupby("start"):
        g = pd.concat([f[(f.start == g.start.iloc[0]) & (f.days >= 80) & (f.days <= 100)], g]).sort_values("end")
        for prev, cur in zip(g.itertuples(), g.iloc[1:].itertuples()):
            if 80 <= (cur.end - prev.end).days <= 100:
                out.setdefault(cur.end.to_period("Q"), cur.val - prev.val)
    return pd.Series(out, dtype=float)


def _instant(f: pd.DataFrame) -> pd.Series:
    f = f[f.start.isna()].copy()
    f["end"] = pd.to_datetime(f.end)
    f = f.sort_values("filed").drop_duplicates("end", keep="first")
    s = pd.Series(f.val.to_numpy(float), index=f.end.dt.to_period("Q"))
    return s[~s.index.duplicated(keep="last")]


def quarterly_canonical(facts: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for name, tags in TAGS.items():
        s = None
        for t in tags:
            f = facts[facts.tag == t]
            if f.empty:
                continue
            v = _quarterly_flow(f) if name in FLOWS else _instant(f)
            s = v if s is None else s.combine_first(v)
        if s is not None:
            cols[name] = s
    q = pd.DataFrame(cols).sort_index()
    if q.empty:
        return q
    q = q.reindex(pd.period_range(q.index.min(), q.index.max(), freq="Q"))
    for c in FLOWS:                                     # trailing-twelve-month flows
        if c in q:
            q[c] = q[c].rolling(4, min_periods=4).sum()
    if "ebit" in q and "depreciation" in q:
        q["ebitda"] = q.ebit + q.depreciation
    if "equity_total" in q:
        q["equity"] = q.equity_total
        q["total_liabilities"] = q.total_assets - q.equity_total
    parts = [c for c in ("ltd_noncurrent", "ltd_current", "st_borrowings") if c in q]
    debt = q[parts].sum(axis=1, min_count=1) if parts else pd.Series(np.nan, index=q.index)
    if "ltd_total" in q:                                # LongTermDebt includes its current portion
        tot = q.ltd_total + (q.st_borrowings.fillna(0) if "st_borrowings" in q else 0)
        debt = tot.combine_first(debt)
    q["total_debt"] = debt                              # no debt tag reported => NaN (never assumed zero)
    st = q["debt_current"] if "debt_current" in q else pd.Series(np.nan, index=q.index)
    st = st.fillna(q[[c for c in ("ltd_current", "st_borrowings") if c in q]].sum(axis=1, min_count=1)) if any(c in q for c in ("ltd_current", "st_borrowings")) else st
    q["st_debt"] = st
    q["fy_end"] = q.index.to_timestamp(how="end").normalize()
    q.index = pd.Index([p.year * 4 + p.quarter - 1 for p in q.index], name="fy")   # consecutive integer periods
    drop = [c for c in ("equity_total", "ltd_noncurrent", "ltd_current", "st_borrowings", "debt_current", "ltd_total") if c in q]
    return q.drop(columns=drop)


def period_id(ts) -> int:
    p = pd.Timestamp(ts).to_period("Q")
    return p.year * 4 + p.quarter - 1


# ------------------------------------------------------------------ orchestration
def pull(cfg, paths: Paths):
    ec = cfg["edgar"]; root = paths.raw / "edgar"; (root / "facts").mkdir(parents=True, exist_ok=True)
    http = Http(ec["user_agent"], ec["requests_per_second"])
    f_ev = root / "item103_filings_2012_2023.csv"
    if not f_ev.exists():
        fetch_item103(http).to_csv(f_ev, index=False)
    ev = pd.read_csv(f_ev)
    f_fr = root / "assets_frames.csv.gz"
    if not f_fr.exists():
        fetch_assets_frames(http).to_csv(f_fr, index=False)
    fr = pd.read_csv(f_fr)
    lo, hi = ec["exclude_sic_range"]
    first = ev.sort_values("file_date").drop_duplicates("cik")
    first = first[(first.file_date >= ec["event_search"]["start"]) & (first.file_date <= ec["event_search"]["end"])]
    first = first[first.sic.notna() & ~first.sic.between(lo, hi)].copy()
    first["event_q"] = first.file_date.map(period_id)
    fr["qid"] = fr.quarter.map(lambda s: int(s[:4]) * 4 + int(s[-1]) - 1)
    have = fr.groupby("cik").qid.apply(set).to_dict()
    need, of = ec["prefilter_assets_quarters"][0], ec["prefilter_assets_quarters"][1]
    first["depth"] = [len({e - i for i in range(1, of + 1)} & have.get(c, set())) for c, e in zip(first.cik, first.event_q)]
    elig = first[first.depth >= need].sort_values("file_date").head(ec["max_failed_firms"])
    print(f"Item 1.03 filers: {ev.cik.nunique()}; non-financial first events in window: {len(first)}; "
          f"with XBRL depth >= {need}/{of}: {(first.depth >= need).sum()}; taken: {len(elig)}")

    # controls
    sic_cache_f = root / "sic_cache.csv"
    cache = dict(pd.read_csv(sic_cache_f).values.tolist()) if sic_cache_f.exists() else {}
    bankrupt = set(ev.cik)
    by_q = {q: g.set_index("cik").assets for q, g in fr.groupby("qid")}
    M, b_lo, b_hi = ec["monitor_quarters"], *ec["controls"]["assets_band"]
    rows, used = [], set()
    for r in elig.itertuples():
        q0 = r.event_q - M
        a = by_q.get(q0, pd.Series(dtype=float))
        if r.cik not in a.index or not a[r.cik] > 0:
            rows.append({**r._asdict(), "role": "failed", "control_cik": None}); continue
        base = a[r.cik]
        cand = a[(a / base).between(b_lo, b_hi) & ~a.index.isin(bankrupt) & ~a.index.isin(used)]
        cand = cand.iloc[np.argsort(np.abs(np.log(cand.to_numpy() / base)))]
        ctrl = None
        for c in cand.index[:60]:
            if len({r.event_q - i for i in range(1, of + 1)} & have.get(c, set())) < need:
                continue
            s = sic_of(http, int(c), cache)
            if s is not None and s // 100 == int(r.sic) // 100:
                ctrl = int(c); used.add(ctrl); break
        rows.append({**r._asdict(), "role": "failed", "control_cik": ctrl})
    pd.Series(cache).rename_axis("cik").rename("sic").to_csv(sic_cache_f)
    panel = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    panel.to_csv(root / "panel_failed.csv", index=False)

    ciks = set(panel.cik) | set(panel.control_cik.dropna().astype(int))
    for i, c in enumerate(sorted(ciks), 1):
        f = root / "facts" / f"CIK{c:010d}.csv.gz"
        if not f.exists():
            fetch_facts(http, c).to_csv(f, index=False)
        if i % 25 == 0:
            print(f"facts {i}/{len(ciks)}", flush=True)
    print(f"panel: {len(panel)} failed firms, {panel.control_cik.notna().sum()} with a matched control")


if __name__ == "__main__":
    cfg = load_config(); pull(cfg, Paths.from_config(cfg).ensure())
