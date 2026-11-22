"""Screener.in company pages -> cohort data files in the canonical schema (data/raw/cohort/).

    python -m src.ingest.screener --firms <firm_id> ...      (firms listed in data/raw/cohort/cohort.csv)
    python -m src.ingest.screener --peers <screener_path>    (peer list for control matching)

Reads the public company page (no login), parses the annual Profit & Loss, Balance Sheet, Cash Flow and
Ratios tables, and writes <firm_id>.csv with the same column mapping as Jyoti (src.ingest.canonical.jyoti).
The raw HTML is cached in data/raw/cohort/html/ so every number can be traced to what was downloaded.

Irregular fiscal periods (a firm changing its year-end) are detected from the column dates. Flow items of
a period that is not 12 months long are scaled to 12 months (x 12 / months); stock items are left as
reported; the period length is written in `period_months` and the scaling in `flow_scale`.
Requests are rate-limited (config `cohort_ingest.seconds_between_requests`).
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd

from src.config import Paths, load_config

BASE = "https://www.screener.in"
SECTIONS = {"profit-loss": "pl", "balance-sheet": "bs", "cash-flow": "cf", "ratios": "ratios"}
# Screener row label -> raw column (same names as data/raw/jyoti_screener.csv)
ROWS = {
    "Sales": "sales", "Revenue": "sales", "Expenses": "expenses", "Operating Profit": "op_profit",
    "Financing Profit": "op_profit", "Other Income": "other_income", "Interest": "interest",
    "Depreciation": "depreciation", "Profit before tax": "pbt", "Net Profit": "net_profit",
    "Equity Capital": "equity_capital", "Reserves": "reserves", "Borrowings": "borrowings",
    "Other Liabilities": "other_liabilities", "Total Liabilities": "total_liabilities", "Fixed Assets": "fixed_assets",
    "CWIP": "cwip", "Investments": "investments", "Other Assets": "other_assets", "Total Assets": "total_assets",
    "Cash from Operating Activity": "cfo", "Cash from Investing Activity": "cfi", "Cash from Financing Activity": "cff",
    "Net Cash Flow": "net_cash_flow", "Debtor Days": "debtor_days", "Inventory Days": "inventory_days",
    "Days Payable": "payable_days", "Cash Conversion Cycle": "ccc_days",
}
FLOWS = ["sales", "expenses", "op_profit", "other_income", "interest", "depreciation", "pbt", "net_profit", "cfo", "cfi", "cff",
         "net_cash_flow"]
MAX_PERIOD = 18   # longest fiscal period treated as one period (Indian companies have used 15- and 18-month years)
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


class Fetcher:
    def __init__(self, cache_dir, wait, user_agent):
        self.cache, self.wait, self.ua, self.last = cache_dir, wait, user_agent, 0.0
        self.cache.mkdir(parents=True, exist_ok=True)

    def get(self, path: str) -> str:
        f = self.cache / (re.sub(r"[^A-Za-z0-9]+", "_", path.strip("/")) + ".html")
        if f.exists():
            return f.read_text(encoding="utf-8")
        for attempt in range(4):
            time.sleep(max(0.0, self.wait - (time.time() - self.last)))
            self.last = time.time()
            try:
                req = urllib.request.Request(BASE + path, headers={"User-Agent": self.ua})
                h = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "ignore")
                f.write_text(h, encoding="utf-8")
                return h
            except Exception as e:  # noqa: BLE001
                if "429" in str(e):
                    time.sleep(self.wait * 4 * (attempt + 1)); continue
                raise
        raise RuntimeError(f"rate-limited repeatedly on {path}")


def _num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_section(html: str, section: str) -> pd.DataFrame:
    """Annual table of one section -> DataFrame indexed by period label ('Mar 2016'), columns = raw names."""
    m = re.search(rf'<section id="{section}".*?(<table.*?</table>)', html, re.S)
    if not m:
        return pd.DataFrame()
    t = pd.read_html(StringIO(m.group(1)))[0]
    t = t.rename(columns={t.columns[0]: "label"})
    t["label"] = t.label.astype(str).str.replace(r"\s*\+\s*$", "", regex=True).str.replace("\xa0", " ").str.strip()
    t = t[t.label.isin(ROWS)].drop_duplicates("label").set_index("label")
    cols = [c for c in t.columns if re.match(r"[A-Z][a-z]{2} \d{4}$", str(c))]   # drops 'TTM'
    out = t[cols].map(_num).T
    out.columns = [ROWS[c] for c in out.columns]
    return out


def parse_company(html: str) -> pd.DataFrame:
    parts = [parse_section(html, s) for s in SECTIONS]
    d = pd.concat([p for p in parts if not p.empty], axis=1)
    d = d.loc[:, ~d.columns.duplicated()]
    ends = pd.to_datetime(["{}-{:02d}-01".format(c.split()[1], MONTHS[c.split()[0]]) for c in d.index]) + pd.offsets.MonthEnd(0)
    d.index = ends
    d = d.sort_index()
    gap = pd.Series(ends.sort_values()).diff().dt.days.div(30.44).round().fillna(12).astype(int)
    # a gap longer than MAX_PERIOD months is missing year(s), not one long period: the period itself is
    # what remains after whole missing years (24 -> 12; 30 -> 18 is impossible to tell apart, so 12 + rest)
    months = gap.where(gap <= MAX_PERIOD, (gap % 12).replace(0, 12)).to_numpy()
    d.insert(0, "period_months", months)
    d.insert(0, "period_end", d.index.strftime("%Y-%m-%d"))
    return d


def to_canonical(d: pd.DataFrame) -> pd.DataFrame:
    """Raw Screener columns -> canonical schema (mapping identical to canonical.jyoti), flows scaled to 12 months."""
    d = d.copy()
    scale = 12.0 / d.period_months
    for c in FLOWS:
        if c in d:
            d[c] = d[c] * scale
    g = lambda c: d[c] if c in d else pd.Series(np.nan, index=d.index)  # noqa: E731
    out = pd.DataFrame({
        "period_end": d.period_end, "period_months": d.period_months, "flow_scale": scale.round(4),
        "revenue": g("sales"), "ebitda": g("op_profit"), "ebit": g("pbt") + g("interest"), "interest": g("interest"),
        "net_income": g("net_profit"), "depreciation": g("depreciation"), "other_income": g("other_income"), "pbt": g("pbt"),
        "ocf": g("cfo"), "capex": -g("cfi"), "cfi": g("cfi"), "cff": g("cff"),
        "current_assets": g("other_assets"), "current_liabilities": g("other_liabilities"), "total_assets": g("total_assets"),
        "total_liabilities": g("borrowings") + g("other_liabilities"), "total_debt": g("borrowings"),
        "equity": g("equity_capital") + g("reserves"), "retained_earnings": g("reserves"),
        "dso": g("debtor_days"), "dio": g("inventory_days"), "dpo": g("payable_days"), "ccc": g("ccc_days"),
    })
    out.insert(0, "fy", pd.DatetimeIndex(out.index).year)
    # a calendar year with two period ends: prefer the 12-month period (a trailing half-year column is
    # an interim figure); otherwise (year-end change) keep the later one
    out = out.assign(_full=(out.period_months == 12).astype(int)).sort_values(["fy", "_full", "period_end"])
    out = out[~out.fy.duplicated(keep="last")].drop(columns="_full").sort_values("fy")
    return out.reset_index(drop=True)


def peers(html: str, fetch: Fetcher) -> pd.DataFrame:
    """Screener's peer table for a company page (same industry), via the page's warehouse id."""
    m = re.search(r'data-warehouse-id="(\d+)"', html)
    if not m:
        return pd.DataFrame()
    h = fetch.get(f"/api/company/{m.group(1)}/peers/")
    t = pd.read_html(StringIO(h))[0]
    links = re.findall(r'href="(/company/[^"]+)"', h)
    t = t[t.iloc[:, 1].astype(str).str.len() > 0].iloc[: len(links)].copy()
    t["path"] = links[: len(t)]
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", nargs="*", default=None)
    ap.add_argument("--path", default=None, help="a single Screener path, e.g. /company/LITL/consolidated/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg)
    ci = cfg["cohort_ingest"]
    fetch = Fetcher(paths.raw / "cohort" / "html", ci["seconds_between_requests"], cfg["edgar"]["user_agent"])
    if a.path:
        d = to_canonical(parse_company(fetch.get(a.path)))
        d.to_csv(a.out, index=False) if a.out else print(d.to_string())
        return
    meta = pd.read_csv(paths.raw / "cohort" / "cohort.csv")
    for r in meta.itertuples():
        if a.firms and r.firm_id not in a.firms:
            continue
        d = to_canonical(parse_company(fetch.get(r.screener_path)))
        d.to_csv(paths.raw / "cohort" / f"{r.firm_id}.csv", index=False)
        print(f"{r.firm_id}: {len(d)} periods, {int((d.period_months != 12).sum())} non-12-month periods scaled")


if __name__ == "__main__":
    main()
