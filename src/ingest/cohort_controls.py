"""Mechanical control matching for the cohort (rule: data/raw/cohort/PROPOSAL.md, incl. Amendment 1).

    python -m src.ingest.cohort_controls

For each failed firm, in event-date order: candidates = first page of its Screener industry list. A
candidate is eligible if it is not a cohort firm, has >= cohort_ingest.control_min_pre_event_periods annual periods
ending before the failed firm's event (consolidated if enough, else standalone), and reports total assets
in the failed firm's reference fiscal year (its last pre-event year with data). The eligible candidate
with the closest total assets (log distance) within cohort_ingest.assets_band is chosen; ties go to the
longer history; already-chosen controls are skipped. Only total assets and period headers are read.
Writes data/raw/cohort/control_matching.csv (every candidate with its reason) and prints the choice.
The IBC check (no CIRP 2015-2023) is done afterwards on the chosen firms and recorded by hand in
cohort.csv, because there is no machine-readable IBBI list.
"""
from __future__ import annotations

import re
from io import StringIO

import numpy as np
import pandas as pd

from src.config import Paths, load_config
from src.ingest import screener as sc

FAILED = [  # firm_id, Screener path, event date, industry list path (from the failed firm's page; Lanco by judgement)
    ("amtek", "/company/AMTEKAUTO/consolidated/", "2017-07-24", "/market/IN02/IN0201/IN020102/IN020102001/"),
    ("bhushan", "/company/TATASTLBSL/consolidated/", "2017-07-26", "/market/IN01/IN0103/IN010301/IN010301004/"),
    ("abg", "/company/ABGSHIP/consolidated/", "2017-08-01", "/market/IN07/IN0702/IN070202/IN070202002/"),
    ("lanco", "/company/LITL/consolidated/", "2017-08-07", "/market/IN07/IN0701/IN070101/IN070101001/"),
    ("era", "/company/ERAINFRA/consolidated/", "2018-05-08", "/market/IN07/IN0701/IN070101/IN070101001/"),
    ("videocon", "/company/VIDEOIND/consolidated/", "2018-06-06", "/market/IN02/IN0202/IN020201/IN020201003/"),
]
EXCLUDE_PATHS = {"AMTEKAUTO", "TATASTLBSL", "ABGSHIP", "LITL", "ERAINFRA", "VIDEOIND", "JETAIRWAYS", "JYOTISTRUC"}


def history(html, event):
    raw = sc.parse_company(html)
    pre = raw[raw.index < pd.Timestamp(event)]
    return raw, pre


WIDER = {"videocon": "/market/IN02/IN0202/IN020201/"}   # broader Screener industry, used only if the specific one yields nothing


def main():
    cfg = load_config(); paths = Paths.from_config(cfg)
    need = cfg["cohort_ingest"]["control_min_pre_event_periods"]; lo, hi = cfg["cohort_ingest"]["assets_band"]
    f = sc.Fetcher(paths.raw / "cohort" / "html", cfg["cohort_ingest"]["seconds_between_requests"], cfg["edgar"]["user_agent"])
    rows, chosen = [], set()
    for fid, path, event, ind in FAILED:
        _, pre = history(f.get(path), event)
        pre = pre[pre.total_assets.notna()]
        ref = pre.index.max(); ref_fy, ref_ta = ref.year, float(pre.loc[ref, "total_assets"])
        h = f.get(ind)
        t = pd.read_html(StringIO(h))[0]
        links = re.findall(r'<a href="(/company/[^"]+)"', h)
        t = t[t["S.No."].astype(str).str.match(r"\d+\.")].reset_index(drop=True).iloc[: len(links)]
        t["path"] = links[: len(t)]
        best = []
        for c in t.itertuples():
            sym = c.path.strip("/").split("/")[1]
            base = re.sub(r"consolidated/$", "", c.path)
            rec = dict(failed=fid, event_date=event, ref_fy=ref_fy, failed_total_assets=ref_ta, candidate=c.Company, path=base,
                       industry_list=ind)
            if sym in EXCLUDE_PATHS:
                rows.append({**rec, "reason": "cohort firm"}); continue
            got = None
            for view in ("consolidated/", ""):
                try:
                    raw, cpre = history(f.get(base + view), event)
                except Exception as e:  # noqa: BLE001
                    rec["reason"] = f"fetch/parse failed: {type(e).__name__}"; continue
                if len(cpre) >= need:
                    got = (view, raw, cpre); break
            if got is None:
                rows.append({**rec, "reason": rec.get("reason", f"< {need} annual periods before event")}); continue
            view, raw, cpre = got
            ta = cpre.total_assets[cpre.index.year == ref_fy]
            if ta.empty or pd.isna(ta.iloc[-1]):
                rows.append({**rec, "view": view or "standalone", "reason": f"no total assets in FY{ref_fy}"}); continue
            r = float(ta.iloc[-1]) / ref_ta
            rec.update(view=view or "standalone", pre_event_periods=len(cpre), candidate_total_assets=float(ta.iloc[-1]), assets_ratio=r)
            if not (lo <= r <= hi):
                rows.append({**rec, "reason": "outside assets band"}); continue
            if base in chosen:
                rows.append({**rec, "reason": "already a control"}); continue
            rows.append({**rec, "reason": "eligible"}); best.append((abs(np.log(r)), -len(cpre), base, c.Company, view))
        if best:
            best.sort(); _, _, base, name, view = best[0]; chosen.add(base)
            for r_ in rows:
                if r_["failed"] == fid and r_["path"] == base:
                    r_["reason"] = "CHOSEN"
            print(f"{fid}: control = {name} ({base}{view}), ref FY{ref_fy}", flush=True)
        elif fid in WIDER and not any(r_.get("widened") for r_ in rows if r_["failed"] == fid):
            print(f"{fid}: no eligible control in the industry list; widening to {WIDER[fid]}", flush=True)
            FAILED.insert(FAILED.index(next(x for x in FAILED if x[0] == fid)) + 1, (fid, path, event, WIDER[fid]))
            for r_ in rows:
                if r_["failed"] == fid:
                    r_["widened"] = False
            WIDER.pop(fid)
        else:
            print(f"{fid}: no eligible control in the industry list", flush=True)
        pd.DataFrame(rows).to_csv(paths.raw / "cohort" / "control_matching.csv", index=False)


if __name__ == "__main__":
    main()
