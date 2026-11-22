"""Real cases and cohort: archetypes, Layers 3-5 where data allow, and the H1-H6 verdicts.

    python -m src.run_real

Writes reports/tables/: real_archetypes.csv, real_layer3_status.csv, real_h2_nmc_flags.csv,
real_monitor_<firm>.csv (if Layers 3-5 ran), cohort_*.csv (if data/raw/cohort/cohort.csv exists),
hypotheses.csv (H1-H6; H3-H5 copied from sim_hypotheses.csv when the simulation has been run).
Nothing is tuned here: every threshold and rule comes from config.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from src.config import Paths, load_config, rng
from src.detectors import event_time, layer2_rules
from src.detectors.monitor import load_inflation, monitor_firm, usable_training_periods
from src.ingest import canonical
from src.masking import archetype
from src.ratios import check_identities, compute_ratios
from src.run_layers import firm_results


def layer3_case(firm, res, cfg, paths, inflation):
    fc = cfg["firms"][firm]; cash = cfg["masking"]["cash_ratio"]; need = cfg["detectors"]["min_periods_layer3"]
    tend = fc.get("training_end_fy")
    r = res["ratios"]
    n_train = int((r.index <= tend).sum()) if tend else 0
    usable = usable_training_periods(r, cash, n_train) if n_train else 0
    row = {"firm": firm, "frequency": "annual", "training_end_fy": tend, "training_periods": n_train,
           "usable_training_periods": usable, "min_required": need}
    if not tend:
        row.update(ran=False, reason="no training window configured; pre-registered to Layers 0-2 only (6 annual points)")
    elif usable < need:
        row.update(ran=False, reason=f"only {usable} usable {cash} value(s) in the training window (FY<= {tend}); "
                                     f"need >= {need}. Requires higher-frequency pre-FY14 data (BSE quarterly FY2010-13)")
    else:
        out, summ = monitor_firm(r, res["flags"], n_train, cfg, rng(zlib.crc32(firm.encode()) % 1000), inflation)
        out.to_csv(paths.tables / f"real_monitor_{firm}.csv"); row.update(ran=True, reason="", **summ)
    return row


def h1_verdict(status, res, cfg):
    s = status.set_index("firm").loc["jyoti"]
    if not s.ran:
        return dict(id="H1", verdict="INCONCLUSIVE", evidence=f"Layer 3 not run for Jyoti: {s.reason}",
                    source="real_layer3_status.csv")
    first = s.first_amber_uncorrected
    efy = res["efy"]
    if first is None or pd.isna(first):
        return dict(id="H1", verdict="REFUTED", evidence="no alarm before the event", source="real_monitor_jyoti.csv")
    lead = efy - int(first)
    return dict(id="H1", verdict="SUPPORTED" if lead >= 2 else "REFUTED",
                evidence=f"first amber alarm FY{int(first)}, event FY{efy}: lead {lead} fiscal years", source="real_monitor_jyoti.csv")


def h2_table(res, cfg):
    canon, r, efy = res["canon"], res["ratios"], res["efy"]
    S = archetype.structural_flags(canon, r, cfg)
    cash = layer2_rules.rule_flags(canon, r, cfg).iloc[:, [0]]
    t = pd.concat([cash, S], axis=1).loc[: cfg["firms"]["nmc"].get("h2_through_fy", 2018)]
    return t


def h2_verdict(t):
    cash, struct = t.iloc[:, 0], t.iloc[:, 1:]
    cash_years = list(t.index[cash == 1])
    s_first = next((y for y in t.index if (struct.loc[y] == 1).any()), None)
    which = "; ".join(f"{c} first FY{next(y for y in t.index if struct.loc[y, c] == 1)}" for c in struct.columns if (struct[c] == 1).any())
    never = "; ".join(c for c in struct.columns if not (struct[c] == 1).any())
    if s_first is None and not cash_years:
        v = "REFUTED"; ev = "both silent through FY2018"
    elif s_first is not None and not cash_years:
        v = "SUPPORTED"; ev = f"cash flag (OCF < 0) silent FY{t.index.min()}-FY{t.index.max()}; structural flags tripped: {which}"
    elif s_first is not None and cash_years and s_first <= min(cash_years) - 1:
        v = "SUPPORTED"; ev = f"structural first FY{s_first}, cash first FY{min(cash_years)}"
    else:
        v = "REFUTED"; ev = f"cash flag tripped in {cash_years}; structural first {s_first}"
    if never:
        ev += f". Never tripped: {never}"
    return dict(id="H2", verdict=v, evidence=ev, source="real_h2_nmc_flags.csv")


def _first(s):
    hit = s.index[s == 1]
    return int(hit[0]) if len(hit) else None


def _order(cash_fy, struct_fy):
    if cash_fy is None and struct_fy is None:
        return "neither"
    if struct_fy is None:
        return "cash only"
    if cash_fy is None:
        return "structural only"
    return "cash first" if cash_fy < struct_fy else ("structural first" if struct_fy < cash_fy else "same year")


def cohort(cfg, paths):
    """Layers 0-2, archetypes and rule-level fingerprints for the cohort (data/raw/cohort/cohort.csv).
    Window for detection / false alarm = Layer 1 window (event FY + window[0] .. event FY - 1), the same
    fiscal years for a control as for its failed firm."""
    f = paths.raw / "cohort" / "cohort.csv"
    if not f.exists():
        return None
    meta = pd.read_csv(f); rows = []
    lo, _ = cfg["layer1"]["window"]
    need = cfg["detectors"]["min_periods_layer3"]; cash = cfg["masking"]["cash_ratio"]
    for m in meta.itertuples():
        canon = canonical.from_csv(paths.raw / "cohort" / f"{m.firm_id}.csv", int(m.fy_end_month))
        check_identities(canon, cfg["identities"], m.firm_id)
        r = compute_ratios(canon, cfg)
        r.to_csv(paths.tables / f"cohort_ratios_{m.firm_id}.csv")
        efy = event_time.event_fy(m.event_date, int(m.fy_end_month))
        F = layer2_rules.rule_flags(canon, r, cfg)
        F.to_csv(paths.tables / f"cohort_layer2_{m.firm_id}.csv")
        S = archetype.structural_flags(canon, r, cfg)
        win = [y for y in range(efy + lo, efy) if y in F.index]
        cash_fy = _first(F.iloc[:, 0].reindex(win))
        struct_fy = _first(S.reindex(win).max(axis=1, skipna=True))
        rules_fy = _first((F.iloc[:, :-1].reindex(win) == 1).any(axis=1).astype(int))
        alt_fy = _first(F.iloc[:, -1].reindex(win))
        pre = r.index[r.index < efy]
        row = {"firm_id": m.firm_id, "name": m.name, "role": m.role,
               "matched_to": m.matched_to if isinstance(m.matched_to, str) else "", "event_date": m.event_date, "event_fy": efy,
               **archetype.classify(canon, r, efy, cfg),
               "pre_event_annual_periods": int(len(pre)), "usable_cash_ratio_periods": int(r.loc[pre, cash].notna().sum()),
               # Layers 3-4 need >= `need` usable TRAINING values before the monitored (Layer 1) window
               "usable_training_before_window": int(r.loc[r.index < efy + lo, cash].notna().sum()),
               "layer3_possible": bool(r.loc[r.index < efy + lo, cash].notna().sum() >= need),
               "window_fy": f"{win[0]}-{win[-1]}" if win else "none", "first_cash_flag_fy": cash_fy,
               "first_structural_flag_fy": struct_fy, "flag_order": _order(cash_fy, struct_fy),
               "first_rule_fy": rules_fy, "first_altman_fy": alt_fy,
               "scaled_periods": int((pd.read_csv(paths.raw / "cohort" / f"{m.firm_id}.csv").get("period_months", pd.Series(dtype=float)) != 12).sum())}
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_summary(coh, cfg, paths):
    """Rule-level detection (failed firms) vs false alarms (matched controls) over the same window."""
    out = []
    f, c = coh[coh.role == "failed"], coh[coh.role == "control"]
    for lab, col in [("any Layer 2 rule", "first_rule_fy"), ("Altman Z''-EM distress", "first_altman_fy"),
                     ("cash flag (OCF < 0)", "first_cash_flag_fy"), ("any structural flag", "first_structural_flag_fy")]:
        out.append({"signal": lab, "failed_firms": len(f), "controls": len(c),
                    "failed_flagged": int(f[col].notna().sum()), "controls_flagged": int(c[col].notna().sum())})
    # same fiscal years for both members of a pair: only years where the failed firm AND its control have data
    pairs = []
    lo, _ = cfg["layer1"]["window"]
    for c_ in c.itertuples():
        fr = f[f.firm_id == c_.matched_to].iloc[0]
        Ff = pd.read_csv(paths.tables / f"cohort_layer2_{fr.firm_id}.csv", index_col=0)
        Fc = pd.read_csv(paths.tables / f"cohort_layer2_{c_.firm_id}.csv", index_col=0)
        yrs = [y for y in range(int(fr.event_fy) + lo, int(fr.event_fy)) if y in Ff.index and y in Fc.index
               and Ff.loc[y].notna().any() and Fc.loc[y].notna().any()]
        rule = lambda F: bool((F.loc[yrs].iloc[:, :-1] == 1).any().any()) if yrs else None  # noqa: E731
        pairs.append({"failed": fr.firm_id, "control": c_.firm_id, "common_years": f"{yrs[0]}-{yrs[-1]}" if yrs else "none",
                      "n_common_years": len(yrs), "failed_rule_flag": rule(Ff), "control_rule_flag": rule(Fc)})
    P = pd.DataFrame(pairs)
    P.to_csv(paths.tables / "cohort_pairs_common_window.csv", index=False)
    out.append({"signal": "any Layer 2 rule, common years only", "failed_firms": len(P), "controls": len(P),
                "failed_flagged": int(P.failed_rule_flag.sum()), "controls_flagged": int(P.control_rule_flag.sum())})
    pd.DataFrame(out).to_csv(paths.tables / "cohort_rule_detection.csv", index=False)
    fp = pd.crosstab(f.archetype, f.flag_order).reset_index()
    fp.to_csv(paths.tables / "cohort_archetype_flag_order.csv", index=False)


def h6_verdict(coh):
    if coh is None:
        return dict(id="H6", verdict="INCONCLUSIVE", evidence="no cohort data: data/raw/cohort/cohort.csv not present",
                    source="cohort_archetypes.csv")
    f = coh[coh.role == "failed"]
    if not f.layer3_possible.any():
        counts = f.archetype.value_counts()
        mx = int(f.usable_training_before_window.max()) if len(f) else 0
        return dict(id="H6", verdict="INCONCLUSIVE",
                    evidence=f"cohort of {len(f)} failed firms collected ({', '.join(f'{k} {v}' for k, v in counts.items())}), but the "
                             f"masking index needs Layers 3-4, which run only with >= 12 usable periods; the most any failed firm has "
                             f"is {mx} usable annual training values before its monitored window. Rule-level exploratory analogue in cohort_archetype_flag_order.csv",
                    source="cohort_archetypes.csv")
    return dict(id="H6", verdict="INCONCLUSIVE", evidence="cohort MI comparison not implemented for this data", source="")


def main():
    cfg = load_config(); paths = Paths.from_config(cfg).ensure()
    inflation = load_inflation(paths.tables)
    res = {f: firm_results(f, cfg, paths) for f in ("nmc", "jyoti")}
    arch = pd.DataFrame([{"firm": f, "event_fy": r["efy"], **archetype.classify(r["canon"], r["ratios"], r["efy"], cfg)}
                         for f, r in res.items()])
    arch.to_csv(paths.tables / "real_archetypes.csv", index=False)
    status = pd.DataFrame([layer3_case(f, res[f], cfg, paths, inflation) for f in res])
    status.to_csv(paths.tables / "real_layer3_status.csv", index=False)
    t2 = h2_table(res["nmc"], cfg); t2.to_csv(paths.tables / "real_h2_nmc_flags.csv")
    coh = cohort(cfg, paths)
    if coh is not None:
        coh.to_csv(paths.tables / "cohort_archetypes.csv", index=False)
        cohort_summary(coh, cfg, paths)
    hyp = [h1_verdict(status, res["jyoti"], cfg), h2_verdict(t2)]
    sim = paths.tables / "sim_hypotheses.csv"
    if sim.exists():
        hyp += pd.read_csv(sim).to_dict("records")
    else:
        hyp += [dict(id=i, verdict="INCONCLUSIVE", evidence="simulation not run yet", source="") for i in ("H3", "H4", "H5")]
    hyp.append(h6_verdict(coh))
    H = pd.DataFrame(hyp).set_index("id").loc[["H1", "H2", "H3", "H4", "H5", "H6"]].reset_index()
    H.to_csv(paths.tables / "hypotheses.csv", index=False)
    print(arch.to_string(index=False)); print(); print(H.to_string(index=False))


if __name__ == "__main__":
    main()
