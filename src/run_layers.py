"""Layer 1 (event-time study) and Layer 2 (rule flags) for the case-study firms.

    python -m src.run_layers [--tables] [--figures] [--firms nmc jyoti] [--out-root DIR]

Tables  -> reports/tables: ratios_<firm>.csv, layer1_event_time_<firm>.csv, layer2_flags_<firm>.csv,
           layer2_first_trip.csv (all firms)
Figures -> reports/figures: layer1_event_time_<firm>.png, layer2_flag_heatmap_<firm>.png
Control firms: list their keys under firms.<firm>.controls in config (each needs a loader in
src.ingest.canonical.LOADERS and its own event_date = the matched failed firm's event date).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import Paths, load_config
from src.detectors import event_time, layer2_rules
from src.ingest import canonical, jyoti_screener
from src.ratios import check_identities, compute_ratios


def firm_results(firm: str, cfg: dict, paths: Paths) -> dict:
    fc = cfg["firms"][firm]
    if firm == "jyoti" and not (paths.processed / jyoti_screener.OUT_FILE).exists():
        jyoti_screener.main(paths)
    canon = canonical.LOADERS[firm](paths, fc["fy_end_month"])
    check_identities(canon, cfg["identities"], firm)
    ratios = compute_ratios(canon, cfg)
    efy = event_time.event_fy(fc["event_date"], fc["fy_end_month"])
    l1 = cfg["layer1"]
    panel = event_time.event_time_panel(ratios, efy, tuple(l1["window"]), l1["ratios"])
    F = layer2_rules.rule_flags(canon, ratios, cfg)
    fy_end = canonical.fy_end(pd.Index(range(F.index.min(), max(F.index.max(), efy) + 1)), fc["fy_end_month"])
    fy_end = pd.Series(fy_end, index=range(F.index.min(), max(F.index.max(), efy) + 1))
    first = layer2_rules.first_trips(F, efy, fc["event_date"], fy_end)
    return dict(canon=canon, ratios=ratios, efy=efy, panel=panel, flags=F, first=first,
                # no-data years: gaps inside the series plus years between the last filing and the event FY
                missing=set(ratios.index[ratios.missing_filing]) | set(range(int(ratios.index.max()) + 1, efy + 1)),
                frac=event_time.event_fraction(fc["event_date"], fc["fy_end_month"]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", action="store_true"); ap.add_argument("--figures", action="store_true")
    ap.add_argument("--firms", nargs="*", default=None)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); paths = Paths.from_config(cfg, out_root=a.out_root).ensure()
    both = not (a.tables or a.figures)
    firms = a.firms or [f for f in cfg["firms"] if f in canonical.LOADERS]
    l1 = cfg["layer1"]
    firsts = []
    for firm in firms:
        res = firm_results(firm, cfg, paths)
        controls = [firm_results(c, cfg, paths)["panel"] for c in cfg["firms"][firm].get("controls", [])]
        band = event_time.control_band(controls, l1["ratios"], tuple(l1["control_band_quantiles"]))
        name, ed = cfg["firms"][firm]["name"], cfg["firms"][firm]["event_date"]
        if a.tables or both:
            res["ratios"].to_csv(paths.tables / f"ratios_{firm}.csv")
            res["panel"].to_csv(paths.tables / f"layer1_event_time_{firm}.csv")
            if band:
                pd.concat(band, names=["ratio"]).to_csv(paths.tables / f"layer1_control_band_{firm}.csv")
            res["flags"].to_csv(paths.tables / f"layer2_flags_{firm}.csv")
            firsts.append(res["first"].assign(firm=firm, event_date=ed, event_fy=res["efy"]).reset_index())
        if a.figures or both:
            from src.reporting import layer_figures
            layer_figures.layer1_event_time(res["panel"], l1["ratios"], name, ed, res["frac"], res["missing"], band,
                                            paths.figures / f"layer1_event_time_{firm}.png")
            layer_figures.layer2_heatmap(res["flags"], name, res["efy"], ed, res["frac"], res["missing"], res["first"],
                                         paths.figures / f"layer2_flag_heatmap_{firm}.png")
        print(f"{name}: event FY{res['efy']} ({ed})\n{res['first'].to_string()}\n")
    if firsts:
        cols = ["firm", "event_date", "event_fy", "rule", "first_trip_fy", "first_trip_tau", "lead_fy", "lead_years",
                "pre_event_years_evaluable", "pre_event_years_tripped"]
        pd.concat(firsts)[cols].to_csv(paths.tables / "layer2_first_trip.csv", index=False)


if __name__ == "__main__":
    main()
