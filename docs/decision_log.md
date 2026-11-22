**Started: 2026-11-21**

# Decision log

Every analysis choice that could have gone another way, with its reason. Newest at the bottom.

| # | Date | Area | Decision | Alternatives | Reason |
|---|---|---|---|---|---|
| 1 | 2026-11-21 | Altman | Use Z''-EM (+3.25; safe > 5.85, distress < 4.35) everywhere in the framework | Original Z'' (2.6 / 1.1) | Emerging-market firms; one version everywhere. The legacy NMC EDA figure (`01_ratio_panels.png`) still draws the 2.6 / 1.1 lines and must not appear in the report as-is |
| 2 | 2026-11-21 | Orientation | All monitored ratios oriented lower = worse; higher-is-worse ones negated (`neg_`) | Mixed orientation | One-sided CUSUM needs a single direction |
| 3 | 2026-11-21 | Orientation | DPO treated as higher = worse (`neg_dpo`) | Higher = better (as in the CCC) | Stretched payables signal supplier financing under stress; this is why CCC components are monitored separately |
| 4 | 2026-11-21 | Masking | Near-zero denominator: \|den\| ≤ ε·\|total assets\|, ε = 0.01 for revenue/COGS, 0.001 otherwise | One ε = 0.01 | One ε would mask interest cover for healthy, low-debt controls (interest < 1% of assets is common) |
| 5 | 2026-11-21 | Masking | Debt/equity masked when equity ≤ 0; OCF/EBITDA and net debt/EBITDA masked when EBITDA ≤ 0 | Keep sign-flipped values | Ratios with a negative denominator reverse their meaning |
| 6 | 2026-11-21 | Missing data | Jyoti FY2017 kept as a gap; no interpolation by default; growth needs consecutive years | Linear interpolation | Inventing a year would leak information; interpolation stays available for robustness |
| 7 | 2026-11-21 | Anomaly | Flag (do not drop) OCF > 1.0 × revenue; Jyoti FY2018 OCF +1,190 on sales 324 | Drop or winsorise | As-reported policy |
| 8 | 2026-11-21 | Identities | "EBITDA ≥ EBIT" checked only where EBIT excludes other income | Always check | Jyoti's EBIT proxy (PBT + interest) includes other income, so the check is invalid there |
| 9 | 2026-11-21 | Proxies | Jyoti working capital = other assets − other liabilities; capex = −CFI; EBIT = PBT + interest | Leave Altman blank for Jyoti | Screener has no current/non-current split. Overstates X1 (excludes short-term borrowings) |
| 10 | 2026-11-21 | Layer 2 | Interest-cover rule evaluated cross-multiplied (EBITDA < 1.2 × interest); interest ≤ 0 ⇒ not tripped | Use masked ratio | Avoids dividing by near-zero interest; matches the legacy EDA rule |
| 11 | 2026-11-21 | Event time | τ = 0 is the fiscal year containing the event date; lead time measured from fiscal year-end | From filing date | Filing dates not yet collected; real warning is shorter by the reporting lag |
| 12 | 2026-11-21 | Bootstrap | Block length b = max(2, ⌈n^(1/3)⌉) | Politis–White | Data-driven selectors are unstable at n ≈ 5–40 |
| 13 | 2026-11-21 | Simulation | S4 = true drift in all ratios; only the reported cash ratio is exponentially smoothed (s = 0.3) | Smooth all ratios | Smoothing a persistent drift only delays it; concealing the cash ratio while structure worsens is the NMC mechanism. Fixed before any run |
| 14 | 2026-11-21 | Simulation | Generator: 4 ratios, AR(1) φ = 0.5, innovation correlation 0.3, onset at monitored period 4, drift 0.25 / 0.1 SD per period | Blueprint gives no values | Fixed before any run (see fingerprint) |
| 15 | 2026-11-21 | Calibration | Parametric bootstrap refits the model on each of 200 simulated series (not 2,000) | 2,000 refits; or no refit (pure Gaussian) | Refitting captures parameter-estimation error (the point of the fix) but costs ~75 ms per fit; 2,000 × 2,500 firms is ~4 days of CPU. Logged as a deviation |
| 16 | 2026-11-21 | Detectors | Crosier MCUSUM kept direction-agnostic, as specified | One-sided projection | Matches the blueprint formula; it can also alarm on improvement (limitation) |
| 17 | 2026-11-21 | Typology | Archetype counts *distinct* structural flags tripped at any time before the event | Same-year co-occurrence | Rule text says "while ≥ 2 flags trip", not "in the same year"; NMC is Other under either reading |
| 18 | 2026-11-21 | Typology | Fingerprint adds "cash first" and "structural only" to the blueprint's four categories | Merge them | Avoids silently merging distinct orders |
| 19 | 2026-11-21 | Data | EDGAR US panel added as an exploratory external-transfer test (blueprint §4.2), rules written to config before download | No external data | Jyoti/NMC lack the periods needed for Layers 3–5; the India/GCC cohort must be collected by hand |
| 20 | 2026-11-21 | Data (EDGAR) | Quarterly; YTD items de-cumulated; flows as trailing-twelve-month sums; first-filed values; controls matched on 2-digit SIC and assets (0.5–2×) at the start of the monitored window | Annual 10-K only | 4× the periods; TTM removes seasonality; first-filed = as-reported policy |
| 21 | 2026-11-21 | Data (EDGAR) | Total debt left blank when no standard debt tag is reported; company-specific tags are not captured | Assume zero debt | Leverage ratios on EDGAR are lower quality; detectors use cash, liquidity, coverage and Z'', which do not need total debt |
| 22 | 2026-11-21 | Baselines | Random-walk + CUSUM baseline run in simulation only; real data compared with Altman Z''-EM and Layer 2 rules | RW on real data | RW threshold needs its own calibration; kept to the matched-FA simulation where it is fair |
| 23 | 2026-11-21 | Data (EDGAR) | (a) Truncate at the last filed quarter if within 4 quarters of the event; (b) structural ratios enter the multivariate detectors only with ≥ 12 usable training values | (a) require a filing in the event quarter; (b) crash / drop firm | First run: 75 of 150 failed firms had no filing in the 8-K quarter (distressed firms stop filing 10-Qs) and 104 runs crashed on all-missing structural ratios. Changed after seeing coverage failures only; the one-firm detection output was discarded |
| 24 | 2026-11-21 | Calibration | A bootstrap threshold ≤ 0 (degenerate fit, zero innovation variance) is treated as undefined: excluded from fitting the inflation factor, and on real firms that detector is skipped | Keep h = 0 | h = 0 makes any positive CUSUM an alarm and makes the masking index infinite. In simulation 27/500 training firms at n = 12 (0 at n ≥ 28) were affected; counts are in sim_inflation_fit.csv |
| 25 | 2026-11-21 | Provenance | Rebuilt exact copies of the fingerprinted config and pre-registration from the session edit history (`docs/prereg_snapshot/`, hashes match); `src.sanity` compares every pre-registered block with them | Re-hash the edited files | The files were legitimately extended (logged deviations), so whole-file hashes no longer match; block-level comparison proves the frozen values did not change |
| 26 | 2026-11-21 | Robustness | Added `src/robustness.py` (EDGAR variants: k grid, ±2-quarter window shift, leave-one-ratio-out, event −4 quarters, threshold multipliers, rule ablation; Jyoti FY2017 fill; alternative event dates; NMC hidden-debt stress test) after the main results | Skip | Required by the blueprint robustness list; exploratory, no verdict depends on it |
| 27 | 2026-11-21 | Robustness | Added threshold-free AUC diagnostics, incl. a non-adaptive CUSUM (deviations from the training mean) and level scores | Report only the pre-registered detectors | Needed to explain why the statistical tiers did not separate failed from healthy US firms; clearly labelled post hoc |
| 28 | 2026-11-21 | Reproducibility | `run_real` seeded Layer 3 with Python's `hash(firm)`, which is salted per process; replaced by `zlib.crc32` | Keep | Would have made Layer 3 on real firms non-reproducible (it never ran, so no output changed) |
| 29 | 2026-11-21 | Event dates | Both dates verified against primary records; NMC's 2020-04-09 order was made by the English High Court, not the ADGM court (which followed on 2020-09-27 for UAE subsidiaries) | Keep "ADGM" | Factual correction; date unchanged |
| 30 | 2026-11-21 | Cohort data | Screener.in public company pages parsed by `src/ingest/screener.py`; raw HTML cached in `data/raw/cohort/html/`; parser pinned by a test that reproduces the hand-transcribed Jyoti data exactly | Manual transcription | Reproducible and auditable; the test caught a bug (a missing year read as a 24-month period) before any cohort data was used |
| 31 | 2026-11-21 | Cohort | Videocon has no control: no firm in its specific or broader Screener industry was within 0.5–2× of its total assets | Relax the band | The rule says record it; detection is still measured for Videocon |
| 32 | 2026-11-21 | Cohort | H6 stays INCONCLUSIVE: annual cohort data give at most 7 usable training values before the monitored window, below the pre-registered 12 for Layers 3–4 | Lower the minimum | The minimum was pre-registered; a rule-level fingerprint is reported as an exploratory analogue |
| 33 | 2026-11-21 | Cohort | Detection vs false alarms also compared on the fiscal years both members of a pair have data | Raw windows only | Controls' histories are shorter (Screener shows 12 years), which would bias their false-alarm rate down |

## Results that differ from the 5-day plan's expectations

- **NMC (Day 1 check):** the plan expects liquidity, interest cover and Z'' to trip. In the data only
  liquidity cover (cash + deposits) / short-term debt < 1 trips, in FY2015, FY2017 and FY2018.
  EBITDA / interest ≥ 4.0, EBIT / interest ≥ 3.1, and Z''-EM ≥ 5.73 (grey zone in FY2018 only, never
  distress). Under the pre-registered archetype rule (≥ 2 structural flags), NMC is **Other**, not Masked.
  Source: `reports/tables/ratios_nmc.csv`.
- **Jyoti (Day 1 check):** matches. OCF < 0, EBITDA / interest < 1.2 and debtor days > 270 all first trip
  in FY2014, before net worth turns negative in FY2016. Z''-EM only reaches distress in FY2018, after
  the event. Source: `reports/tables/layer2_first_trip.csv`.
- **H2 vs the archetype rule:** H2 is SUPPORTED on NMC (cash flag silent FY2013–FY2018, liquidity cover < 1
  from FY2015). But only one of the three structural flags ever trips, so the pre-registered archetype rule
  (≥ 2 flags) classifies NMC as **Other**, not Masked. Both results are reported as they stand.
