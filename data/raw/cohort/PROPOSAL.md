# Cohort proposal (written 2026-11-21, BEFORE any cohort financial figure was downloaded)

Only event dates and the list of fiscal-year column headers on Screener.in were looked at to build this.
No ratio, balance-sheet or cash-flow number of any candidate has been read. Commit this file before the
financial data is collected; the commit timestamp is the evidence.

## Sampling frame (in this order)

1. RBI's first list of 12 large defaulters referred to the IBC (June 2017).
2. Accounts from RBI's second list (Aug 2017) that press reports name as referred to the NCLT
   (RBI did not publish the second list itself): Castex Technologies, IVRCL, Orchid Pharma,
   SEL Manufacturing, Uttam Galva Steels, Visa Steel, Videocon Industries.
3. Pre-registered negative controls: Jet Airways, Reliance Communications.

## Eligibility (pre-registered rule, PREREGISTRATION §2)

Listed, non-financial, India or GCC; event in 2015–2023; ≥ 8 annual filings before the event, checked on
the Screener.in annual P&L headers. Consolidated statements are used when they cover ≥ 8 pre-event years,
otherwise standalone (same convention as Jyoti). Event = NCLT admission date (start of CIRP).

## Result

| Firm | Event (NCLT admission) | Source for date | Screener history | Status |
|---|---|---|---|---|
| Bhushan Steel (now Tata Steel BSL) | 2017-07-26 | IBBI resolution table; IBBI order list | FY2010–FY2021 consolidated | **Failed firm** |
| Amtek Auto | 2017-07-24 | IBBI resolution table | Jun 2008 – Mar 2019 consolidated (year-end changes: Jun → Sep → Mar) | **Failed firm** (see note) |
| ABG Shipyard | 2017-08-01 | IBBI order list (CP (IB) 53/7/AHM/2017) | FY2006–FY2015 consolidated (FY2008 missing) | **Failed firm** |
| Lanco Infratech | 2017-08-07 | IBBI-hosted order (CP(IB) 111/7/HDB/2017) | FY2006–FY2017 | **Failed firm** |
| Era Infra Engineering | 2018-05-08 | IBBI order list (CP IB-190(PB)-2017) | FY2008–FY2017 (then FY2025–26) | **Failed firm** |
| Videocon Industries | 2018-06-06 | Company filing; IBBI order | Sep 2006 – Mar 2018 consolidated (year-end changes: Sep → Dec → Jun → Dec → Mar) | **Failed firm** (see note) |
| Jet Airways | 2019-06-20 | NCLT Mumbai order (press, company filing) | FY2008–FY2018 consolidated | **Negative control** |
| Jyoti Structures | 2017-07-04 | — | — | Excluded: deep case study |
| Essar Steel, Bhushan Power & Steel | 2017 | — | — | Excluded: not listed |
| Monnet Ispat, Alok Industries, Jaypee Infratech | 2017 | IBBI | Screener starts FY2012/FY2014/FY2015 | Excluded: < 8 pre-event years |
| IVRCL, Orchid Pharma, SEL Manufacturing | 2017–2018 | — | Screener starts FY2013–FY2015 | Excluded: < 8 pre-event years |
| Electrosteel Steels, Castex, Uttam Galva, Visa Steel | — | — | not on Screener | Excluded: no data source |
| Reliance Communications | — | — | Screener starts FY2015 | Excluded negative control: < 8 pre-event years |

**6 failed firms** (meets the pre-registered target of 6–8) and **1 negative control**.

Note: Amtek Auto and Videocon changed their fiscal year-end, so some periods last 15–18 months. Their flow
items (sales, OCF, EBITDA, interest) are not 12-month figures in those years.

## Control-matching rule (fixed now, applied mechanically after approval)

For each failed firm: candidates = Screener's peer list for that company (same industry). Drop any
candidate with a CIRP admission in 2015–2023 (checked on IBBI) or with < 8 annual filings before the
failed firm's event. Choose the candidate whose total assets in the last fiscal year before the event are
closest (in log terms) to the failed firm's, within 0.5–2×; ties go to the longer history. If none
qualifies, widen to the broader Screener sector and record that. Each control is aligned at its failed
firm's event date.

## Known limitations of this cohort

- India only: no free, machine-readable source was found for GCC listed firms (Arabtec and Drake & Scull
  would need PDF extraction from DFM annual reports).
- All events fall in 2017–2019 and most firms are in steel, infrastructure or heavy manufacturing,
  because the frame is RBI's large-defaulter lists.
- Screener data are vendor-standardised (same proxies as Jyoti).

## Decisions (approved by the project owner, 2026-11-21)

- The list above is approved as the cohort: 6 failed firms + Jet Airways as negative control.
- Amtek Auto and Videocon are kept. Flow items of any period longer or shorter than 12 months are scaled
  to 12 months (× 12 / months in the period); stock items (balance sheet) are used as reported. Each
  scaled period is flagged in the data file.
- India only. GCC firms are not included (no machine-readable source); recorded as a limitation.

## Amendment 1 (2026-11-21, before any control candidate's figures were downloaded)

Screener's peer list does not work for delisted companies (for Lanco it returns unrelated firms), so the
candidate pool in the control-matching rule is replaced by: **the first page (25 largest by current market
capitalisation) of the failed firm's Screener industry list** (the most specific industry level shown on the
failed firm's page). Lanco Infratech's page carries no industry tag; its candidates come from **Civil
Construction** (its main segment was EPC/construction), a judgement recorded here. Screener classifies ABG
Shipyard under Commercial Vehicles; that classification is used as is. Era Infra and Lanco share an
industry, so controls are assigned without replacement in event-date order. Everything else in the rule is
unchanged (no IBC case 2015–2023, ≥ 8 annual filings before the event, closest total assets in the failed
firm's last pre-event fiscal year within 0.5–2×, ties to the longer history). Jet Airways is a negative
control and gets no matched control.

## Amendment 2 (2026-11-21, after a first matching run that read only year headers, no candidate figures)

The "≥ 8 annual filings before the event" condition was wrongly applied to controls. The pre-registered
rule (PREREGISTRATION §2) requires it of failed firms only; controls are "matched on sector and size band"
with no distress in the window. It also cannot be met: Screener's public page shows only the latest 12
fiscal years, so every firm still listed today starts around FY2015 and all candidates were rejected.
Controls now need: total assets reported in the failed firm's reference fiscal year, and at least
`cohort_ingest.control_min_pre_event_periods` annual periods before the event (so the Layer 1/2 window can
be partly evaluated). The shorter control history is a stated limitation: for 2017 events, controls cover
about three of the five pre-event window years.
