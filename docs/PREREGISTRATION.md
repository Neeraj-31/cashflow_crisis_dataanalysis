**Date: 2026-11-21**

# Pre-registration: Early Warning of Cash-Flow Crises: A Layered, Self-Referential Monitoring Framework

**Status: complete; locks when committed.** H1–H6 (§9) are copied verbatim from the blueprint (v2).
Commit this file, with the config, before any detector backtest or cohort data. Every number below is mirrored in `configs/config.yaml`
(`prereg_date: "2026-11-21"`). Code reads only the config; if the two disagree, that is a bug. Any change
after locking goes in the deviations log (§10) with its date and reason.

## 1. Confirmatory vs exploratory

- **Exploratory:** the case studies, NMC Health and Jyoti Structures. Their data were examined before this
  document was written, so they motivate the framework but cannot test it.
- **Confirmatory:** the validation cohort (§2), which will be assembled and evaluated only under the rules
  below.

## 2. Cohort selection rule (`cohort`)

- **Rescoped target:** the blueprint's 8–12 failed firms is rescoped to 6–8 by the 5-day plan.
- **Failed firms:** listed non-financial companies in India or the GCC that entered formal insolvency,
  payment default or restructuring with the event dated in **2015–2023** (inclusive).
  - Each must have **≥ 8 years of annual filings before the event**.
  - Financials (banks, NBFCs, insurers, REITs) are excluded.
- **Target:** **6–8 failed firms.**
- **Controls:** **one matched healthy control per failed firm**, matched on sector and size band.
  - A control must have no distress event in the same window.
  - Each control is aligned in event time at its matched failed firm's event date.

## 3. Event definition and alignment (`firms`, `layer1`)

- **Event date:** the date of the first formal event (insolvency admission, administration, default
  announcement or restructuring).
- **Event time** τ is counted in fiscal years. τ = 0 is the fiscal year containing the event date; the
  Layer 1 window is τ = −5…0.
- **Lead time:**
  - *lead_fy* = event FY − first-trip FY.
  - *lead_years* = (event date − first-trip fiscal year-end) / 365.25.
  - This is measured from fiscal year-end, not from when the accounts were published, so the usable
    warning is shorter by the reporting lag.
- **Case-study event dates** (verify against primary sources before locking):

| Firm | Event date | Event | Event FY |
|---|---|---|---|
| NMC Health | 2020-04-09 | NMC Health PLC placed into administration by the English High Court (the ADGM court followed for the UAE operating companies on 2020-09-27) | FY2020 (Dec year-end) |
| Jyoti Structures | 2017-07-04 | NCLT Mumbai admits CIRP under the IBC | FY2018 (Mar year-end) |

## 4. Data handling (`ratios`, `identities`)

- **Orientation:** every monitored ratio is oriented so that **lower = worse**. Higher-is-worse
  quantities are negated (`neg_` prefix).
  - Payable days (DPO) are treated as higher = worse, since stretched payables signal supplier financing
    under stress.
  - CCC components (DSO, DIO, DPO) are monitored separately from the CCC.
- **Missing years** stay missing and are never interpolated (`fill_missing: none`). Growth rates need
  two consecutive fiscal years.
- **Near-zero denominators** are masked as missing, never divided. The test is |denominator| ≤ ε ×
  |total assets|:
  - ε = 0.01 for revenue/COGS denominators (margins, days);
  - ε = 0.001 for all other denominators.
- **Ratios undefined for a non-positive denominator** are also masked: debt/equity when equity ≤ 0, and
  net debt/EBITDA or OCF/EBITDA when EBITDA ≤ 0.
- **Accounting identities** are checked before any analysis, and any breach stops the run:
  - assets = liabilities + equity;
  - EBITDA + other income − interest − depreciation = PBT, where available;
  - DSO + DIO − DPO = CCC.
  - Tolerance: max(3 units, 0.1% of total assets), and 3 days for the CCC identity.
- **Anomaly flag:** OCF greater than 1.0 × revenue in a year sets `anomaly_ocf_gt_revenue`. The value is
  kept, not dropped.

## 5. Altman Z'' variant (`altman_z2_em`)

We use the **emerging-market variant** (Altman, 2005):

> Z''-EM = **3.25** + 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4

- X1 = working capital / total assets
- X2 = retained earnings / total assets
- X3 = EBIT / total assets
- X4 = book equity / total liabilities

**Zones:** safe if Z > **5.85**; grey if 4.35 ≤ Z ≤ 5.85; distress if Z < **4.35**. These are the
original Z'' cut-offs of 2.6 and 1.1 plus the 3.25 constant.

Where a source lacks a current/non-current split, proxies are documented in
`src/ingest/canonical.PROXIES`. For Jyoti, working capital uses "other assets − other liabilities",
which excludes short-term borrowings and therefore overstates X1.

## 6. Archetype rule (`archetype`)

Each failed firm is classified on its pre-event years:

- **Visible:** OCF < 0 in **≥ 2 consecutive** pre-event years.
- **Masked:** OCF stays positive while **≥ 2** of these structural flags trip:
  - liquidity cover, (cash + deposits) / short-term debt, < **1**;
  - EBITDA / interest < **1.5**;
  - Altman Z''-EM in the distress zone (< 4.35).
- **Other:** everything else.

## 7. Layer 2 rule thresholds (`layer2.rules`)

| Rule | Trips when |
|---|---|
| Operating cash flow | OCF < **0** |
| Interest cover | EBITDA / interest < **1.2** (interest ≤ 0 ⇒ not tripped; evaluated cross-multiplied) |
| Collections | Debtor days (DSO) > **270** |
| Altman | Z''-EM < 4.35 (distress zone, §5) |

Each rule outputs tripped, not tripped, or not evaluable (inputs missing or masked). Per firm we report
the first pre-event trip year and the lead time for each rule, and for "any rule".

## 8. Detector parameters, Layer 3 (`detectors`)

- **CUSUM:** reference value **k = 0.5**, in training-window standard-deviation units.
- **Nominal false-alarm rates:** **5% and 10%**. Both are reported.
- **Training window:** pseudo out-of-sample split, using the **first 60%** of each firm's pre-event
  years.
- **Bootstrap:** **circular block bootstrap** with **2,000 resamples** and seed 20261121.
  - **Block length rule:** b = max(2, ⌈n_train^(1/3)⌉).
  - Data-driven selectors such as Politis–White are not used, because they are unstable at the short
    annual lengths here (n_train ≈ 5–9).
- **Calibrator** (blueprint §7, Layer 5):
  - Refit the model on the first 60% of the training window.
  - Filter the remaining 40% with frozen parameters to get pseudo-out-of-sample innovations, and centre
    the pool.
  - Draw 2,000 circular-block-bootstrap monitoring paths of length 16.
  - h_α = the (1 − α) quantile of max_t S_t over those paths.
- **Univariate detector (Layer 3):** a statsmodels UnobservedComponents local level with AR(1) noise,
  fitted on the training window. The full series is then filtered with the parameters **frozen**, and
  the standardised one-step innovations z_t feed a one-sided CUSUM that accumulates deterioration:
  S_t = max(0, S_{t−1} − z_t − k), alarm when S_t > h. Missing years are handled natively by the Kalman
  filter.
- **Baselines:** ARIMA(1,0,0) with the same CUSUM; a naive random-walk forecast with CUSUM; and the
  Altman Z''-EM zones.
- **Multivariate detector (Layer 4):** the univariate model is run on each of 4–6 oriented ratios.
  Covariance Σ is the Ledoit–Wolf estimate on training innovations, and the alarm is Crosier's MCUSUM:
  C_t = √((S_{t−1}+z_t)ᵀ Σ⁻¹ (S_{t−1}+z_t)); S_t = 0 if C_t ≤ k, else (S_{t−1}+z_t)(1 − k/C_t).
- **Alert tiers** (blueprint §7):
  - **Amber:** either detector crosses its lower threshold (h at 10%).
  - **Red:** the multivariate detector crosses h at 5% *and* at least one Layer 2 rule trips.
- **Minimum data:** Layers 3 and 4 run only where ≥ 12 usable periods exist. NMC (6 annual points)
  stays at Layers 0–2 unless half-yearly data is added, in which case it runs with a thin-baseline
  warning.

## 8a. Simulation and calibration study (`simulation`)

- **Generator:** AR(1) ratios with correlated innovations and a known onset.
- **Scenarios:**
  - S1: drift in the target ratio.
  - S2: target ratio healthy, other ratios deteriorating.
  - S3: small drift in all ratios.
  - S4: concealment. True deterioration hits all ratios at the S1/S2 drift rate, but the reported
    target (cash) ratio is an exponentially smoothed version of its true value over the whole series:
    reported_t = 0.3 · true_t + 0.7 · reported_{t−1}. The other ratios are reported as is. This models
    NMC: structural ratios worsen while the reported cash ratio lags and looks calm. Set on 2026-11-21,
    before any simulation run.
  - Ratio 0 is the target (cash) ratio. Ratios 1–3 are structural. The univariate detectors watch
    ratio 0. The multivariate detector for H3 watches all four. The structural detector for MI watches
    ratios 1–3.
  - **Alarm definitions:**
    - A healthy firm's false alarm is any alarm in its 16 monitored periods.
    - A detection is a first alarm at or after onset. An alarm before onset counts as a false alarm,
      not a detection.
  - **Multivariate direction:** Crosier's MCUSUM is direction-agnostic (as specified), so it can also
    alarm on improvement. This is accepted and stated as a limitation.
  - S5 (heavy tails) and S6 (training-period breaks) are cut, following the blueprint's cut order.
- **Generator settings**, set before any run:
  - 4 ratios, AR(1) with φ = 0.5 and innovation equicorrelation 0.3;
  - 16 monitored periods, with onset at monitored period 4;
  - drift 0.25 SD per period (S1, S2) and 0.1 SD per period (S3);
  - power grid {0.05, 0.1, 0.15, 0.25, 0.4}.
- **Experiment A (calibration curve):**
  - Healthy firms only. Compare the empirical false-alarm rate with the nominal 5% and 10%, for training
    lengths n ∈ {12, 16, 20, 28, 40}.
  - At least 500 firms per cell, with a bootstrap CI on every rate.
  - The result is reported as it comes out; no outcome is assumed.
- **Fixes** are fitted on training firms and evaluated on **held-out** firms (separate seed):
  - (a) an inflation factor on h, as a function of n;
  - (b) a parametric bootstrap;
  - (c) a conformal threshold with finite-sample quantile ⌈(m+1)(1−α)⌉/m.
  - For each fix we report the corrected false-alarm rate with CIs and its cost in detection rate.
- **Matched-false-alarm comparison:** every detector is tuned to the same empirical false-alarm rate on
  held-out healthy firms before detection rates are compared across S1–S4.

## 8b. Masking index (`masking`)

- **Formula:** MI_t = S_struct,t / h_struct − S_cash,t / h_cash.
  - The structural run is the multivariate detector on liquidity cover, interest cover and Z''-EM.
  - The cash run is the univariate detector on OCF / current liabilities.
- **Detector-order fingerprint:** cash only / structural first / both together / neither.
- **Check:** in simulation, peak MI must separate S1 from S2 (blueprint) and S1 from S4 (5-day plan).
  We report the effect size (Cohen's d and AUC) with bootstrap CIs.

## 8c. Cohort extras, robustness, cost

- **Archetype first:** the archetype rule (§6) is applied **before** MI is computed.
- **Negative controls:** Jet Airways and Reliance Communications are reported separately, as stated
  limits of the method.
- **Robustness checks:**
  - training window ± 2 periods;
  - a (k, h) grid;
  - leave-one-ratio-out ablation;
  - Jyoti FY2017 handled by Kalman-missing vs linear interpolation;
  - alternative event definitions: first covenant breach, auditor emphasis of matter, default,
    restructuring, formal insolvency.
- **Expected cost:** C_FA = 1 and C_M ∈ {5, 10, 20}. We report the range of C_M / C_FA over which the
  framework beats the Altman Z''-EM baseline.

## 9. Hypotheses

Copied verbatim from the project blueprint (v2, §3). Each hypothesis is marked SUPPORTED, REFUTED or
INCONCLUSIVE strictly by these criteria. INCONCLUSIVE is used only when the required data or run does
not exist, and the reason is stated.

| ID | Question | Hypothesis | Supported if | Refuted if |
|---|---|---|---|---|
| RQ1 / H1 | Does a firm's own cash-flow history warn early in an operational crisis? | For Jyoti, a univariate state-space + CUSUM monitor alarms at least two fiscal years before formal insolvency, at a pre-set false-alarm rate | Lead time of 2 years or more | No alarm, or alarm after the event |
| RQ2 / H2 | Does cash flow hide distress when liabilities are financed away? | For NMC, cash-flow flags stay silent through 2018 while structural flags (liquidity cover, interest cover, Z'') trip at least one year earlier | Cash flags silent, structural flags trip first | Both silent, or both trip together |
| RQ3 / H3 | Does joint monitoring of several ratios beat monitoring one? | The multivariate detector wins when the target ratio is masked and loses slightly when the drift sits in the target ratio | Simulation shows this pattern | Opposite pattern |
| RQ4 / H4 | Can alarm thresholds be trusted with so little data? | Bootstrap-calibrated thresholds are over-liberal at about 28 training periods | Empirical false-alarm rate exceeds nominal | At or below nominal |
| RQ5 / H5 | Can the miscalibration be corrected cheaply? | A simulation-learned inflation factor or conformal threshold brings the empirical false-alarm rate within ±3 points of nominal at n = 28 | Rate within ±3 points on held-out simulated firms | Remaining gap larger than 3 points |
| RQ6 / H6 | Does the masking index separate visible from masked crises? | On the cohort, firms classified as masked by the pre-set rule have a higher peak masking index than visible firms, and the multivariate detector alarms first for them | Index separates the groups; detector order matches | No separation, or order reversed |

**Operational notes, fixed before any evaluation:**
- **H1:** the "pre-set false-alarm rate" is the 10% nominal rate, using the blueprint's uncorrected
  calibrator; the n-corrected threshold is reported alongside. The training window ends at FY2013
  (pre-FY14 healthy baseline). Lead time = event FY − first-alarm FY.
- **H2:** cash-flow flags are the Layer 2 rule OCF < 0. Structural flags are liquidity cover < 1,
  EBITDA / interest < 1.5 and Z''-EM < 4.35. "Trip first" means the first structural trip comes at
  least one fiscal year before any cash-flag trip, and no cash flag trips through FY2018.
- **H3 "loses slightly":** in S1, multivariate detection is below univariate detection at the matched
  false-alarm rate. In S2 and S4, multivariate detection is above univariate detection. Both judgements
  use point estimates, with CIs reported.
- **H4:** judged at n = 28 for both nominal rates. "Exceeds" means the lower 95% CI bound is above
  nominal.
- **H5:** judged on held-out firms. SUPPORTED if either fix (inflation factor or conformal) lands within
  ±3 points at the 10% nominal rate.

## 10. Deviations log

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-21 | §8a fix (b) | Parametric bootstrap uses 200 refitted resamples per firm instead of 2,000 | Each resample needs a model refit (~75 ms); 2,000 × 2,500 held-out firms is about 4 CPU-days. Decided before results were seen |
| 2026-11-21 | addendum | EDGAR US panel (config `edgar`) added as an exploratory external-transfer test | Not part of the pre-registered cohort test; rules written before download |
| 2026-11-21 | §2 event table | NMC event source corrected from "ADGM courts" to "English High Court"; the date (2020-04-09) is unchanged | Verified against primary records; the ADGM order (2020-09-27) covered the UAE subsidiaries |
| 2026-11-21 | robustness | Added `robustness` grids (CUSUM k, threshold multipliers, alternative event dates, NMC hidden-debt stress test) after the main results were seen | Exploratory sensitivity analyses only; no verdict uses them |
| 2026-11-21 | §2 cohort | Sampling frame fixed as RBI's 2017 large-defaulter lists (first list; second-list names confirmed in press), event = NCLT admission date; India only (no machine-readable GCC source) | The rule named markets but no frame; fixed and fingerprinted (`data/raw/cohort/PROPOSAL.md`) before any figure was read |
| 2026-11-21 | §2 cohort | Controls: candidate pool = first page of the failed firm's Screener industry list (peer list fails for delisted firms); ≥ 8-year history required of failed firms only, controls need ≥ 3 pre-event years | Amendments 1–2 in PROPOSAL.md, each fingerprinted before candidates' figures were read; Screener shows only 12 years for listed firms |
| 2026-11-21 | §2 cohort | Flows of non-12-month fiscal periods scaled to 12 months (Amtek Auto, Videocon changed year-ends) | Approved by the project owner before data collection |
| 2026-11-22 | addendum (retrospective) | Config blocks `cohort_ingest` and `sanity` (both added 2026-11-21) named here retrospectively. `cohort_ingest` is used only by data collection, `sanity` only by `src/sanity.py`; no analysis or verdict reads either | A new sanity check requires every block added after the fingerprint to be named in this log; these two had been described but not named |
| 2026-11-22 | addendum (post-hoc) | New config block `evaluation` (`ci_resamples: 2000`), the value previously hard-coded in `src/run_edgar.py`; every EDGAR detection / false-alarm CI now comes from one module (`src/evaluation.py`) and one table (`edgar_detection_summary.csv`). The main table is byte-identical | The explainer and the sanity table quoted different CIs for the same quantity because they used two bootstrap seed streams |
| 2026-11-22 | addendum (post-hoc) | New config block `post_hoc` and scripts `src/post_hoc.py`, `src/simulate/self_starting.py`: paired matched-unit bootstrap (rules vs each method), fixed-horizon evaluation, persistence variants, filing-date lead times, TTM-overlap re-run of Layer 3, winsorisation at 1%/99%, industry mix, healthy-centred masking index, self-starting CUSUM benchmark in the H4/H5 setting. Added after every result had been seen; post-hoc / exploratory; no verdict uses them | Response to a review of `reports/explainer.html` (decision log #34-#47) |
| 2026-11-22 | reporting (post-hoc) | Figure 8 (EDGAR event time) and `robust_edgar_level_vs_change.csv` restricted to the analysed sample (91 failed / 46 controls); the wider sample is kept as a second set of rows. The quoted median training OCF / current liabilities changed from 0.35 / 0.65 to 0.33 / 0.51 (failed / controls) | The figure and the quoted medians described firms that are not in the analysed sample |
| 2026-11-22 | reporting | Wording only, no verdict changed: H3 gets an 'Interpretation: partially supported' line beside the unchanged code-computed verdict; 'rules beat Altman' became 'not distinguishable' (paired CIs include zero); 'statistical alarms barely beat chance' became 'no detectable advantage over chance (CI includes zero)'; the H6 cohort order-of-flags claim is restricted to non-Visible firms (Visible is definitional) | Over-claims found in the review |
| 2026-11-25 | addendum (post-hoc) | New config block `eda` and script `src/eda.py`: descriptive EDA of the data properties behind the design choices (training quarters only), read by the explainer's design-choice table. No detector, threshold or verdict reads it | Document why each fixed setting suits the data; the rationales are retrospective and labelled so (decision log #48) |
