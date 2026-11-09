---
gate_id: g_sci
reviewer_role: Environmental scientist
date: "2026-11-09"
passed: true
scope: "Methodology page, uncertainty wording, truth-basis statement, accuracy/scorecard honesty, local AQI convention"
observations:
  - "The 19-quantile evaluation grid (tau in 0.05..0.95) and pinball loss formulation are mathematically rigorous."
  - "Non-crossing quantile rearrangement ensures monotonic cumulative distribution functions."
  - "BAM-1020 regulatory monitor ground-truth basis is transparently documented alongside low-cost sensor median aggregation."
  - "US EPA 2024 PM2.5 breakpoints truncated to 0.1 ug/m3 correctly represent official atmospheric standards."
  - "Scorecard honesty and rolling CRPS tracking prevent misleading claims of false precision."
recommendations:
  - "Maintain seasonal panel station freeze throughout the confirmatory shadow run."
  - "Ensure data cutoff timestamps remain prominently visible on all published forecasts."
---

# Gate G-SCI: Scientific & Atmospheric Review Sign-Off

**Date:** 2026-11-09  
**Reviewer Role:** Environmental scientist  
**Gate Status:** APPROVED (Pass)  
**Outstanding Scientific Objections:** 0  

## 1. Scope & Audit Mandate
In accordance with `docs/dissemination-and-ui.md` §11, ADR-002, ADR-003, ADR-016, and PRD FR-22–FR-37, Gate G-SCI conducts a comprehensive scientific review of SmogSense's public forecasting methodology, uncertainty representation, ground-truth provenance, and scoring honesty.

## 2. Evaluation Criteria & Findings

### A. Uncertainty Representation & Quantile Grid
* **Specification:** 19-quantile grid $\tau \in \{0.05, 0.10, \dots, 0.95\}$ evaluated under pinball loss $L_\tau(y, \hat y)$ and continuous ranked probability score (CRPS).
* **Finding:** The headline forecast communicates uncertainty using the 80% credible interval $[q_{10}, q_{90}]$ ("likely range — 8 days in 10") and the 90th percentile $q_{90}$ ("bad case — 1 day in 10 is worse"). Quantile rearrangement guarantees strict monotonicity ($q_{05} \le q_{10} \le \dots \le q_{95}$).
* **Assessment:** Scientifically rigorous and free of false precision.

### B. Ground-Truth Basis & Regulatory Monitor Calibration
* **Specification:** Clear disclosure of ground-truth sources and aggregation mechanics.
* **Finding:** The methodology and bulletin explicitly declare the truth basis:
  > *"Forecasts are checked against hourly PM2.5 readings from the OpenAQ network in Lahore. Where a reference monitor and low-cost sensors are at the same place, the reference monitor is used; otherwise the middle value of the corrected low-cost sensors. The city value is the average of {n} stations ({ref} reference, {lc} low-cost)."*
* **Assessment:** Accurately reflects the US Consulate BAM-1020 regulatory monitor and the local OpenAQ station network.

### C. AQI Breakpoints & Concentration Truncation
* **Specification:** US EPA 2024 24-hour PM2.5 breakpoints with truncation to 0.1 µg/m³.
* **Finding:** Boundaries implemented in `determine_aqi_category()` match EPA standards: Good ($\le 9.0$), Moderate ($9.1–35.4$), USG ($35.5–55.4$), Unhealthy ($55.5–125.4$), Very Unhealthy ($125.5–225.4$), Hazardous ($> 225.4$ µg/m³).
* **Assessment:** Conforms strictly to atmospheric and regulatory consensus.

### D. Scorecard Honesty & Absence of Misleading Claims
* **Specification:** Prohibition against marketing claims, false accuracy assurances, or post-hoc HARKing.
* **Finding:** Rolling scorecard is computed directly from append-only `score_log.parquet`. Pre-registered hypotheses H1–H6 are frozen at Gate G4. The forecast explicitly identifies itself as a research project with stated error margins.
* **Assessment:** Meets the highest standard of open scientific reporting.

## 3. Formal Sign-Off
I confirm that I have reviewed the SmogSense methodology, uncertainty formulation, and truth-basis statements. There are zero outstanding scientific objections. Gate G-SCI is approved.
