# SmogSense Gate G4 Confirmatory Pre-Registration: Probabilistic 72-Hour PM2.5 Transfer Forecasting

## Cryptographic Provenance & Freeze Register Binding

| Item | Value |
|---|---|
| Document Title | SmogSense Gate G4 Confirmatory Pre-Registration: Probabilistic 72-Hour PM2.5 Transfer Forecasting |
| Registration Date | 2026-12-04 (Gate G4 Freeze) |
| Created At (UTC) | `2026-11-09T12:11:30.643839+00:00` |
| Git Release Tag | `v0.3.0-gate-g4` |
| Git Commit SHA | `0709f9daffdbf65c42272b137132c67811c90af8` |
| Deterministic Config Hash | `4a907bee381036b4a4f472cf9be7985574c0e118327876f0f521ff60ab200dcd` |
| Freeze Register Digest | `36563dc7f2911b76cfd95336daee46311d273a698e5a49a7f5714a56991975e7` |
| Candidate Calibrator (M7) SHA-256 | `cda76200533985b86fb2c8c09ae7349d6ab3ab69e19f695cef2e751dc23bab7b` |
| Shared Physics Base (M5) SHA-256 | `be0f18fbf0d02d51cf03e30b247daa678461b4b6146fe57cf52a8986da1a60c5` |
| Headline Candidate Model | `m7_hybrid_maml` |

---

## 1. Context & Research Questions

Ahead of Gate G4 (4 December 2026) and the prospective confirmatory shadow evaluation window (7 December 2026 - 31 January 2027), SmogSense commits this immutable pre-registration record. In accordance with modern empirical open science, scores are never recomputed retroactively, and candidate models are evaluated against pre-registered criteria.

| RQ | Core Research Question | Primary Rung Comparison | Evaluation Metric |
|---|---|---|---|
| **RQ1** | Does the learned system beat the fair physics baseline (bias-corrected CAMS) for Lahore at 24, 48, and 72 h? | M7 vs M2 (CAMS-BC) | Continuous Ranked Probability Score (CRPS) |
| **RQ2** | How does probabilistic forecast skill degrade as available ground stations $N \to 0$ (the Sensor-Sparsity Curve)? | M7, M6, M5, M4, M3b | $\mathrm{CRPS}(N) = c_\infty + (c_0 - c_\infty) e^{-N/\nu}$ |
| **RQ3** | Does meta-learning (Meta-SGD) add value over plain fine-tuning of a Delhi-pretrained base in sparse settings? | M7 vs M6 | CRPS Ratio (M7 / M6) on Delhi Season T |
| **RQ4** | Are 80% central prediction intervals honest and calibrated under prospective operational conditions? | Headline Model | 80% Interval Coverage and Median PIT |

---

## 2. Pre-Registered Hypotheses & Scientific Tiers

Hypotheses are structured into strict scientific tiers to separate confirmatory power from exploratory diagnostics:
- **Tier 1, Powered (H5)**: Delhi Season T contains 49 regulatory monitors and multi-year data, providing adequate sample size ($100.0\%$ power at $\alpha = 0.05$) to detect a $3\%$ skill gain.
- **Tier 1, Indicative (H1-H4)**: Evaluated prospectively on the Lahore City Panel. Given single-season meteorological correlation ($n_{\text{eff}} \approx 19\text{--}28$), tests are reported with confidence intervals and Holm-Bonferroni corrections. Tests degrade to descriptive if $< 35$ confirmatory days are observed.
- **Tier 2, Descriptive (H6)**: Evaluated as descriptive calibration diagnostics without hypothesis rejection penalties.

| ID | Tier | Statement | Domain | Population | Metric | Horizon | $N$ | Threshold / Operator | Family |
|---|---|---|---|---|---|---|---|---|---|
| `H1` | Tier 1 Indicative | M7 beats CAMS-BC at 24 h with N=5 available stations | `lahore` | Lahore City Panel (leave-stations-out, N=5) | `crpss_vs_m2` | `24` | `5` | `>= 0.15` | `family_1_lahore` |
| `H2` | Tier 1 Indicative | M7 beats CAMS-BC at 72 h with N=5 available stations | `lahore` | Lahore City Panel (leave-stations-out, N=5) | `crpss_vs_m2` | `72` | `5` | `>= 0.1` | `family_1_lahore` |
| `H3` | Tier 1 Indicative | M7 beats raw CAMS at every horizon | `lahore` | Lahore City Panel (leave-stations-out, N=5) | `crpss_vs_m1` | `all` | `5` | `>= 0.25` | `family_1_lahore` |
| `H4` | Tier 1 Indicative | Transfer alone (N=0) still adds skill over CAMS-BC; otherwise report negative transfer | `lahore` | Lahore City Panel (pure zero-shot / sensor-free mode, N=0) | `crpss_vs_m2` | `24` | `0` | `>= 0.05` | `family_1_lahore` |
| `H5` | Tier 1 Powered | Meta-learning adds value over plain fine-tuning at N in {1,2} | `delhi` | Delhi Season T Regulatory Monitors (>= 20 stations) | `crps_ratio_m7_over_m6` | `all` | `[1, 2]` | `<= 0.97` | `family_2_delhi_h5` |
| `H6` | Tier 2 Descriptive | Calibration: pooled coverage of [q10,q90] within +/-0.05 of 0.80; P(y<=q50) within 0.07 of 0.50 | `lahore` | Pooled Spatial-Temporal City Panel | `coverage_80` | `all` | `0` | `+/-0.05 0.8` | `unadjusted_h6` |

---

## 3. Multiplicity Control Protocol

To control family-wise error rate (FWER) without compromising statistical discovery:
1. **Family 1 (Lahore Confirmatory Hypotheses)**: Step-down Holm-Bonferroni correction applied over the family $\{H_1, H_2, H_3(24), H_3(48), H_3(72), H_4\}$ at $\alpha = 0.05$.
2. **Family 2 (Delhi Transfer Power Test)**: Independent confirmatory test for $H_5$ evaluated at $\alpha = 0.05$ one-sided.
3. **Family 3 (Calibration Diagnostics)**: Descriptive evaluation of $H_6$ coverage (target $80\% \pm 5\%$) and median coverage (target $50\% \pm 7\%$) without rejection penalty.

---

## 4. Statistical Power & Sample Size Note

Empirical parameters established during the Task P2-14 pilot study:

| Parameter | Empirical Value | Specification Note |
|---|---|---|
| H5 Test Domain | `delhi` | Delhi airshed regulatory monitoring network |
| Evaluated Split Season | `2025-2026` | Confirmatory Season T split |
| Eligible Reference Monitors | **49 / 49** | Exceeds $\ge 20$ monitor threshold requirement |
| Daily Log-Ratio SD ($\sigma$) | **0.0004** | Measured variance across pilot evaluation |
| Effective Sample Size ($n_{\text{eff}}$) | **8.00** | From 25 pilot days, lag-1 autocorrelation $\rho_1 = +0.515$ |
| Analytical Statistical Power | **100.0%** | At $\alpha = 0.05$ one-sided for target gain $\delta = 0.03046$ |

> [!NOTE]
> **Lahore Confirmatory Power Note**: Confirmatory window spans 56 days (7 Dec 2026 - 31 Jan 2027). Under AR(1) temporal autocorrelation rho_1 ≈ 0.5, effective sample size is n_eff ≈ 19 - 28. Tests H1-H4 are evaluated as indicative with Holm-Bonferroni FWER control, and degrade to descriptive if fewer than 35 valid days are observed or n_eff < 25.

---

## 5. Confirmatory Evaluation Calendar & Invariants

| Milestone | Target Dates | Protocol Description |
|---|---|---|
| Development History End | `2026-10-18` | Historical baseline data cut-off date |
| Dev Replay Window | `2026-10-19 to 2026-12-06` | 49-day replay shadow window over point-in-time logged inputs |
| Gate G4 Freeze Date | `2026-12-04` | Model weights, configurations, and hypotheses permanently frozen |
| Confirmatory Shadow Window | `2026-12-07 to 2027-01-31` | 56-day prospective confirmatory evaluation period |
| Settled Truth Archive Delay | `96 hours` | Settlement delay for OpenAQ AWS historical archive validation |
| Analysis Freeze (Gate G7) | `2027-02-08` | Final scoring freeze and thesis analysis compilation |

### Invariant Data Integrity Rules
- **18/24h Completeness Rule**: Observations require $\ge 18/24$ valid, non-imputed hourly readings to form a valid block mean. Imputed values never count towards completeness.
- **Point-in-Time Correctness**: All features at forecast issuance $T$ use only observations satisfying $\text{available\_at} \le T$.
- **Zero Retroactive Rescoring**: Prospective forecasts logged to `.state/forecast_log/` are permanently immutable.
- **Cryptographic Binding**: Candidate residual calibrator must declare matching `base_sha256` matching the frozen physics base.

---

## 6. Negative Results & Occam's Razor Policy

SmogSense adheres to a strict hierarchical degradation and negative-results reporting policy:
1. If Model M7 (Meta-SGD) does not demonstrate statistically significant gain over Model M6 (fine-tuning) on Delhi Season T ($H_5$ not met), the simpler fine-tuning architecture (Model M6) is favored.
2. If neither calibrator demonstrates superior skill over the frozen physics base (Model M5) under the deployment acceptance gate, Model M5 serves as the headline model, demonstrating the bounds of pure zero-shot physics transfer.
3. If transfer skill falls below bias-corrected CAMS ($H_4$ not met), negative transfer is formally documented.
4. All negative, null, or confirmatory results are reported with identical scientific prominence in the final capstone thesis.
