# Model Card — SmogSense Production Model Bundle

Based on *Model Cards for Model Reporting* (Mitchell et al., 2019), standardized for operational probabilistic air-quality forecasting.

---

## 1. Model Details

### Basic Metadata

| Attribute | Specification |
|---|---|
| Model Name | SmogSense Physics-Residual Meta-Quantile Forecaster (Models M5, M6, M7) |
| Architecture Version | Revision 3 (Physics-Residual Meta-Quantile Learning, ADR-024) |
| Intended Domain | Lahore, Pakistan (Latitude 31.52° N, Longitude 74.35° E) |
| Issuance Cadence | Daily at 00:00 UTC (05:00 PKT) |
| Target Quantity | 24-hour block-mean PM2.5 concentration at horizons $h \in \{24, 48, 72\}\,\text{hours}$ |
| Output Format | 19 canonical quantile levels $\tau \in \{0.05, 0.10, \dots, 0.95\}$ in $\mu\text{g/m}^3$ |
| Primary Developers | SmogSense Engineering & Science Pair |
| License | MIT License |
| Execution Environment | Zero-cost container parity: Ubuntu 24.04 LTS, Python 3.12, PyTorch 2.14.1 (CPU), LightGBM 4.7.0 |

### Architectural Stack

The system decouples regional airshed advection from sparse local micro-climate calibration into two distinct components:

1. **Frozen Physics Base (Model M5)**:
   - 57 independent gradient-boosted trees ($3\text{ horizons} \times 19\text{ quantiles}$) trained on the dense Delhi reference network (Block E).
   - Trained with quantile pinball loss, validation early stopping on Block V, and stochastic feature group dropout on FIRMS fire exposure ($p=0.10$) and CAMS AOD ($p=0.10$) for graceful degradation.
   - Non-crossing quantile guarantee enforced via post-hoc monotone rearrangement with physical clipping $[0.0, 2000.0]\,\mu\text{g/m}^3$ (Chernozhukov et al., 2010).
2. **Neural Residual Calibrator (Models M6 & M7)**:
   - Architecture: 6-channel sequence encoder (`LocalGRUEncoder`: 2 layers, hidden size 64, latent dimension 32) coupled with a 14-dimensional tabular context vector into a 2-layer MLP head (`ResidualQuantileHead`: 46 $\to$ 64 $\to$ 57).
   - Parameters: Exactly **47,577** neural weights. Model M7 additionally learns **14** per-parameter-tensor Meta-SGD step sizes $\boldsymbol{\alpha}$.
   - Meta-Learning: First-Order MAML (FOMAML) with Meta-SGD trained across sparse station subsets ($N \in \{1, 2, 4\}$).
   - Zero-Initialization Invariant: Output projection initialized to zero, guaranteeing untrained/fallback $\Delta\hat{z} = 0$ bit-for-bit.

---

## 2. Intended Use & Scope

### Primary Intended Use
- Daily operational probabilistic 72-hour air quality guidance for the citizens and public health stakeholders of Lahore, Pakistan.
- Automated publication of interactive forecast cards, bulletin JSON feeds, and uncertainty intervals (10th–90th percentile) to GitHub Pages.

### Out of Scope & Misuse
- **Not a substitute for official regulatory monitoring**: SmogSense provides atmospheric research guidance and cannot replace certified government legal instruments.
- **Not for clinical triage**: Atmospheric forecasts should not be used as the sole basis for acute individual medical interventions.
- **Geographic Limits**: Calibrated specifically for the Lahore airshed (radius $\le 30\,\text{km}$). Application to other cities requires local support window adaptation.
- **Horizons Beyond 72 Hours**: Atmospheric dispersion predictability degrades sharply beyond day 3; projections beyond 72 hours are strictly prohibited.

---

## 3. Factors & Airshed Physics

The Indo-Gangetic Plain (IGP) experiences extreme winter pollution episodes governed by coupled meteorological and anthropogenic phenomena:

| Physical Factor | Atmospheric Mechanism | SmogSense Representation |
|---|---|---|
| Nocturnal Inversions | Shallow planetary boundary layer ($< 200\,\text{m}$) traps surface emissions | Log-ventilation coefficient $\ln(\text{VC}) = \ln(\text{PBLH} \times U_{10})$ across 24h, 48h, 72h |
| Transboundary Fire Plumes | Post-monsoon paddy residue burning in Punjab/Haryana | Target-centric forward advection exposure $E_{24}, E_{48}, E_{72}$ derived from FIRMS MODIS/VIIRS |
| Aerosol Optical Depth | Column-integrated aerosol loading | CAMS EAC4 / operational AOD at 550 nm with satellite feature dropout |
| Hygroscopic Swelling | Relative humidity $> 70\%$ causes aerosol deliquescence and sensor scattering bias | Trailing RH features; support-hour humidity audit and pre-correction |
| Sensor Divergence | Low-cost optical sensors exhibit gain/offset drift relative to reference beta-attenuation monitors | Empirical humidity-aware pre-correction ($a \cdot x / (1 + b \cdot \text{RH}) + c$) |

---

## 4. Training Data & Lineage

The training regime follows strict chronological season blocking to eliminate data leakage:

| Block | Seasons Covered | Domain | Role & Purpose |
|---|---|---|---|
| Block E (Source) | 2018–2019 through 2023–2024 | Delhi CAAQMS | Primary base training and out-of-fold cross-fitting ($K=3$ folds) |
| Block V (Tuning) | 2024–2025 | Delhi CAAQMS | Early stopping validation, hyperparameter tuning, residual bound calibration ($\rho_{99}$) |
| Block T (Confirmatory) | 2025–2026 | Delhi CAAQMS | Pre-registered confirmatory testing for Hypothesis H5 ($\ge 20$ stations) |
| Transfer Target | 2024–2025 dev window | Lahore (8664) | Few-shot deployment adaptation evaluation and ladder benchmarking |

### Complete Observation Lineage
- Ground observations: OpenAQ AWS S3 public archive (hourly PM2.5, temperature, relative humidity).
- Meteorology: ECMWF ERA5 reanalysis and operational IFS forecasts via Copernicus Climate Data Store.
- Atmospheric chemistry: ECMWF CAMS EAC4 reanalysis and global atmospheric composition forecasts.
- Satellite thermal anomalies: NASA FIRMS active fire hotspots (MODIS Terra/Aqua and VIIRS Suomi-NPP/NOAA-20).

---

## 5. Evaluation & Transfer Benchmarks

### Lahore Transfer Ladder (US Consulate Monitor 8664)

Evaluated across the prospective winter development window on Lahore reference monitor 8664:

| Model Rung | Description | MAE ($\mu\text{g/m}^3$) | CRPS ($\mu\text{g/m}^3$) | Skill vs CAMS M1 |
|---|---|---|---|---|
| M0 | 30-day trailing climatology | 89.40 | 64.20 | -53.6% |
| M1 | Raw operational CAMS IFS | 56.01 | 41.80 | Baseline (0.0%) |
| M2 | CAMS with 7-day bias correction (CAMS-BC) | 32.40 | 24.10 | +42.3% |
| M3b | Gradient-boosted bias corrector | 24.15 | 18.05 | +56.8% |
| M5 | Frozen LightGBM physics base (Delhi Block E $\to$ Lahore) | **17.62** | **12.85** | **+69.3%** |
| M7 | Physics-residual meta-quantile learning (FOMAML + Meta-SGD) | **16.10** | **11.92** | **+71.5%** |

### Key Benchmark Findings
1. **Physics Base Transfer (M5)**: Achieved a **68.6% error reduction** over raw CAMS ($56.01 \to 17.62\,\mu\text{g/m}^3$) without consuming a single Lahore training label, proving cross-border transferability of meteorological advection representations.
2. **Residual Bound Calibration ($\rho_{99}$)**: Evaluated across 200 deployment simulations on Block V; threshold calibrated at $\rho_{99} = 0.362452\,\text{log1p units}$, guarding against erratic neural adaptation overshoots.
3. **Hypothesis H5 Confirmed**: Meta-SGD adaptation achieved a point CRPS ratio of $0.927 \le 0.97$ over supervised fine-tuning M6 on Delhi Season T across 49 eligible reference stations ($p < 0.001$, analytical power $> 99\%$).

---

## 6. Operational Guardrails & Failure Modes

### Acceptance Gate Engine (Condition C6 & C7)
Before deploying adapted residual weights $\theta^*$, the engine executes a three-way check on a chronological holdout window:
- Relative pinball loss improvement over M5 $\ge 2\%$.
- Daily win fraction over M5 $\ge 60\%$.
- Mean absolute residual $R \le \rho_{99} = 0.362452$.

If any condition fails, the pipeline sets $\Delta\hat{z} = 0$ identically, falling back to Model M5 with **degradation level 0**.

### Operational Degradation Ladder

| Level | Condition | Active Pipeline Engine |
|---|---|---|
| Level 0 | Normal operations | Model M7 (or M5 fallback if check gate rejects) |
| Level 1 | Late CAMS delivery ($> 02:47\,\text{UTC}$) | Previous CAMS cycle ($\Delta t + 12\,\text{h}$) |
| Level 2 | Missing satellite feeds or bundle mismatch | Model M5 with feature group dropout, or CAMS-BC (M2) |
| Level 3 | Complete OpenAQ / CAMS outage | Trailing 30-day station climatology (M0) |

---

## 7. Ethical, Public Health & Communication Standards

1. **Uncertainty Communication**: Point forecasts can create false precision. SmogSense mandates communicating the full 80% credible interval ($q_{10}$ to $q_{90}$) alongside median expectations.
2. **Health Breakpoints**: Health category color coding strictly follows US EPA 2024 revised annual PM2.5 standards:
   - Good: $0.0 - 9.0\,\mu\text{g/m}^3$ (Green)
   - Moderate: $9.1 - 35.4\,\mu\text{g/m}^3$ (Yellow)
   - Unhealthy for Sensitive Groups: $35.5 - 55.4\,\mu\text{g/m}^3$ (Orange)
   - Unhealthy: $55.5 - 125.4\,\mu\text{g/m}^3$ (Red)
   - Very Unhealthy: $125.5 - 225.4\,\mu\text{g/m}^3$ (Purple)
   - Hazardous: $\ge 225.5\,\mu\text{g/m}^3$ (Maroon)
3. **Bilingual Accessibility**: Bulletins are formatted with parallel English and Urdu advisory copy to ensure wide accessibility across local communities.

---

## 8. Verification & Maintenance

- **Cryptographic Binding**: Calibrator bundles enforce `base_sha256` matching against the active physics base booster trees. Mismatched binaries trigger immediate fallback to Level 2.
- **Rollback Protocol**: Changing `promoted_bundle: none` in `configs/model.yaml` instantly reverts the operational forecast to the baseline physics base.
- **Scheduled Retraining**: Retraining is automatically triggered upon any CAMS model cycle upgrade or if rolling 14-day 80% coverage deviates outside $[0.70, 0.90]$.
