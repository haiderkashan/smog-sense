# Model card — SmogSense bundle `<bundle_version>`

> Based on *Model Cards for Model Reporting* (Mitchell et al., 2019), adapted to probabilistic air-quality forecasting.
> Copy this file into the bundle as `model_card.md` and fill in every field. A card with an unfilled field blocks promotion.

## 1. Model details

| Field | Value |
|---|---|
| Bundle version / git SHA / config hash | |
| Architecture | GRU/TCN encoder + non-crossing neural head (meta-learned) + LightGBM quantile ensemble + optional CQR/ACI calibration |
| Output | 19 quantile levels (0.05–0.95) of the 24-h block-mean PM2.5 for h in {24, 48, 72}, µg/m³ |
| Training domains and seasons | |
| Meta-learning algorithm and hyperparameters | |
| Training date, hardware, wall-clock | |
| CAMS model cycles present in training data | |

## 2. Intended use

Daily probabilistic PM2.5 guidance for Lahore as a *research* product with a public bulletin. **Out of scope:** regulatory
reporting, individual medical decisions, forecasts beyond 72 h, locations other than Lahore without re-validation.

## 3. Data

Sources and licences; station counts per domain; fraction of target blocks that pass the 18/24-hour completeness rule;
known gaps (sensor outages, MODIS/VIIRS platform changes).

## 4. Evaluation (all numbers from `evaluation/`)

| Metric | h=24 | h=48 | h=72 |
|---|---|---|---|
| CRPS (µg/m³) — this model | | | |
| CRPSS vs CAMS-BC (M2) | | | |
| CRPSS vs raw CAMS (M1) | | | |
| Coverage of [q10, q90] (target 0.80 ± 0.05) | | | |
| P(y ≤ q50) (target 0.50 ± 0.07) | | | |
| Brier skill, P(y > 125.5 µg/m³) | | | |

Sensor-sparsity curve (CRPS vs N available stations, with 95 % block-bootstrap bands): attach figure.
Pre-registered hypotheses H1–H6 (`configs/evaluation.yaml`): state PASS / FAIL for each, including negative results.

## 5. Failure modes and limitations

Systematic error by regime (e.g. dense-smog events with sensor saturation above ~500 µg/m³), by station class
(reference vs low-cost), by horizon; behaviour after a CAMS upgrade; behaviour when AOD/fire inputs are degraded.

## 6. Ethical and public-health considerations

How uncertainty is communicated; wording reviewed by (name, role, date); risk of false reassurance vs alarm fatigue;
equity (sensor coverage is denser in wealthier neighbourhoods — report coverage by area).

## 7. Maintenance

Retraining trigger (CAMS cycle change, CRPSS vs CAMS-BC below 0.05 for 14 days, coverage outside 0.70–0.90 for 14 days),
owner, rollback procedure (`promoted_bundle: none`).
