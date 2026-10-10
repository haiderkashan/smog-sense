# Product & Scientific Requirements Document (PRD)

> **Version 0.2 · revised 10 October 2026 (original 0.1: 4 October 2026).** Requirement IDs (`FR-xx`, `NFR-xx`) are referenced from commit messages, pull requests and tests. Priorities: **MUST** (ships in the capstone), **SHOULD**, **COULD**.
> Phases refer to the [roadmap](../README.md#status-and-roadmap) (0 scaffolding · 1 ETL · 2 features + transfer learning · 3 probabilistic engine + evaluation · 4 delivery · 5 shadow season and defence).

## 1. Purpose and scope

**Problem.** Every winter, Lahore and the wider Indo-Gangetic Plain are enveloped in hazardous PM2.5. Regulatory-grade monitoring is sparse relative to the population and to the extreme spatial heterogeneity of the pollution, so a hyper-local 24–72 hour forecast built the usual way (dense sensors → machine learning)
degrades exactly where it is needed. The work supports **SDG 3** (health and well-being), **SDG 11** (sustainable cities) and **SDG 13** (climate action).

**Product.** SmogSense is a fully automated, zero-cost pipeline that every morning (05:00 Pakistan time) publishes a **probabilistic, bilingual (English/Urdu) 72-hour PM2.5 forecast for Lahore** as (a) a lightweight website, (b) a machine-readable JSON, and (c) WhatsApp-ready image cards. It combines satellite fire detections, CAMS atmospheric forecasts and a
few-shot transfer-learning model pre-trained on Delhi's dense monitoring network. The learned model is a **decoupled design**: a meta-learned neural encoder adapts to the few available Lahore stations, and an independent LightGBM quantile stacker maps the adapted latent plus physical features to the forecast quantiles.

**Research contributions** (what the capstone must demonstrate): the **Sensor-Sparsity Curve** (forecast skill vs number of available stations), **few-shot transfer** from a data-rich to a data-sparse city (with honest negative-transfer reporting), and **calibrated quantile forecasts** verified prospectively against a fair baseline.

**In scope (MVP).** Lahore metropolitan area; Delhi as source domain (optionally Amritsar/Ludhiana); horizons 24/48/72 h as 24-hour block means; English and Urdu; GitHub-only infrastructure.
**Out of scope.** All of Pakistan (see [verification log](verification-log.md) and the strategic-scaling rationale: sensors are Punjab-clustered, micro-climates differ; reserved for post-graduate work with federated or regional models); hourly forecasts; forecasts beyond 72 h; other pollutants; mobile apps; automated WhatsApp broadcasting (billed); user accounts; any paid service.

## 2. Stakeholders and personas

| Stakeholder | Primary need | Success looks like |
|---|---|---|
| **Citizen** (parent, commuter, worker; Urdu-first, phone-only) | "What do I do today?" in five seconds | Correctly states the protective action, that "1 day in 10 is worse", and until when the forecast is valid, after a 15-minute test (≥ 80 %) |
| **Climate scientist / reviewer** | Methodological credibility | Pre-registered hypotheses, honest negative results, reproducible bundle, prospective verification |
| **Journalist** | Quotable, sourced numbers | Units, issuance time, validity, provenance, truth basis and track record on every page; open JSON |
| **Municipal / hospital planner** | Upper-tail risk 24–72 h ahead | `P(exceed 125.5 / 225.5 µg m⁻³)` per horizon (rounded to 5 %) |
| **Student maintainer** (owner) | Pipeline that runs unattended and fails loudly | Alerts reach the phone; one-command local reproduction; clear runbooks |
| **Supervisor / examiners** | Rigour, originality, feasibility | Ladder of models with ablations, sparsity curve, data-contract tests, decision log |

## 3. Functional requirements

### 3.1 Data acquisition

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-01 | Ingest hourly PM2.5 (and RH, temperature where present) for all stations inside the Lahore box from OpenAQ v3 every cycle within 48 req/min and 1,600 req/h, under the quota partitions (live 0.60 / scoring 0.20 / discovery 0.10) | MUST | 1 | contract + unit tests; run manifest counters |
| FR-02 | Ingest the latest *knowable* CAMS global forecast run (single as-of rule: `available_at ≤ T`) over the IGP box with the configured variables; map licence/queue failures to typed errors | MUST | 1 | as-of unit tests; contract test |
| FR-03 | Ingest FIRMS VIIRS NOAA-21/NOAA-20 detections for the last 72 h; never query Suomi-NPP after 2026-11-01T13:00Z | MUST | 1 | contract test |
| FR-04 | Resumable backfill of Delhi and Lahore history (OpenAQ archive, CAMS archive, FIRMS archive) stored as Release assets | MUST | 2 | workflow run; idempotency test |
| FR-05 | ERA5 hindcast ingestion for parity diagnostics only (first scope cut) | COULD | 2 | diagnostic report |
| FR-06 | MAIAC missingness study on history (research, disabled by default) | COULD | 2 | notebook |
| FR-07 | Land every raw payload with a manifest (parameters, status, sha256, time); re-ingestion is a no-op when unchanged | MUST | 1 | integration test |

### 3.2 Preprocessing and quality control

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-08 | Bit-flagged QC (range, flat-line, spike, spatial outlier, humidity, coverage, drift); never delete rows | MUST | 1 | unit tests incl. regression cases |
| FR-09 | Fill gaps ≤ 3 h with PCHIP in log1p space; leave longer gaps masked; never use imputed hours as targets or toward the 18/24 completeness rule | MUST | 1 | property tests (no overshoot) |
| FR-10 | Versioned station registry with eligibility (uptime, history), station identity = `sensor_id`, and co-location groups | MUST | 1 | schema test |
| FR-11 | Fit, validate and (only if beneficial) enable a humidity-aware low-cost correction | SHOULD | 2 | cross-validation report |
| FR-37 | **Observation capture and revision policy:** one canonical row per (sensor, hour) with `n_revisions`; first-seen times logged in `obs_pull_log` for latency measurement; a re-ingested overlapping pull never creates duplicates; scoring uses the first issuance and settled truth | MUST | 1 | dedup and schema-negative tests; latency capture report |

### 3.3 Features and models

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-12 | Compute ventilation coefficient, dilution index, stagnation run-lengths, stability and RH from CAMS fields | MUST | 2 | unit tests (monotonicity, floors) |
| FR-13 | Target-centric, wind-aligned fire exposure with platform normalisation, one angle convention (β = bearing fire→target; φ = direction air moves toward) | MUST | 2 | unit tests (worked example a ≈ 0.97 / 0; step removal) |
| FR-14 | 72 h lookback tensors with mask and delta channels; forecast age δ as an input; all features point-in-time correct | MUST | 2 | as-of tests |
| FR-15 | Sparse-network task generator for $N\in\{0,1,2,3,5,8\}$ with modality dropout, twin-aware splits and E/S/V/T season blocks | MUST | 2 | unit tests (seed determinism, no twin leakage) |
| FR-16 | Baselines M0–M3 and **M3b** (linear quantile regression) emitting 19-level quantiles, computed from the available set A only | MUST | 1 | unit tests; shadow log |
| FR-17 | Delhi supervised pre-training and **MAML (FOMAML) meta-learning of the neural encoder and a temporary 3-level × 3-horizon head** | MUST | 2 | ladder |
| FR-18 | **LightGBM quantile ensemble (3 × 19 = 57 boosters) as a stacker on [adapted latent ‖ tabular ‖ δ]**, trained on rows from simulated deployments, with monotone rearrangement; bound to its encoder by `encoder_sha256`; no neural/GBM blend and no warm-start boosting in the product path | MUST | 3 | property tests; binding test; ladder |
| FR-19 | CQR + ACI online calibration (M8, γ = 0.02) — **backlog**, only if time allows | COULD | 3 | coverage tests |
| FR-20 | Few-shot adaptation to Lahore from the fixed meta-initialisation (shared recipe, pinball loss with gradient clip 1.0) | MUST | 3 | unit + sparsity experiment |
| FR-21 | Versioned model bundles with sha256 manifest and encoder↔stacker binding; promotion by reviewed PR | MUST | 3 | loader tests |
| FR-38 | **City-panel target:** the city value is a pseudo-location equal to the mean of group-collapsed station block means over a panel frozen at season freeze (≥ 3 valid stations; reference first, else median of corrected low-cost); modelled directly | MUST | 1 | unit tests; panel recorded in the freeze register |
| FR-39 | **Adaptation acceptance and `adaptation_status`:** chronological fit / 72-h embargo / check split; accept iff check pinball improves ≥ 2 % on ≥ 60 % of days **and** latent drift ≤ the 99th percentile over simulated tasks; status ∈ {`adapted`, `skipped_n0`, `skipped_insufficient_support`, `rejected_no_gain`, `rejected_latent_drift`} logged per forecast; rejection falls back to meta-initial latents without changing the degradation level | MUST | 3 | unit tests; status-enum schema test; threshold pilot |

### 3.4 Evaluation

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-22 | CRPS via quantile function with tail model, validated against analytic CRPS (< 0.5 % at K = 19) | MUST | 3 | unit tests |
| FR-23 | Sensor-sparsity experiment: curve over $N$ with bootstrap bands and `adaptation_status` distribution (the support-duration surface is backlog) | MUST | 3 | reproducible report |
| FR-24 | Append-only shadow-scoring ledger, provisional and settled, first valid issuance only (`run_id`, `is_rerun`) | MUST | 1 | integration test |
| FR-25 | DM tests, block bootstrap, Holm correction over {H1–H4} | MUST | 3 | unit tests |

### 3.5 Delivery

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-26 | Publish `forecast/latest.json` conforming to `bulletin.schema.json` (including `valid_until_utc` and `basis`) | MUST | 4 | schema contract test |
| FR-27 | Bilingual static site (forecast, methodology) within page budgets, usable without JavaScript; accuracy and archive HTML pages are deferred | MUST | 4 | site validation |
| FR-28 | WhatsApp post card (1080×1350), Open Graph image, correct Urdu Nastaliq shaping | MUST | 4 | rendering test (Raqm), size budgets |
| FR-29 | Post the card to a Telegram channel — **deferred** | COULD | 4 | manual + dry-run |
| FR-30 | Visible banners for degraded modes; horizons labelled as clock windows; exceedance probabilities rounded to 5 % and clamped (<5 % / >95 %); truth-basis statement | MUST | 4 | e2e test |
| FR-31 | Public rolling scorecard (JSON) rendered from `score_log` with sample sizes; HTML accuracy page deferred | MUST (JSON) / SHOULD (HTML) | 4 | e2e test |
| FR-32 | Methodology and limitations pages in plain language, both languages | SHOULD | 4–5 | expert review |

### 3.6 Operations

| ID | Requirement | Pri | Phase | Verified by |
|---|---|---|---|---|
| FR-33 | Daily workflow with three idempotent triggers, host pre-check before any image pull, content-hash image tag, typed exit codes, per-stage log groups | MUST | 1 | workflow runs |
| FR-34 | Alerting (issue per failure/degradation) and dead-man's-switch watchdog | MUST | 1 | forced-failure drill |
| FR-35 | Daily point-in-time input snapshot to the `state` branch (enables replay shadow) | MUST | 1 | integration test |
| FR-36 | `make check` reproduces every CI gate locally | MUST | 0 | CI |
| FR-40 | **Expiry notice via three paths:** every bulletin carries `valid_until_utc` = issuance + 30 h; a static "valid until" line, a client-side JS banner, and a watchdog-published static expiry notice | MUST | 4 | e2e test with a stale bulletin |

## 4. Non-functional requirements

| ID | Requirement | Target / rule | Verified by |
|---|---|---|---|
| NFR-01 | **Cost** | **USD 0.00**; no card on file; permanent free tiers only; no paid SDK dependency | `test_no_paid_or_billable_dependencies`; monthly quota review |
| NFR-02 | **Compute fit** | Fits 2 vCPU / 7 GB / 14 GB disk; peak RAM ≤ 5 GB; runtime image ≤ 5 GB (target ≈ 3 GB) | compose limits; CI size gate |
| NFR-03 | **Latency (free compute)** | ≤ 1,100 Actions minutes/month for scheduled workflows (daily ≈ 870); CI extra | run manifest; job summary |
| NFR-04 | **Timeliness** | never presented as current after issuance + 30 h | shadow ledger; watchdog; FR-40 |
| NFR-05 | **Idempotency** | Re-running an issuance is a safe no-op (exit 11) | integration test |
| NFR-06 | **Reproducibility** | `uv.lock`; config hash; fixed seeds; deterministic LightGBM; bundle manifests with sha256; library versions recorded | CI lock check; loader |
| NFR-07 | **Security** | Secrets only via environment; third-party actions pinned by SHA; least-privilege tokens; repo token never in a container; strict CSP; no `pull_request_target` | `test_repo_hygiene.py` |
| NFR-08 | **Accessibility** | WCAG 2.2 AA; text contrast ≥ 4.5:1 (design tokens ≥ 5.25); keyboard operable; skip link; reduced motion | design-token tests; manual pass |
| NFR-09 | **Web performance** | HTML ≤ 60 KB, CSS ≤ 25 KB, JS ≤ 30 KB, page ≤ 500 KB excl. fonts, Urdu font ≤ 160 KB; cards ≤ 400 KB; OG ≤ 300 KB; LCP < 2.5 s on slow 3G | `site validate` |
| NFR-10 | **Localization** | Every user-facing string exists in `en` and `ur` (key, structure and placeholder parity; 79 keys after revision 2); Urdu Nastaliq line-height ≥ 2.0; native-speaker and health review before launch | `test_i18n.py`; gates G-LANG, G-HLTH |
| NFR-11 | **Maintainability** | `ruff` clean, `mypy --strict` clean, every module has a contract docstring; coverage ≥ 85 % on pure-function packages from the end of Phase 1 (the roadmap names 70 % as the first cut if time is short) | CI |
| NFR-12 | **Observability** | Run manifest per command (including `adaptation_status`, peak memory, `run_id`); job summary; freshness metrics in the bulletin provenance | manifest schema |
| NFR-13 | **Licensing and attribution** | Respect OpenAQ (CC BY 4.0), Copernicus (CC BY 4.0 with the mandated wording), NASA FIRMS acknowledgement; MIT for code | footer text test |
| NFR-14 | **Privacy** | No cookies, analytics or third-party requests; no personal data stored | CSP; review |
| NFR-15 | **Portability** | Site deployable on any static host; development on Linux, macOS, Windows via Docker | CI; docs |
| NFR-16 | **Scientific integrity** | Pre-registered hypotheses; freeze register; negative results reported; no retroactive rescoring | evaluation protocol |
| NFR-17 | **Memory constraint** | measured peak RSS of the adaptation stage ≤ 5 GB, and growth over a loop of repeated adaptations under a stated bound | profiling; peak-memory test |

## 5. Accuracy and scientific targets

All scores use the 24-hour block-mean target, settled truth, the 19-level CRPS estimator, and the fair reference **M2 (bias-corrected CAMS)**. These are *pre-registered hypotheses*, not guarantees; **thresholds are provisional until the freeze register (4 Dec 2026)**. The hypothesis table below mirrors the [evaluation strategy](evaluation-strategy.md#1-evaluation-questions-and-hypotheses), which is authoritative.

| ID | Statement | Tier | Window | Target |
|---|---|---|---|---|
| H1 | HM vs M2, 24 h | 1, indicative | Lahore confirmatory (7 Dec–31 Jan), $N_H=\min(5,\text{eligible}-3)$ | CRPSS ≥ 0.15, lower 95 % bound > 0 |
| H2 | HM vs M2, 72 h | 1, indicative | same | CRPSS ≥ 0.10, lower bound > 0 |
| H3 | HM vs raw CAMS (M1), each horizon | 1, indicative | same | CRPSS ≥ 0.25, lower bound > 0 |
| H4 | sensor-free ($N=0$) vs M1 | 1, indicative | same | ≥ 0.05 (deterministic CRPS) — else report negative transfer |
| H5 | meta-learned vs supervised init (M7 vs M6) | 1, **powered** | **Delhi season T**, $N\in\{1,2\}$, ≥ 20 stations | CRPS ratio M7/M6: upper CI bound < 1 and point ≤ 0.97 — or the finding is that simple fine-tuning suffices |
| H6 | calibration of the 80 % interval | 2, descriptive | pooled | coverage reported with CI; flagged if the CI excludes [0.70, 0.90] (monitoring target 0.80 ± 0.05; $P(y\le q_{.50})$ within 0.50 ± 0.07) |

| Horizon | CRPSS vs CAMS-BC (M2) | CRPSS vs raw CAMS (M1) | At $N$ |
|---|---|---|---|
| 24 h | **≥ 0.15** (H1) | **≥ 0.25** (H3) | $N_H$ |
| 48 h | ≥ 0.125 (interpolated; descriptive) | ≥ 0.25 (H3) | $N_H$ |
| 72 h | **≥ 0.10** (H2) | ≥ 0.25 (H3) | $N_H$ |
| 24 h, sensor-free | **≥ 0.05** vs M1 (H4) | – | 0 |

Fewer than 35 confirmatory days ⇒ H1–H4 are reported descriptively. If the stacker is not robust to latent shift, or adaptation adds nothing, **M5 becomes the headline and the negative result is reported**.

**Operational gates.** *Promote* a bundle only if, on validation and the Lahore support-block test, CRPSS vs M2 ≥ 0.05 at all three horizons and 80 % coverage ∈ [0.75, 0.85]. *Retrain trigger:* CAMS cycle change, CRPSS vs M2 < 0.05 for 14 days, or coverage outside [0.70, 0.90] for 14 days.
*Kill-switch:* CRPSS vs M2 < 0 for 14 consecutive days ⇒ set `promoted_bundle: none` (baseline-only) via one-line PR.

## 6. Localization and typography rules

1. **Languages:** English (default) and Urdu; equal feature parity; reciprocal `hreflang` links; no auto-redirect.
2. **Urdu typeface and shaping:** Noto Nastaliq Urdu; text drawn only through engines that shape (browser; Pillow + Raqm). Matplotlib never renders Urdu.
3. **Line-height** 2.1 (minimum 2.0); no letter-spacing, italics, synthetic bold or ellipsis truncation; allow 30–40 % text expansion.
4. **Digits:** Western digits in both languages by default; units `µg/m³` (Latin) in charts, spelled out in Urdu running text; numbers and units are isolated LTR runs.
5. **Direction:** `dir="rtl"`; logical CSS properties; charts keep a left→right time axis (to be validated with readers).
6. **Uncertainty wording:** natural frequencies ("8 days in 10", "1 day in 10"); never "worst case"; percentages only for exceedance, rounded to 5 % and clamped (<5 % / >95 %).
7. **AQI scheme:** US EPA 2024 PM2.5 breakpoints on 24-h means, named on the page; no index above 500. A local (Punjab EPA) convention is an open owner decision (gate G-SCI).
8. **Review gates:** native Urdu editor (G-LANG), public-health physician (G-HLTH), environmental scientist (G-SCI), comprehension test with lay users (G-UX).

## 7. Constraints and assumptions

| Constraint / assumption | Detail |
|---|---|
| Budget | USD 0.00 forever; if a service ever asks for a card, the design changes, not the budget |
| Repository | Public, on a personal Free plan (needed for Pages, unmetered Actions, free GHCR) |
| Calendar (assumed) | Build started early October 2026; smog season late October–January; shadow window 19 Oct 2026 – 31 Jan 2027; capstone defence in the first half of 2027 — **confirm the exact date**; the roadmap is built backwards from the season, not from the defence |
| Effort | ≈ 504 h in total (Oct–Dec ≈ 384 h against ≈ 360 h of capacity, about 24 h short; scope cuts documented in the roadmap) |
| Team | One student maintainer (a second maintainer is recommended before peak season) |
| Data access | Free credentials for OpenAQ, ADS/CDS, FIRMS obtained in Phase 0 |
| Provider drift | Limits and satellites change (see Suomi-NPP, MODIS); every external fact is dated in the verification log |
| Honesty | One season is one climate realisation; results are reported with their sample sizes |

## 8. Risks (top eleven)

R1 negative transfer from Delhi · R2 CAMS model-cycle upgrades · R3 sensor right-censoring in extreme smog · R4 late model vs early season (mitigated by baselines from day 1 and replay shadow) · R5 ADS publication delays · R6 platform retirements (Suomi-NPP, MODIS) ·
R7 GitHub schedule drops or auto-disable (mitigated by retries, watchdog, keep-alive) · R8 public misinterpretation of intervals (comprehension test) · R9 single-maintainer bus factor · R10 scope creep to "whole Pakistan" · R11 latent shift or negligible adaptation gain in the decoupled model (mitigated by the acceptance rule, drift guard and the M5 fallback). Details, owners and mitigations: [ML §12](ml-architecture.md#12-failure-analysis-and-known-risks), [system §7](system-architecture.md#7-failure-handling-and-degradation-ladder), [ops §11](deployment-and-ops.md#11-disaster-recovery). (Numbers here are the PRD's own and differ from the ML risk table's.)

## 9. Definition of done (project level)

* A scheduled workflow has published a valid bilingual bulletin, with a stated expiry, on ≥ 95 % of days of the shadow window with zero secrets leaked and $0 spent.
* The sparsity curve, ladder table and H1–H6 verdicts exist for the settled shadow window, with negative results reported.
* All review gates passed and recorded; the verification log is current; the repository reproduces from a clean clone with `make check`.

## 10. Glossary

| Term | Meaning |
|---|---|
| **AOD** | Aerosol optical depth (here: CAMS total AOD at 550 nm) |
| **ACI / CQR** | Adaptive conformal inference / conformalised quantile regression |
| **`adaptation_status`** | Outcome of the daily adaptation: `adapted`, `skipped_n0`, `skipped_insufficient_support`, `rejected_no_gain`, `rejected_latent_drift` |
| **CAMS** | Copernicus Atmosphere Monitoring Service; **CAMS-BC** = trailing-window bias-corrected CAMS (reference M2) |
| **CRPS / CRPSS** | Continuous ranked probability score / skill score vs a reference |
| **δ (forecast age)** | $T-B^\*$: age of the CAMS cycle used (12 or 18 h at level 0), an input to the model |
| **E/S/V/T** | Source-season blocks: encoder training / stacker rows / tuning / Delhi confirmatory test |
| **FRP** | Fire radiative power (MW) |
| **IGP** | Indo-Gangetic Plain |
| **Issuance $T_0$** | Forecast reference time, 00:00 UTC (05:00 PKT) |
| **MAML / FOMAML / ANIL** | Model-agnostic meta-learning / first-order variant / adapt-head-only variant (used here only as a negative control) |
| **$N$** | Number of *available* stations (labels for adaptation and observed-history input) |
| **PBLH / VC / DI** | Planetary boundary-layer height / ventilation coefficient (PBLH × wind) / dilution index (mean of 1/VC) |
| **Pinball loss** | Quantile loss $\rho_\tau(u)=u(\tau-\mathbf 1\{u<0\})$ |
| **Replay vs live shadow** | Scoring a late model on logged inputs vs forecasts issued before outcomes existed |
| **Stacker** | The independent LightGBM quantile model that consumes the frozen encoder's latent plus tabular features |
| **`valid_until_utc`** | Issuance + 30 h; after it the page shows an expiry notice |
