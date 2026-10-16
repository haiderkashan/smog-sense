# Test Specifications

Contract tests to implement for Phase 0 and Phase 1, plus the protocols for experiments that cannot be tests yet. Each test has an ID, the property it pins, and the exact expected values. Fixtures are recorded once and kept under `tests/fixtures/`.
Items marked † depend on revision-2 or revision-3 contract changes (schema, manifest or CLI) that need owner approval first.

## Phase 0 — contracts and structure

### T0-01 As-of parity
On one recorded snapshot, the rule-reconstructed feature inputs equal the snapshot, and the eligible set is identical at 00:17, 02:47 and 05:47 (the three triggers must see the same `available_at ≤ T` rows, because eligibility depends on `available_at`, never on `ingested_at_utc`).

### T0-02 Clip
No eligible row has `hour_end > T`, and none has `available_at > T`. For `T = 00:00Z` and `L_oaq = 3.0 h` the latest eligible hourly label is 20:00Z of the previous day.

### T0-03 CAMS as-of and forecast-age table
`asof_cams_run(T)` returns exactly these values at the four lattice points, and level 1 adds 12 h:

| *T* | *B*\* | δ = T − B\* (level 0) | δ (level 1) |
|---|---|---|---|
| 00Z | 12Z of day D−1 | 12 h | 24 h |
| 06Z | 12Z of day D−1 | 18 h | 30 h |
| 12Z | 00Z of day D | 12 h | 24 h |
| 18Z | 00Z of day D | 18 h | 30 h |

Boundary cases: at exactly `T = B + 10 h` the run is eligible; one second earlier it is not.

### T0-04 Stitched lookback
For each hour *t* in `[T−72 h, T]` the selected cycle satisfies `B ≤ t` and `B ≤ B*(T)`; no cap on lead is applied. For `T = 06Z` the selected cycle is 12Z of D−1 for every *t* ≥ that base time, and the maximum lead in the window is 18 h. A cycle with `B > B*(T)` is never selected, even if present in the archive.

### T0-05 Bearing and fire exposure
Worked example (Lahore 31.5204°N 74.3587°E; fire at Amritsar 31.634°N 74.8723°E): θ (target→fire) = 75.5° ± 0.2°, β = θ + 180° = 255.5° ± 0.2°.
Easterly wind (`u = −5`, `v = 0`): φ = 270°, **a = 0.97 ± 0.01**. Westerly wind (`u = +5`): φ = 90°, **a = 0 ± 0.01** (clipped). A calm-wind input uses `w_min = 0.5 m s⁻¹` and never divides by zero. Upwind exposure exceeds downwind exposure for every wind direction within ±60° of the fire→target bearing.

### T0-06 Dedup and revisions
Re-ingesting an overlapping 72-h pull leaves exactly one canonical row per (`sensor_id`, `ts_utc`) and `n_revisions` increments when the value changed; an unchanged re-pull leaves `n_revisions` unchanged (idempotent).

### T0-07 Co-location
A group {1 reference + 2 low-cost sensors within 50 m} resolves to the reference value and records `basis = reference`. A group of 3 low-cost sensors resolves to the **median of the corrected values** and records `basis = lowcost_median`. The city value counts one value per group.

### T0-08 Schema negatives †
A duplicate (`sensor_id`, `ts_utc`, `pull_id`) in `obs_pull_log` is rejected; a rerun with a new `run_id` is accepted and written with `is_rerun = true`; a `forecast_log` row with `adaptation_status` outside {`adapted`, `skipped_n0`, `skipped_insufficient_support`, `rejected_no_gain`, `rejected_residual_bound`} is rejected (`rejected_latent_drift`, the revision-2 value, is no longer valid); a bulletin without `valid_until_utc` fails the JSON schema.
`validate_frame()` on a frame with a **duplicate index** raises `SchemaError`/`SchemaErrors` (not an opaque `ValueError`) and does not mutate the caller's frame.

### T0-09 Config cross-consistency
Loaded together, the YAML configs satisfy:
- `adaptation.inner.steps` (deployment) equals `meta.inner_steps`; `adaptation.inner.support_days` equals the set used in meta-training ({7, 14, 30}); both use `loss: pinball` and `grad_clip: 1.0`.
- OpenAQ quota partitions are 0.60 / 0.20 / 0.10 with the sum ≤ 1.0; effective limits are 48/min and 1,600/h.
- The method enum contains exactly the **ten** ids M0, M1, M2, M3, M3b, M4, M5, M6, M7, M8.
- The quantile grid has 19 levels 0.05…0.95; the public levels {0.10, 0.50, 0.90} are a subset; the temporary head levels are the same three.
- `calibration.aci_gamma` (0.02) is a member of `aci_gamma_grid` {0.01, 0.02, 0.05}.
- `rh_mask.enabled` is `false`; `rh_mask.threshold_pct` is 85; `min_support_fraction` is 0.5.
- `bulletin.valid_until_hours` is 30 and is the only expiry setting.
- `sources.yaml` carries a latency for every source with `available_at` semantics (OpenAQ 3.0 h assumed, FIRMS 3 h, CAMS 10 h, archive 72 h).
- Model bundle config references `bind_base_sha256: true`; the residual head has 57 outputs (3 horizons × 19 levels) and its last layer is zero-initialised; `meta.learn_step_size: true` with `alpha_init: 0.01` and per-tensor granularity; `cross_fit.folds` ≤ 3 and season-blocked.
- `corruption` block present with `gain: [0.8, 1.3]`, `rh_slope: [0, 1.5]`, `offset: [-10, 25]`, `rh_ref_pct: 70`, `p: 1.0`, `support_labels: clean`; M6 pre-training reads the same block.
- The M5 feature list contains neither `recent_cams_bias`, absolute `lat`/`lon`, nor any event flag; M4's may contain coordinates.

### T0-10 i18n
`en` and `ur` catalogues have identical keys, nesting, list lengths and `{placeholder}` sets (73 existing keys; **79** once the six revision-2 keys are approved: `horizon.window`, `status.valid_until`, `status.expired_banner`, `uncertainty.chance_exceed_low`, `uncertainty.chance_exceed_high`, `methodology.truth`). The placeholders of the six new keys match between languages.

### T0-11 Workflow structure
- The daily workflow, its two retry triggers, the scoring workflow, the keep-alive workflow and the 06:07 watchdog all set the concurrency group `smogsense-state` with `cancel-in-progress: false`.
- Every downstream step of the daily workflow is guarded by `skip != 'true'`, and the host pre-check runs **before** the image login/pull step.
- No cron uses minute 0; the daily crons are `17 0`, `47 2`, `47 5`; the watchdog is `7 6`.
- The image tag format is `img-<hash>`; no workflow references `sha-<commit>` or `:latest` for the daily job.
- Every job has `timeout-minutes`; no `pull_request_target`; third-party actions are pinned by SHA.

### T0-12 Documentation hygiene
- No public document refers to git-ignored internal files; all relative links and anchors resolve; document headings referenced by tests exist (for example "LightGBM quantile ensemble" in the ML architecture).
- The planning arithmetic reconciles: phase hours sum to the stated totals, and the weekly table sums to the phase table.
- Stale phrases are absent from public docs: "36 h", "sha-<commit>", "nine methods", "warm-start" outside the ablation mentions, "worst case", and (outside verification-log history entries) "encoder_sha256", "rejected_latent_drift", "stacker", "simulated deployments on block S", "E/S/V/T".

### T0-13 Repository hygiene
Pins, permissions, non-root containers, secret redaction, strict CSP and the "no paid dependency" check (existing tests) still pass after the revision-2 changes.

## Phase 1 — ingestion, QC, ledger

### T1-01 Rate limiter and quota partitions
The token bucket enforces both windows (48/min, 1,600/h); after a response with `x-ratelimit-remaining: 3` the bucket never allows more than 3 further requests before `x-ratelimit-reset`. A partition cannot borrow from another partition; it can borrow from the unassigned reserve.

### T1-02 Retry policy and circuit breaker
Retries only on 408/429/500/502/503/504, maximum six attempts, full-jitter delay `U(0,1)·min(120 s, 2 s·2ⁿ)`; `Retry-After` larger than the computed delay wins; three consecutive 429s open the circuit for 15 minutes; other 4xx are never retried.

### T1-03 ADS behaviour
An HTTP 403 "required licences not accepted" maps to a typed error and exit 20 with the licence message; queue polling stops at the 20-minute budget; request keys match the recorded *Show API request* of the dataset.

### T1-04 QC regressions
- Cubic-spline overshoot regression: a 3-h gap between 520 and 480 µg/m³ is filled by PCHIP within [480, 520] (the spline would exceed 960); a trough→rise gap never produces a negative value.
- Flat-line (≥ 6 identical hourly values above 0), spike (centred 7-h median + max(8·MAD, 100)), range, negative clipping and `IMPUTED` flag behave as specified; rows are never deleted.
- Imputed hours never count toward the 18-of-24 completeness rule and never become targets.

### T1-05 Distribution and CRPS
The quantile-function object reproduces analytic CRPS with **< 0.5 % mean relative error at K = 19** and **< 5 % at K = 7** (normal and lognormal σ ∈ {0.3, 0.5, 0.8}); the identity CRPS = 2∫ρ_τ reproduces 25.618 for N(100, 30²), y = 140; a point forecast scores exactly |y − ŷ|.

### T1-06 Rearrangement (property test)
For arbitrary quantile vectors, sorting never increases total pinball loss; output is monotone and within [0, 2000] µg/m³.

### T1-07 Baselines on the available set only
M0, M2, M3 and M3b computed from a set A never read any other station; at N = 0, M0/M2/M3/M3b are undefined (not silently imputed) and M1 is deterministic. M0 uses the last complete 24-h block **available at T** (ending at `T − L_oaq`).

### T1-08 Idempotency and pre-check
Running an issuance twice exits 0 then 11; with `SMOGSENSE_FORCE=1` the second run overwrites safely and the manifest records `is_rerun = true` with a new `run_id`. The workflow pre-check exits before any image pull when the manifest says `published: true`.

### T1-09 Scoring ledger †
Only the **first valid issuance** of a day is scored; reruns are logged but never scored; provisional scores appear at window end + 6 h (API) and settled scores at +96 h (archive); both are append-only.

### T1-10 Forecast log and bulletin contracts †
`forecast_log` rows contain 19 levels × 3 horizons × (stations + city) with `adaptation_status`; the bulletin passes `bulletin.schema.json`, carries `valid_until_utc = issuance + 30 h` and a `basis` (truth basis) field; exceedance probabilities are multiples of 5 % clamped to "<5 %" / ">95 %" (a 93 % internal value is shown as 95 %); horizon labels are clock windows.

### T1-11 Expiry paths
(a) the static "valid until" line renders; (b) with the clock set past `valid_until_utc` the JS banner replaces the forecast; (c) when no bulletin exists for today the watchdog publishes a static expiry notice built from the last good bulletin; all three work with JavaScript disabled where applicable (a and c).

### T1-12 State branch protections †
`state verify` warns at 400 MB and alerts at 700 MB; the clone is sparse/partial; a yearly rotation creates a new orphan branch without losing the previous one (checked against a bare repository).

### T1-13 FIRMS continuity
The client refuses to query Suomi-NPP at or after `2026-11-01T13:00Z`; fire features are platform-normalised so removing one platform from a fixture changes the normalised counts by less than the raw counts do.

### T1-14 Leakage guards
No feature uses `available_at > T`; the E/V/T season blocks and the cross-fit folds of block E have no overlapping sample (lookback or target) across a boundary and a 72-h embargo separates them; scalers are fitted on block E only; co-located stations never straddle the A/B split (twin-aware).

### T1-15 Bundle integrity and binding †
A bundle whose file hashes differ from its manifest is refused; a calibrator whose `base_sha256` differs from the loaded base is refused and the pipeline serves level 2 (CAMS-BC).

### T1-16 Adaptation outcomes †
With a synthetic support window: N = 0 ⇒ `skipped_n0`; fewer than `min_support_days` valid days ⇒ `skipped_insufficient_support`; no check improvement ≥ 2 % over the zero-residual forecast on ≥ 60 % of check days ⇒ `rejected_no_gain`; improvement criteria met but $R=\operatorname{mean}|\Delta\hat z|$ above $\rho_{99}$ (the 99th percentile over simulated deployments on block V) ⇒ `rejected_residual_bound`; otherwise `adapted`. When both rejection tests fail the status is `rejected_no_gain`. Every rejection and skip sets $\Delta\hat z=0$, so the published forecast equals M5 bit for bit, with degradation level 0.

### T1-17 Memory
The adaptation stage on a synthetic worst-case support set stays below the stated peak-RSS bound, and repeated adaptations in a loop do not grow memory beyond the stated bound.

### T1-18 Parameter count †
The residual stack (GRU over C = 6 channels, hidden 64, two layers, 64→32 projection, head 46→64→57) has exactly **47,577** parameters (encoder 40,864; head 6,713), plus 14 step-size scalars. The test changes only together with the channel set or head input and with [ML architecture §4](ml-architecture.md#4-sequence-encoder).

### T1-19 Zero-initialised residual †
A freshly built calibrator returns $\Delta\hat z=0$ for every input, so its output equals the base quantiles exactly (M6/M7 untrained = M5); after one optimiser step the last-layer gradient is non-zero; the base receives no gradient.

### T1-20 Out-of-fold integrity †
For every row of block E, the base prediction given to the calibrator comes from boosters trained on folds that do not contain that row's season; folds are season-blocked with a 72-h embargo and purged lookbacks; the partition hash in the bundle manifest matches the folds used; the final base (fit on all of E) is never used to produce calibrator training inputs.

### T1-21 Corruption augmentation †
For a fixed seed the per-task draws $(a_i,b_i,c_i)$ lie inside the configured ranges and are identical across the task's 72-h window and its support and query windows; $\tilde x=a\,x+b\max(0,\mathrm{RH}-70)+c$ is reproduced exactly on a synthetic series; RH and query labels are unchanged; evaluation rows are never corrupted except under the explicit H5-C flag; M6 pre-training uses the same function.

## Pending experiments (protocols only — not yet tests)

| ID | Experiment | Protocol and decision rule |
|---|---|---|
| X-01 | Residual-overshoot and bound test | Meta-train on corrupted Delhi tasks; run the deployment recipe on Lahore dev support and on Delhi V tasks with extra-noisy support; compare check-window pinball and $R$ with and without the residual bound. If overshoot that the bound does not repair raises pinball by more than 5 % against M5, ship M5 |
| X-02 | In-sample vs out-of-fold base predictions | Train one calibrator on in-sample M5 predictions (trees fit on the same rows) and one on out-of-fold predictions; evaluate both on block T and on Lahore dev. Expect the in-sample calibrator to under-correct and to be over-confident; it must not be used |
| X-03 | Pinball vs quantile-Huber, with and without clipping | 2 × 2 on block V; report pinball, coverage and tail behaviour; Huber stays an ablation unless it improves the upper quantiles |
| X-04 | RH-mask audit | Share of Lahore low-cost support hours with RH > 85 % (device vs CAMS RH); upper-quantile pinball with the mask on/off, RH-stratified (none / 75–85 / above 85) |
| X-05 | Peak-RSS loop | Run 100 consecutive adaptations; record peak and growth of RSS |
| X-06 | State-branch growth and compaction | Simulate 365 daily commits; measure size, partial-clone time, rotation and rebase exercise |
| X-07 | API-vs-archive diff | For the same stations and days compare API (provisional) and archive (settled) values; report the distribution of differences |
| X-08 | Backfill pilot | 500 archive files; measure throughput, 404 rate, memory; fix the sharding plan |
| X-09 | ACI γ backtest | Backtest γ ∈ {0.01, 0.02, 0.05} on Delhi seasons with delayed feedback; confirm γ = 0.02 or change the default with a dated log entry |
| X-10 | OpenAQ latency capture | 14 days of `first_seen_utc − hour_end_utc`; replace the assumed 3.0 h by the measured p95 (dated entry in the verification log) |
| X-11 | Corruption-label identifiability | On Delhi block T with synthetic corruption (known truth), compare `support_labels: clean` vs `same_as_inputs`, corruption on/off for M7 and M6, and priors from the co-location audit vs the default ranges. Decide the label policy and whether `corruption.p` < 1 is needed; report H5-C |
