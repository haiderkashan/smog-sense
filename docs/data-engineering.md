# Data Engineering

> **Audience:** engineers building the Phase 1–2 pipelines. **Rule zero:** every number in this document that describes an external
> service was verified against provider documentation in October 2026 (see the [verification log](verification-log.md)). Re-verify before relying on it.

## 1. Scope and data-contract principles

1. **Contracts, not conventions.** Every table has a Pandera schema in `data/schemas/*.schema.yaml` (column, dtype, unit, valid range, meaning). A stage validates its
   output *before* writing it. Production code must call `validate_frame()`, which (a) resets the index, because Pandera 0.33 on pandas 3 raises an opaque
   `ValueError` instead of `SchemaError` when the frame has a duplicate index, and (b) catches both `SchemaError` and `SchemaErrors`. Both behaviours are pinned by tests.
2. **Raw is immutable.** Clients land payloads unchanged under `data/raw/` with a manifest (request URL without credentials, parameters, HTTP status, response sha256, UTC timestamp). Cleaning never edits raw.
3. **Atomic and idempotent.** Writes go to a temporary name and are renamed; re-ingesting a day with an unchanged hash is a no-op.
4. **UTC everywhere.** Local time (Asia/Karachi UTC+5; Asia/Kolkata UTC+5:30) exists only for display and day-boundary conventions. Neither zone observes DST today, but no code may assume that.
5. **Hour-start labels.** An hourly value labelled `ts_utc = 18:00` describes `[18:00, 19:00)`. A 24-hour block is the half-open interval `[start, end)` of hour-start labels
   (equivalently `(start, end]` on hour-end labels), so there is never an off-by-one between targets, windows and verification.
6. **No secrets in artefacts.** Credentials come from environment variables only; the HTTP audit log and the run manifest never contain them.
7. **One as-of rule.** What a feature may see is decided by `available_at` (§11), never by when a row happened to be ingested. `ingested_at_utc` is stored for audit only.

## 2. Source inventory and verified limits

| Source | Role | Auth | Verified limits and behaviour (Oct 2026) | Latency | Licence / attribution |
|---|---|---|---|---|---|
| **OpenAQ API v3** | Live PM2.5 and co-measured RH/temperature | `X-API-Key` header (mandatory in v3) | **60 requests/min and 2,000 requests/h**; headers `x-ratelimit-used/reset/limit/remaining`; repeated 429s may lead to a ban | provider-dependent; **3.0 h assumed** until measured (§11) | CC BY 4.0 (OpenAQ and providers) |
| **OpenAQ AWS archive** | Historical backfill, settled scoring (never a feature source) | none (public bucket, anonymous HTTPS) | Objects `records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>-<yyyymmdd>.csv.gz` (grouped by local day); **written 72–96 h after the end of the local day** | 72–96 h | as above |
| **Copernicus ADS** (CAMS) | PM2.5, AOD, boundary-layer height, winds, T/Td, pressure-level T | personal access token; per-dataset licence acceptance | 0.4° (~40 km), runs at 00/12 UTC, 5-day lead, archive from 2015; **00Z guaranteed by 10:00 UTC, 12Z by 22:00 UTC**; ADS is not an operational service | see as-of rule | CC BY 4.0; "Generated using Copernicus Atmosphere Monitoring Service information [year]" |
| **Copernicus CDS** (ERA5) | Hindcast diagnostics only | personal access token; licence acceptance | 0.25°; ERA5T ≈ 5-day latency; `…-timeseries` dataset is *experimental, not for operations* | ~5 days | "Generated using Copernicus Climate Change Service information [year]" |
| **NASA FIRMS** | Active fire detections | free `MAP_KEY` | **5,000 transactions per 10 min**; CSV only; ≤ 10 days per request; sources `VIIRS_NOAA21_NRT`, `VIIRS_NOAA20_NRT/SP`, `VIIRS_SNPP_NRT/SP`, `MODIS_NRT/SP` | NRT ≈ 3 h | NASA open data, acknowledge FIRMS |
| NASA LAADS (MAIAC) | **Research only**, disabled | Earthdata token | MODIS Aqua/Terra end-of-mission planned 2026–27 | days | NASA open data |

> **Not applicable to this project:** the "4 concurrent connections / 20 MB/s / 10-minute token" quotas in the original plan belong to the Copernicus **Data Space Ecosystem**
> (Sentinel downloads), a different service from the CDS/ADS used here.

## 3. OpenAQ v3 client

**Station discovery** (weekly, cached in the registry): `GET /v3/locations` with the domain's bounding box, `parameters_id` = PM2.5, `limit=1000`, paginated.
Pagination terminates by arithmetic (`found` vs `limit × page`), never by guessing from an empty page. Each location yields: id, name, provider, coordinates, `isMonitor`
(regulatory-grade), sensor ids per parameter, first/last datetime. *(Exact field and path names are pinned by `tests/contract/test_openaq_contract.py` in Phase 1.)*
**Station identity** is the PM2.5 `sensor_id`; the location id is kept as an attribute (a location can host several sensors).

**Hourly data:** `GET /v3/sensors/{id}/hours` for the PM2.5 sensor with `datetime_from/to`, plus RH and temperature sensors of the same location when present.
Daily operation needs 72 h per sensor ⇒ one page per sensor. With *S* sensors the request count is *S* (+ discovery); at the budgeted ≤ 48 requests/min, 100 sensors take about 2 minutes.

**Rate budget.** Effective limits are `safety_margin × published` = **48/min and 1,600/h**. A token bucket enforces both windows; after every response the bucket is reconciled with
`x-ratelimit-remaining` and `x-ratelimit-reset` (seconds until reset), so a second process or a manual query cannot silently exhaust the quota.
**Quota partitions** of the hourly budget (defaults in `configs/sources.yaml`): live ingestion **0.60**, scoring and reconciliation **0.20**, discovery and ad-hoc **0.10**, unassigned reserve 0.10. A partition may borrow from the reserve but never from another partition. Backfill uses the archive and consumes no API quota.

**Retry policy.** Only idempotent GETs, on `408, 429, 500, 502, 503, 504`. Delay for attempt *n* (full jitter): `d_n = U(0,1) · min(120 s, 2 s · 2^n)`; if `Retry-After` or `x-ratelimit-reset` is larger, that wins.
Maximum six attempts. **Circuit breaker:** three consecutive 429s open the circuit for 15 minutes (the documentation warns that repeated violations may result in a ban). 4xx other than 408/429 are never retried.

**First-seen log (`obs_pull_log`, proposed).** Every pull appends one row per (sensor, hour): `first_seen_utc`, `hour_end_utc`, `pull_id`. This is the only way to *measure* the real latency of the API (§11) instead of assuming it. It is append-only, written to `state/obs_pull_log/`, and is never read by feature code.

**Hygiene, co-location and truth basis.** Co-located duplicates (≤ 50 m, different providers) remain separate stations but share a `colocated_group`. The value used for a group, in the city target and in verification, is:
**the reference monitor if the group contains one (`isMonitor`); otherwise the median of the corrected low-cost sensors.** Discovery results are snapshotted with `registry_version`. Station identity, the co-location hierarchy, the truth basis, and the quota partitions are maintained here.
The **city target** is a pseudo-location whose value is the mean of group-collapsed station values over a panel frozen at the season freeze, requiring ≥ 3 valid stations (definition in [ML architecture §1](ml-architecture.md#1-problem-formulation)).

## 4. OpenAQ archive backfill

Used for Phase 2 history (Delhi source seasons, Lahore history) and for *settled* scoring. Because files appear only 72–96 h after the local day ends it can never feed a live forecast; the most recent days of any history are topped up from the API.

* Anonymous HTTPS GET of the key template in `configs/sources.yaml`; 8 concurrent connections; no API key, no rate limit, no AWS account, no cost.
* Backfill plan: pilot of 500 files first, shard by (domain, year), cache 404s, consolidate to monthly Parquet, and treat any throughput estimate as unmeasured.
* Convert the timestamps to UTC, map to `observations_hourly.schema.yaml`, set `source = openaq_archive`. `available_at` is computed by the §11 rule (`local_day_end + 72 h`); `ingested_at_utc` is the actual download time (audit only).
* Jobs stop cleanly before the 6-hour runner limit (`--max-wall-minutes`) and are resumable; results are stored as Parquet assets of the rolling `data-backfill` GitHub Release. Seed history: archive up to the latest available day plus an API top-up, refreshed weekly.

## 5. Quality control and low-cost sensor handling

Rules **never delete rows**: they set bits in `qc_flags` and null `pm25_ugm3`, so raw and cleaned values stay auditable. Order of operations matters (cheap temporal checks first, spatial checks after).

| Bit | Flag | Rule (defaults in `configs/preprocessing.yaml`) | Effect |
|---|---|---|---|
| 1 | `RANGE_REJECT` | value < −5 or > 1,500 µg/m³ | cleaned = null |
| 2 | `NEGATIVE_CLIPPED` | value in [−5, 0) | clipped to 0 |
| 4 | `FLATLINE` | ≥ 6 identical consecutive hourly values above 0 | null |
| 8 | `SPIKE` | exceeds centred 7-h rolling median by max(8·MAD, 100 µg/m³) | null |
| 16 | `SPATIAL_OUTLIER` | exceeds neighbours' median (≤ 15 km, ≥ 3 neighbours) by max(6·MAD, 80 µg/m³) | null |
| 32 | `HIGH_RH` | RH > 90 % (hygroscopic growth inflates optical readings) | flag; corrected if the correction is enabled |
| 64 | `LOW_COVERAGE` | < 75 % of sub-hourly readings present | null |
| 128 | `HIGH_RANGE` | > 500 µg/m³ (optical sensors typically lose accuracy near their effective range; check your device datasheet) | flag only |
| 256 | `DRIFT_SUSPECT` | weekly regression slope vs neighbours outside [0.5, 2.0] | flag only |
| 512 | `IMPUTED` | filled by short-gap imputation (§6) | flag |

**Why the flags are not all deletions.** In a genuine dense-smog episode a *true* value can exceed the spatial-outlier rule at one station; the rule therefore applies only
when at least three neighbours exist, and `HIGH_RANGE` keeps (but flags) the values that matter most for public health.

**Low-cost correction.** The correction has the linear form `pm25_corrected = a·pm25_raw + b·RH + c` (the structure of Barkjohn et al., 2021, developed for PurpleAir sensors).
**Coefficients are not hard-coded**: they are fitted per sensor model against reference monitors in Lahore (`isMonitor`, ≤ 8 km, ≥ 500 hourly pairs, 60-day window, Huber loss), validated in Phase 2, and switched on in
`configs/preprocessing.yaml` only if cross-validated error improves. The regulatory-vs-low-cost calibration gap between Delhi and Lahore is also handled downstream by few-shot adaptation.

**RH mask for adaptation support (not a QC flag; off by default).** `HIGH_RH` above is a *QC* flag on stored data. Separately, the adaptation recipe can optionally drop *support* readings from low-cost sensors when RH > 85 % (device RH if present, otherwise CAMS RH), provided at least 50 % of the support window survives. This filter acts only on the labels used to adapt the encoder; it never edits stored data, targets or evaluation rows. It is disabled until a Phase 1–2 audit (share of Lahore support hours affected, effect on upper-quantile pinball) shows it helps; specification in [ML architecture §8](ml-architecture.md#8-meta-learning-maml-and-adaptation).

**Station registry and eligibility.** A station is *eligible* for training/evaluation if its 90-day uptime ≥ the domain threshold and it has ≥ 30 days of history. Ineligible stations may still be *displayed* live.
The registry is versioned (`registry_version`) and every experiment records the version it used.

## 6. Gap handling and imputation

* **≤ 3 consecutive missing hours:** shape-preserving **PCHIP interpolation in log1p space**; edges (gap touching the start or end of the series) stay missing.
* **> 3 hours:** stay missing, with an explicit mask channel and a time-since-last-observation channel for the encoder (GRU-D-style inputs).
* **Imputed hours are inputs only.** They are never targets, never scored, and do not count toward the 18-of-24 completeness rule.
* **Latency gap.** The final hours before *T* that are not yet `available_at ≤ T` (§11) are *not imputed*: they are simply missing, in training and in serving alike.

**Why not cubic splines.** The original plan calls spline interpolation "mathematically appropriate". It is not, for PM2.5: smog peaks are sharp and splines ring. Tested on a three-hour gap
between 520 and 480 µg/m³, a cubic spline fills the gap with values up to **962 µg/m³** (+443 above both neighbours); on a trough-to-rise gap it produced **negative** concentrations (−64 µg/m³).
PCHIP and linear interpolation stayed inside the neighbours' envelope in both cases. `tests/unit/test_qc_and_imputation.py` contains this as a regression test.

**Why not fill long gaps with model values** (as the original plan suggests). Filling the *observed* channel with CAMS and then forecasting against CAMS inflates skill through circularity and hides sensor outages from the model.
Instead, CAMS enters as its own channel, and the model learns how much to trust each source from the mask.

## 7. Copernicus ADS: CAMS global forecasts

**Dataset** `cams-global-atmospheric-composition-forecasts` (ADS). Verified content: PM1/PM2.5/PM10, total and speciated aerosol optical depth (including total AOD at 550 nm), trace gases, and a set of meteorological
variables including **boundary-layer height, 10 m wind components, 2 m temperature and dewpoint, surface pressure, total precipitation, cloud cover, surface solar radiation** and multi-level temperature. Grid 0.4°, hourly
single-level output, runs at 00 and 12 UTC, 5-day lead, archive from 2015, licence CC BY 4.0, GRIB (NetCDF conversion available).

> **Not the same as the "CAMS regional forecast" in the original plan.** The CAMS regional ensemble covers **Europe only**; for the Indo-Gangetic Plain the global forecast is the only CAMS option.

**Why this is the backbone.** One product, one login, supplies the PM2.5 baseline *and* the meteorology (boundary-layer height, winds, T, Td) *and* a gap-free AOD. Training on the **archive of forecasts** (2015–present) and serving the
**live forecast** removes the reanalysis→forecast train/serve skew (§11 and [ML architecture](ml-architecture.md)).

**Request pattern** (keys mirror the dataset page's *Show API request*; a contract test compares them): `type=forecast`, `data_format=grib`, `date`, `time ∈ {00:00, 12:00}`, `leadtime_hour` list, `area=[N,W,S,E]` from `igp_gridded_subset`,
single-level variables (`particulate_matter_2.5um`, `total_aerosol_optical_depth_550nm`, `boundary_layer_height`, `10m_u/v_component_of_wind`, `2m_temperature`, `2m_dewpoint_temperature`, …) and pressure-level `temperature` at 925/850 hPa.
Operational leads: `lead_max = δ + 72 h`, with δ = T − B*(T): **84 h** for a 00 UTC issuance at level 0 and **96 h** at degradation level 1 (δ = 24 h). Training-lattice issuances at 06Z/18Z need up to 90 h (102 h at level 1). A run for the Indo-Gangetic box is on the order of 1–2 MB.

**Operational details.**
* **Authentication.** Personal access token from the ADS profile page; the Python `cdsapi` client reads `url` and `key` (the old `UID:key` format is gone). CDS and ADS are *separate data stores with separate endpoints*; keep
  `ADS_API_KEY` and `CDS_API_KEY` as separate secrets and construct one client per store.
* **Licence acceptance** is a one-time, manual, per-dataset action in the web UI; until done, requests fail with HTTP 403 *required licences not accepted*. `doctor --online` detects this.
* **Queueing.** Requests queue; poll every 30 s within a 20-minute wall-clock budget. Backfill requests are chunked per day (or per month for short lead lists) and are resumable.
* **Non-operational service.** The ADS is not an operational ECMWF service; delays occur (user reports after the spring-2026 model upgrade confirm this). The degradation ladder (level 1: use the previous cycle) exists for that reason.
* **Model cycles change yearly**, shifting forecast distributions. Every extracted record carries `cams_cycle` (when exposed) and each model bundle records the cycles it was trained on; evaluation reports skill before and after an upgrade.

**Station extraction** (`preprocessing/gridded.py`): bilinear interpolation from the four surrounding nodes to each station and to the city centroid; record `grid_distance_km`. Unit conversion: PM2.5 kg m⁻³ → µg m⁻³ (× 10⁹);
RH from 2 m temperature *T* and dewpoint *T_d* (°C) by the Magnus relation `e_s(T) = 6.1094·exp(17.625·T/(T+243.04))` hPa, `RH = 100·e_s(T_d)/e_s(T)`. Lower-tropospheric stability uses the 925/850 hPa temperatures.

## 8. Copernicus CDS: ERA5 hindcast

ERA5 (0.25°, ≈ 5-day latency for ERA5T) can **never** be an operational input. It is kept for: (1) **parity diagnostics**: how close CAMS-IFS boundary-layer height and winds are to the reanalysis (a sanity check, not a training target; this diagnostic is among the first scope cuts);
(2) filling any gap in the CAMS archive; (3) the ERA5-only "physics-prior" ablation; (4) point extraction for station hindcasts via `reanalysis-era5-single-levels-timeseries` — flagged *experimental, not recommended for operational systems* by ECMWF — with a fallback to
monthly gridded subsets of `reanalysis-era5-single-levels` over `igp_gridded_subset`. Variables: `boundary_layer_height`, `10m_u/v_component_of_wind`, `2m_temperature`, `2m_dewpoint_temperature`, `surface_pressure`, `total_precipitation`, `surface_solar_radiation_downwards`, `total_cloud_cover`.

## 9. NASA FIRMS fire ingestion

**API:** `GET /api/area/csv/<MAP_KEY>/<SOURCE>/<west,south,east,north>/<DAY_RANGE ≤ 10>/<YYYY-MM-DD>`; CSV only; **5,000 transactions per 10 minutes** per key (larger requests can count as several). The client budgets 50 %.
Availability per source is queried at run time (`/api/data_availability/csv/<MAP_KEY>/<SOURCE>`), never hard-coded.

**VIIRS columns:** `latitude, longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_ti5, frp, daynight`. Keep `confidence ∈ {n, h}`; hour bin from `acq_date` + `acq_time` (UTC, HHMM).

**Platform continuity policy** (a design driver, not a footnote):

| Platform | Status | Policy |
|---|---|---|
| VIIRS **NOAA-21** | operating | operational primary |
| VIIRS **NOAA-20** | operating | operational secondary |
| VIIRS **Suomi-NPP** | **product delivery ends 2026-11-01 13:00 UTC** (NOAA/NASA notice, Aug 2026) | historical only; the client refuses to query it after that instant |
| MODIS Aqua / Terra | planned end of science collection 2026–27 (LAADS transition notices; the documents differ on exact dates); orbits drifting | legacy/at-risk; excluded from the operational feature set |

Because the number of contributing platforms changes over time (earlier seasons have fewer VIIRS platforms; more once NOAA-21 data begin; fewer again after Suomi-NPP ends; exact dates come from the availability endpoint), raw counts would create an artificial trend. All fire features are therefore **platform-normalised**
(divide by `platforms_active`, derived from per-platform coverage), and `platforms_active` is itself a feature-store column for diagnostics.

**Aggregation (target-centric).** For each target, compute geodesic distance *d* and the **target→fire azimuth θ** to every detection (`pyproj.Geod`, clockwise from north). Detections are binned into rings `[0,50), [50,150), [150,300), [300,600]` km and eight 45° sectors **by θ**, accumulating platform-normalised counts and FRP per hour.
The transport-weighted exposure indices `E24/E48/E72` use the opposite vector: **β = bearing of fire→target = θ + 180° (mod 360°)**, compared with the direction the air moves toward, φ = atan2(u, v). The full definition and one worked example (Amritsar → Lahore: ΔE ≈ 48.8 km, ΔN ≈ 12.6 km, θ ≈ 75.5°, β ≈ 255.5°; an easterly wind gives a ≈ 0.97, a westerly wind a = 0) are in [ML architecture §3](ml-architecture.md#3-fire-transport-exposure) and are not repeated here, so there is a single convention.
Target-centric geometry is deliberately
**not** a fixed "upwind box": Lahore sits on the border, with Indian Punjab to its east and south-east, so a single north-west box (as in the original plan, which is right for Delhi) would miss the dominant sources.

**NRT vs standard processing.** Historical training uses standard-processing (SP) data; operations use NRT. The two differ slightly; `source_product` is stored and a Phase 2 diagnostic compares NRT and SP counts on the overlap.

## 10. Aerosol optical depth strategy

* **Operational AOD = CAMS total AOD at 550 nm.** Gap-free, available at every lead time (so it is a legitimate *forecast* covariate), same product as the meteorology.
* **MAIAC (MCD19A2) is research-only and disabled.** Reasons: MODIS Aqua/Terra end-of-mission is planned for 2026–27 with drifting orbits; the product is HDF4 in sinusoidal tiles (GDAL HDF4 driver, Earthdata token that expires);
  its 1 km resolution is irrelevant against a ~40 km model grid; and daily composites arrive with a delay of days.
* **The cloud-masking paradox is a testable hypothesis, not a design assumption.** The plan states that dense smoke is misclassified as cloud, so valid-pixel fraction falls exactly when pollution peaks. A Phase 2 notebook
  quantifies this on historical MAIAC (valid-pixel fraction vs reference PM2.5, by season). If confirmed, it is a publishable remote-sensing finding; it does not change operations.

## 11. Temporal alignment

Every table carries **valid time**; knowledge time is *derived*, not recorded. The alignment layer implements one rule:

> **A feature at issuance *T* uses a row only if `available_at ≤ T`.**
> `available_at = hour_end + L_source` for hourly observation-like data (hour_end = `ts_utc + 1 h`; FIRMS: `acq_datetime + 3 h`); `available_at = B + 10 h` for a CAMS cycle with base time *B*; `available_at = local_day_end + 72 h` for the archive.
> `ingested_at_utc` is an audit field and never enters this rule.

1. **Latencies** live in `configs/sources.yaml`. The OpenAQ latency starts as an **assumed 3.0 h** and is replaced by the **measured p95** of `first_seen_utc − hour_end_utc` from `obs_pull_log` once ≥ 14 days have been captured (earliest adoption about 2 Nov, after capture starts ~16 Oct); FIRMS NRT ≈ 3 h; archive 72 h; CAMS by the base-time rule. Changing a latency is a dated entry in the verification log, and bundles record the latency table they were trained with.
2. **Same rule live and backfilled.** For live data the rule *excludes* rows that arrived earlier than the modelled latency, and rows that arrive later are simply absent; for backfilled data `available_at` is reconstructed from the same formula. Training and serving therefore see identical information. The input snapshot records exactly what a live run used.
3. **CAMS:** `B*(T) = max{ B ∈ {00Z, 12Z} : B + 10 h ≤ T }`; forecast age **δ = T − B\*** (a model input, `cams_lead_offset_h`); lead of a target block end = `T + h − B*`.

   | *T* | *B*\* | δ (level 0) | δ (level 1, +12 h) |
   |---|---|---|---|
   | 00Z | 12Z of D−1 | 12 h | 24 h |
   | 06Z | 12Z of D−1 | 18 h | 30 h |
   | 12Z | 00Z of D | 12 h | 24 h |
   | 18Z | 00Z of D | 18 h | 30 h |

4. **Stitched lookback:** for each hour *t* in `[T − 72 h, T]`, take the value from the **latest cycle *B* with `B ≤ t` and `B ≤ B*(T)`**. No cap on lead is imposed (leads reach δ at the end of the window for 06Z/18Z issuances).
5. **Join:** features are produced with a backward as-of join on `available_at` per source. Unit tests assert: the four lattice points of the table above; stitched lookback never selects a cycle with `B > B*(T)` or `B > t`; and no feature uses data with `available_at > T`.
6. **Train/serve parity** is enforced structurally: training reads the same product (CAMS forecast archive, both 00Z and 12Z bases) through the same alignment function as serving. ERA5 is *not* substituted for the lookback. Training augments δ with an extra {0, 12, 24} h offset (probabilities 0.70 / 0.20 / 0.10) so the model has seen stale cycles.
7. **Consequence for baselines.** Because the last ~`L_oaq` hours are unavailable at *T*, persistence-type baselines use the last complete 24-h block **available at *T*** (ending at `T − L_oaq`), not the block ending at *T*.

## 12. Storage layout and the `state` branch

Directory layout and what is tracked are specified in [`data/README.md`](../data/README.md). Daily journals on the orphan `state` branch (~150 KB/day):

| Path | Content | Schema |
|---|---|---|
| `obs/<domain>/dt=YYYY-MM-DD.parquet` | QC'd hourly observations | `observations_hourly` |
| `obs_pull_log/dt=YYYY-MM-DD.parquet` | first-seen log for latency measurement (proposed) | `obs_pull_log` |
| `inputs/<domain>/dt=YYYY-MM-DD.parquet` | point-in-time input snapshot for replay | `feature_store` + CAMS/fire series |
| `forecasts/dt=…/forecast_log.parquet` | 19 quantile levels × horizons × points × methods; `run_id`, `is_rerun`, `adaptation_status` | `forecast_log` |
| `scores/dt=…/score_log.parquet` | provisional and settled scores (first valid issuance only) | `score_log` |
| `manifests/YYYY-MM-DD.json`, `site_json/forecast/YYYY-MM-DD.json` | run manifest; public bulletin (with `valid_until_utc`, `basis`) | JSON / `bulletin.schema.json` |

The new columns and the `obs_pull_log` table are schema changes that require owner approval before the schemas are edited. The branch is cloned sparsely, size-guarded (warn 400 MB / alert 700 MB) and rotated yearly (see [system architecture §6](system-architecture.md#6-state-storage-and-persistence)).

## 13. Phase 1 data audit (acceptance criteria)

Before any modelling, run `notebooks/01_station_audit_*` and record the results in the verification log:

| Question | Acceptance / decision rule |
|---|---|
| How many **eligible** Lahore stations exist on OpenAQ, by provider, class (reference vs low-cost) and 90-day uptime? | The sparsity curve needs `N_max = eligible − 3` held-out. If eligible < 4, widen the *target family* to Lahore + Islamabad + Indian-Punjab cities and report the limitation |
| How many Delhi stations/seasons are available, and what fraction of smog-season days have ≥ 18 valid hours? | ≥ 20 stations and ≥ 5 seasons ⇒ Delhi-only source is viable (E/S/V/T needs at least 4 seasons, one per block; H5 needs ≥ 20 stations in season T); else enable `amritsar`/`ludhiana` (`configs/domains.yaml`) |
| Which Lahore stations are `isMonitor` (regulatory-grade)? | ≥ 1 required as the calibration anchor for the low-cost correction and as the truth anchor of co-located groups |
| Winter-day completeness in Lahore | < 50 % ⇒ Phase 2 uses a longer support window and wider intervals |
| Does `/v3/sensors/{id}/hours` carry coverage metadata? | If yes, use it in `LOW_COVERAGE`; if not, derive from raw counts |
| Phase 1 tasks | latency capture (P1-18), backfill pilot (P1-19), Lahore archive/API reconciliation (P1-20), co-location and truth-basis audit (P1-21), as-of lattice and parity audit (P1-22), licence audit, CAMS bias audit at Lahore reference monitors (this resolves the untagged "raw CAMS under-predicts" claim), and the RH-mask audit (share of low-cost support hours with RH > 85 %). |
