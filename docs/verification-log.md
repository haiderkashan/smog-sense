# Verification Log and Errata — Research Plan v1

> **What this is.** The original [`SmogSense_Project_Research_Plan_v1.pdf`](research/SmogSense_Project_Research_Plan_v1.pdf) was explicitly a *starting point that might contain errors*. This log records, claim by claim, what was checked,
> how, what was found, and what changed in the design. **Verification date: 1–4 October 2026.** Provider behaviour changes: when you re-check something, append a dated entry rather than editing history.
>
> **Legend.** ✅ confirmed · 🔧 refined (true but incomplete or needs a design change) · ❌ incorrect or inconsistent (design corrected) · ❓ not verified (reason given; re-check scheduled) · 🆕 new fact the plan did not contain.
> **Evidence** types: **[T]** experiment run in this repository's tooling · **[D]** provider documentation or notice read during verification · **[A]** analytical argument (geometry, mathematics).

## Summary: the ten changes that matter most

| # | Plan v1 said | Reality | Design consequence |
|---|---|---|---|
| 1 | Run daily at 00:00 UTC using CAMS | CAMS 00Z is guaranteed only by **10:00 UTC**; 12Z by 22:00 UTC | Use the *previous day's 12Z* run; explicit as-of rule; 00:17 UTC trigger + two idempotent retries |
| 2 | Upwind fire box to the north-west of Lahore | Indian Punjab lies **east / south-east** of Lahore; NW is right for Delhi, wrong for Lahore | Target-centric, wind-aligned fire exposure (rings × sectors) instead of a fixed box |
| 3 | "CAMS regional forecast" as the baseline | CAMS regional ensemble is **Europe-only** | Use **CAMS global** (0.4°) — which also carries PBLH, winds, T/Td, AOD |
| 4 | Copernicus quotas: 4 connections, 20 MB/s, 10-minute token | Those are **Copernicus Data Space Ecosystem** (Sentinel) limits, not CDS/ADS | CDS/ADS: personal tokens, queues, **per-dataset licence acceptance** |
| 5 | LightGBM does multi-quantile via an `alpha` array | `alpha` is a single double; list **rejected** [T] | 3 × 19 boosters + monotone rearrangement |
| 6 | MAML adapts the LightGBM model | Trees are not differentiable | MAML only for the neural encoder/head; LightGBM warm-start residual boosting |
| 7 | MODIS MAIAC AOD as an operational feature | MODIS Aqua/Terra end-of-mission planned 2026–27; **Suomi-NPP data ends 2026-11-01** | CAMS AOD550 operational; MAIAC research-only; platform-normalised fire features |
| 8 | Cubic-spline gap filling is "mathematically appropriate" | Spline overshot to **962 µg/m³** between 520 and 480 [T] | PCHIP in log space for ≤ 3 h; mask otherwise |
| 9 | Sparsity curve starts at "zero stations = pure zero-shot" | The encoder consumes the local PM2.5 history — none exists at N = 0 | Sensor-free mode, sparse-network task construction, N defined for labels *and* inputs |
| 10 | GitHub runner is 2-core / 7 GB; deploy by committing to `gh-pages` | Public repos get 4 vCPU / 16 GB; Pages needs a *public* repo; daily commits bloat the repo | Public repo; design to 2 vCPU / 7 GB anyway; single-orphan-commit publish; separate `state` branch |

## A. Claim-by-claim

### A1. Ground-truth data and OpenAQ

| ID | Plan v1 claim | Status | Finding and evidence | Impact |
|---|---|---|---|---|
| V-01 | OpenAQ v3 free tier: 60 requests/min | ✅🔧 | Confirmed, **plus 2,000 requests/hour**; API key mandatory; headers `x-ratelimit-used/reset/limit/remaining`; repeated violations can lead to a ban **[D]** | Budget 80 % (48/min, 1,600/h); header-aware limiter; circuit breaker |
| V-02 | OpenAQ is the primary ground-truth source | ✅🆕 | Also an **AWS Open Data archive**: anonymous HTTPS, `records/csv.gz/locationid=…/year=…/month=…/location-<id>-<yyyymmdd>.csv.gz`, **written 72 h after the end of the local day** **[D]** | Backfill and *settled* scoring only; live scoring uses the API (two truth stages) |
| V-03 | The Urban Unit installed 160 low-cost sensors in 14 cities | ❓ | Multiple independent operators serve Lahore (Urban Unit, Pakistan Air Quality Initiative, Punjab EPA, schools/community) according to public network listings **[D]**; OpenAQ's *Lahore* count, uptime and device classes are unknown | Phase 1 station audit decides the maximum $N$ of the sparsity curve ([data engineering §13](data-engineering.md#13-phase-1-data-audit-acceptance-criteria)) |
| V-04 | Delhi has a dense regulatory network | ✅❓ | Press reports cite ~38 continuous stations in Delhi **[D]**; CEEW states OpenAQ archives CPCB feeds; *current* OpenAQ coverage not audited | Audit in Phase 1; fall back to Punjab source cities if thin |
| V-05 | Low-cost sensors drift and over-read at high humidity | ✅ | Consistent with the literature; the plan's cited few-shot calibration paper (arXiv:2108.00640) exists but was not re-derived | RH flag, fitted correction (not hard-coded), drift flag |
| V-06 | Interpolate gaps < 3 h with splines; fill longer gaps from numerical models | ❌ | **[T]** Cubic spline across a 3-hour gap between 520 and 480 µg/m³ produced up to **962 µg/m³** (+443 above both neighbours); on a trough→rise gap it produced **−64 µg/m³**. PCHIP and linear stayed inside the envelope. Filling the observed channel with model values also makes a later "beats CAMS" comparison circular | PCHIP in log1p ≤ 3 h; mask beyond; CAMS is a separate channel; imputed hours are never targets |

### A2. Copernicus (CDS/ADS), CAMS, ERA5

| ID | Plan v1 claim | Status | Finding and evidence | Impact |
|---|---|---|---|---|
| V-07 | CDS free-tier "4 concurrent connections, 20 MB/s bandwidth, 10-minute token" | ❌ | The cited page is `documentation.dataspace.copernicus.eu`: the **Data Space Ecosystem** (Sentinel EO downloads). Those quotas (4 concurrent S3 connections at 20 MB/s; token 10 min / refresh 60 min) do not govern CDS/ADS **[D]** | CDS/ADS use personal access tokens, a request queue and **per-dataset licence acceptance** (HTTP 403 otherwise) **[D]** |
| V-08 | Use a "CAMS PM2.5 regional forecast" | ❌ | The CAMS *regional* ensemble covers Europe; for South Asia only the **global** forecast exists (0.4°, 00/12 UTC, 5 days, archive from 2015, CC BY 4.0) **[D]** | Global product is the baseline |
| V-09 | Real-time meteorology must come from GFS or CAMS | 🔧 | The CAMS global dataset itself includes **boundary-layer height, 10 m winds, 2 m T and dewpoint, pressure-level T, total AOD550, cloud, radiation, precipitation** **[D]** | One product supplies baseline + meteorology + AOD ⇒ no second NWP model |
| V-10 | Train on ERA5, serve with a forecast model (implicit) | ❌ (risk) | Boundary-layer height is model-dependent; reanalysis→forecast skew is a classic failure | Train on the **CAMS forecast archive**, serve the live forecast; ERA5 hindcast-only |
| V-11 | ERA5 hourly, 0.25°, ≈ 5-day latency | ✅🆕 | Confirmed. The new `reanalysis-era5-single-levels-timeseries` dataset is flagged **experimental / not for operations** **[D]** | Hindcast diagnostics only |
| V-12 | Daily pipeline at 00:00 UTC | ❌ | **00 UTC CAMS data guaranteed by 10:00 UTC; 12 UTC by 22:00 UTC** **[D]**; the ADS is **not an operational service**, and user reports show delays after the spring-2026 model upgrade **[D]** | As-of rule $B^\*(T)$; degradation level 1; 02:47/05:47 retries |

### A3. Fire and aerosol remote sensing

| ID | Plan v1 claim | Status | Finding and evidence | Impact |
|---|---|---|---|---|
| V-13 | FIRMS provides NRT MODIS/VIIRS fires | ✅🆕 | Free MAP_KEY; **5,000 transactions / 10 min**; CSV only; ≤ 10-day windows; sources include `VIIRS_NOAA21_NRT`, `VIIRS_NOAA20_NRT/SP`, `VIIRS_SNPP_NRT/SP`, `MODIS_NRT/SP` **[D]** | Client budgets 50 %; availability queried at run time |
| V-14 | (Silent on satellite retirements) | 🆕 | **Suomi-NPP product delivery ends 2026-11-01 13:00 UTC** (NOAA/NASA notice, Aug 2026) — mid-season. NASA plans to end Aqua/Terra MODIS science collection in 2026–27 (LAADS notices; dates differ between documents) and both are drifting **[D]** | NOAA-21/NOAA-20 operational; platform-normalised features; MODIS/SNPP legacy only |
| V-15 | MAIAC MCD19A2 (1 km daily AOD) is a key spatial feature | 🔧 | Product depends on MODIS; HDF4 sinusoidal tiles; Earthdata token; days of latency; 1 km irrelevant against a 40 km model | **Research-only**, disabled; operational AOD = CAMS AOD550 |
| V-16 | Dense smog is masked as cloud ⇒ missing AOD at peak pollution | ❓ | Plausible and widely discussed; not tested here | Testable Phase 2 hypothesis; publishable if confirmed; not an operational assumption |
| V-17 | Fire box upwind (north-west) of Lahore; "NW winds from Indian Punjab toward the south-east" | ❌ | **[A]** Amritsar is ≈ 50 km **east** of Lahore, Ludhiana east-south-east; Indian-Punjab smoke reaches Lahore under easterly/south-easterly flow; NW flow carries Pakistani-Punjab air. The statement is correct for **Delhi** | Target-centric polar rings/sectors + wind-aligned exposure ([ML §3](ml-architecture.md#3-fire-transport-exposure)) |
| V-18 | Partial-convolution imputation of masked AOD | 🔧 | Unnecessary once AOD comes gap-free from CAMS | Dropped |

### A4. Modelling

| ID | Plan v1 claim | Status | Finding and evidence | Impact |
|---|---|---|---|---|
| V-19 | LightGBM supports multi-quantile regression with an `alpha` array | ❌ | **[T]** `LGBMRegressor(objective="quantile", alpha=[0.1,0.5,0.9])` raises *"Parameter alpha should be of type double"*. One booster per level works; independent boosters crossed in **7/500 rows** | 57 boosters; rearrangement (sorting never increases pinball loss) |
| V-20 | MAML makes the (GRU + LightGBM) model adapt in few steps | ❌ | **[A]** MAML needs gradients through the inner update; tree ensembles have none | MAML for encoder/head; warm-start residual boosting for LightGBM |
| V-21 | Sensor-sparsity curve: $N=0$ is "zero-shot transfer"; add $1,2,5,N$ stations to the support set | ❌ | **[A]** The encoder's main input is the target station's PM2.5 history: at $N=0$ there is none; "labels for adaptation" and "observed history at inference" were conflated | $N$ defined for both; sensor-free mode; sparse-network tasks; own-history vs leave-stations-out ([ML §7](ml-architecture.md#7-sparse-network-task-construction)) |
| V-22 | "10th/50th/90th percentile … worst-case scenario" | 🔧 | **[A]** $[q_{.10},q_{.90}]$ is an **80 %** interval; $q_{.90}$ is exceeded ~1 day in 10 | Labels: "Bad case (1 day in 10 is worse)"; "8 days in 10" |
| V-23 | CRPS is the premier strictly proper score | ✅🔧 | Confirmed. But CRPS from only three quantiles is crude: **[T]** the quantile-score identity holds (analytic 25.618 vs 2∫ρ = 25.618) and a 7-level estimator without a tail model was off by up to **13 %**; with a tail model 1.0–3.5 % mean at $K=7$ and **0.2–0.3 % (≤ 0.5 %) at $K=19$** | 19-level grid; tail model; unit-tested estimator |
| V-24 | Baselines: persistence and raw CAMS | 🔧 | Raw CAMS under-predicts Lahore's peaks at 40 km, so beating it says little | Add CAMS-BC (fair), climatology, probabilistic forms; M2 is the reference |
| V-25 | Delhi is the optimal source: "identical" forcing | 🔧 | Same airshed, but Delhi is **downwind** of the burn belt while Lahore is adjacent: lag structure and source mix differ | Negative-transfer risk (R1); H4/H5; optional Amritsar/Ludhiana sources |
| V-26 | Shadow-run for prospective validation | ✅🔧 | Sound; but a model finished after the season starts would otherwise be untestable | Daily input snapshots, live vs replay shadow, freeze register ([evaluation §9](evaluation-strategy.md#9-prospective-shadow-run-protocol)) |
| V-27 | Standard 24/48/72 h point targets | 🔧 | Hourly low-cost values are noisy; AQI is a 24-h concept | 24-h block-mean targets with an 18/24-hour completeness rule |

### A5. Infrastructure, hosting, dissemination

| ID | Plan v1 claim | Status | Finding and evidence | Impact |
|---|---|---|---|---|
| V-28 | GitHub Actions runner: 2 cores, 7 GB | 🔧 | That is the **private-repo** runner (newest docs say 2 vCPU / **8 GB**, older ones 7 GB). **Public** repos: **4 vCPU / 16 GB / 14 GB SSD, free and unlimited** **[D]** | Design to 2 vCPU / 7 GB (compose emulates it); keep the repo public |
| V-29 | Scheduled workflows run reliably | ❌🆕 | Disabled after **60 days without repository activity** in public repos; scheduled runs may be delayed at the top of the hour **[D]** | Off-hour minutes; three idempotent triggers; keep-alive; watchdog |
| V-30 | Commit outputs daily to `gh-pages` | 🔧 | GitHub Pages: ≤ 1 GB site, ≈ 100 GB/month soft bandwidth, ≈ 10 builds/h soft, 10-min deploy timeout; **only public repos on the Free plan** **[D]**. Daily PNG commits grow the repository unboundedly | Single orphan commit per deploy ([T]: two deploys ⇒ one commit); journals on a separate `state` branch |
| V-31 | Hosting on Vercel/Cloudflare free tier | ✅🔧 | Cloudflare Pages Free: 500 builds/month (Git builds), 20,000 files, 25 MiB/file **[D]**. Vercel Hobby terms not re-verified ❓ | GitHub Pages primary; Cloudflare optional mirror |
| V-32 | WhatsApp is the primary dissemination vector | ✅🔧 | True as a channel, but the **WhatsApp Business Platform bills per delivered template message** outside a user-initiated window **[D]** | Share-ready cards + links; Telegram (free) can be automated |
| V-33 | Render the Urdu infographic with Matplotlib/Pillow | ❌🔧 | **[T]** Pillow + **Raqm** renders correctly joined Nastaliq (the Pillow wheel bundles Raqm/HarfBuzz/FriBiDi); the basic layout — what Matplotlib uses — gives disconnected, reversed glyphs. Font in Ubuntu `fonts-noto-core` (571 KB TTF → **107 KB** WOFF2 subset) | Pillow+Raqm for all public text; Matplotlib English-only; no headless browser |
| V-34 | (Silent on AQI scheme) | 🆕 | **US EPA 2024** PM2.5 breakpoints: 0–9.0, 9.1–35.4, 35.5–55.4, 55.5–125.4, 125.5–225.4, 225.5+ (24-h); index ends at 500 **[D]**; many apps still use the older breakpoints | Scheme named on the page; µg m⁻³ + category; never an extrapolated index |
| V-35 | "In November 2024 the AQI in Lahore reached 1900" | ❓ | Not re-verified. The EPA index is defined to 500; larger numbers appear only on consumer apps' extended scales | Not used anywhere in the pipeline; avoided in public copy |
| V-36 | Secondary organic aerosol "up to 60 % of PM2.5" | ❓ | Not verified; not needed | Not used |
| V-37 | Ventilation coefficient "up to four times lower in winter" | ❓ | Plausible; not verified here. Note: the textbook VC uses the *mean wind in the mixed layer*; the plan's 10 m wind is a proxy | Qualitative use only; $VC_{10}$ with optional $VC_{925}$; box-model justification added |
| V-38 | Ingest with the `openaq` Python SDK | 🔧 | Acceptable for exploration; a thin `httpx` client gives header-aware throttling and trivially mockable contracts | Thin client; SDK optional |

## B. Experiments run during verification (all reproducible from the repository's pinned tooling)

| ID | Test | Result |
|---|---|---|
| T-01 | LightGBM 4.7 with `alpha=[0.1,0.5,0.9]` | **Rejected** (`Parameter alpha should be of type double`). Separate boosters work; 7 of 500 rows crossed |
| T-02 | Gap interpolation on a PM2.5 spike | Cubic spline max **962** vs neighbours 520/480; PCHIP and linear bounded; trough→rise: spline **−64** |
| T-03 | CRPS from quantiles | Identity verified (analytic 25.618 equals the quantile-score integral 25.618; a point forecast gives exactly its absolute error, 40). Estimator error: K=7 with tails 1.0–3.5 % mean (≤ 6.2 % max); **K=19: 0.2–0.3 % (≤ 0.5 %)**; K=7 without tails up to 13.1 % |
| T-04 | Urdu shaping | Pillow Raqm **correct**; basic layout **broken** (visual comparison); web subset **109,632 bytes** |
| T-05 | GRIB stack | `eccodes` 2.49 is a pure-Python wheel that pulls **`eccodeslib`** (bundled C library): no apt `libeccodes`; GRIB2 → `cfgrib` → `xarray` round-trip succeeded |
| T-06 | Dependency resolution | The full set (runtime, dev, optional) resolves **wheels-only** on Python 3.12 (`uv pip compile --no-build`): pandas 3.0, NumPy 2.5, PyTorch 2.14, LightGBM 4.7, scikit-learn 1.9, xarray 2026.9, Pandera 0.33, Pillow 12.3 |
| T-07 | Apt packages | All names used in the Dockerfile exist on Ubuntu 24.04 (`libgdal34t64` pulls the HDF4/HDF5/NetCDF drivers) |
| T-08 | `.gitignore` | `git check-ignore` confirms secrets, AI-context files, data, weights are ignored while `.gitkeep`, schemas, `.env.example` and fixtures stay tracked |
| T-09 | Operational scripts | Against a bare repository: orphan `state` creation, idempotent commit, retry; `gh-pages` stays at **one** commit over repeated deploys; incomplete site refused; token never written to git config |
| T-10 | Static analysis | `hadolint` (Dockerfile), `shellcheck` (all scripts), `actionlint` + official schemas (workflows, Dependabot, issue forms, Compose spec, CITATION.cff): clean. Strict `mypy` over 73 modules, `ruff`, and the test suite pass |
| T-11 | WCAG contrast | Computed for every colour pair in the design tokens (min 5.25:1 for chips; chrome 6.45–14.67) |
| T-12 | Pandera 0.33 on pandas 3.0 | Failure reporting raises an opaque `ValueError` when the validated frame has a **duplicate index**; strict/lazy failures raise `SchemaErrors` (plural). Mitigation: `validate_frame()` resets the index and catches both |

## C. New risks the plan did not contain

CAMS model-cycle upgrades shift forecast distributions (and the ADS lagged after the spring-2026 upgrade) · MODIS and Suomi-NPP retirements · GitHub schedule drops and 60-day auto-disable · Pages requires a public repository · GHCR private-package quota (~500 MB) is smaller than the image ·
Git LFS quotas (never use LFS) · WhatsApp automation is billed · optical-sensor right-censoring biases upper quantiles low · Pandera/pandas 3 failure-reporting quirk · one season is one climate realisation (inference limits).

## D. Still unverified — owner and when

| Item | Why it matters | When |
|---|---|---|
| Exact OpenAQ v3 field/path names (`isMonitor`, `/sensors/{id}/hours` coverage metadata) | Registry and QC | Phase 1 contract tests |
| ADS request keys and variable names (and 925 hPa winds) | Ingestion correctness; $VC_{925}$ | Phase 1: compare with the dataset page's *Show API request* |
| OpenAQ Lahore/Delhi station counts, uptime, providers | Maximum $N$; source viability | Phase 1 audit |
| Cloudflare *direct-upload* build accounting; Vercel Hobby terms | Optional mirror | Before enabling the mirror |
| GitHub Pages CORS header on JSON | Third-party consumption | Phase 4 `curl -I` |
| Local (Punjab EPA) AQI categories | Public wording | Before launch (G-SCI) |
| Diwali dates file | Event flag for Indian domains | Phase 2 |
| Archive grouping, ADS limits, Booster.predict behaviour, SmoothL1/Huber relation, OpenAQ licence terms | Various unverified items added | Pending |
| Archive grouping, ADS limits, Booster.predict behaviour, SmoothL1/Huber relation, OpenAQ licence terms | Various unverified items added | Pending |

## E. Sources (accessed 1–4 October 2026)

OpenAQ documentation — *Rate limits* (`docs.openaq.org/using-the-api/rate-limits`) and *About Open Data on AWS* (`docs.openaq.org/aws/about`). · Copernicus ADS — dataset *CAMS global atmospheric composition forecasts* (`ads.atmosphere.copernicus.eu`, overview tab) and the ADS/CDS *How to use the API* pages. ·
ECMWF Confluence — CAMS global dissemination schedule; ECMWF Forum thread on ADS publication delays after the 50r1 upgrade. · Copernicus Data Space Ecosystem — *Quotas and Limitations* (`documentation.dataspace.copernicus.eu/Quotas.html`). · CDS dataset pages for ERA5 and the experimental ERA5 time-series dataset. ·
GitHub Docs — *GitHub-hosted runners*, *Events that trigger workflows* (schedule), *GitHub Pages limits*. · NASA FIRMS — *Area API* and *MAP_KEY* pages (`firms.modaps.eosdis.nasa.gov/api`). · NASA Earthdata / LAADS DAAC — MODIS-to-VIIRS transition notices; NOAA/NASA notice on the end of Suomi-NPP products (Aug 2026). ·
Cloudflare Docs — *Pages limits*. · Meta for Developers — *WhatsApp Business Platform pricing*. · US EPA — AQI technical assistance document (2024 revision of the PM2.5 breakpoints). · Open-Meteo — *Air Quality API* documentation. ·
IQAir / aqicn network listings for Lahore; press coverage of Delhi's monitoring network. · PyPI and GitHub release tags (library and action versions). · Tools: `uv`, `hadolint`, `actionlint`, `shellcheck`, `check-jsonschema`.

## F. Template for future entries

```
### V-NN · <short title> · <YYYY-MM-DD>
Claim:      <what was believed, and where it came from>
Check:      <how it was verified: [T] test / [D] document / [A] argument>
Result:     <confirmed / corrected / refined / unverified>
Impact:     <files and sections changed>
```

## G. Review round 2 (reasoning-only)
- Reasoning-only tables indicating new methods.
