# data/

**Nothing in `raw/`, `interim/`, `processed/` or `external/` is ever committed** (see `.gitignore`); only the directory
structure (`.gitkeep`) and `schemas/` are tracked. Durable, small journals live on the orphan `state` branch; large
artefacts (backfilled Parquet, model bundles) live in GitHub Releases.

| Directory | Contents | Produced by | Lifetime |
|---|---|---|---|
| `raw/openaq/` | Raw OpenAQ API JSON pages, one file per request, with a request manifest | `smogsense ingest openaq` | CI: job only; local: until you delete |
| `raw/openaq_archive/` | Daily `location-<id>-<yyyymmdd>.csv.gz` files from the OpenAQ AWS archive | `ingest openaq-archive` | local |
| `raw/cams/` | CAMS global forecast GRIB (one file per base time and lead-time chunk) | `ingest cams` | CI: Actions cache 7 d |
| `raw/era5/` | ERA5 monthly NetCDF/GRIB subsets (hindcast only) | `ingest era5` | local |
| `raw/firms/` | FIRMS CSV per source and date window | `ingest firms` | local |
| `raw/maiac/` | MCD19A2 HDF4 tiles (research only, disabled by default) | `ingest maiac` | local |
| `interim/qc/` | Observations after QC with `qc_flags` | `preprocessing/qc.py` | regenerable |
| `interim/aligned/` | As-of aligned inputs per issuance (point-in-time snapshot) | `preprocessing/alignment.py` | regenerable |
| `interim/station_series/` | CAMS/ERA5 fields extracted at stations | `preprocessing/gridded.py` | regenerable |
| `processed/feature_store/` | Versioned Parquet feature tables (`feature_set_version`, `config_hash`) | `features build/backfill` | regenerable |
| `processed/training_sets/` | Sparse-network task tables for meta-learning | `features backfill` | regenerable |
| `processed/forecasts/` | `forecast_log` Parquet | `forecast run` | mirrored to `state` |
| `processed/scores/` | `score_log` Parquet | `score shadow` | mirrored to `state` |
| `processed/run_manifests/` | Run manifest JSON + `latest_summary.md` | every command | mirrored to `state` (manifests only) |
| `external/station_registry/` | Registry snapshots (`registry_version`) | `data_ingestion/station_registry.py` | versioned |
| `external/static_covariates/` | DEM-derived elevation, built-up fraction, Diwali dates | one-off scripts (Phase 2) | versioned |
| `external/boundaries/` | City polygons used for station assignment | one-off | versioned |
| `schemas/` | **Tracked.** Pandera YAML contracts + `bulletin.schema.json` | humans | forever |

## Data contracts

`schemas/*.schema.yaml` are Pandera schemas (`pandera.DataFrameSchema.from_yaml`) and are the only definition of column
names, dtypes, units and valid ranges. Every stage validates its output against its schema before writing.
Two implementation rules were learned the hard way (Pandera 0.33 with pandas 3.0):

1. Call `df.reset_index(drop=True)` before `validate()`. A duplicate *index* turns a clean `SchemaError` into an opaque
   `ValueError` in the failure-reporting path.
2. Catch **both** `pandera.errors.SchemaError` and `pandera.errors.SchemaErrors` (strict-mode and lazy failures raise the
   plural form). `utils/io.validate_frame()` normalises both into a single contract-violation error (CLI exit code 30).

## Bitemporal rule

Observation tables carry two times: `ts_utc` (when it was true) and `ingested_at_utc` (when we learned it). A feature at
issuance *T* may use a row only if `ingested_at_utc <= T`. Historical backfills have no true knowledge times, so they are
assigned `ts_utc + source_latency` from `configs/sources.yaml`; this is documented in `docs/data-engineering.md`.

## The `state` branch (not in this working tree)

Created automatically by `scripts/state_attach.sh`, cloned into `.state/` (git-ignored). Layout:

```
state/
  obs/<domain>/dt=YYYY-MM-DD.parquet          QC'd hourly observations (~20 KB/day)
  inputs/<domain>/dt=YYYY-MM-DD.parquet       point-in-time input snapshot used by that day's forecast
  forecasts/dt=YYYY-MM-DD/forecast_log.parquet
  scores/dt=YYYY-MM-DD/score_log.parquet
  manifests/YYYY-MM-DD.json
  site_json/forecast/YYYY-MM-DD.json          the public bulletin JSON, for rebuilding the archive
  heartbeat.json
```

Expected growth is ~150 KB/day (~55 MB/year), far below GitHub's 1 GB recommendation.
