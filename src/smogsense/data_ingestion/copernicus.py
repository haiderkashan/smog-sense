"""smogsense.data_ingestion.copernicus — CDS and ADS client wrapper (CAMS global forecasts, ERA5 hindcast).

Wraps cdsapi for two distinct data stores with separate endpoints and personal access tokens:
ADS (cams-global-atmospheric-composition-forecasts) and CDS (reanalysis-era5-single-levels).
Handles licence-not-accepted errors, queue polling with bounded wall-clock budget, area
subsetting and GRIB/NetCDF landing.

Public contract (implemented in Phase 1):
- latest_available_cams_run(now_utc) -> base_time_utc using the 10 h availability rule.
- fetch_cams(base_time_utc, leadtimes, variables, area) -> Path
- fetch_era5(month, variables, area) -> Path

Specification: docs/data-engineering.md → 'Copernicus ADS: CAMS global forecasts'
"""
