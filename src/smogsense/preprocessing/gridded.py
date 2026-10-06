"""smogsense.preprocessing.gridded — GRIB/NetCDF to station time series.

Reads CAMS/ERA5 files via xarray+cfgrib, converts units (kg m-3 to ug m-3, K to degC), and
extracts station series by bilinear interpolation recording grid distance.

Public contract (implemented in Phase 1):
- to_station_series(ds, stations) -> DataFrame validated by cams_station_series.schema.yaml

Specification: docs/data-engineering.md → 'Copernicus ADS: CAMS global forecasts'
"""
