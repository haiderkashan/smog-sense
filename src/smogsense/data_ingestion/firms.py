"""smogsense.data_ingestion.firms — NASA FIRMS area-API client (active fire detections).

Queries /api/area/csv/<MAP_KEY>/<SOURCE>/<west,south,east,north>/<DAY_RANGE>/<DATE> in windows
of at most 10 days for VIIRS NOAA-21 and NOAA-20 (operational) plus legacy MODIS/Suomi-NPP
(historical only). CSV only; 5,000 transactions per 10 minutes per MAP_KEY.

Public contract (implemented in Phase 1):
- detections(source, bbox, start, end) -> DataFrame
- Platform availability table guards against requesting Suomi-NPP after 2026-11-01T13:00Z.

Specification: docs/data-engineering.md → 'NASA FIRMS fire ingestion'
"""
