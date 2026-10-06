"""smogsense.data_ingestion.openaq_archive — OpenAQ Open Data on AWS reader (bulk historical backfill).

Reads s3://openaq-data-archive style objects over anonymous HTTPS:
records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>-<yyyymmdd>.csv.gz. Files
appear 72 hours after local end-of-day, so this source is for backfill and settled scoring only,
never for live inference.

Public contract (implemented in Phase 2):
- fetch_day(location_id, date) -> DataFrame
- Idempotent: skips days already present in data/raw/openaq_archive/ with matching sha256.

Specification: docs/data-engineering.md → 'OpenAQ archive backfill'
"""
