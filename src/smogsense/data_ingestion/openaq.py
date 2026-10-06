"""smogsense.data_ingestion.openaq — OpenAQ v3 REST client (near-real-time observations).

Discovers PM2.5 sensors inside a domain bounding box, paginates /v3/locations and
/v3/sensors/{id}/hours, and returns hourly records plus coverage metadata. Free tier is 60
requests/minute and 2,000/hour per API key; the client budgets 80% of both.

Public contract (implemented in Phase 1):
- list_locations(domain) -> DataFrame (registry candidates).
- hourly(sensor_ids, start_utc, end_utc) -> DataFrame (raw, un-QC'd).
- Respects pagination via limit/page and stops on found/limit arithmetic, never on empty-page
  guessing.

Specification: docs/data-engineering.md → 'OpenAQ v3 client'

Contract update: pull ledger, quota partition.
"""
