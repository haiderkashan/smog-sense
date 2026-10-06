"""smogsense.data_ingestion — Source clients (network I/O only; no scientific logic).

One module per external source. Clients return raw, typed records and persist raw payloads under
data/raw/ with a manifest; they never clean, impute or aggregate.

Specification: docs/data-engineering.md
"""
