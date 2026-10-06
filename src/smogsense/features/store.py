"""smogsense.features.store — Versioned Parquet feature store.

Writes partitioned Parquet under data/processed/feature_store/ keyed by feature_set_version and
config hash; refuses to overwrite a partition with different hashes.

Specification: docs/data-engineering.md → 'Storage layout and the state branch'
"""
