"""smogsense.publishing.state — State-branch synchronisation and compaction.

Reads/writes append-only Parquet under the orphan 'state' branch; monthly squash keeps
repository size bounded.

Specification: docs/system-architecture.md → 'State, storage, and persistence'
"""
