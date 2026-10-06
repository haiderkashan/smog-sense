"""smogsense.utils.io — Atomic file writes, Parquet helpers, content hashing.

Write-then-rename semantics so that a killed job never leaves a half-written artefact.

Specification: docs/data-engineering.md → 'Storage layout and the state branch'
"""
