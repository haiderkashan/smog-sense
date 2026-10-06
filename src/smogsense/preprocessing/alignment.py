"""smogsense.preprocessing.alignment — Bitemporal as-of joins.

Implements the availability rule B*(T) = max{B in {00Z, 12Z} : B + 10 h <= T} for CAMS and per-
source latency tables so that features at issuance T only use information that was knowable at
T.

Public contract (implemented in Phase 1):
- asof_cams_run(issuance) -> base_time
- stitch_cams_series(issuance, window) -> hourly series from the most recent knowable cycle for
  each valid hour

Specification: docs/data-engineering.md → 'Temporal alignment'

Contract update: unified rule and corrected stitching.
"""
