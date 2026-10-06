"""smogsense.inference.degradation — Degradation ladder.

Chooses the best available mode: full, stale-CAMS, observations-only, baseline-only, and
publishes the mode on the bulletin.

Specification: docs/system-architecture.md → 'Failure handling and degradation ladder'

Contract update: stale notice.
"""
