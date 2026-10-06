"""smogsense.contracts — Shared value objects: Issuance, Horizon, QuantileGrid, DomainId.

Frozen dataclasses for the vocabulary used across every stage: issuance time T0 (UTC), forecast
horizons h in {24, 48, 72}, the 19-level quantile grid and the three public levels.

Public contract (implemented in Phase 0):
- Issuance.block_window(h) -> (start_utc, end_utc) for the 24 h block ending at T0 + h.
- QuantileGrid.public == (0.10, 0.50, 0.90).

Specification: docs/ml-architecture.md → 'Problem formulation'
"""
