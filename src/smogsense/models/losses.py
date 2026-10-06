"""smogsense.models.losses — Pinball (quantile) loss and aggregates.

rho_tau(u) = u (tau - 1[u<0]); masked mean over levels, horizons and valid targets.

Specification: docs/ml-architecture.md → 'Quantile heads and distributional output'
"""
