"""smogsense.features.sparse_network — Sparse-network simulation and neighbour-history signal.

Samples available-station subsets of size N from a dense network, builds the inverse-distance-
weighted history signal from available stations, and applies modality dropout so the model
learns to forecast with N = 0, 1, 2, 5, ... observed stations.

Public contract (implemented in Phase 2):
- sample_available(stations, n, rng)
- idw_history(available, target, window)

Specification: docs/ml-architecture.md → 'Sparse-network task construction'
"""
