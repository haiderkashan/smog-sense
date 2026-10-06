"""smogsense.features.windowing — Lookback tensors with masks and time-since-observation channels.

Builds the L = 72 h hourly input window X (values, mask, delta) per (station, issuance) on the
6-hourly training lattice.

Specification: docs/ml-architecture.md → 'Sequence encoder'
"""
