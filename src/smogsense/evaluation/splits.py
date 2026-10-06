"""smogsense.evaluation.splits — Purged, embargoed, blocked time-series splits.

72 h embargo between train/validation/test blocks; spatial hold-out by station; no sample whose
lookback or target window overlaps another split.

Specification: docs/evaluation-strategy.md → 'Data splits and leakage control'
"""
