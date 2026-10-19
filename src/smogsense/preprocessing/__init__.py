"""smogsense.preprocessing — Cleaning, harmonisation and alignment.

Pure functions over DataFrames/xarray objects; no network access.

Specification: docs/data-engineering.md
"""

from smogsense.preprocessing.low_cost_correction import LowCostCorrectionModel

__all__ = ["LowCostCorrectionModel"]
