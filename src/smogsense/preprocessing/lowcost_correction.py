"""smogsense.preprocessing.lowcost_correction — Alias for smogsense.preprocessing.low_cost_correction.

Provides backward and alternate naming compatibility.
Specification: docs/data-engineering.md §5, docs/ml-architecture.md §8, and PRD FR-11.
"""

from __future__ import annotations

from smogsense.preprocessing.low_cost_correction import (
    LowCostCorrectionModel,
    audit_corruption_priors,
    build_paired_observations,
    find_candidate_stations,
    haversine_distance_km,
    update_preprocessing_config_decision,
)

__all__ = [
    "LowCostCorrectionModel",
    "audit_corruption_priors",
    "build_paired_observations",
    "find_candidate_stations",
    "haversine_distance_km",
    "update_preprocessing_config_decision",
]
