"""smogsense.features — Physics-informed feature engineering.

Deterministic, vectorised transformations that turn aligned inputs into model-ready tensors and
tables.

Specification: docs/ml-architecture.md → 'Feature engineering (physics-informed)'
"""

from smogsense.features.backfill import (
    BackfillResult,
    compute_oof_partition_hash,
    get_season,
    get_season_date_range,
    get_seasons_for_interval,
    is_boundary_purged,
    partition_block_e_folds,
    partition_seasons,
    run_feature_backfill,
)
from smogsense.features.store import (
    build_feature_table,
    load_feature_table,
    save_feature_table,
)

__all__ = [
    "BackfillResult",
    "build_feature_table",
    "compute_oof_partition_hash",
    "get_season",
    "get_season_date_range",
    "get_seasons_for_interval",
    "is_boundary_purged",
    "load_feature_table",
    "partition_block_e_folds",
    "partition_seasons",
    "run_feature_backfill",
    "save_feature_table",
]
