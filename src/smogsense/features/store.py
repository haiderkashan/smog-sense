"""smogsense.features.store — Versioned Parquet feature store.

Writes partitioned Parquet under data/processed/feature_store/ keyed by feature_set_version and
config hash; manages canonical point-in-time input snapshots under .state/inputs/{domain}/dt=YYYY-MM-DD.parquet.

Specification: docs/data-engineering.md → 'Storage layout and the state branch'
"""

from smogsense.features.snapshot import (
    InputSnapshot,
    build_input_snapshot,
    load_input_snapshot,
    save_input_snapshot,
)

__all__ = [
    "InputSnapshot",
    "build_input_snapshot",
    "load_input_snapshot",
    "save_input_snapshot",
]
