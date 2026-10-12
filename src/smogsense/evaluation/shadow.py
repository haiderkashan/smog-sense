"""smogsense.evaluation.shadow — Prospective scoring ledger.

Provisional scoring from the OpenAQ API at T0+h+6 h and settled scoring from the archive after
the 72 h publication lag; append-only score_log.

Specification: docs/evaluation-strategy.md → 'Prospective shadow-run protocol'

Contract update: first-issuance scoring.
"""

from smogsense.pipeline.shadow import (
    load_observations,
    run_shadow_scoring,
    save_score_log,
    score_forecast_targets,
    score_provisional,
    score_settled,
)

__all__ = [
    "load_observations",
    "run_shadow_scoring",
    "save_score_log",
    "score_forecast_targets",
    "score_provisional",
    "score_settled",
]
