"""smogsense.models — Forecast models and adaptation.

Neural encoders, quantile heads, LightGBM quantile ensemble, meta-learning and calibration.

Specification: docs/ml-architecture.md
"""

from smogsense.models.ablations import (
    CANONICAL_ABLATIONS,
    MAMLAblationRunner,
    apply_task_label_policy,
    corrupt_support_labels,
    evaluate_ablation_condition,
    run_maml_ablations,
)
from smogsense.models.diagnostics import (
    RINGS_4,
    SECTORS_8,
    compute_atmospheric_lag_correlations,
    compute_lag_cross_correlation,
    compute_sector_importance,
    generate_diagnostics_report,
    run_atmospheric_diagnostics,
)
from smogsense.models.encoders.gru import LocalGRUEncoder
from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
    ResidualQuantileHead,
    rearrange_quantiles,
)
from smogsense.models.ladder import (
    LADDER_RUNGS,
    RUNG_NAMES,
    ModelLadderEvaluator,
    evaluate_predictions,
    run_model_ladder,
)
from smogsense.models.lgbm_quantile import (
    LGBMQuantileEnsemble,
    fit_cross_fit_oof,
)
from smogsense.models.losses import MaskedResidualPinballLoss, pinball_loss
from smogsense.models.maml import (
    PARAM_TENSOR_NAMES,
    MAMLTask,
    MetaSGDStepSizes,
    TaskSampler,
    evaluate_maml,
    inner_loop_adapt,
    maml_outer_step,
    train_maml,
)
from smogsense.models.pretraining import (
    ResidualPretrainingDataset,
    apply_sensor_corruption,
    evaluate_calibrator,
    train_m6_calibrator,
)

__all__ = [
    "CANONICAL_ABLATIONS",
    "CANONICAL_HORIZONS",
    "CANONICAL_QUANTILES",
    "LADDER_RUNGS",
    "PARAM_TENSOR_NAMES",
    "RINGS_4",
    "RUNG_NAMES",
    "SECTORS_8",
    "LGBMQuantileEnsemble",
    "LocalGRUEncoder",
    "MAMLAblationRunner",
    "MAMLTask",
    "MaskedResidualPinballLoss",
    "MetaSGDStepSizes",
    "ModelLadderEvaluator",
    "ResidualCalibrator",
    "ResidualPretrainingDataset",
    "ResidualQuantileHead",
    "TaskSampler",
    "apply_sensor_corruption",
    "apply_task_label_policy",
    "compute_atmospheric_lag_correlations",
    "compute_lag_cross_correlation",
    "compute_sector_importance",
    "corrupt_support_labels",
    "evaluate_ablation_condition",
    "evaluate_calibrator",
    "evaluate_maml",
    "evaluate_predictions",
    "fit_cross_fit_oof",
    "generate_diagnostics_report",
    "inner_loop_adapt",
    "maml_outer_step",
    "pinball_loss",
    "rearrange_quantiles",
    "run_atmospheric_diagnostics",
    "run_maml_ablations",
    "run_model_ladder",
    "train_m6_calibrator",
    "train_maml",
]
