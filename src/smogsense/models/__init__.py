"""smogsense.models — Forecast models and adaptation.

Neural encoders, quantile heads, LightGBM quantile ensemble, meta-learning and calibration.

Specification: docs/ml-architecture.md
"""

from smogsense.features.backfill import (
    compute_oof_partition_hash,
    verify_twin_aware_folds,
)
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
from smogsense.models.residual_bound import (
    AdaptationAcceptanceResult,
    AdaptationStatus,
    compute_residual_statistic,
    evaluate_adaptation_acceptance,
    simulate_deployments_on_block_v,
    split_support_window,
)
from smogsense.models.rh_mask_audit import (
    DEFAULT_MIN_SUPPORT_FRACTION,
    DEFAULT_RH_THRESHOLD,
    ExtremeSmogCoincidenceSummary,
    RHDistributionSummary,
    RHMaskAuditReport,
    RHMaskImpactSummary,
    analyze_rh_distribution_and_extremes,
    apply_rh_support_mask,
    evaluate_rh_mask_impact,
    generate_rh_mask_report,
    load_lahore_audit_data,
    run_rh_mask_audit,
)

__all__ = [
    "CANONICAL_ABLATIONS",
    "CANONICAL_HORIZONS",
    "CANONICAL_QUANTILES",
    "DEFAULT_MIN_SUPPORT_FRACTION",
    "DEFAULT_RH_THRESHOLD",
    "LADDER_RUNGS",
    "PARAM_TENSOR_NAMES",
    "RINGS_4",
    "RUNG_NAMES",
    "SECTORS_8",
    "AdaptationAcceptanceResult",
    "AdaptationStatus",
    "ExtremeSmogCoincidenceSummary",
    "LGBMQuantileEnsemble",
    "LocalGRUEncoder",
    "MAMLAblationRunner",
    "MAMLTask",
    "MaskedResidualPinballLoss",
    "MetaSGDStepSizes",
    "ModelLadderEvaluator",
    "RHDistributionSummary",
    "RHMaskAuditReport",
    "RHMaskImpactSummary",
    "ResidualCalibrator",
    "ResidualPretrainingDataset",
    "ResidualQuantileHead",
    "TaskSampler",
    "analyze_rh_distribution_and_extremes",
    "apply_rh_support_mask",
    "apply_sensor_corruption",
    "apply_task_label_policy",
    "compute_atmospheric_lag_correlations",
    "compute_lag_cross_correlation",
    "compute_oof_partition_hash",
    "compute_residual_statistic",
    "compute_sector_importance",
    "corrupt_support_labels",
    "evaluate_ablation_condition",
    "evaluate_adaptation_acceptance",
    "evaluate_calibrator",
    "evaluate_maml",
    "evaluate_predictions",
    "evaluate_rh_mask_impact",
    "fit_cross_fit_oof",
    "generate_diagnostics_report",
    "generate_rh_mask_report",
    "inner_loop_adapt",
    "load_lahore_audit_data",
    "maml_outer_step",
    "pinball_loss",
    "rearrange_quantiles",
    "run_atmospheric_diagnostics",
    "run_maml_ablations",
    "run_model_ladder",
    "run_rh_mask_audit",
    "simulate_deployments_on_block_v",
    "split_support_window",
    "train_m6_calibrator",
    "train_maml",
    "verify_twin_aware_folds",
]
