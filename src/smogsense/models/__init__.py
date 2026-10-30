"""smogsense.models — Forecast models and adaptation.

Neural encoders, quantile heads, LightGBM quantile ensemble, meta-learning and calibration.

Specification: docs/ml-architecture.md
"""

from smogsense.models.encoders.gru import LocalGRUEncoder
from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
    ResidualQuantileHead,
    rearrange_quantiles,
)
from smogsense.models.lgbm_quantile import (
    LGBMQuantileEnsemble,
    fit_cross_fit_oof,
)
from smogsense.models.losses import MaskedResidualPinballLoss, pinball_loss
from smogsense.models.pretraining import (
    ResidualPretrainingDataset,
    apply_sensor_corruption,
    evaluate_calibrator,
    train_m6_calibrator,
)

__all__ = [
    "CANONICAL_HORIZONS",
    "CANONICAL_QUANTILES",
    "LGBMQuantileEnsemble",
    "LocalGRUEncoder",
    "MaskedResidualPinballLoss",
    "ResidualCalibrator",
    "ResidualPretrainingDataset",
    "ResidualQuantileHead",
    "apply_sensor_corruption",
    "evaluate_calibrator",
    "fit_cross_fit_oof",
    "pinball_loss",
    "rearrange_quantiles",
    "train_m6_calibrator",
]
