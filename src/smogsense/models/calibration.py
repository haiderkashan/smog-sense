"""smogsense.models.calibration — Online recalibration: conformalised quantile regression with adaptive conformal inference.

Adjusts the [q10, q90] interval using a rolling calibration window and the ACI update
alpha_{t+1} = alpha_t + gamma(alpha - err_t), tolerant of delayed label feedback.

Specification: docs/ml-architecture.md → 'Calibration'

Contract update: gamma default 0.02.
"""
