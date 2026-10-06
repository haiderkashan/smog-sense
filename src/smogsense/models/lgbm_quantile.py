"""smogsense.models.lgbm_quantile — LightGBM quantile ensemble (one booster per level and horizon).

LightGBM accepts a single float alpha, so K levels x H horizons boosters are trained
independently, then rearranged to enforce monotonicity. Supports init_model warm starts and
init_score residual boosting.

Specification: docs/ml-architecture.md → 'LightGBM quantile stacker'

Contract update: stacker bound to the encoder hash; remove warm-start and blend.
"""
