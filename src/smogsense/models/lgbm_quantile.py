"""smogsense.models.lgbm_quantile — LightGBM quantile ensemble (one booster per level and horizon).

LightGBM accepts a single float alpha, so K levels x H horizons boosters are trained
independently, then rearranged to enforce monotonicity. Supports init_model warm starts and
init_score residual boosting.

Specification: docs/ml-architecture.md → 'LightGBM quantile ensemble (the frozen physics base)'

Contract update: frozen physics base M5 (57 boosters), season-blocked cross-fit OOF base for calibrator training.
"""
