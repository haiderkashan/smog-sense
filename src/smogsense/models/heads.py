"""smogsense.models.heads — Residual quantile head (46 -> 64 -> 57).

MLP 46 -> 64 (ReLU) -> 57 (linear) predicting local quantile residuals Delta z_{tau, h}
for 19 levels x 3 horizons. Last layer is zero-initialised so an unadapted calibrator
reproduces M5 exactly. Monotone rearrangement applied at inference.

Specification: docs/ml-architecture.md → 'Quantile loss, the residual head, and the quantile function'
"""
