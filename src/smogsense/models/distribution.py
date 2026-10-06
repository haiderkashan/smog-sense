"""smogsense.models.distribution — Quantile-function object with tail model.

Piecewise-linear interior, exponential upper tail (linear in -ln(1-tau)), log-linear lower tail
floored at 0; provides ppf, cdf, prob_exceed and exact-integral CRPS. Includes monotone
rearrangement of crossing quantiles.

Specification: docs/evaluation-strategy.md → 'CRPS and quantile scoring'
"""
