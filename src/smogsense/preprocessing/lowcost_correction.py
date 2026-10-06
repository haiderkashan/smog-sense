"""smogsense.preprocessing.lowcost_correction — Humidity-aware correction of low-cost optical PM2.5 readings.

Fits and applies a linear RH-dependent correction (form of Barkjohn et al. 2021) against
reference monitors; coefficients are fitted on Lahore data, never hard-coded.

Public contract (implemented in Phase 2):
- fit(pairs) -> CorrectionModel
- apply(df, model) -> df

Specification: docs/data-engineering.md → 'Quality control and low-cost sensor handling'
"""
