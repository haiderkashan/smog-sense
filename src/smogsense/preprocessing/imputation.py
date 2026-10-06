"""smogsense.preprocessing.imputation — Gap handling: shape-preserving short-gap fill, masked long gaps.

Gaps of at most 3 consecutive hours are filled with PCHIP in log1p space (cubic splines are
rejected because they overshoot); longer gaps stay missing with an explicit mask. Imputed values
are inputs only and are never used as training or scoring targets.

Public contract (implemented in Phase 1):
- impute_short_gaps(series, max_gap_h=3) -> (series, imputed_mask)

Specification: docs/data-engineering.md → 'Gap handling and imputation'
"""
