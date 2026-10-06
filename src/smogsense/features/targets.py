"""smogsense.features.targets — 24-hour block-mean targets with completeness rule.

y_h is the mean PM2.5 over the 24 hours [T0+h-24h, T0+h) (hour-start labels) for h in {24,48,72}; a block is valid only with at
least 18 of 24 valid, non-imputed hourly values (75% completeness).

Public contract (implemented in Phase 2):
- block_targets(obs, issuance) -> DataFrame with y_h and valid_h

Specification: docs/ml-architecture.md → 'Problem formulation'
"""
