"""smogsense.models.adaptation — Few-shot adaptation to a target domain.

Runs J inner-loop steps on the neural residual calibrator (GRU + head) using only the
N available stations' labels and strictly earlier data. LightGBM physics base M5 is frozen.
Includes fit/embargo/check split, residual-bound check (R <= rho_99), and exact fallback to M5.

Specification: docs/ml-architecture.md → 'Meta-learning (MAML, Meta-SGD) and adaptation'

Contract update: residual calibrator, residual bound, statuses, memory hygiene.
"""
