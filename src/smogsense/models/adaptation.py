"""smogsense.models.adaptation — Few-shot adaptation to a target domain.

Runs J inner-loop steps for the neural part and warm-start residual boosting for the tree part
using only the N available stations' labels and strictly earlier data.

Specification: docs/ml-architecture.md → 'Meta-learning (MAML) and adaptation'

Contract update: the recipe, fit/embargo/check, drift guard, statuses, memory hygiene.
"""
