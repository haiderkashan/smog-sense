"""smogsense.models.heads — Non-crossing neural quantile head.

q_1 = a_1, q_k = q_{k-1} + softplus(a_k) for 19 levels x 3 horizons in log1p space.

Specification: docs/ml-architecture.md → 'Quantile loss, the temporary neural head, and the quantile function'
"""
