"""smogsense.evaluation.sparsity_curve — Sensor-sparsity experiment runner.

For N in the configured grid and K Monte-Carlo station draws, adapts on N available stations and
scores on held-out stations in a later block; also supports the support-duration axis.

Specification: docs/evaluation-strategy.md → 'Sensor-sparsity experiment'
"""
