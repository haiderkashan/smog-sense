"""smogsense.models.encoders.gru — Two-layer GRU encoder with missingness-aware inputs.

Consumes [x*m, m, delta] per hour and returns a latent vector z (default 32-d).

Specification: docs/ml-architecture.md → 'Sequence encoder'
"""
