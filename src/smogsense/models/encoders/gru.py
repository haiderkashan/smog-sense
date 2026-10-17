"""smogsense.models.encoders.gru — Two-layer GRU encoder with missingness-aware inputs.

Consumes the six local observation channels (C = 6) per hour and returns latent vector z in R^{32}.

Specification: docs/ml-architecture.md → 'Sequence encoder'
"""
