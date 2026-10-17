"""smogsense.models.maml — Meta-learning of the neural encoder and head (first-order MAML, ANIL, optional second-order).

Task sampler, inner loop on support set, outer loop on query set, using torch.func functional
calls. LightGBM is not differentiable and is therefore adapted separately.

Specification: docs/ml-architecture.md → 'Meta-learning (MAML, Meta-SGD) and adaptation'

Contract update: neural residual calibrator, permanent head (57 outputs, zero-initialised last layer), Meta-SGD with per-tensor step sizes.
"""
