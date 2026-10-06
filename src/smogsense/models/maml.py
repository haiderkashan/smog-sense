"""smogsense.models.maml — Meta-learning of the neural encoder and head (first-order MAML, ANIL, optional second-order).

Task sampler, inner loop on support set, outer loop on query set, using torch.func functional
calls. LightGBM is not differentiable and is therefore adapted separately.

Specification: docs/ml-architecture.md → 'Meta-learning (MAML) and adaptation'

Contract update: neural only, temporary head, shared recipe.
"""
