"""smogsense.models.registry — Model artefact registry (GitHub Releases).

Resolves, downloads and verifies (sha256) model bundles; refuses a bundle whose
feature_set_version or config hash is incompatible with the running code.

Specification: docs/deployment-and-ops.md → 'Release and model promotion'

Contract update: encoder <-> stacker binding.
"""
