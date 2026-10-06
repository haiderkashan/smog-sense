"""Idempotent daily run.

Test specification (cases to implement):
- Second run for the same issuance is a no-op
- Concurrent runs cannot corrupt the state branch
"""
