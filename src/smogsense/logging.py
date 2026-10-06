"""smogsense.logging — Structured JSON logging with secret redaction.

Configures stdlib logging to emit one JSON object per line carrying run_id, stage, issuance and
domain. A redaction filter masks every configured secret value and any string matching known key
patterns.

Public contract (implemented in Phase 0):
- get_logger(stage) -> Logger
- Redaction is applied to message, args and exception text.

Specification: docs/deployment-and-ops.md → 'Alerting'
"""
