"""smogsense.errors — Exception taxonomy mapped to CLI exit codes.

Defines SourceUnavailable, QuotaExceeded, StaleInput, SchemaViolation, DegradedMode and
InternalError. The degradation ladder in inference/degradation.py reacts to these types, never
to message text.

Specification: docs/system-architecture.md → 'Failure handling and degradation ladder'
"""
