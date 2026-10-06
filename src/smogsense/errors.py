"""smogsense.errors - Exception taxonomy mapped to CLI exit codes."""

class SmogSenseError(Exception):
    """Base exception for all smogsense errors."""
    exit_code: int = 50

class DegradedMode(SmogSenseError):
    """Raised when the system must fall back to a simpler method but still publish."""
    exit_code = 10

class AlreadyPublished(SmogSenseError):
    """Raised when the run is an idempotent no-op because an issuance was already made."""
    exit_code = 11

class SourceUnavailable(SmogSenseError):
    """Raised when an external API or data source is unreachable or empty."""
    exit_code = 20

class SchemaViolation(SmogSenseError):
    """Raised when data fails validation (missing columns, duplicates, bounds)."""
    exit_code = 30

class QuotaExceeded(SmogSenseError):
    """Raised when an API rate limit or quota is exhausted."""
    exit_code = 40

class InternalError(SmogSenseError):
    """Raised when an unexpected or logic error occurs."""
    exit_code = 50
