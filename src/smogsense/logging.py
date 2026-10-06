"""smogsense.logging - JSON logging with secret redaction."""

import os
import structlog
import logging

def _get_secrets() -> set[str]:
    secrets = set()
    for key in ["OPENAQ_API_KEY", "ADS_API_KEY", "CDS_API_KEY", "FIRMS_MAP_KEY"]:
        if val := os.getenv(key):
            secrets.add(val)
    return secrets

def redact_secrets(logger, log_method, event_dict):
    """Processor to redact secrets from the event dict."""
    secrets = _get_secrets()
    if not secrets:
        return event_dict

    def _redact(obj):
        if isinstance(obj, str):
            for secret in secrets:
                if secret in obj:
                    obj = obj.replace(secret, "***REDACTED***")
            return obj
        elif isinstance(obj, dict):
            return {k: _redact(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_redact(i) for i in obj]
        return obj

    return _redact(event_dict)

def setup_logging(json_format: bool = True):
    """Initialize structured logging."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_secrets,
    ]
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
