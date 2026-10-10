"""smogsense.logging - JSON logging with secret redaction.

Specification: docs/deployment-and-ops.md -> 'Observability'
"""

import logging
import os
from typing import Any, cast

import structlog


def _get_secrets() -> set[str]:
    secrets = set()
    for key in ["OPENAQ_API_KEY", "ADS_API_KEY", "CDS_API_KEY", "FIRMS_MAP_KEY"]:
        if val := os.getenv(key):
            secrets.add(val)
    return secrets


def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Processor to redact secrets from the event dict."""
    secrets = _get_secrets()
    if not secrets:
        return event_dict

    def _redact(obj: Any) -> Any:
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

    return cast(dict[str, Any], _redact(event_dict))


def setup_logging(json_format: bool = True) -> None:
    """Initialize structured logging."""
    processors = cast(
        Any,
        [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_secrets,
        ],
    )
    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
    )

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
