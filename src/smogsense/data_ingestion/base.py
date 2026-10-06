"""smogsense.data_ingestion.base — Resilient HTTP foundation shared by all REST clients.

httpx client factory with explicit timeouts, tenacity retry (exponential backoff with full
jitter, honours Retry-After), a token-bucket rate limiter driven by configured per-minute and
per-hour budgets, a circuit breaker, and an append-only request audit log.

Public contract (implemented in Phase 1):
- RateBudget(per_minute, per_hour, safety_margin) with header-aware reconciliation
  (x-ratelimit-*).
- Retries only idempotent GETs; 4xx other than 408/429 are never retried.
- Every request is logged without credentials.

Specification: docs/data-engineering.md → 'OpenAQ v3 client'
"""
