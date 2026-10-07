import asyncio
import contextlib
import datetime
import hashlib
import random
import time
from typing import Any

import httpx
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
)
from tenacity.wait import wait_base


class CircuitBreakerError(Exception):
    pass

class RateBudget:
    def __init__(self, per_minute: int, per_hour: int, safety_margin: float = 0.8) -> None:
        self.capacity_minute = int(per_minute * safety_margin)
        self.capacity_hour = int(per_hour * safety_margin)
        self.tokens_minute = float(self.capacity_minute)
        self.tokens_hour = float(self.capacity_hour)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

        self.rate_minute = self.capacity_minute / 60.0
        self.rate_hour = self.capacity_hour / 3600.0

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.last_update = now

                self.tokens_minute = min(float(self.capacity_minute), self.tokens_minute + elapsed * self.rate_minute)
                self.tokens_hour = min(float(self.capacity_hour), self.tokens_hour + elapsed * self.rate_hour)

                if self.tokens_minute >= tokens and self.tokens_hour >= tokens:
                    self.tokens_minute -= tokens
                    self.tokens_hour -= tokens
                    return

                wait_minute = max(0.0, (tokens - self.tokens_minute) / self.rate_minute)
                wait_hour = max(0.0, (tokens - self.tokens_hour) / self.rate_hour)
                wait_time = max(wait_minute, wait_hour)

                await asyncio.sleep(wait_time)

    async def reconcile(self, remaining: int, reset_seconds: float) -> None:
        async with self._lock:
            if reset_seconds <= 60:
                self.tokens_minute = min(self.tokens_minute, float(remaining))
            else:
                self.tokens_hour = min(self.tokens_hour, float(remaining))

class CircuitBreaker:
    def __init__(self, consecutive_429_to_open: int = 3, open_seconds: int = 900) -> None:
        self.limit = consecutive_429_to_open
        self.open_duration = open_seconds
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = asyncio.Lock()
        self.logger = structlog.get_logger("circuit_breaker")

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.limit and self.opened_at is None:
                self.opened_at = time.monotonic()
                self.logger.warning("circuit_breaker_opened", duration=self.open_duration)

    async def record_success(self) -> None:
        async with self._lock:
            if self.failures > 0:
                self.failures = 0
                if self.opened_at is not None:
                    self.opened_at = None
                    self.logger.info("circuit_breaker_closed")

    async def check(self) -> None:
        async with self._lock:
            if self.opened_at is not None:
                elapsed = time.monotonic() - self.opened_at
                if elapsed >= self.open_duration:
                    self.opened_at = None
                    self.failures = 0
                    self.logger.info("circuit_breaker_closed")
                else:
                    raise CircuitBreakerError(f"Circuit open for another {self.open_duration - elapsed:.1f}s")

class AuditLogger:
    def __init__(self) -> None:
        self.logger = structlog.get_logger("audit")

    def log(self, response: httpx.Response) -> None:
        parsed_url = response.request.url
        scrubbed_url = str(parsed_url.copy_with(password=None, username=None))
        params = dict(parsed_url.params)

        body_bytes = response.content
        sha256 = hashlib.sha256(body_bytes).hexdigest()

        self.logger.info(
            "http_request_audit",
            url=scrubbed_url,
            params=params,
            status_code=response.status_code,
            response_sha256=sha256,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat()
        )

class WaitRetryAfter(wait_base):
    def __call__(self, retry_state: RetryCallState) -> float:
        attempt = retry_state.attempt_number - 1
        jitter = random.SystemRandom().uniform(0, 1)
        calc_wait = float(jitter * min(120.0, 2.0 * (2 ** attempt)))

        header_wait = 0.0
        if retry_state.outcome is not None:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError):
                headers = exc.response.headers
                if "retry-after" in headers:
                    with contextlib.suppress(ValueError):
                        header_wait = float(headers["retry-after"])
                elif "x-ratelimit-reset" in headers:
                    with contextlib.suppress(ValueError):
                        header_wait = float(headers["x-ratelimit-reset"])

        return max(calc_wait, header_wait)

def _should_retry(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 429, 500, 502, 503, 504}
    return False

def create_client(connect_timeout: float = 10.0, read_timeout: float = 60.0) -> httpx.AsyncClient:
    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0, pool=10.0)
    return httpx.AsyncClient(timeout=timeout)

class ResilientClient:
    def __init__(self, client: httpx.AsyncClient, rate_budget: RateBudget, circuit_breaker: CircuitBreaker) -> None:
        self.client = client
        self.rate_budget = rate_budget
        self.circuit_breaker = circuit_breaker
        self.audit = AuditLogger()

    @retry(
        retry=retry_if_exception(_should_retry),
        wait=WaitRetryAfter(),
        stop=stop_after_attempt(6)
    )
    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        await self.circuit_breaker.check()
        await self.rate_budget.acquire()

        try:
            resp = await self.client.get(url, **kwargs)
            await resp.aread()
            resp.raise_for_status()

            remaining = resp.headers.get("x-ratelimit-remaining")
            reset = resp.headers.get("x-ratelimit-reset")
            if remaining is not None and reset is not None:
                with contextlib.suppress(ValueError):
                    await self.rate_budget.reconcile(int(remaining), float(reset))

            await self.circuit_breaker.record_success()
            self.audit.log(resp)
            return resp

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await self.circuit_breaker.record_failure()
            with contextlib.suppress(Exception):
                await e.response.aread()
                self.audit.log(e.response)
            raise

