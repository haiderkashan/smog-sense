import asyncio
import datetime
import email.utils
import hashlib
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx
import structlog
from tenacity import RetryError

from smogsense.data_ingestion.base import (
    AuditLogger,
    CircuitBreaker,
    CircuitBreakerError,
    RateBudget,
    ResilientClient,
    create_client,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

@pytest.fixture
def rate_budget() -> RateBudget:
    return RateBudget(per_minute=6000, per_hour=120000, safety_margin=1.0)

@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    return CircuitBreaker(consecutive_429_to_open=3, open_seconds=900)

@pytest.fixture
async def resilient_client(rate_budget: RateBudget, circuit_breaker: CircuitBreaker) -> AsyncGenerator[ResilientClient, None]:
    client = create_client()
    res_client = ResilientClient(client, rate_budget, circuit_breaker)
    yield res_client
    await client.aclose()


@pytest.mark.anyio
@respx.mock
async def test_429_with_retry_after(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []

    async def _mock_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "5.5"}),
            httpx.Response(200, json={"ok": True}),
        ]
    )

    resp = await resilient_client.get("https://api.example.com/data")
    assert resp.status_code == 200
    assert route.call_count == 2
    assert route.call_count == 2

    assert len(sleep_calls) >= 1
    assert any(delay >= 5.5 for delay in sleep_calls)


@pytest.mark.anyio
@respx.mock
async def test_retry_after_http_date(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    async def _mock_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        sleep_calls.append(delay)
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    future = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
    http_date = email.utils.format_datetime(future)

    route = respx.get("https://api.example.com/data").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": http_date}),
            httpx.Response(200, json={"ok": True}),
        ]
    )

    resp = await resilient_client.get("https://api.example.com/data")
    assert resp.status_code == 200
    assert route.call_count == 2
    assert route.call_count == 2
    assert any(delay >= 3590 for delay in sleep_calls)


@pytest.mark.anyio
@respx.mock
async def test_circuit_breaker_opens_after_3_consecutive_429s(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(429)
    )

    with pytest.raises(CircuitBreakerError) as exc_info:
        await resilient_client.get("https://api.example.com/data")

    assert "Circuit open for another" in str(exc_info.value)
    assert route.call_count == 3


@pytest.mark.anyio
@respx.mock
async def test_circuit_breaker_recovery(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    cb = resilient_client.circuit_breaker
    cb.failures = 3
    cb.opened_at = time.monotonic() - 901

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200)
    )

    await resilient_client.get("https://api.example.com/data")
    assert cb.opened_at is None
    assert cb.failures == 0
    assert route.call_count == 1


@pytest.mark.anyio
@respx.mock
async def test_reconcile_known_minute_window(resilient_client: ResilientClient) -> None:
    rb = resilient_client.rate_budget
    rb.published_minute = 60
    rb.published_hour = 2000
    rb.tokens_minute = 48.0
    rb.tokens_hour = 1600.0

    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, headers={
            "x-ratelimit-remaining": "2",
            "x-ratelimit-limit": "60"
        })
    )

    await resilient_client.get("https://api.example.com/data")
    assert rb.tokens_minute <= 2.0
    assert rb.tokens_hour >= 1599.0

@pytest.mark.anyio
@respx.mock
async def test_reconcile_ambiguous_window(resilient_client: ResilientClient) -> None:
    rb = resilient_client.rate_budget
    rb.published_minute = 60
    rb.published_hour = 2000
    rb.tokens_minute = 48.0
    rb.tokens_hour = 1600.0

    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, headers={
            "x-ratelimit-remaining": "2"
        })
    )

    await resilient_client.get("https://api.example.com/data")
    assert rb.tokens_minute >= 47.0
    assert rb.tokens_hour >= 1599.0


@pytest.mark.parametrize("status_code,should_retry", [
    (408, True), (429, True), (500, True), (502, True), (503, True), (504, True),
    (400, False), (401, False), (403, False), (404, False)
])
@pytest.mark.anyio
@respx.mock
async def test_retry_status_matrix(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch, status_code: int, should_retry: bool) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(status_code)
    )

    with pytest.raises((RetryError, httpx.HTTPStatusError, CircuitBreakerError)):
        await resilient_client.get("https://api.example.com/data")

    if should_retry:
        # Tenacity makes 6 attempts. But wait! If it is 429, the circuit breaker opens after 3!
        if status_code == 429:
            assert route.call_count == 3
        else:
            assert route.call_count == 6
    else:
        assert route.call_count == 1


@pytest.mark.anyio
@respx.mock
async def test_timeouts_and_connection_errors_are_retried(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        side_effect=httpx.ConnectTimeout("Timeout")
    )

    with pytest.raises(RetryError):
        await resilient_client.get("https://api.example.com/data")

    assert route.call_count == 6


@pytest.mark.anyio
async def test_rate_budget_concurrency() -> None:
    # 6000 tokens per min -> 100 tokens per sec capacity
    rb = RateBudget(per_minute=6000, per_hour=360000, safety_margin=1.0)
    rb.tokens_minute = 0.0
    rb.tokens_hour = 0.0

    start = time.monotonic()

    async def worker() -> None:
        await rb.acquire(1)
        assert rb.tokens_minute >= -0.01
        assert rb.tokens_hour >= -0.01

    # Ask for 150 tokens. Capacity is 100. It should take >= 0.5s to fulfill.
    await asyncio.gather(*(worker() for _ in range(150)))

    elapsed = time.monotonic() - start
    assert elapsed >= 0.45


def test_audit_logger_redaction() -> None:
    audit = AuditLogger()
    req = httpx.Request("GET", "https://firms.modaps.eosdis.nasa.gov/api/area/csv/MY_SECRET_MAP_KEY/VIIRS_SNPP_NRT/world/1/2026-10-07?api_key=SECRET123&q=lahore&token=XYZ")
    resp = httpx.Response(200, request=req, content=b"{}")

    with structlog.testing.capture_logs() as cap_logs:
        audit.log(resp)

    assert len(cap_logs) == 1
    assert "MY_SECRET_MAP_KEY" not in cap_logs[0]["url"]
    assert "***" in cap_logs[0]["url"]
    params = cap_logs[0]["params"]
    assert params["q"] == "lahore"
    assert params["api_key"] == "***"
    assert params["token"] == "***"

    url = cap_logs[0]["url"]
    assert "SECRET123" not in url
    assert "XYZ" not in url


@pytest.mark.anyio
@respx.mock
async def test_audit_logger_failure_isolation(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_log(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Audit logger crashed")

    monkeypatch.setattr(resilient_client.audit, "log", broken_log)

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200)
    )

    resp = await resilient_client.get("https://api.example.com/data")
    assert resp.status_code == 200
    assert route.call_count == 1


def test_sha256_deterministic() -> None:
    audit = AuditLogger()
    body = b"hello world"
    req = httpx.Request("GET", "https://api.example.com/")
    resp = httpx.Response(200, request=req, content=body)

    with structlog.testing.capture_logs() as cap_logs:
        audit.log(resp)

    expected_sha256 = hashlib.sha256(body).hexdigest()
    assert cap_logs[0]["response_sha256"] == expected_sha256

