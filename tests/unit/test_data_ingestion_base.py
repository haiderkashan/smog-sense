from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
import respx
from tenacity import RetryError

from smogsense.data_ingestion.base import (
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

    assert len(sleep_calls) >= 1
    assert any(delay >= 5.5 for delay in sleep_calls)


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
async def test_reconcile_rate_budget(resilient_client: ResilientClient) -> None:
    resilient_client.rate_budget.tokens_minute = 10.0
    resilient_client.rate_budget.tokens_hour = 10.0

    respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, headers={
            "x-ratelimit-remaining": "2",
            "x-ratelimit-reset": "30"
        })
    )

    await resilient_client.get("https://api.example.com/data")
    assert resilient_client.rate_budget.tokens_minute <= 2.0


@pytest.mark.anyio
@respx.mock
async def test_403_raises_immediately(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(403)
    )

    with pytest.raises(httpx.HTTPStatusError):
        await resilient_client.get("https://api.example.com/data")

    assert route.call_count == 1


@pytest.mark.anyio
@respx.mock
async def test_502_retries_6_times(resilient_client: ResilientClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _mock_sleep(*args: Any, **kwargs: Any) -> None:
        pass
    monkeypatch.setattr("asyncio.sleep", _mock_sleep)

    route = respx.get("https://api.example.com/data").mock(
        return_value=httpx.Response(502)
    )

    with pytest.raises(RetryError):
        await resilient_client.get("https://api.example.com/data")

    assert route.call_count == 6

