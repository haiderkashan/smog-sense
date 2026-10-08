import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from smogsense.data_ingestion.base import CircuitBreaker, RateBudget, ResilientClient, create_client
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations
from smogsense.data_ingestion.station_registry import StationRegistry


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def resilient_client() -> AsyncGenerator[ResilientClient, None]:
    client = create_client()
    rb = RateBudget(per_minute=60, per_hour=2000)
    cb = CircuitBreaker()
    res_client = ResilientClient(client, rb, cb)
    yield res_client
    await client.aclose()


@pytest.mark.anyio
@respx.mock
async def test_openaq_list_locations_contract(resilient_client: ResilientClient) -> None:
    # Load fixture
    fixture_path = Path("tests/fixtures/openaq/locations_page.json")
    with fixture_path.open() as f:
        data = json.load(f)

    route = respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(200, json=data)
    )

    df = await list_locations(resilient_client, "lahore")

    assert route.called
    assert len(df) == 1

    row = df.iloc[0]
    assert row["location_id"] == 1001
    assert row["name"] == "US Embassy Lahore"
    assert bool(row["is_monitor"]) is True
    assert row["provider"] == "State Department"
    assert row["lat"] == 31.55
    assert row["lon"] == 74.33
    assert row["sensor_id_pm25"] == 5001
    assert row["sensor_id_rh"] == 5002
    assert row["sensor_id_temp"] == 5003
    assert pd.to_datetime(row["first_datetime"]).isoformat() == "2020-01-01T00:00:00+00:00"
    assert row["lifecycle_uptime"] == 0.95


@pytest.mark.anyio
@respx.mock
async def test_openaq_fetch_hourly_contract(resilient_client: ResilientClient) -> None:
    fixture_path = Path("tests/fixtures/openaq/hours_page.json")
    with fixture_path.open() as f:
        data = json.load(f)

    route = respx.get("https://api.openaq.org/v3/sensors/5001/hours").mock(
        return_value=httpx.Response(200, json=data)
    )

    start_utc = datetime(2026, 10, 7, 10, 0, tzinfo=UTC)
    end_utc = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)

    locations_df = pd.DataFrame(
        [
            {
                "location_id": 1001,
                "sensor_id_pm25": 5001,
                "sensor_id_temp": 5003,
                "sensor_id_rh": 5002,
            }
        ]
    )
    df = await fetch_hourly(resilient_client, locations_df, start_utc, end_utc)

    assert route.called
    assert len(df) == 2

    assert "ts_utc" in df.columns
    assert "pm25_ugm3" in df.columns

    row0 = df.iloc[0]
    assert row0["location_id"] == 1001
    assert row0["pm25_ugm3"] == 138.1
    assert pd.to_datetime(row0["ts_utc"]).isoformat() == "2026-10-07T11:00:00+00:00"


@pytest.mark.anyio
async def test_station_registry(tmp_path: Path) -> None:
    registry = StationRegistry(tmp_path)

    # 3 stations: A and B are close but same provider -> not colocated.
    # B and C are close and different providers -> colocated.
    df_in = pd.DataFrame(
        [
            {
                "location_id": 1,
                "provider": "Prov1",
                "lat": 31.55000,
                "lon": 74.33000,
                "first_datetime": pd.to_datetime("2026-01-01T00:00:00Z"),
                "last_datetime": pd.to_datetime("2026-10-07T00:00:00Z"),
                "pm25_uptime_90d": 0.8,
            },
            {
                "location_id": 2,
                "provider": "Prov1",
                "lat": 31.55010,  # ~11m away
                "lon": 74.33000,
                "first_datetime": pd.to_datetime("2026-01-01T00:00:00Z"),
                "last_datetime": pd.to_datetime("2026-10-07T00:00:00Z"),
                "pm25_uptime_90d": 0.4,  # Below uptime threshold
            },
            {
                "location_id": 3,
                "provider": "Prov2",
                "lat": 31.55015,  # ~16m away from Prov1(2), ~22m away from Prov1(1)
                "lon": 74.33000,
                "first_datetime": pd.to_datetime("2026-09-15T00:00:00Z"),  # < 30 days history!
                "last_datetime": pd.to_datetime("2026-10-07T00:00:00Z"),
                "pm25_uptime_90d": 0.9,
            },
        ]
    )

    df_out = registry.build_and_save(df_in, "test_domain", min_uptime=0.5)

    assert "eligible" in df_out.columns
    assert "colocated_group_id" in df_out.columns

    assert bool(df_out.iloc[0]["eligible"]) is True
    assert bool(df_out.iloc[1]["eligible"]) is False  # uptime < 0.5
    assert bool(df_out.iloc[2]["eligible"]) is False  # history < 30 days

    g1 = df_out.iloc[0]["colocated_group_id"]
    g2 = df_out.iloc[1]["colocated_group_id"]
    g3 = df_out.iloc[2]["colocated_group_id"]

    # Provider 1 stations cannot be co-located with each other
    # But Provider 1 stations can be co-located with Provider 2
    # So 1, 2, 3 should all get the same group id because 1 is close to 3, and 2 is close to 3.
    # My simple logic assigned them if dist <= 50 and diff provider.
    assert g1 == g3
    assert g2 == g3
    assert g1 == g2
