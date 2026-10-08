from datetime import UTC, datetime
from typing import ClassVar

import httpx
import pandas as pd
import pytest
import respx

from smogsense.data_ingestion.base import CircuitBreaker, RateBudget, ResilientClient, create_client
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations


@pytest.fixture
def mock_settings(monkeypatch):
    class MockSettings:
        domains: ClassVar[dict] = {
            "domains": {
                "lahore": {
                    "station_bbox": {"west": 74.0, "south": 31.0, "east": 75.0, "north": 32.0}
                }
            }
        }

        @classmethod
        def load(cls, path):
            return cls()

    monkeypatch.setattr("smogsense.data_ingestion.openaq.Settings", MockSettings)


@pytest.fixture
async def client():
    budget = RateBudget(per_minute=60, per_hour=2000, safety_margin=1.0)
    cb = CircuitBreaker()
    http_client = create_client()
    res_client = ResilientClient(http_client, budget, cb)
    yield res_client
    await http_client.aclose()


@respx.mock
@pytest.mark.anyio
async def test_list_locations_semantics(client, mock_settings):
    # Test isMonitor missing, invalid coords, multiple sensors logic
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 3, "limit": 1000, "page": 1},
                "results": [
                    {
                        "id": 1,
                        "name": "Loc1",
                        "provider": {"name": "P1"},
                        "coordinates": {"latitude": 31.5, "longitude": 74.5},
                        # isMonitor missing
                        "sensors": [
                            {
                                "id": 10,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 80.0},
                            },
                            {
                                "id": 11,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 90.0},
                            },  # Highest cov wins
                        ],
                    },
                    {
                        "id": 2,
                        "name": "Loc2",
                        "provider": {"name": "P2"},
                        "coordinates": {"latitude": 200.0, "longitude": 74.5},  # Invalid lat
                        "sensors": [
                            {
                                "id": 20,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 90.0},
                            }
                        ],
                    },
                    {
                        "id": 3,
                        "name": "Loc3",
                        "provider": {"name": "P3"},
                        "coordinates": {"latitude": 31.6, "longitude": 74.6},
                        "isMonitor": True,
                        "sensors": [
                            {
                                "id": 30,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 95.0},
                            },
                            {
                                "id": 31,
                                "parameter": {"id": 98, "name": "rh"},
                                "coverage": {"percentComplete": 90.0},
                            },
                            {
                                "id": 32,
                                "parameter": {"id": 100, "name": "temp"},
                                "coverage": {"percentComplete": 90.0},
                            },
                        ],
                    },
                ],
            },
        )
    )

    df = await list_locations(client, "lahore")
    assert len(df) == 2  # Loc2 filtered out due to invalid lat

    # Loc1 checks
    loc1 = df[df["location_id"] == 1].iloc[0]
    assert pd.isna(loc1["is_monitor"])  # Missing preserved
    assert loc1["sensor_id_pm25"] == 11  # Higher coverage picked
    assert loc1["lifecycle_uptime"] == 0.90
    assert pd.isna(loc1["sensor_id_rh"])

    # Loc3 checks
    loc3 = df[df["location_id"] == 3].iloc[0]
    assert loc3["is_monitor"] is True
    assert loc3["sensor_id_pm25"] == 30
    assert loc3["sensor_id_rh"] == 31
    assert loc3["sensor_id_temp"] == 32


@respx.mock
@pytest.mark.anyio
async def test_pagination_and_duplicates(client, mock_settings):
    # Test strict pagination boundary and duplicate location resolution
    # Found=2, Limit=1. Will require exactly 2 pages.
    # Duplicating loc1 on page 2 to ensure uniqueness logic.
    respx.get("https://api.openaq.org/v3/locations", params__contains={"page": 1}).mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 1500, "limit": 1000, "page": 1},
                "results": [{"id": 1, "coordinates": {"latitude": 31.0, "longitude": 74.0}}],
            },
        )
    )
    respx.get("https://api.openaq.org/v3/locations", params__contains={"page": 2}).mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 1500, "limit": 1000, "page": 2},
                "results": [
                    {"id": 1, "coordinates": {"latitude": 31.0, "longitude": 74.0}},  # Dup
                    {"id": 2, "coordinates": {"latitude": 31.1, "longitude": 74.1}},
                ],
            },
        )
    )

    df = await list_locations(client, "lahore")
    assert len(df) == 2
    assert set(df["location_id"].tolist()) == {1, 2}


@respx.mock
@pytest.mark.anyio
async def test_fetch_hourly_pivot(client):
    locations_df = pd.DataFrame(
        [{"location_id": 1, "sensor_id_pm25": 10, "sensor_id_rh": 11, "sensor_id_temp": 12}]
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 1, tzinfo=UTC)

    respx.get("https://api.openaq.org/v3/sensors/10/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 1},
                "results": [{"period": {"datetimeFrom": {"utc": "2026-01-01T00:00:00Z"}}, "value": 50.0}],
            },
        )
    )
    respx.get("https://api.openaq.org/v3/sensors/11/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 1},
                "results": [{"period": {"datetimeFrom": {"utc": "2026-01-01T00:00:00Z"}}, "value": 60.0}],
            },
        )
    )
    respx.get("https://api.openaq.org/v3/sensors/12/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"found": 1},
                "results": [{"period": {"datetimeFrom": {"utc": "2026-01-01T00:00:00Z"}}, "value": 25.0}],
            },
        )
    )

    df = await fetch_hourly(client, locations_df, start, end)
    assert len(df) == 1
    assert df.iloc[0]["location_id"] == 1
    assert df.iloc[0]["pm25_ugm3"] == 50.0
    assert df.iloc[0]["rh_pct"] == 60.0
    assert df.iloc[0]["temperature_c"] == 25.0
    assert df["ts_utc"].dt.tz == UTC


@respx.mock
@pytest.mark.anyio
async def test_fetch_hourly_429_integration(client):
    # Ensures concurrent fetches use the ResilientClient correctly
    locations_df = pd.DataFrame(
        [{"location_id": 1, "sensor_id_pm25": 10, "sensor_id_rh": None, "sensor_id_temp": None}]
    )
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, 1, tzinfo=UTC)

    route = respx.get("https://api.openaq.org/v3/sensors/10/hours")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"meta": {"found": 0}, "results": []}),
    ]

    df = await fetch_hourly(client, locations_df, start, end)
    assert route.call_count == 2
    assert len(df) == 0
