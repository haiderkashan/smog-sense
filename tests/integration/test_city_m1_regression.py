"""Regression tests for City M1 baseline integration (Phase 1a.7).

Verifies that:
1. City M1 forecast uses city/centroid-specific residual quantiles rather than
   accidentally inheriting the last station's M1 object.
2. Station loop ordering does not affect the city forecast.
3. N=0 deterministic behavior is preserved when centroid residual quantiles are absent.
"""

import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
import respx

from smogsense.pipeline.orchestrator import run_daily_pipeline


@pytest.fixture
def clean_state():
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
    ]
    files = [
        Path(".state/artifacts/seed_history.json"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()
    yield
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_city_m1_uses_centroid_not_last_station(mock_extract, mock_cams_fetch, clean_state):
    """Verify city M1 uses centroid quantiles, not station 1 or station 2 quantiles."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # 1. Provide a seed_history.json with deliberately different quantiles:
    # Station 1: residuals = +10.0 (q05=10.0, q95=10.0)
    # Station 2: residuals = -20.0 (q05=-20.0, q95=-20.0)
    # Centroid: residuals = +5.0  (q05=5.0,  q95=5.0)
    seed_data = {
        "metadata": {"horizons": [24, 48, 72]},
        "quantiles": {
            "1": {
                "24": [10.0] * 19,
                "48": [10.0] * 19,
                "72": [10.0] * 19,
            },
            "2": {
                "24": [-20.0] * 19,
                "48": [-20.0] * 19,
                "72": [-20.0] * 19,
            },
            "centroid": {
                "24": [5.0] * 19,
                "48": [5.0] * 19,
                "72": [5.0] * 19,
            },
        },
        "counts": {
            "1": {"24": 10, "48": 10, "72": 10},
            "2": {"24": 10, "48": 10, "72": 10},
            "centroid": {"24": 10, "48": 10, "72": 10},
        },
    }
    seed_path = Path(".state/artifacts/seed_history.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("w", encoding="utf-8") as f:
        json.dump(seed_data, f)

    # Mock OpenAQ list_locations with 2 stations
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"name": "pm25"}}],
                    },
                    {
                        "id": 2,
                        "name": "Station 2",
                        "coordinates": {"latitude": 31.6, "longitude": 74.4},
                        "sensors": [{"id": 11, "parameter": {"name": "pm25"}}],
                    },
                ],
                "meta": {"found": 2},
            },
        )
    )

    # Mock OpenAQ hours
    respx.get(url__regex=r".*/v3/sensors/\d+/hours.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeTo": {"utc": (issuance - timedelta(hours=i)).isoformat()}
                        },
                        "value": 40.0,
                    }
                    for i in range(1, 25)
                ]
            },
        )
    )

    # Mock CAMS fetch
    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy")
        return dest_path

    mock_cams_fetch.side_effect = fake_fetch

    # CAMS value = 50.0 for Station 1, Station 2, and Centroid
    cams_rows = [
        {
            "location_id": loc,
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 50.0,
        }
        for loc in [1, 2, "centroid"]
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    exit_code = run_daily_pipeline(issuance, force=True)
    assert exit_code == 10

    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    assert log_path.exists()
    df = pd.read_parquet(log_path)

    # Total rows: 3 horizons * (2 stations + 1 city) = 9 rows
    assert len(df) == 9

    # Station 1: CAMS 50 + 10 = 60.0
    st1 = df[df["point_id"] == "station:1"]
    assert (st1["q05"] == 60.0).all()
    assert (st1["q95"] == 60.0).all()

    # Station 2: CAMS 50 - 20 = 30.0
    st2 = df[df["point_id"] == "station:2"]
    assert (st2["q05"] == 30.0).all()
    assert (st2["q95"] == 30.0).all()

    # City: must use centroid CAMS 50 + 5 = 55.0!
    # It must NOT be 30.0 (Station 2, the last station) and NOT 60.0 (Station 1)!
    city = df[df["level"] == "city"]
    assert len(city) == 3
    assert (city["q05"] == 55.0).all(), (
        f"City forecast q05 was {city['q05'].tolist()}, expected 55.0 from centroid!"
    )
    assert (city["q50"] == 55.0).all()
    assert (city["q95"] == 55.0).all()


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_city_m1_independent_of_station_order(mock_extract, mock_cams_fetch, clean_state):
    """Verify city forecast is identical regardless of station iteration order."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    seed_data = {
        "metadata": {"horizons": [24, 48, 72]},
        "quantiles": {
            "1": {"24": [10.0] * 19, "48": [10.0] * 19, "72": [10.0] * 19},
            "2": {"24": [-20.0] * 19, "48": [-20.0] * 19, "72": [-20.0] * 19},
            "centroid": {"24": [7.0] * 19, "48": [7.0] * 19, "72": [7.0] * 19},
        },
        "counts": {
            "1": {"24": 10, "48": 10, "72": 10},
            "2": {"24": 10, "48": 10, "72": 10},
            "centroid": {"24": 10, "48": 10, "72": 10},
        },
    }
    seed_path = Path(".state/artifacts/seed_history.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("w", encoding="utf-8") as f:
        json.dump(seed_data, f)

    # Stations ordered [2, 1] (reversed order)
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 2,
                        "name": "Station 2",
                        "coordinates": {"latitude": 31.6, "longitude": 74.4},
                        "sensors": [{"id": 11, "parameter": {"name": "pm25"}}],
                    },
                    {
                        "id": 1,
                        "name": "Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"name": "pm25"}}],
                    },
                ],
                "meta": {"found": 2},
            },
        )
    )

    respx.get(url__regex=r".*/v3/sensors/\d+/hours.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeTo": {"utc": (issuance - timedelta(hours=i)).isoformat()}
                        },
                        "value": 40.0,
                    }
                    for i in range(1, 25)
                ]
            },
        )
    )

    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy")
        return dest_path

    mock_cams_fetch.side_effect = fake_fetch

    cams_rows = [
        {
            "location_id": loc,
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 50.0,
        }
        for loc in [2, 1, "centroid"]
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    exit_code = run_daily_pipeline(issuance, force=True)
    assert exit_code == 10

    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df = pd.read_parquet(log_path)

    city = df[df["level"] == "city"]
    # Centroid CAMS 50 + 7 = 57.0, regardless of whether Station 1 or Station 2 was processed last
    assert (city["q05"] == 57.0).all()
    assert (city["q95"] == 57.0).all()


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_city_m1_n_zero_deterministic(mock_extract, mock_cams_fetch, clean_state):
    """Verify city M1 falls back to deterministic scalar when centroid quantiles are missing."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # Seed history contains stations, but NO centroid entry
    seed_data = {
        "metadata": {"horizons": [24, 48, 72]},
        "quantiles": {
            "1": {"24": [10.0] * 19, "48": [10.0] * 19, "72": [10.0] * 19},
        },
        "counts": {
            "1": {"24": 10, "48": 10, "72": 10},
        },
    }
    seed_path = Path(".state/artifacts/seed_history.json")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("w", encoding="utf-8") as f:
        json.dump(seed_data, f)

    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )

    respx.get(url__regex=r".*/v3/sensors/\d+/hours.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeTo": {"utc": (issuance - timedelta(hours=i)).isoformat()}
                        },
                        "value": 40.0,
                    }
                    for i in range(1, 25)
                ]
            },
        )
    )

    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy")
        return dest_path

    mock_cams_fetch.side_effect = fake_fetch

    cams_rows = [
        {
            "location_id": loc,
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 52.0,
        }
        for loc in [1, "centroid"]
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    exit_code = run_daily_pipeline(issuance, force=True)
    assert exit_code == 10

    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df = pd.read_parquet(log_path)

    # Station 1 has quantiles -> 52.0 + 10.0 = 62.0
    st1 = df[df["point_id"] == "station:1"]
    assert (st1["q05"] == 62.0).all()

    # City has NO centroid quantiles -> N=0 deterministic scalar = 52.0 for all quantiles
    city = df[df["level"] == "city"]
    assert (city["q05"] == 52.0).all()
    assert (city["q50"] == 52.0).all()
    assert (city["q95"] == 52.0).all()
