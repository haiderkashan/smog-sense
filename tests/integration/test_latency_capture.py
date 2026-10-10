"""Integration test for Task P1-18: Latency Capture and obs_pull_log Recording.

Specification:
- docs/data-engineering.md §11 (Temporal alignment & latency capture)
- docs/PRD.md FR-37 (Observation capture and revision policy)
- data/schemas/obs_pull_log.schema.yaml
"""

import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pandas as pd
import pytest
import respx

from smogsense.pipeline.orchestrator import load_pandera_schema, run_daily_pipeline
from smogsense.utils.io import validate_frame


@pytest.fixture
def clean_state():
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
        Path(".state/inputs/lahore"),
        Path(".state/obs_pull_log"),
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
def test_latency_capture_obs_pull_log_created_and_valid(mock_extract, mock_cams_fetch, clean_state):
    """Test that daily run creates schema-valid .state/obs_pull_log/dt=YYYY-MM-DD.parquet."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)
    issuance_date = issuance.strftime("%Y-%m-%d")

    # Mock OpenAQ list_locations with 2 stations
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 101,
                        "name": "Station A",
                        "provider": {"name": "AirGradient"},
                        "coordinates": {"latitude": 31.55, "longitude": 74.35},
                        "sensors": [
                            {"id": 1001, "parameter": {"id": 2, "name": "pm25"}},
                            {"id": 1002, "parameter": {"id": 98, "name": "rh"}},
                            {"id": 1003, "parameter": {"id": 100, "name": "temp"}},
                        ],
                    },
                    {
                        "id": 102,
                        "name": "Station B",
                        "provider": {"name": "Clarity"},
                        "coordinates": {"latitude": 31.50, "longitude": 74.30},
                        "sensors": [
                            {"id": 2001, "parameter": {"id": 2, "name": "pm25"}},
                        ],
                    },
                ],
                "meta": {"found": 2},
            },
        )
    )

    # Mock OpenAQ hourly endpoints for station A
    respx.get("https://api.openaq.org/v3/sensors/1001/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T19:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T20:00:00Z"},
                        },
                        "value": 45.2,
                    },
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T20:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T21:00:00Z"},
                        },
                        "value": 52.8,
                    },
                ]
            },
        )
    )
    respx.get("https://api.openaq.org/v3/sensors/1002/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T19:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T20:00:00Z"},
                        },
                        "value": 65.0,
                    },
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T20:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T21:00:00Z"},
                        },
                        "value": 68.0,
                    },
                ]
            },
        )
    )
    respx.get("https://api.openaq.org/v3/sensors/1003/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T19:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T20:00:00Z"},
                        },
                        "value": 22.5,
                    },
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T20:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T21:00:00Z"},
                        },
                        "value": 21.0,
                    },
                ]
            },
        )
    )

    # Mock OpenAQ hourly endpoints for station B
    respx.get("https://api.openaq.org/v3/sensors/2001/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeFrom": {"utc": "2026-10-18T19:00:00Z"},
                            "datetimeTo": {"utc": "2026-10-18T20:00:00Z"},
                        },
                        "value": 38.0,
                    },
                ]
            },
        )
    )

    # Mock CAMS
    mock_cams_fetch.side_effect = lambda *a, **kw: kw.get("dest_path")
    mock_extract.return_value = pd.DataFrame(
        [
            {
                "location_id": "centroid",
                "target_hour_utc": issuance + timedelta(hours=h),
                "pm25_ugm3": 50.0,
            }
            for h in [24, 48, 72]
        ]
    )

    # Run daily pipeline
    exit_code = run_daily_pipeline(issuance)
    assert exit_code == 10

    # Verify obs_pull_log parquet exists
    pull_log_file = Path(f".state/obs_pull_log/dt={issuance_date}.parquet")
    assert pull_log_file.exists(), f"Expected {pull_log_file} to exist"

    # Read and inspect DataFrame
    df = pd.read_parquet(pull_log_file)
    assert len(df) == 3  # 2 hours for station A + 1 hour for station B

    # Required columns check
    expected_cols = [
        "sensor_id",
        "location_id",
        "domain",
        "ts_utc",
        "hour_end_utc",
        "first_seen_utc",
        "pull_id",
        "value_raw",
        "rh_pct",
        "temp_c",
        "provider",
        "source",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Required non-null checks
    for col in [
        "sensor_id",
        "location_id",
        "domain",
        "ts_utc",
        "hour_end_utc",
        "first_seen_utc",
        "pull_id",
        "provider",
        "source",
    ]:
        assert df[col].notna().all(), f"Found unexpected NaN in column {col}"

    # Verify Pandera validation roundtrip
    schema = load_pandera_schema("obs_pull_log.schema.yaml")
    validated_df = validate_frame(df, schema)
    assert len(validated_df) == len(df)

    # Verify latency calculation
    latency_series = df["first_seen_utc"] - df["hour_end_utc"]
    assert len(latency_series) == 3
    assert pd.api.types.is_timedelta64_dtype(latency_series.dtype)

    # Verify providers
    providers = set(df["provider"].tolist())
    assert "AirGradient" in providers
    assert "Clarity" in providers


def test_obs_pull_log_append_and_deduplicate(clean_state):
    """Test that multiple pulls on the same day append new observations and deduplicate identical rows."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    now = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)
    t1 = now - timedelta(hours=5)
    t2 = now - timedelta(hours=4)

    df1 = pd.DataFrame(
        [
            {
                "sensor_id": "sensor_101",
                "location_id": "loc_1",
                "domain": "lahore",
                "ts_utc": t1,
                "hour_end_utc": t1 + timedelta(hours=1),
                "first_seen_utc": now,
                "pull_id": "pull_run_1",
                "pm25_ugm3": 45.0,
                "rh_pct": 55.0,
                "temperature_c": 22.0,
                "provider": "AirGradient",
            }
        ]
    )

    from smogsense.pipeline.orchestrator import record_obs_pull_log

    dest1 = record_obs_pull_log(df1, issuance, "pull_run_1", domain="lahore")
    assert dest1.exists()
    saved1 = pd.read_parquet(dest1)
    assert len(saved1) == 1

    # Second pull with duplicate of t1 with same pull_id plus new observation t2
    df2 = pd.DataFrame(
        [
            {
                "sensor_id": "sensor_101",
                "location_id": "loc_1",
                "domain": "lahore",
                "ts_utc": t1,
                "hour_end_utc": t1 + timedelta(hours=1),
                "first_seen_utc": now,
                "pull_id": "pull_run_1",
                "pm25_ugm3": 45.0,
                "rh_pct": 55.0,
                "temperature_c": 22.0,
                "provider": "AirGradient",
            },
            {
                "sensor_id": "sensor_101",
                "location_id": "loc_1",
                "domain": "lahore",
                "ts_utc": t2,
                "hour_end_utc": t2 + timedelta(hours=1),
                "first_seen_utc": now + timedelta(minutes=30),
                "pull_id": "pull_run_2",
                "pm25_ugm3": 50.0,
                "rh_pct": 60.0,
                "temperature_c": 21.0,
                "provider": "AirGradient",
            },
        ]
    )

    dest2 = record_obs_pull_log(df2, issuance, "pull_run_2", domain="lahore")
    saved2 = pd.read_parquet(dest2)
    assert len(saved2) == 2  # The duplicate was dropped, the new one was appended
