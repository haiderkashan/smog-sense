"""smogsense.tests.integration.test_daily_run"""

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
        Path(".state/inputs/lahore"),
        Path("gh-pages"),
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
def test_run_daily_full(mock_extract, mock_cams_fetch, clean_state):
    """Test full Level 0 (mode=full) run with real dependencies except HTTP boundaries."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)
    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    manifest_path = Path(f".state/manifests/{run_id}.json")
    if manifest_path.exists():
        manifest_path.unlink()

    # Mock OpenAQ list_locations
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Test Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )
    # Mock OpenAQ fetch_hourly for sensor 10
    respx.get("https://api.openaq.org/v3/sensors/10/hours").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "period": {
                            "datetimeTo": {"utc": (issuance - timedelta(hours=i)).isoformat()}
                        },
                        "value": 42.0,
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
        dest_path.write_text("dummy_grib_data")
        return dest_path

    mock_cams_fetch.side_effect = fake_fetch

    # Mock extract_stations (it normally uses cfgrib to read the grib and xarray to interpolate)
    # We return a dummy CAMS timeseries DataFrame
    # Note: For M1Cams, extract_stations must return target_hour_utc and pm25_ugm3
    cams_rows = [
        {
            "location_id": loc,
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 55.0,
        }
        for loc in [1, "centroid"]
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    # Execute
    exit_code = run_daily_pipeline(issuance)

    assert exit_code == 10
    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"

    # Verify Manifest
    manifest_path = Path(f".state/manifests/{run_id}.json")
    assert manifest_path.exists()
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["published"] is True
    assert manifest["degradation_level"] == 2

    # Verify Forecast Log
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    assert log_path.exists()
    df = pd.read_parquet(log_path)
    # 3 horizons x (1 station + 1 city) = 6 rows
    assert len(df) == 6
    assert (df["method"] == "m1_cams_raw").all()
    # N=0 deterministic M1 -> step function -> q05=q95=55.0
    assert (df["q05"] == 55.0).all()

    # Verify Bulletin
    bulletin_path = Path(f"gh-pages/{run_id}.json")
    assert bulletin_path.exists()
    with bulletin_path.open("r", encoding="utf-8") as f:
        bulletin = json.load(f)
    assert bulletin["mode"] == "baseline_only"
    assert len(bulletin["horizons"]) == 3

    # Verify Idempotency (Already published -> exit 11)
    exit_code2 = run_daily_pipeline(issuance)
    assert exit_code2 == 11

    # Verify Force rerun -> exit 10
    exit_code3 = run_daily_pipeline(issuance, force=True)
    assert exit_code3 == 10
    rerun_manifests = sorted(Path(".state/manifests").glob(f"{run_id}_rerun_*.json"))
    assert len(rerun_manifests) == 1
    with rerun_manifests[0].open("r", encoding="utf-8") as f:
        manifest3 = json.load(f)
    assert manifest3["is_rerun"] is True
    # Verify original first issuance manifest is preserved and not overwritten
    with manifest_path.open("r", encoding="utf-8") as f:
        orig_manifest = json.load(f)
    assert orig_manifest["is_rerun"] is False
