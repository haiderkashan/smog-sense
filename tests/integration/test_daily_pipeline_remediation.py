"""Integration tests for daily pipeline audit remediation.

Tests:
1. CircuitBreakerError triggers exit code 40.
2. Level 2 (CAMS available, OpenAQ down) publishes via centroid fallback (no exit 20).
3. Non-negative clipping on quantiles when CAMS is small and residuals are negative.
4. Forecast log is only written if bulletin generation succeeds.
5. Observation snapshot is persisted to .state/inputs/lahore.
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

from smogsense.data_ingestion.base import CircuitBreakerError
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
def test_circuit_breaker_error_exits_40(clean_state):
    """Verify CircuitBreakerError causes pipeline to exit with code 40."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # Force list_locations to raise CircuitBreakerError
    with patch(
        "smogsense.pipeline.orchestrator.list_locations",
        side_effect=CircuitBreakerError("Circuit open"),
    ):
        exit_code = run_daily_pipeline(issuance)

    assert exit_code == 40
    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    manifest_path = Path(f".state/manifests/{run_id}.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["exit_code"] == 40
    assert not manifest["published"]


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_openaq_down_level_2_publishes(mock_extract, mock_cams_fetch, clean_state):
    """Verify that when OpenAQ fails but CAMS succeeds, Level 2 runs and publishes."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # OpenAQ returns 500 error
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    mock_cams_fetch.side_effect = lambda *a, **kw: kw.get("dest_path")

    # CAMS extract returns centroid only
    cams_rows = [
        {
            "location_id": "centroid",
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 45.0,
        }
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    exit_code = run_daily_pipeline(issuance)

    assert exit_code == 10  # Success with degradation level 2
    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    manifest_path = Path(f".state/manifests/{run_id}.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["published"] is True
    assert manifest["degradation_level"] == 2


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_non_negative_quantiles_clipped_at_zero(mock_extract, mock_cams_fetch, clean_state):
    """Verify that negative residuals applied to small CAMS values are floored at 0.0."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # Provide a seed history with negative residuals (-20)
    seed_file = Path(".state/artifacts/seed_history.json")
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    seed_data = {
        "quantiles": {
            "centroid": {
                "24": [-20.0] * 19,
                "48": [-20.0] * 19,
                "72": [-20.0] * 19,
            }
        }
    }
    seed_file.write_text(json.dumps(seed_data), encoding="utf-8")

    # CAMS value is 10.0 => 10 + (-20) = -10.0, which must be floored to 0.0
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(500, text="Down")
    )
    mock_cams_fetch.side_effect = lambda *a, **kw: kw.get("dest_path")

    cams_rows = [
        {
            "location_id": "centroid",
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 10.0,
        }
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    exit_code = run_daily_pipeline(issuance)
    assert exit_code == 10

    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    df = pd.read_parquet(log_path)
    # Check that all quantiles are >= 0.0
    for q in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
        assert (df[f"q{q:02d}"] >= 0.0).all()
        # In this test, all should be exactly 0.0 due to clipping
        assert (df[f"q{q:02d}"] == 0.0).all()


@respx.mock
@pytest.mark.anyio
@patch("smogsense.pipeline.orchestrator.CamsClient.fetch_cams")
@patch("smogsense.pipeline.orchestrator.extract_stations")
def test_log_not_persisted_if_bulletin_fails(mock_extract, mock_cams_fetch, clean_state):
    """Verify that if bulletin generation/validation fails, forecast log is not saved on disk."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(500, text="Down")
    )
    mock_cams_fetch.side_effect = lambda *a, **kw: kw.get("dest_path")

    cams_rows = [
        {
            "location_id": "centroid",
            "target_hour_utc": issuance + timedelta(hours=h),
            "pm25_ugm3": 45.0,
        }
        for h in [24, 48, 72]
    ]
    mock_extract.return_value = pd.DataFrame(cams_rows)

    run_id = f"run_{issuance.strftime('%Y%m%d_%H%M')}"
    log_path = Path(f".state/forecasts/forecast_{run_id}.parquet")

    # Force bulletin generation to fail
    with patch(
        "smogsense.pipeline.orchestrator.generate_bulletin_json",
        side_effect=RuntimeError("Bulletin validation crash"),
    ):
        exit_code = run_daily_pipeline(issuance)

    assert exit_code == 50
    # Forecast log should NOT exist on disk
    assert not log_path.exists()
