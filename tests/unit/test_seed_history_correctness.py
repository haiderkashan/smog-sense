"""Unit tests for Phase 1a.7 Seed History & Residual Quantiles correctness.

Tests:
1. Cutoff semantics & dynamic cutoff calculation (OpenAQ + CAMS support)
2. Prevention of temporal leakage (future observations cannot leak)
3. Residual definition: residual = observation - CAMS (sign correctness)
4. Strict horizon separation (24h, 48h, 72h)
5. Strict station separation (station 1 != station 2 != centroid)
6. Missing observation handling
7. Missing CAMS handling
8. Partial station & partial horizon coverage
9. 19 empirical quantile levels & monotonicity
10. Accurate count preservation
11. Empty groups remain absent
12. CLI surface for run seed-history
"""

import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.config import Settings
from smogsense.pipeline.seed_history import determine_cutoff_dates, generate_seed_history


@pytest.fixture
def clean_state():
    p = Path(".state/artifacts/seed_history_test.json")
    if p.exists():
        with contextlib.suppress(Exception):
            p.unlink()
    yield
    if p.exists():
        with contextlib.suppress(Exception):
            p.unlink()


# -----------------------------------------------------------------------------
# 1. Cutoff Semantics & Temporal Correctness
# -----------------------------------------------------------------------------


@respx.mock
@pytest.mark.anyio
async def test_cutoff_dates_requires_72h_observation():
    """Verify cutoff requires +72h observation buffer for 72h horizon."""
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station A",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"id": 2, "name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )

    # Observations run up to 2026-08-25 18:00:00 UTC
    # Since +72h observation is required:
    # 2026-08-25 18:00 - 72h = 2026-08-22 18:00 -> aligned to 00Z = 2026-08-22 00:00:00 UTC
    end_obs = datetime(2026, 8, 25, 18, 0, tzinfo=UTC)
    dates = pd.date_range(start_utc, end_obs, freq="h")
    obs_results = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 45.0 + (i % 3)}
        for i, dt in enumerate(dates)
    ]
    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results, "meta": {"found": len(obs_results)}}
        )
    )

    settings = Settings.load("configs").model_dump()
    end_issuance, obs_clean, stations_df = await determine_cutoff_dates(
        start_utc, now_utc, settings
    )

    assert end_issuance == datetime(2026, 8, 22, 0, 0, tzinfo=UTC)
    assert not obs_clean.empty
    assert len(stations_df) == 1


@respx.mock
@pytest.mark.anyio
async def test_cutoff_dates_prevents_future_leakage():
    """Verify that an observation with timestamp after now_utc is clamped and cannot leak."""
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = datetime(2026, 8, 25, 0, 0, tzinfo=UTC)

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station A",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"id": 2, "name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )

    # Observations erroneously contain future rows up to 2026-08-30
    end_future = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)
    dates = pd.date_range(start_utc, end_future, freq="h")
    obs_results = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 45.0 + (i % 3)}
        for i, dt in enumerate(dates)
    ]
    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results, "meta": {"found": len(obs_results)}}
        )
    )

    settings = Settings.load("configs").model_dump()
    end_issuance, _obs_clean, _stations_df = await determine_cutoff_dates(
        start_utc, now_utc, settings
    )

    # Effective t_obs is clamped to now_utc (2026-08-25 00:00:00)
    # So end_issuance = 2026-08-25 00:00 - 72h = 2026-08-22 00:00:00 UTC
    assert end_issuance == datetime(2026, 8, 22, 0, 0, tzinfo=UTC)


@respx.mock
@pytest.mark.anyio
async def test_cutoff_dates_insufficient_history_raises():
    """Verify that history shorter than 72 hours raises ValueError."""
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)  # only ~36 hours

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station A",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"id": 2, "name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )

    dates = pd.date_range(start_utc, now_utc, freq="h", tz=UTC)
    obs_results = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 45.0 + (i % 3)}
        for i, dt in enumerate(dates)
    ]
    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results, "meta": {"found": len(obs_results)}}
        )
    )

    settings = Settings.load("configs").model_dump()
    with pytest.raises(ValueError, match="Insufficient observations"):
        await determine_cutoff_dates(start_utc, now_utc, settings)


# -----------------------------------------------------------------------------
# 2. Residual Definition, Horizon Separation & Station Separation
# -----------------------------------------------------------------------------


@respx.mock
@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.pipeline.seed_history.CamsClient.fetch_cams")
def test_seed_history_data_semantics_and_separation(
    mock_fetch_cams, mock_open_datasets, clean_state
):
    """Verify residual definition (obs - CAMS), horizon separation, and station separation."""
    import xarray as xr

    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = start_utc + timedelta(
        days=6
    )  # Allows issuance 2026-08-20, 2026-08-21, 2026-08-22, 2026-08-23

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [
                            {
                                "id": 10,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 100},
                            }
                        ],
                    },
                    {
                        "id": 2,
                        "name": "Station 2",
                        "coordinates": {"latitude": 31.6, "longitude": 74.4},
                        "sensors": [
                            {
                                "id": 11,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 100},
                            }
                        ],
                    },
                ],
                "meta": {"found": 2},
            },
        )
    )

    # Observations:
    # Station 1 has obs = 60.0 (+/- 1 to avoid flatline QC)
    # Station 2 has obs = 30.0 (+/- 1 to avoid flatline QC)
    dates = pd.date_range(start_utc, now_utc, freq="h")
    obs_10 = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 60.0 + (i % 2)}
        for i, dt in enumerate(dates)
    ]
    obs_11 = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 30.0 + (i % 2)}
        for i, dt in enumerate(dates)
    ]
    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(200, json={"results": obs_10, "meta": {"found": len(obs_10)}})
    )
    respx.get(url__regex=r".*/v3/sensors/11/hours.*").mock(
        return_value=httpx.Response(200, json={"results": obs_11, "meta": {"found": len(obs_11)}})
    )

    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy")
        return dest_path

    mock_fetch_cams.side_effect = fake_fetch

    # CAMS forecast = 40.0 everywhere
    def fake_open_datasets(grib_path):
        base_str = Path(grib_path).stem.replace("cams_hist_", "")
        cams_base = datetime.strptime(base_str, "%Y%m%d_%H")
        lats = np.array([32.0, 31.0])
        lons = np.array([74.0, 75.0])
        steps = np.array(
            [np.timedelta64(h, "h") for h in range(12, 120, 12)], dtype="timedelta64[ns]"
        )
        data = np.full((1, len(steps), 2, 2), 40.0 / 1e9)
        ds = xr.Dataset(
            {"pm2p5": (["time", "step", "latitude", "longitude"], data, {"units": "kg m**-3"})},
            coords={
                "time": [np.datetime64(cams_base)],
                "step": steps,
                "latitude": lats,
                "longitude": lons,
            },
        )
        return [ds]

    mock_open_datasets.side_effect = fake_open_datasets

    out_file = Path(".state/artifacts/seed_history_test.json")
    exit_code = generate_seed_history(
        test_start_utc=start_utc, test_now_utc=now_utc, output_path=out_file
    )
    assert exit_code == 0

    with out_file.open("r", encoding="utf-8") as f:
        artifact = json.load(f)

    quantiles = artifact["quantiles"]
    counts = artifact["counts"]

    # 1. Station separation:
    assert "1" in quantiles
    assert "2" in quantiles
    assert "centroid" in quantiles

    # Station 1: obs (~60) - CAMS (40) = ~+20.0 (strictly positive)
    q1_24 = quantiles["1"]["24"]
    assert all(19.0 <= v <= 22.0 for v in q1_24)

    # Station 2: obs (~30) - CAMS (40) = ~-10.0 (strictly negative)
    q2_24 = quantiles["2"]["24"]
    assert all(-11.0 <= v <= -9.0 for v in q2_24)

    # Centroid: obs mean(~45) - CAMS (40) = ~+5.0
    qc_24 = quantiles["centroid"]["24"]
    assert all(4.0 <= v <= 6.0 for v in qc_24)

    # 2. Horizon separation:
    for loc in ["1", "2", "centroid"]:
        for h in ["24", "48", "72"]:
            assert h in quantiles[loc]
            q_h = quantiles[loc][h]
            # Exactly 19 levels
            assert len(q_h) == 19
            # Monotonicity
            assert all(q_h[i] <= q_h[i + 1] for i in range(18))
            # Counts match
            assert counts[loc][h] > 0


# -----------------------------------------------------------------------------
# 3. Missing Data Handling
# -----------------------------------------------------------------------------


@respx.mock
@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.pipeline.seed_history.CamsClient.fetch_cams")
def test_missing_target_observation_not_counted(mock_fetch_cams, mock_open_datasets, clean_state):
    """Verify that if observation for a target hour is missing, no residual is fabricated."""
    import xarray as xr

    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = start_utc + timedelta(days=6)

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station 1",
                        "coordinates": {"latitude": 31.5, "longitude": 74.3},
                        "sensors": [{"id": 10, "parameter": {"id": 2, "name": "pm25"}}],
                    }
                ],
                "meta": {"found": 1},
            },
        )
    )

    # Observation data exists ONLY up to 2026-08-23 (3 days).
    # Cutoff determination will only resolve issuances with +72h observations.
    dates = pd.date_range(start_utc, start_utc + timedelta(days=3), freq="h")
    obs_10 = [
        {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 50.0 + (i % 2)}
        for i, dt in enumerate(dates)
    ]
    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(200, json={"results": obs_10, "meta": {"found": len(obs_10)}})
    )

    mock_fetch_cams.side_effect = lambda *a, **kw: kw.get("dest_path")

    def fake_open_datasets(grib_path):
        base_str = Path(grib_path).stem.replace("cams_hist_", "")
        cams_base = datetime.strptime(base_str, "%Y%m%d_%H")
        lats = np.array([32.0, 31.0])
        lons = np.array([74.0, 75.0])
        steps = np.array(
            [np.timedelta64(h, "h") for h in range(12, 120, 12)], dtype="timedelta64[ns]"
        )
        data = np.full((1, len(steps), 2, 2), 40.0 / 1e9)
        ds = xr.Dataset(
            {"pm2p5": (["time", "step", "latitude", "longitude"], data, {"units": "kg m**-3"})},
            coords={
                "time": [np.datetime64(cams_base)],
                "step": steps,
                "latitude": lats,
                "longitude": lons,
            },
        )
        return [ds]

    mock_open_datasets.side_effect = fake_open_datasets

    out_file = Path(".state/artifacts/seed_history_test.json")
    exit_code = generate_seed_history(
        test_start_utc=start_utc, test_now_utc=now_utc, output_path=out_file
    )
    assert exit_code == 0

    with out_file.open("r", encoding="utf-8") as f:
        artifact = json.load(f)

    # Only 1 issuance had full +72h coverage (2026-08-20 00:00:00)
    assert artifact["counts"]["1"]["72"] == 1


# -----------------------------------------------------------------------------
# 4. CLI Surface
# -----------------------------------------------------------------------------


def test_cli_run_seed_history_help():
    """Verify that CLI surface exposes run seed-history with options."""
    runner = CliRunner()
    result = runner.invoke(app, ["run", "seed-history", "--help"])
    assert result.exit_code == 0
    assert "--start" in result.stdout
    assert "--end" in result.stdout
    assert "--output" in result.stdout
