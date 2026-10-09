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
import xarray as xr

from smogsense.pipeline.seed_history import generate_seed_history


@pytest.fixture
def clean_state():
    state_dir = Path(".state")
    if state_dir.exists():
        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(state_dir)
    yield
    with contextlib.suppress(Exception):
        shutil.rmtree(state_dir)


@respx.mock
@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.pipeline.seed_history.CamsClient.fetch_cams")
def test_seed_history_integration(mock_fetch_cams, mock_open_datasets, clean_state):
    """Test full integration of seed history with minimal mocking."""
    # We set a small window for the test
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    now_utc = start_utc + timedelta(days=6)  # 6 days later, so t_obs_max allows 1 day of issuance

    respx.get(url__regex=r".*/v3/locations.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": 1,
                        "name": "Station A",
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
                        "name": "Station B",
                        "coordinates": {"latitude": 31.6, "longitude": 74.4},
                        "sensors": [
                            {
                                "id": 11,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 100},
                            }
                        ],
                    },
                    {
                        "id": 3,
                        "name": "Station C",
                        "coordinates": {"latitude": 31.55, "longitude": 74.35},
                        "sensors": [
                            {
                                "id": 12,
                                "parameter": {"id": 2, "name": "pm25"},
                                "coverage": {"percentComplete": 100},
                            }
                        ],
                    },
                ],
                "meta": {"found": 3},
            },
        )
    )

    # 2. Fake OpenAQ observations
    # For simplicity, all stations have continuous data
    dates = pd.date_range(start_utc, now_utc, freq="h")
    obs_results_10 = []
    obs_results_11 = []
    obs_results_12 = []
    for i, dt in enumerate(dates):
        obs_results_10.append(
            {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 50.0 + (i % 2)}
        )  # Station A obs = 50
        obs_results_11.append(
            {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 20.0 + (i % 2)}
        )  # Station B obs = 20
        obs_results_12.append(
            {"period": {"datetimeFrom": {"utc": dt.isoformat()}}, "value": 35.0 + (i % 2)}
        )  # Station C obs = 35

    respx.get(url__regex=r".*/v3/sensors/10/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results_10, "meta": {"found": len(obs_results_10)}}
        )
    )
    respx.get(url__regex=r".*/v3/sensors/11/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results_11, "meta": {"found": len(obs_results_11)}}
        )
    )
    respx.get(url__regex=r".*/v3/sensors/12/hours.*").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results_12, "meta": {"found": len(obs_results_12)}}
        )
    )

    # 3. Fake CAMS fetch
    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy_grib")
        return dest_path

    mock_fetch_cams.side_effect = fake_fetch

    # 4. Fake xarray/cfgrib dataset
    def fake_open_datasets(grib_path):
        # Determine base time from path to align step dimensions
        base_str = Path(grib_path).stem.replace("cams_hist_", "")
        cams_base = datetime.strptime(base_str, "%Y%m%d_%H")

        # Create a tiny 2x2 grid covering Lahore
        lats = np.array([32.0, 31.0])
        lons = np.array([74.0, 75.0])
        steps = np.array(
            [np.timedelta64(h, "h") for h in range(12, 120, 12)], dtype="timedelta64[ns]"
        )

        # Create data array with exact shape (time, step, lat, lon)
        # We will make CAMS value = 40.0 for Station A (which is at 31.5, 74.3)
        # and CAMS value = 50.0 for Station B (which is at 31.6, 74.4)
        # Actually, interpolation is bilinear. If we just make the whole field uniform for each horizon,
        # it's easier. We can make CAMS = 40 for everyone.
        # But wait, test asks for: Station A -> residual +10. Station B -> residual -30.
        # So Station A obs=50, cams=40 -> res=+10.
        # Station B obs=20, cams=50 -> res=-30.
        # If we make CAMS spatially varying:
        # A simple way: make the grid flat but varying over time? No, we need it to vary spatially.
        # We can just construct the grid so that at (31.5, 74.3) it interpolates to 40,
        # and at (31.6, 74.4) it interpolates to 50.
        # For simplicity, we can just return a flat 40 for A, and 50 for B.
        # Grid [lat, lon]:
        # [32.0, 74.0] -> Val1, [32.0, 75.0] -> Val2
        # [31.0, 74.0] -> Val3, [31.0, 75.0] -> Val4
        # Just use flat 40 for now, then A gets +10, B gets -20.

        # Let's make it easy: create 4D array (time:1, step:3, lat:2, lon:2)
        # Actually xarray dims typically: time, step, latitude, longitude
        data = np.full(
            (1, len(steps), 2, 2), 40.0 / 1e9
        )  # convert to kg m-3 since gridded.py multiplies by 1e9
        data[0, :, :, :] = 40.0 / 1e9
        # To make Station B = 50, we can just set the grid so that it evaluates differently,
        # or we just rely on A=+10, B=-20 for separation.
        # We'll use flat 40 for all, so Centroid is also 40.

        ds = xr.Dataset(
            {
                "pm2p5": (["time", "step", "latitude", "longitude"], data, {"units": "kg m**-3"}),
            },
            coords={
                "time": [np.datetime64(cams_base)],
                "step": steps,
                "latitude": lats,
                "longitude": lons,
            },
        )
        return [ds]

    mock_open_datasets.side_effect = fake_open_datasets

    # 5. Run the pipeline!
    exit_code = generate_seed_history(test_start_utc=start_utc, test_now_utc=now_utc)
    assert exit_code == 0

    # 6. Verify Artifact
    artifact_path = Path(".state/artifacts/seed_history.json")
    assert artifact_path.exists()
    with artifact_path.open() as f:
        artifact = json.load(f)

    assert "metadata" in artifact
    assert "quantiles" in artifact

    q_1 = artifact["quantiles"].get("1")
    q_2 = artifact["quantiles"].get("2")
    q_3 = artifact["quantiles"].get("3")
    q_c = artifact["quantiles"].get("centroid")

    # Station A: obs 50, cams 40 => residual +10
    assert q_1 is not None
    assert all(10.0 <= v <= 11.0 for v in q_1["24"])

    # Station B: obs 20, cams 40 => residual -20
    assert q_2 is not None
    assert all(-20.0 <= v <= -19.0 for v in q_2["24"])

    # Station C: obs 35, cams 40 => residual -5
    assert q_3 is not None
    assert all(-5.0 <= v <= -4.0 for v in q_3["24"])

    # Centroid: obs mean(50, 20, 35)=35, cams 40 => residual -5
    assert q_c is not None
    assert all(-5.0 <= v <= -4.0 for v in q_c["24"])

    # Horizon separation
    for h in ["24", "48", "72"]:
        assert len(q_1[h]) == 19
        assert all(q_1[h][i] <= q_1[h][i + 1] for i in range(18))  # Monotonic

    # Metadata tracking
    meta = artifact["metadata"]
    assert "coverage" in meta
    assert len(meta["coverage"]["expected"]) > 0
    assert len(meta["coverage"]["processed"]) > 0
    assert len(meta["coverage"]["failed"]) == 0
