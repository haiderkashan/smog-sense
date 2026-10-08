import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from smogsense.pipeline.seed_history import determine_cutoff_dates, generate_seed_history


@pytest.fixture
def clean_state():
    state_dir = Path(".state")
    if state_dir.exists():
        import shutil
        try:
            shutil.rmtree(state_dir)
        except Exception:
            pass
    yield
    try:
        shutil.rmtree(state_dir)
    except Exception:
        pass


@respx.mock
@patch("smogsense.pipeline.seed_history.extract_stations")
@patch("smogsense.pipeline.seed_history.CamsClient.fetch_cams")
def test_seed_history_integration(mock_fetch_cams, mock_extract, clean_state):
    """Test full integration of seed history."""
    now = datetime.now(UTC)
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    
    # Fake OpenAQ response
    respx.get("https://api.openaq.org/v3/locations").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"id": 1, "name": "Test Station 1", "coordinates": {"latitude": 31.5, "longitude": 74.3}, "sensors": [{"id": 10, "parameter": {"name": "pm25"}}]}
                ],
                "meta": {"found": 1}
            }
        )
    )
    
    # We will fake the obs to end exactly 3 days ago.
    t_obs_max = now - timedelta(days=3)
    t_obs_max = t_obs_max.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # We will simulate 3 days of valid observations.
    dates = pd.date_range(t_obs_max - timedelta(days=5), t_obs_max, freq="h")
    obs_results = []
    for dt in dates:
        obs_results.append({
            "period": {"datetimeTo": {"utc": dt.isoformat()}},
            "value": 50.0
        })
        
    respx.get("https://api.openaq.org/v3/sensors/10/hours").mock(
        return_value=httpx.Response(
            200, json={"results": obs_results}
        )
    )

    def fake_fetch(*args, **kwargs):
        dest_path = kwargs.get("dest_path")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text("dummy_grib")
        return dest_path

    mock_fetch_cams.side_effect = fake_fetch

    def fake_extract(*args, **kwargs):
        cams_path = args[0]
        # deduce cams base from path
        base_str = cams_path.stem.replace("cams_hist_", "")
        cams_base = datetime.strptime(base_str, "%Y%m%d_%H").replace(tzinfo=UTC)
        
        rows = []
        for loc in [1, "centroid"]:
            for lead in kwargs.get("settings", {}).get("leadtime_hours", [36, 60, 84]): # Wait, mock does not have leadtime_hours
                pass
            # Just return some dummy targets
            for h in [24, 48, 72]:
                rows.append({
                    "location_id": loc,
                    "target_hour_utc": cams_base + timedelta(hours=h + 12), # Approximating target
                    "pm25_ugm3": 40.0
                })
        return pd.DataFrame(rows)
        
    # Better fake_extract to guarantee a match
    def fake_extract_better(*args, **kwargs):
        cams_path = args[0]
        base_str = cams_path.stem.replace("cams_hist_", "")
        cams_base = datetime.strptime(base_str, "%Y%m%d_%H").replace(tzinfo=UTC)
        # We need to return targets that match the current loop issuance.
        # But wait, in the loop, we call extract_stations, and match on target = issuance + 24.
        # Let's just generate all targets for the next 4 days.
        rows = []
        for i in range(1, 100):
            target = cams_base + timedelta(hours=i)
            rows.append({
                "location_id": 1,
                "target_hour_utc": target,
                "pm25_ugm3": 40.0
            })
        return pd.DataFrame(rows)
        
    mock_extract.side_effect = fake_extract_better

    # We will monkeypatch start_utc to be closer to avoid long test
    start_utc_test = t_obs_max - timedelta(days=5)
    start_utc_test = start_utc_test.replace(hour=0, minute=0, second=0, microsecond=0)
    
    with patch("smogsense.pipeline.seed_history.datetime") as mock_dt:
        mock_dt.now.return_value = now
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Override start_utc in the script
        import smogsense.pipeline.seed_history as sh
        original_generate = sh.generate_seed_history
        
        def fake_generate():
            sh.datetime = mock_dt
            # But the start_utc is hardcoded in generate_seed_history
            pass
            
    # Since start_utc is hardcoded, I will just patch determine_cutoff_dates to return our dates
    with patch("smogsense.pipeline.seed_history.determine_cutoff_dates") as mock_det, patch("smogsense.pipeline.seed_history.Settings.load") as mock_settings_load:
        # Mock settings
        mock_settings = mock_settings_load.return_value
        mock_settings.model_dump.return_value = {}
        mock_settings.domains = {
            "domains": {
                "lahore": {
                    "station_bbox": {"north": 32, "south": 31, "east": 75, "west": 74}
                }
            }
        }
        
        stations_df = pd.DataFrame([{"location_id": 1, "lat": 31.5, "lon": 74.3}])
        start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
        end_issuance = start_utc + timedelta(hours=48)

        obs_df = pd.DataFrame([
            {"location_id": 1, "ts_utc": start_utc + timedelta(hours=h), "pm25_ugm3": 50.0}
            for h in range(0, 100)
        ])
        async def dummy_det(*args, **kwargs):
            return (end_issuance, obs_df, stations_df)
        mock_det.side_effect = dummy_det
        
        exit_code = generate_seed_history()
        assert exit_code == 0

    # Verify Artifact
    artifact_path = Path(".state/artifacts/seed_history.json")
    assert artifact_path.exists()
    with artifact_path.open() as f:
        artifact = json.load(f)
        
    assert "metadata" in artifact
    assert "quantiles" in artifact
    
    q_1 = artifact["quantiles"].get("1")
    print("ARTIFACT:")
    print(artifact)
    assert q_1 is not None
    assert "24" in q_1
    assert "48" in q_1
    assert "72" in q_1
    
    # Residual = obs (50.0) - cams (40.0) = 10.0
    for h in ["24", "48", "72"]:
        assert all(np.isclose(v, 10.0) for v in q_1[h])
        assert len(q_1[h]) == 19
        # Monotonicity check
        assert all(q_1[h][i] <= q_1[h][i+1] for i in range(18))
        
    assert artifact["counts"]["1"]["24"] > 0
