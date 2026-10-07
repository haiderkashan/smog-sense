import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from pathlib import Path

from smogsense.data_ingestion.copernicus import CamsClient
from smogsense.errors import SourceUnavailable

@pytest.fixture
def base_config():
    return {
        'ads': {
            'url': 'test', 'key_env': 'ADS_API_KEY',
            'dataset': 'test_dataset',
            'request': {},
            'queue': {'max_wall_minutes_operational': 20, 'poll_interval_s': 30}
        }
    }

def test_fetch_cams_validation(base_config):
    client = CamsClient(base_config)
    dest_path = Path("test.grib")
    
    with pytest.raises(ValueError, match="must be on the hour"):
        client.fetch_cams(datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc), [0], ["var1"], [10, 2, 3, 10], dest_path)
        
    with pytest.raises(ValueError, match="must be 00Z or 12Z"):
        client.fetch_cams(datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc), [0], ["var1"], [10, 2, 3, 10], dest_path)
        
    with pytest.raises(ValueError, match="Area must contain exactly 4 numbers"):
        client.fetch_cams(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), [0], ["var1"], [1, 2, 3], dest_path)
        
    with pytest.raises(ValueError, match="leadtime_hours must not be empty"):
        client.fetch_cams(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), [], ["var1"], [10, 2, 3, 10], dest_path)
        
    with pytest.raises(ValueError, match="must be non-negative integer"):
        client.fetch_cams(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), [-1], ["var1"], [10, 2, 3, 10], dest_path)
        
    with pytest.raises(ValueError, match="variables must not be empty"):
        client.fetch_cams(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc), [0], [], [10, 2, 3, 10], dest_path)

def test_fetch_cams_403_licence_error(base_config):
    client = CamsClient(base_config)
    
    with patch.dict('os.environ', {'ADS_API_KEY': '123'}):
        with patch('cdsapi.Client') as MockClient:
            mock_cds = MagicMock()
            MockClient.return_value = mock_cds
            mock_cds.retrieve.side_effect = Exception("please accept the terms of the dataset")
            
            with pytest.raises(SourceUnavailable, match="Accept the dataset licence on the ADS website once"):
                client.fetch_cams(
                    datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                    [0, 1], ["var1"], [10, 0, 0, 10], Path("test.grib")
                )

@patch('time.sleep', return_value=None)
@patch('time.monotonic', side_effect=[0, 10, 20 * 60 + 1])
def test_fetch_cams_queue_timeout(mock_mono, mock_sleep, base_config):
    client = CamsClient(base_config)
    
    with patch.dict('os.environ', {'ADS_API_KEY': '123'}):
        with patch('cdsapi.Client') as MockClient:
            mock_cds = MagicMock()
            MockClient.return_value = mock_cds
            mock_result = MagicMock()
            mock_cds.retrieve.return_value = mock_result
            mock_result.reply = {"state": "running"}
            
            with pytest.raises(SourceUnavailable, match="ADS queue timeout"):
                client.fetch_cams(
                    datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                    [0, 1], ["var1"], [10, 0, 0, 10], Path("test.grib")
                )

def test_atomic_landing(tmp_path, base_config):
    client = CamsClient(base_config)
    dest_path = tmp_path / "final.grib"
    
    with patch.dict('os.environ', {'ADS_API_KEY': '123'}):
        with patch('cdsapi.Client') as MockClient:
            mock_cds = MagicMock()
            MockClient.return_value = mock_cds
            mock_result = MagicMock()
            mock_cds.retrieve.return_value = mock_result
            mock_result.reply = {"state": "completed"}
            
            def fake_download(target_path):
                Path(target_path).write_text("grib_data")
            mock_result.download.side_effect = fake_download
            
            with patch('time.monotonic', return_value=0):
                res = client.fetch_cams(
                    datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                    [0], ["var1"], [10, 0, 0, 10], dest_path
                )
                
            assert res == dest_path
            assert dest_path.read_text() == "grib_data"
            # check that no tmp files are left
            assert len(list(tmp_path.glob("*.tmp.grib"))) == 0

def test_atomic_landing_cleanup_on_fail(tmp_path, base_config):
    client = CamsClient(base_config)
    dest_path = tmp_path / "final.grib"
    
    with patch.dict('os.environ', {'ADS_API_KEY': '123'}):
        with patch('cdsapi.Client') as MockClient:
            mock_cds = MagicMock()
            MockClient.return_value = mock_cds
            mock_result = MagicMock()
            mock_cds.retrieve.return_value = mock_result
            mock_result.reply = {"state": "completed"}
            
            def fake_download_fail(target_path):
                Path(target_path).write_text("partial")
                raise Exception("Network error")
            mock_result.download.side_effect = fake_download_fail
            
            with patch('time.monotonic', return_value=0):
                with pytest.raises(SourceUnavailable, match="Failed to download GRIB"):
                    client.fetch_cams(
                        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                        [0], ["var1"], [10, 0, 0, 10], dest_path
                    )
                
            assert not dest_path.exists()
            assert len(list(tmp_path.glob("*.tmp.grib"))) == 0

def test_concurrent_atomic_landing(tmp_path, base_config):
    import concurrent.futures
    import time
    
    client = CamsClient(base_config)
    dest_path = tmp_path / "final.grib"
    
    # We will simulate two threads downloading the same file.
    # To ensure they overlap, we'll make the mock download sleep slightly,
    # then write to its specific target_path.
    
    with patch.dict('os.environ', {'ADS_API_KEY': '123'}):
        with patch('cdsapi.Client') as MockClient:
            mock_cds = MagicMock()
            MockClient.return_value = mock_cds
            mock_result = MagicMock()
            mock_cds.retrieve.return_value = mock_result
            mock_result.reply = {"state": "completed"}
            
            def fake_download(target_path):
                # Write marker so we know which thread wrote it
                # Because both threads will call os.replace on the same dest_path,
                # the "last writer wins", but neither should crash.
                time.sleep(0.1)
                Path(target_path).write_text("grib_data_thread")
                
            mock_result.download.side_effect = fake_download
            
            with patch('time.monotonic', return_value=0):
                def run_fetch():
                    return client.fetch_cams(
                        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
                        [0], ["var1"], [10, 0, 0, 10], dest_path
                    )
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(run_fetch) for _ in range(2)]
                    
                    for f in concurrent.futures.as_completed(futures):
                        f.result() # should not raise exceptions
                        
                assert dest_path.exists()
                assert dest_path.read_text() == "grib_data_thread"
                assert len(list(tmp_path.glob("*.tmp.grib"))) == 0
