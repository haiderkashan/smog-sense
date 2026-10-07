from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from smogsense.preprocessing.gridded import calc_rh, extract_stations


def test_calc_rh():
    # Known values: T=25C, Td=15C -> RH ~54%
    rh = calc_rh(np.array([25.0]), np.array([15.0]))
    assert np.isclose(rh[0], 53.8, atol=0.5)

@pytest.fixture
def mock_grib_simple():
    lats = [32.0, 31.0]
    lons = [74.0, 75.0]

    data = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ])

    t2m_da = xr.DataArray(
        data=data + 273.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"}
    )
    pm2p5_da = xr.DataArray(
        data=data * 1e-9,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "kg m**-3"}
    )

    ds = xr.Dataset(
        {"t2m": t2m_da, "pm2p5": pm2p5_da},
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")}
    )
    return ds

@patch('smogsense.preprocessing.gridded.xr.open_dataset')
@patch('smogsense.preprocessing.gridded.get_centroid_from_config')
def test_extract_stations_bilinear(mock_centroid, mock_open_ds, mock_grib_simple):
    mock_open_ds.return_value.__enter__.return_value = mock_grib_simple
    mock_centroid.return_value = {"lat": 31.5, "lon": 74.5}

    stations = pd.DataFrame([
        {"location_id": "loc1", "lat": 31.5, "lon": 74.5},
        {"location_id": "loc2", "lat": 32.0, "lon": 74.0},
        {"location_id": "loc_out", "lat": 40.0, "lon": 80.0},
    ])

    df = extract_stations(Path("dummy.grib"), stations)

    # Assert
    assert df.index.name is None
    assert "pm25_ugm3" in df.columns
    assert "temperature_c" in df.columns
    assert "outside_grid" in df.columns

    # Centroid gets added
    assert len(df) == 4

    # Check loc1 (31.5, 74.5)
    row_loc1 = df[df["location_id"] == "loc1"].iloc[0]
    assert np.isclose(row_loc1["temperature_c"], 25.0)
    assert np.isclose(row_loc1["pm25_ugm3"], 25.0)
    assert bool(row_loc1["outside_grid"]) is False

    # Check loc_out
    row_out = df[df["location_id"] == "loc_out"].iloc[0]
    assert np.isnan(row_out["temperature_c"])
    assert bool(row_out["outside_grid"]) is True

def test_extract_stations_validation():
    stations = pd.DataFrame([{"id": "loc1", "lat": 31.5}]) # missing lon and location_id
    with pytest.raises(ValueError, match="missing required columns"):
        extract_stations(Path("dummy.grib"), stations)

@pytest.fixture
def mock_grib_time_step():
    lats = [31.0, 32.0] # ascending this time
    lons = [74.0, 75.0]

    # Shape: (time: 2, step: 2, latitude: 2, longitude: 2)
    # We will use dummy data
    data = np.ones((2, 2, 2, 2))

    t2m_da = xr.DataArray(
        data=data,
        dims=["time", "step", "lat", "lon"],
        coords={
            "time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
            "step": [pd.Timedelta("0h"), pd.Timedelta("1h")],
            "lat": lats,
            "lon": lons
        },
        attrs={"units": "C"} # Try C directly
    )

    ds = xr.Dataset({"t2m": t2m_da})
    return ds

@patch('smogsense.preprocessing.gridded.xr.open_dataset')
@patch('smogsense.preprocessing.gridded.get_centroid_from_config')
def test_extract_stations_time_step(mock_centroid, mock_open_ds, mock_grib_time_step):
    mock_open_ds.return_value.__enter__.return_value = mock_grib_time_step
    mock_centroid.return_value = {"lat": 31.5, "lon": 74.5}

    stations = pd.DataFrame([{"location_id": "loc1", "lat": 31.5, "lon": 74.5}])
    df = extract_stations(Path("dummy.grib"), stations)

    # 2 times * 2 steps = 4 samples. For 2 locations (loc1 + centroid) = 8 rows total
    assert len(df) == 8

    # Verify the cross-product of time+step generated correct ts_utc
    times = df["ts_utc"].unique()
    assert len(times) == 4

    # Data is all 1.0, and units were C, so temp should be 1.0
    assert (df["temperature_c"] == 1.0).all()

@pytest.fixture
def mock_grib_360():
    # Lats: 32, 31 (descending)
    # Lons: 359.0, 360.0 (which is -1.0, 0.0) -> actually let's use 358.0, 359.0
    lats = [32.0, 31.0]
    lons = [358.0, 359.0]

    data = np.array([
        [10.0, 20.0],
        [30.0, 40.0]
    ])

    t2m_da = xr.DataArray(
        data=data + 273.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"}
    )

    ds = xr.Dataset(
        {"t2m": t2m_da},
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")}
    )
    return ds

@patch('smogsense.preprocessing.gridded.xr.open_dataset')
@patch('smogsense.preprocessing.gridded.get_centroid_from_config')
def test_extract_stations_longitude_normalization(mock_centroid, mock_open_ds, mock_grib_360):
    mock_open_ds.return_value.__enter__.return_value = mock_grib_360
    # Station uses standard -180/180 convention.
    # 358.5 in [0, 360] is -1.5 in [-180, 180].
    mock_centroid.return_value = {"lat": 31.5, "lon": -1.5}

    stations = pd.DataFrame([
        {"location_id": "loc1", "lat": 31.5, "lon": -1.5},
        {"location_id": "loc_out", "lat": 31.5, "lon": -10.0},
    ])

    df = extract_stations(Path("dummy.grib"), stations)

    assert len(df) == 3
    row_loc1 = df[df["location_id"] == "loc1"].iloc[0]
    # It should interpolate successfully because -1.5 is normalized to 358.5
    assert np.isclose(row_loc1["temperature_c"], 25.0)
    assert bool(row_loc1["outside_grid"]) is False

    row_out = df[df["location_id"] == "loc_out"].iloc[0]
    assert np.isnan(row_out["temperature_c"])
    assert bool(row_out["outside_grid"]) is True
