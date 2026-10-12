from pathlib import Path
from typing import Any
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

    data = np.array([[10.0, 20.0], [30.0, 40.0]])

    t2m_da = xr.DataArray(
        data=data + 273.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"},
    )
    pm2p5_da = xr.DataArray(
        data=data * 1e-9,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "kg m**-3"},
    )

    ds = xr.Dataset(
        {"t2m": t2m_da, "pm2p5": pm2p5_da},
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")},
    )
    return ds


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.preprocessing.gridded.get_centroid_from_config")
def test_extract_stations_bilinear(mock_centroid, mock_open_ds, mock_grib_simple):
    mock_open_ds.return_value = [mock_grib_simple]
    mock_centroid.return_value = {"lat": 31.5, "lon": 74.5}

    stations = pd.DataFrame(
        [
            {"location_id": "loc1", "lat": 31.5, "lon": 74.5},
            {"location_id": "loc2", "lat": 32.0, "lon": 74.0},
            {"location_id": "loc_out", "lat": 40.0, "lon": 80.0},
        ]
    )

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
    stations = pd.DataFrame([{"id": "loc1", "lat": 31.5}])  # missing lon and location_id
    with pytest.raises(ValueError, match="missing required columns"):
        extract_stations(Path("dummy.grib"), stations)


@pytest.fixture
def mock_grib_time_step():
    lats = [31.0, 32.0]  # ascending this time
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
            "lon": lons,
        },
        attrs={"units": "C"},  # Try C directly
    )

    ds = xr.Dataset({"t2m": t2m_da})
    return ds


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.preprocessing.gridded.get_centroid_from_config")
def test_extract_stations_time_step(mock_centroid, mock_open_ds, mock_grib_time_step):
    mock_open_ds.return_value = [mock_grib_time_step]
    mock_centroid.return_value = {"lat": 31.5, "lon": 74.5}

    stations = pd.DataFrame([{"location_id": "loc1", "lat": 31.5, "lon": 74.5}])
    df = extract_stations(Path("dummy.grib"), stations)

    # 2 times * 2 steps = 4 samples. For 2 locations (loc1 + centroid) = 8 rows total
    assert len(df) == 8

    # Verify the cross-product of time+step generated correct ts_utc
    times = df["target_hour_utc"].unique()
    assert len(times) == 4

    # Data is all 1.0, and units were C, so temp should be 1.0
    assert (df["temperature_c"] == 1.0).all()


@pytest.fixture
def mock_grib_360():
    # Lats: 32, 31 (descending)
    # Lons: 359.0, 360.0 (which is -1.0, 0.0) -> actually let's use 358.0, 359.0
    lats = [32.0, 31.0]
    lons = [358.0, 359.0]

    data = np.array([[10.0, 20.0], [30.0, 40.0]])

    t2m_da = xr.DataArray(
        data=data + 273.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"},
    )

    ds = xr.Dataset({"t2m": t2m_da}, coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")})
    return ds


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
@patch("smogsense.preprocessing.gridded.get_centroid_from_config")
def test_extract_stations_longitude_normalization(mock_centroid, mock_open_ds, mock_grib_360):
    mock_open_ds.return_value = [mock_grib_360]
    # Station uses standard -180/180 convention.
    # 358.5 in [0, 360] is -1.5 in [-180, 180].
    mock_centroid.return_value = {"lat": 31.5, "lon": -1.5}

    stations = pd.DataFrame(
        [
            {"location_id": "loc1", "lat": 31.5, "lon": -1.5},
            {"location_id": "loc_out", "lat": 31.5, "lon": -10.0},
        ]
    )

    df = extract_stations(Path("dummy.grib"), stations)

    assert len(df) == 3
    row_loc1 = df[df["location_id"] == "loc1"].iloc[0]
    # It should interpolate successfully because -1.5 is normalized to 358.5
    assert np.isclose(row_loc1["temperature_c"], 25.0)
    assert bool(row_loc1["outside_grid"]) is False

    row_out = df[df["location_id"] == "loc_out"].iloc[0]
    assert np.isnan(row_out["temperature_c"])
    assert bool(row_out["outside_grid"]) is True


def test_extract_stations_real_grib_no_indexpath(tmp_path: Path):
    eccodes = pytest.importorskip("eccodes")
    grib_file = tmp_path / "test_sample.grib"
    with grib_file.open("wb") as f:
        for step in [0, 1]:
            gid = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            eccodes.codes_set(gid, "shortName", "pm2p5")
            eccodes.codes_set(gid, "step", step)
            eccodes.codes_set(gid, "dataDate", 20261010)
            eccodes.codes_set(gid, "dataTime", 0)
            eccodes.codes_write(gid, f)
            eccodes.codes_release(gid)

    stations = pd.DataFrame([{"location_id": "loc1", "lat": 31.5, "lon": 74.3}])
    res = extract_stations(grib_file, stations, domain="lahore")

    # Verify no .idx index file was created on disk
    idx_files = list(tmp_path.glob("*.idx"))
    assert len(idx_files) == 0

    assert not res.empty
    assert "location_id" in res.columns
    assert set(res["location_id"].unique()) == {"loc1", "centroid"}


def test_get_centroid_from_config_variations() -> None:
    from smogsense.preprocessing.gridded import get_centroid_from_config

    # 1. Custom dict with nested domains: {"domains": {"domains": {"delhi": ...}}}
    settings_nested = {"domains": {"domains": {"delhi": {"centroid": {"lat": 28.6, "lon": 77.2}}}}}
    c1 = get_centroid_from_config("delhi", settings=settings_nested)
    assert c1 == {"lat": 28.6, "lon": 77.2}

    # 2. Custom dict with single domains level: {"domains": {"delhi": ...}}
    settings_single = {"domains": {"delhi": {"centroid": {"lat": 28.65, "lon": 77.25}}}}
    c2 = get_centroid_from_config("delhi", settings=settings_single)
    assert c2 == {"lat": 28.65, "lon": 77.25}

    # 3. Object with .domains attribute
    class DummySettings:
        def __init__(self) -> None:
            self.domains = {"domains": {"amritsar": {"centroid": {"lat": 31.63, "lon": 74.87}}}}

    c3 = get_centroid_from_config("amritsar", settings=DummySettings())
    assert c3 == {"lat": 31.63, "lon": 74.87}

    class DummySettingsSingle:
        def __init__(self) -> None:
            self.domains = {"amritsar": {"centroid": {"lat": 31.635, "lon": 74.875}}}

    c3_single = get_centroid_from_config("amritsar", settings=DummySettingsSingle())
    assert c3_single == {"lat": 31.635, "lon": 74.875}

    # 4. Canonical fallbacks
    assert "lat" in get_centroid_from_config("amritsar")
    assert "lat" in get_centroid_from_config("ludhiana")
    assert get_centroid_from_config("nonexistent_domain") == {}


def test_normalize_longitudes_edge_cases() -> None:
    from smogsense.preprocessing.gridded import _normalize_longitudes

    # Empty array
    empty = np.array([])
    assert len(_normalize_longitudes(empty, np.array([70.0, 80.0]))) == 0

    # Grid in [-180, 180] and points > 180
    grid_lons = np.array([-10.0, 10.0])
    pts_lons = np.array([190.0])
    norm = _normalize_longitudes(pts_lons, grid_lons)
    assert np.isclose(norm[0], -170.0)


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
def test_extract_stations_error_conditions(mock_open_ds: Any) -> None:
    # 1. Centroid missing raises ValueError
    with pytest.raises(ValueError, match="Centroid configuration missing"):
        extract_stations(Path("dummy.grib"), pd.DataFrame(), domain="unknown_domain_xyz")

    # 2. No datasets found in GRIB
    mock_open_ds.return_value = []
    with pytest.raises(ValueError, match="No datasets found in GRIB"):
        extract_stations(
            Path("dummy.grib"),
            pd.DataFrame([{"location_id": "1", "lat": 31.5, "lon": 74.3}]),
            domain="lahore",
        )

    # 3. Spatial dimensions missing
    ds_no_spatial = xr.Dataset(
        {"var": ("x", [1, 2])}, coords={"valid_time": pd.Timestamp("2026-01-01")}
    )
    mock_open_ds.return_value = [ds_no_spatial]
    with pytest.raises(ValueError, match="Could not identify spatial dimensions"):
        extract_stations(
            Path("dummy.grib"),
            pd.DataFrame([{"location_id": "1", "lat": 31.5, "lon": 74.3}]),
            domain="lahore",
        )


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
def test_extract_stations_units_and_dewpoint(mock_open_ds: Any) -> None:
    lats = [32.0, 31.0]
    lons = [74.0, 75.0]
    data = np.array([[10.0, 20.0], [30.0, 40.0]])

    # Dataset with t2m, d2m (dewpoint), pm25
    t2m_da = xr.DataArray(
        data=data + 273.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"},
    )
    d2m_da = xr.DataArray(
        data=data + 263.15,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "K"},
    )
    pm25_da = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        attrs={"units": "ug m-3"},
    )
    ds = xr.Dataset(
        {"t2m": t2m_da, "d2m": d2m_da, "pm25": pm25_da},
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")},
    )
    mock_open_ds.return_value = [ds]

    # Test stations with NaN lat/lon (valid_mask False branch)
    stations = pd.DataFrame(
        [
            {"location_id": "loc1", "lat": 31.5, "lon": 74.5},
            {"location_id": "loc_nan", "lat": np.nan, "lon": np.nan},
        ]
    )
    res = extract_stations(Path("dummy.grib"), stations, domain="lahore")
    assert not res.empty
    assert "rh_pct" in res.columns
    assert "pm25_ugm3" in res.columns
    row_nan = res[res["location_id"] == "loc_nan"].iloc[0]
    assert bool(row_nan["outside_grid"]) is True

    # Test unknown units raise ValueError
    ds_bad_pm25 = xr.Dataset(
        {
            "pm25": xr.DataArray(
                data,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                attrs={"units": "unknown_pm25_unit"},
            )
        },
        coords={"valid_time": pd.Timestamp("2026-01-01")},
    )
    mock_open_ds.return_value = [ds_bad_pm25]
    with pytest.raises(ValueError, match=r"Unknown PM2\.5 units"):
        extract_stations(
            Path("dummy.grib"),
            pd.DataFrame([{"location_id": "1", "lat": 31.5, "lon": 74.3}]),
            domain="lahore",
        )

    ds_bad_t = xr.Dataset(
        {
            "t2m": xr.DataArray(
                data,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                attrs={"units": "bad_temp_unit"},
            )
        },
        coords={"valid_time": pd.Timestamp("2026-01-01")},
    )
    mock_open_ds.return_value = [ds_bad_t]
    with pytest.raises(ValueError, match="Unknown temperature units"):
        extract_stations(
            Path("dummy.grib"),
            pd.DataFrame([{"location_id": "1", "lat": 31.5, "lon": 74.3}]),
            domain="lahore",
        )

    ds_bad_d = xr.Dataset(
        {
            "d2m": xr.DataArray(
                data,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                attrs={"units": "bad_dew_unit"},
            )
        },
        coords={"valid_time": pd.Timestamp("2026-01-01")},
    )
    mock_open_ds.return_value = [ds_bad_d]
    with pytest.raises(ValueError, match="Unknown dewpoint units"):
        extract_stations(
            Path("dummy.grib"),
            pd.DataFrame([{"location_id": "1", "lat": 31.5, "lon": 74.3}]),
            domain="lahore",
        )


@patch("smogsense.preprocessing.gridded.cfgrib.open_datasets")
def test_extract_stations_empty_stations_and_multiple_datasets(mock_open_ds: Any) -> None:
    lats = [32.0, 31.0]
    lons = [74.0, 75.0]
    data = np.array([[10.0, 20.0], [30.0, 40.0]])

    ds1 = xr.Dataset(
        {
            "t2m": xr.DataArray(
                data=data + 273.15,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                attrs={"units": "K"},
            )
        },
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")},
    )
    ds2 = xr.Dataset(
        {
            "pm25": xr.DataArray(
                data=data * 1e-9,
                dims=["latitude", "longitude"],
                coords={"latitude": lats, "longitude": lons},
                attrs={"units": "kg m-3"},
            )
        },
        coords={"valid_time": pd.Timestamp("2026-01-01T00:00:00Z")},
    )
    mock_open_ds.return_value = [ds1, ds2]

    # Test stations is None
    res = extract_stations(Path("dummy.grib"), None, domain="lahore")  # type: ignore[arg-type]
    assert len(res) == 1
    assert res.iloc[0]["location_id"] == "centroid"
    assert "pm25_ugm3" in res.columns
    assert "temperature_c" in res.columns
