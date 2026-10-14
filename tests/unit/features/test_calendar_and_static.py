"""Unit tests for calendar, static station covariates, and aerosol feature extraction."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from smogsense.features.aerosol import extract_block_aerosol_features
from smogsense.features.calendar import (
    compute_cyclical_time_features,
    extract_block_calendar_features,
    is_diwali_window,
)
from smogsense.features.static_covariates import (
    DOMAIN_CENTROIDS,
    compute_distance_to_centroid,
    extract_station_static_covariates,
)


def test_cyclical_time_features_geometry() -> None:
    """Validate mathematical properties of cyclical sin/cos encodings."""
    # 00:00 local time (midnight in UTC)
    t_midnight = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    feats_00 = compute_cyclical_time_features(t_midnight, timezone="UTC")
    assert feats_00["sin_hour"] == pytest.approx(0.0, abs=1e-6)
    assert feats_00["cos_hour"] == pytest.approx(1.0, abs=1e-6)

    # 06:00 local time
    t_06 = datetime(2026, 11, 1, 6, 0, tzinfo=UTC)
    feats_06 = compute_cyclical_time_features(t_06, timezone="UTC")
    assert feats_06["sin_hour"] == pytest.approx(1.0, abs=1e-6)
    assert feats_06["cos_hour"] == pytest.approx(0.0, abs=1e-6)

    # 12:00 local time (noon)
    t_12 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    feats_12 = compute_cyclical_time_features(t_12, timezone="UTC")
    assert feats_12["sin_hour"] == pytest.approx(0.0, abs=1e-6)
    assert feats_12["cos_hour"] == pytest.approx(-1.0, abs=1e-6)

    # Check bounds and Pythagoras identity across all 24 hours
    for h in range(24):
        t_h = datetime(2026, 11, 1, h, 30, tzinfo=UTC)
        f = compute_cyclical_time_features(t_h, timezone="UTC")
        assert -1.0 <= f["sin_hour"] <= 1.0
        assert -1.0 <= f["cos_hour"] <= 1.0
        assert f["sin_hour"] ** 2 + f["cos_hour"] ** 2 == pytest.approx(1.0, rel=1e-5)
        assert f["sin_dow"] ** 2 + f["cos_dow"] ** 2 == pytest.approx(1.0, rel=1e-5)
        assert f["sin_doy"] ** 2 + f["cos_doy"] ** 2 == pytest.approx(1.0, rel=1e-5)


def test_diwali_dates_detection() -> None:
    """Test Diwali event window detection for known calendar dates."""
    # Diwali 2023 fell on 2023-11-12
    t_diwali_2023 = datetime(2023, 11, 12, 12, 0, tzinfo=UTC)
    assert is_diwali_window(t_diwali_2023, timezone="Asia/Kolkata", window_days=1) is True

    # 1 day before Diwali 2023
    t_eve = datetime(2023, 11, 11, 18, 0, tzinfo=UTC)
    assert is_diwali_window(t_eve, timezone="Asia/Kolkata", window_days=1) is True

    # 1 day after Diwali 2023
    t_post = datetime(2023, 11, 13, 6, 0, tzinfo=UTC)
    assert is_diwali_window(t_post, timezone="Asia/Kolkata", window_days=1) is True

    # 3 days before Diwali 2023 (outside window)
    t_far = datetime(2023, 11, 9, 12, 0, tzinfo=UTC)
    assert is_diwali_window(t_far, timezone="Asia/Kolkata", window_days=1) is False

    # Arbitrary summer date
    t_summer = datetime(2023, 6, 15, 12, 0, tzinfo=UTC)
    assert is_diwali_window(t_summer, timezone="Asia/Kolkata", window_days=1) is False


def test_domain_diwali_feature_exclusion_policy() -> None:
    """Strict Domain Adaptation Rule:

    The Diwali feature is included for Indian domains (delhi, amritsar, ludhiana),
    and is STRICTLY DROPPED (not present in columns) for Lahore, never zero-filled.
    """
    t_diwali = datetime(2023, 11, 12, 0, 0, tzinfo=UTC)

    # Indian domains must have f_cal_diwali_h24
    for indian_dom in ("delhi", "amritsar", "ludhiana"):
        feats_in = extract_block_calendar_features(t_diwali, domain=indian_dom)
        assert "f_cal_diwali_h24" in feats_in
        assert feats_in["f_cal_diwali_h24"] == 1.0

    # Pakistani domain (Lahore) must strictly NOT have f_cal_diwali_h24
    feats_lahore = extract_block_calendar_features(t_diwali, domain="lahore")
    assert "f_cal_diwali_h24" not in feats_lahore
    assert "f_cal_diwali_h48" not in feats_lahore
    assert "f_cal_diwali_h72" not in feats_lahore


def test_static_covariates_distance_and_descriptors() -> None:
    """Test station coordinate and centroid distance extraction."""
    c_lat, c_lon = DOMAIN_CENTROIDS["lahore"]

    # Distance to centroid for the exact centroid coordinate must be ~0.0 km
    dist_zero = compute_distance_to_centroid(c_lat, c_lon, domain="lahore")
    assert dist_zero == pytest.approx(0.0, abs=1e-3)

    # Distance to centroid for a station 10 km away
    dist_offset = compute_distance_to_centroid(c_lat + 0.1, c_lon, domain="lahore")
    assert dist_offset > 5.0

    # Test full static covariates extraction
    station_data = {
        "latitude": c_lat,
        "longitude": c_lon,
        "elevation": 215.0,
        "is_reference": True,
    }
    covs = extract_station_static_covariates(station_data, domain="lahore")

    assert covs["f_stat_lat"] == pytest.approx(c_lat)
    assert covs["f_stat_lon"] == pytest.approx(c_lon)
    assert covs["f_stat_elevation_m"] == pytest.approx(215.0)
    assert covs["f_stat_dist_city_center_km"] == pytest.approx(0.0, abs=1e-3)
    assert covs["f_stat_is_reference"] == 1.0


def test_extract_block_aerosol_features() -> None:
    """Test CAMS AOD550 horizon and lookback feature extraction."""
    t_iss = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    # 24 hours lookback + 72 hours forecast = 96 hours
    timestamps = pd.date_range("2026-10-31 00:00:00", periods=96, freq="1h", tz="UTC")
    cams_df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "aod550": np.linspace(0.2, 1.2, 96),
        }
    )

    feats = extract_block_aerosol_features(cams_df, t_iss, horizons=(24, 48, 72))

    assert "f_aod_trailing_24h_mean" in feats
    assert "f_aod_mean_h24" in feats
    assert "f_aod_max_h24" in feats
    assert "f_aod_mean_h48" in feats
    assert "f_aod_mean_h72" in feats

    assert feats["f_aod_trailing_24h_mean"] > 0.0
    assert feats["f_aod_mean_h24"] > 0.0
    assert feats["f_aod_max_h24"] >= feats["f_aod_mean_h24"]
