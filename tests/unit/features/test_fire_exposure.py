"""Unit tests for wind-aligned, target-centric fire exposure kernel."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from smogsense.features.fire_exposure import (
    compute_advective_kernel,
    compute_hourly_fire_exposure,
    compute_single_angle_geometry,
    compute_wind_alignment_weight,
    compute_wind_direction_and_speed,
    exposure,
    extract_block_fire_features,
)


def test_amritsar_to_lahore_worked_example() -> None:
    """Validate worked example from docs/ml-architecture.md §3:

    Target Lahore: 31.5204°N, 74.3587°E.
    Fire at Amritsar: 31.6340°N, 74.8723°E.
    Offset: dE ~ 48.8 km, dN ~ 12.6 km.
    Target -> Fire azimuth theta ~ 75.5° clockwise from North.
    Fire -> Target transport bearing beta ~ 255.5°.
    Easterly wind (u = -5, v = 0): phi = 270°, alignment a ~ 0.97 (high exposure).
    Westerly wind (u = +5, v = 0): phi = 90°, cos(-165.5°) < 0, clipped to a = 0.0.
    """
    lahore_lat, lahore_lon = 31.5204, 74.3587
    amritsar_lat, amritsar_lon = 31.6340, 74.8723

    theta, beta, dist_km, sector = compute_single_angle_geometry(
        lahore_lat, lahore_lon, [amritsar_lat], [amritsar_lon]
    )

    assert theta[0] == pytest.approx(75.5, abs=1.0)
    assert beta[0] == pytest.approx(255.5, abs=1.0)
    assert dist_km[0] == pytest.approx(50.4, abs=2.0)
    # Sector index: 75.5° falls in ENE/East sector (sector 1 or 2: N=0, NE=1, E=2)
    assert sector[0] in (1, 2)

    # 1. Easterly wind (blows from East to West, u = -5, v = 0 -> phi = 270°)
    phi_east, speed_east = compute_wind_direction_and_speed(u10=-5.0, v10=0.0)
    assert phi_east == pytest.approx(270.0, abs=0.1)
    assert speed_east == pytest.approx(5.0, abs=0.1)

    a_east = compute_wind_alignment_weight(phi_east, beta[0])
    assert a_east == pytest.approx(0.968, abs=0.02)
    assert a_east > 0.95

    # 2. Westerly wind (blows from West to East, u = +5, v = 0 -> phi = 90°)
    phi_west, speed_west = compute_wind_direction_and_speed(u10=5.0, v10=0.0)
    assert phi_west == pytest.approx(90.0, abs=0.1)
    assert speed_west == pytest.approx(5.0, abs=0.1)

    a_west = compute_wind_alignment_weight(phi_west, beta[0])
    assert a_west == 0.0


def test_exposure_decays_with_distance() -> None:
    """Closer fire produces strictly higher exposure for fixed wind and equal FRP."""
    t0 = pd.Timestamp("2026-11-01 12:00:00", tz="UTC")
    w_speed = 5.0
    alignment = 1.0

    # Fire at 30 km vs Fire at 150 km vs Fire at 400 km
    k_30 = compute_advective_kernel(
        dist_km=30.0,
        wind_speed_ms=w_speed,
        alignment_weight=alignment,
        detection_utc=t0 - timedelta(hours=2),
        valid_time_utc=t0,
        issuance_utc=t0,
    )
    k_150 = compute_advective_kernel(
        dist_km=150.0,
        wind_speed_ms=w_speed,
        alignment_weight=alignment,
        detection_utc=t0 - timedelta(hours=2),
        valid_time_utc=t0,
        issuance_utc=t0,
    )
    k_400 = compute_advective_kernel(
        dist_km=400.0,
        wind_speed_ms=w_speed,
        alignment_weight=alignment,
        detection_utc=t0 - timedelta(hours=2),
        valid_time_utc=t0,
        issuance_utc=t0,
    )

    assert k_30 > k_150 > k_400 > 0.0


def test_exposure_decays_with_detection_age() -> None:
    """Recent fire detection produces higher exposure than older fire detection."""
    t0 = pd.Timestamp("2026-11-01 12:00:00", tz="UTC")
    w_speed = 5.0
    alignment = 1.0
    dist = 50.0

    k_recent = compute_advective_kernel(
        dist_km=dist,
        wind_speed_ms=w_speed,
        alignment_weight=alignment,
        detection_utc=t0 - timedelta(hours=1),
        valid_time_utc=t0,
        issuance_utc=t0,
    )
    k_old = compute_advective_kernel(
        dist_km=dist,
        wind_speed_ms=w_speed,
        alignment_weight=alignment,
        detection_utc=t0 - timedelta(hours=36),
        valid_time_utc=t0,
        issuance_utc=t0,
    )

    assert k_recent > k_old > 0.0


def test_platform_normalisation_removes_satellite_density_step() -> None:
    """When number of active platforms doubles, platform normalisation keeps exposure invariant."""
    t_iss = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    valid_hours = [t_iss + timedelta(hours=h) for h in range(24)]

    # 1 platform scenario: 1 detection with FRP 100 MW
    dets_1plat = pd.DataFrame(
        {
            "latitude": [31.634],
            "longitude": [74.872],
            "frp": [100.0],
            "acq_datetime_utc": [t_iss - timedelta(hours=2)],
        }
    )

    # 2 platforms scenario: 2 detections with FRP 100 MW each (same fire cluster detected by both)
    dets_2plat = pd.DataFrame(
        {
            "latitude": [31.634, 31.634],
            "longitude": [74.872, 74.872],
            "frp": [100.0, 100.0],
            "acq_datetime_utc": [t_iss - timedelta(hours=2), t_iss - timedelta(hours=2)],
        }
    )

    wind_df = pd.DataFrame(
        {
            "ts_utc": valid_hours,
            "u10_ms": np.full(24, -5.0),
            "v10_ms": np.full(24, 0.0),
        }
    )

    exp_1 = compute_hourly_fire_exposure(
        detections_df=dets_1plat,
        target_lat=31.5204,
        target_lon=74.3587,
        wind_df=wind_df,
        issuance_utc=t_iss,
        valid_hours=valid_hours,
        platforms_active=1,
    )

    exp_2 = compute_hourly_fire_exposure(
        detections_df=dets_2plat,
        target_lat=31.5204,
        target_lon=74.3587,
        wind_df=wind_df,
        issuance_utc=t_iss,
        valid_hours=valid_hours,
        platforms_active=2,
    )

    # Normalised exposures must match exactly
    for vh in valid_hours:
        assert exp_1[vh] == pytest.approx(exp_2[vh], rel=1e-5)


def test_extract_block_fire_features_rings_and_sectors() -> None:
    """Test concentric ring and sector binning in block feature extraction."""
    t_iss = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    lahore_lat, lahore_lon = 31.5204, 74.3587

    # Create detections placed in distinct rings and sectors
    # 1. In ring 0-50 km (Amritsar is ~50.4 km, let's use closer point: 31.55, 74.45 ~ 10 km)
    # 2. In ring 50-150 km (Amritsar ~ 50.4 km)
    # 3. Beyond 600 km (should be excluded)
    dets = pd.DataFrame(
        {
            "latitude": [31.55, 31.634, 40.0],
            "longitude": [74.45, 74.872, 85.0],
            "frp": [20.0, 80.0, 500.0],
            "acq_datetime_utc": [
                t_iss - timedelta(hours=3),
                t_iss - timedelta(hours=4),
                t_iss - timedelta(hours=1),
            ],
        }
    )

    wind_df = pd.DataFrame(
        {
            "ts_utc": pd.date_range("2026-11-01 00:00:00", periods=72, freq="1h", tz="UTC"),
            "u10_ms": np.full(72, -4.0),
            "v10_ms": np.full(72, 0.0),
        }
    )

    feats = extract_block_fire_features(
        detections_df=dets,
        target_lat=lahore_lat,
        target_lon=lahore_lon,
        wind_df=wind_df,
        issuance_utc=t_iss,
        horizons=(24, 48, 72),
        platforms_active=1,
    )

    # Detections within 600 km total 2 (the 3rd at 40N, 85E is >1000 km away)
    assert feats["f_fire_detections_total"] == 2.0
    assert feats["f_fire_frp_total"] == 100.0  # 20 + 80
    assert feats["f_fire_count_r0_50"] == 1.0
    assert feats["f_fire_frp_r0_50"] == 20.0
    assert feats["f_fire_count_r50_150"] == 1.0
    assert feats["f_fire_frp_r50_150"] == 80.0
    assert feats["f_fire_count_r150_300"] == 0.0
    assert feats["f_fire_count_r300_600"] == 0.0

    # Sector assignments (both points are north-east of Lahore)
    assert feats["f_fire_count_sector_NE"] + feats["f_fire_count_sector_E"] == 2.0

    # Block exposures must be positive
    assert feats["f_fire_exposure_e24"] > 0.0
    assert feats["f_fire_exposure_e48"] > 0.0
    assert feats["f_fire_exposure_e72"] > 0.0


def test_exposure_schema_validation() -> None:
    """Test that exposure() produces DataFrame conforming to fire_exposure.schema.yaml."""
    t_iss = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    dets = pd.DataFrame(
        {
            "latitude": [31.634],
            "longitude": [74.872],
            "frp": [120.0],
            "acq_datetime_utc": [t_iss - timedelta(hours=3)],
        }
    )

    hours = [t_iss + timedelta(hours=i) for i in range(24)]
    wind_df = pd.DataFrame(
        {
            "ts_utc": hours,
            "u10_ms": np.full(24, -3.0),
            "v10_ms": np.full(24, 1.0),
        }
    )

    res_df = exposure(
        detections=dets,
        target="lahore",
        wind_series=wind_df,
        params={
            "issuance_utc": t_iss,
            "platforms_active": 2,
            "coverage_complete": True,
        },
    )

    assert len(res_df) == 24
    assert "fire_count_r0_50" in res_df.columns
    assert "frp_mw_r50_150" in res_df.columns
    assert "exposure_e24" in res_df.columns
    assert res_df["domain"].iloc[0] == "lahore"
