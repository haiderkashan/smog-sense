"""tests.unit.test_qc_full - Comprehensive test suite for full QC pipeline.

Tests all 10 QC bits from docs/data-engineering.md §5:
1. RANGE_REJECT (1)
2. NEGATIVE_CLIPPED (2)
3. FLATLINE (4)
4. SPIKE (8)
5. SPATIAL_OUTLIER (16)
6. HIGH_RH (32)
7. LOW_COVERAGE (64)
8. HIGH_RANGE (128)
9. DRIFT_SUSPECT (256)
10. IMPUTED (512)
"""

import numpy as np
import pandas as pd

from smogsense.preprocessing.qc import (
    DRIFT_SUSPECT,
    FLATLINE,
    HIGH_RANGE,
    HIGH_RH,
    IMPUTED,
    LOW_COVERAGE,
    NEGATIVE_CLIPPED,
    RANGE_REJECT,
    SPATIAL_OUTLIER,
    SPIKE,
    apply_qc,
    clean_observations,
)


def test_qc_bits_definition():
    """Verify exact bit mask integers matching schema observations_hourly."""
    assert RANGE_REJECT == 1
    assert NEGATIVE_CLIPPED == 2
    assert FLATLINE == 4
    assert SPIKE == 8
    assert SPATIAL_OUTLIER == 16
    assert HIGH_RH == 32
    assert LOW_COVERAGE == 64
    assert HIGH_RANGE == 128
    assert DRIFT_SUSPECT == 256
    assert IMPUTED == 512


def test_qc_high_rh():
    """Verify that RH > 90% flags HIGH_RH without nulling PM2.5."""
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=3, freq="h"),
            "pm25_ugm3": [45.0, 50.0, 55.0],
            "rh_pct": [85.0, 91.0, 95.5],
        }
    )
    out = apply_qc(df)

    assert out.loc[0, "qc_flags"] == 0
    assert out.loc[0, "pm25_ugm3"] == 45.0

    assert bool(out.loc[1, "qc_flags"] & HIGH_RH)
    assert out.loc[1, "pm25_ugm3"] == 50.0  # Retained

    assert bool(out.loc[2, "qc_flags"] & HIGH_RH)
    assert out.loc[2, "pm25_ugm3"] == 55.0  # Retained


def test_qc_low_coverage():
    """Verify that sub-hourly coverage < 75% flags LOW_COVERAGE and nulls PM2.5."""
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=3, freq="h"),
            "pm25_ugm3": [45.0, 50.0, 55.0],
            "coverage_fraction": [0.80, 0.70, 0.50],
        }
    )
    out = apply_qc(df)

    assert out.loc[0, "qc_flags"] == 0
    assert out.loc[0, "pm25_ugm3"] == 45.0

    assert bool(out.loc[1, "qc_flags"] & LOW_COVERAGE)
    assert pd.isna(out.loc[1, "pm25_ugm3"])

    assert bool(out.loc[2, "qc_flags"] & LOW_COVERAGE)
    assert pd.isna(out.loc[2, "pm25_ugm3"])


def test_qc_high_range():
    """Verify that PM2.5 > 500 ug/m3 flags HIGH_RANGE without nulling."""
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1],
            "ts_utc": pd.date_range("2026-01-01", periods=3, freq="h"),
            "pm25_ugm3": [499.0, 500.0, 550.0],
        }
    )
    out = apply_qc(df)

    assert out.loc[0, "qc_flags"] == 0
    assert out.loc[0, "pm25_ugm3"] == 499.0

    assert out.loc[1, "qc_flags"] == 0
    assert out.loc[1, "pm25_ugm3"] == 500.0

    assert bool(out.loc[2, "qc_flags"] & HIGH_RANGE)
    assert out.loc[2, "pm25_ugm3"] == 550.0


def test_qc_spatial_outlier_detected():
    """Verify that an extreme reading among >= 3 close neighbours is flagged SPATIAL_OUTLIER and nulled."""
    # 4 stations around Lahore within 5 km of each other
    stations_meta = pd.DataFrame(
        [
            {"sensor_id": 101, "lat": 31.520, "lon": 74.350},
            {"sensor_id": 102, "lat": 31.525, "lon": 74.355},
            {"sensor_id": 103, "lat": 31.515, "lon": 74.345},
            {"sensor_id": 104, "lat": 31.530, "lon": 74.360},
        ]
    )

    t0 = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    obs_df = pd.DataFrame(
        [
            {"sensor_id": 101, "ts_utc": t0, "pm25_ugm3": 50.0},
            {"sensor_id": 102, "ts_utc": t0, "pm25_ugm3": 52.0},
            {"sensor_id": 103, "ts_utc": t0, "pm25_ugm3": 48.0},
            {"sensor_id": 104, "ts_utc": t0, "pm25_ugm3": 350.0},  # Spatial outlier
        ]
    )

    out = apply_qc(obs_df, stations_df=stations_meta)

    # 101, 102, 103 should remain valid
    for s_id in [101, 102, 103]:
        row = out[out["sensor_id"] == s_id].iloc[0]
        assert not (row["qc_flags"] & SPATIAL_OUTLIER)
        assert pd.notna(row["pm25_ugm3"])

    # 104 should be flagged and nulled
    outlier_row = out[out["sensor_id"] == 104].iloc[0]
    assert bool(outlier_row["qc_flags"] & SPATIAL_OUTLIER)
    assert pd.isna(outlier_row["pm25_ugm3"])


def test_qc_spatial_outlier_skipped_when_under_min_neighbors():
    """Verify spatial check is skipped when fewer than 3 neighbours exist (e.g. only 2 stations)."""
    stations_meta = pd.DataFrame(
        [
            {"sensor_id": 201, "lat": 31.520, "lon": 74.350},
            {"sensor_id": 202, "lat": 31.525, "lon": 74.355},
        ]
    )

    t0 = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    obs_df = pd.DataFrame(
        [
            {"sensor_id": 201, "ts_utc": t0, "pm25_ugm3": 50.0},
            {"sensor_id": 202, "ts_utc": t0, "pm25_ugm3": 350.0},
        ]
    )

    out = apply_qc(obs_df, stations_df=stations_meta)

    # Neither should be flagged with SPATIAL_OUTLIER because min_neighbors=3
    for s_id in [201, 202]:
        row = out[out["sensor_id"] == s_id].iloc[0]
        assert not (row["qc_flags"] & SPATIAL_OUTLIER)


def test_qc_drift_suspect():
    """Verify that weekly regression slope outside [0.5, 2.0] flags DRIFT_SUSPECT."""
    # 4 stations with 75 overlapping hours
    # stations 1, 2, 3 have concordant signals: x_t = 50 + 20*sin(t/5)
    # station 4 has drifted calibration: y_t = 0.2 * x_t (slope ~ 0.2 < 0.5)
    hours = 75
    timestamps = pd.date_range("2026-01-01 00:00", periods=hours, freq="h", tz="UTC")

    t_arr = np.arange(hours)
    base_signal = 50.0 + 20.0 * np.sin(t_arr / 5.0)

    rows = []
    for dt, val in zip(timestamps, base_signal, strict=True):
        rows.append({"sensor_id": 301, "ts_utc": dt, "pm25_ugm3": val, "lat": 31.52, "lon": 74.35})
        rows.append(
            {"sensor_id": 302, "ts_utc": dt, "pm25_ugm3": val + 2.0, "lat": 31.52, "lon": 74.36}
        )
        rows.append(
            {"sensor_id": 303, "ts_utc": dt, "pm25_ugm3": val - 2.0, "lat": 31.53, "lon": 74.35}
        )
        # Drifted station (slope ~ 0.25)
        rows.append(
            {"sensor_id": 304, "ts_utc": dt, "pm25_ugm3": 0.25 * val, "lat": 31.53, "lon": 74.36}
        )

    df = pd.DataFrame(rows)
    out = apply_qc(df)

    drift_rows = out[out["sensor_id"] == 304]
    assert (drift_rows["qc_flags"] & DRIFT_SUSPECT).any()
    # DRIFT_SUSPECT is flag-only: PM2.5 remains valid
    assert drift_rows["pm25_ugm3"].notna().all()


def test_clean_observations_integration():
    """Verify end-to-end clean_observations pipeline with full QC and short-gap PCHIP."""
    df = pd.DataFrame(
        {
            "sensor_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "ts_utc": [
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                pd.Timestamp("2026-01-01 02:00", tz="UTC"),
                pd.Timestamp("2026-01-01 03:00", tz="UTC"),
                pd.Timestamp("2026-01-01 04:00", tz="UTC"),
                pd.Timestamp("2026-01-01 05:00", tz="UTC"),
                pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                pd.Timestamp("2026-01-01 02:00", tz="UTC"),
            ],
            "pm25_ugm3": [40.0, 42.0, np.nan, np.nan, 45.0, -10.0, 510.0, 520.0, 515.0],
            "rh_pct": [60.0, 65.0, 70.0, 75.0, 95.0, 50.0, 60.0, 95.0, 60.0],
        }
    )

    out = clean_observations(df)

    # Station 1: Short gap of 2 NaNs (rows 2, 3) should be imputed with PCHIP
    s1 = out[out["sensor_id"] == 1].reset_index(drop=True)
    assert pd.notna(s1.loc[2, "pm25_ugm3"])
    assert pd.notna(s1.loc[3, "pm25_ugm3"])
    assert bool(s1.loc[2, "qc_flags"] & IMPUTED)
    assert bool(s1.loc[3, "qc_flags"] & IMPUTED)

    # Row 4 (45 ug/m3, 95% RH) should have HIGH_RH
    assert bool(s1.loc[4, "qc_flags"] & HIGH_RH)
    assert s1.loc[4, "pm25_ugm3"] == 45.0

    # Row 5 (-10 ug/m3) should be RANGE_REJECT and NaN
    assert bool(s1.loc[5, "qc_flags"] & RANGE_REJECT)
    assert pd.isna(s1.loc[5, "pm25_ugm3"])

    # Station 2: Sustained severe smog (> 500 ug/m3) should have HIGH_RANGE
    s2 = out[out["sensor_id"] == 2].reset_index(drop=True)
    assert bool(s2.loc[0, "qc_flags"] & HIGH_RANGE)
    assert s2.loc[0, "pm25_ugm3"] == 510.0
    assert bool(s2.loc[1, "qc_flags"] & HIGH_RANGE)
    assert bool(s2.loc[1, "qc_flags"] & HIGH_RH)
    assert s2.loc[1, "pm25_ugm3"] == 520.0
