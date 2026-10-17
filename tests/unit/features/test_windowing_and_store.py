"""Unit tests for sequence windowing tensors, target engine, and versioned feature store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smogsense.features.store import (
    DEFAULT_FEATURE_SET_VERSION,
    SCHEMA_VERSION,
    build_feature_table,
    compute_config_hash,
    load_feature_table,
    save_feature_table,
)
from smogsense.features.targets import (
    assert_no_target_leakage,
    block_targets,
)
from smogsense.features.windowing import (
    SEQUENCE_CHANNELS,
    build_sequence_tensor,
    compute_elapsed_time_since_observation,
)


def test_compute_elapsed_time_since_observation() -> None:
    """Test Che et al. (2018) GRU-D elapsed time delta_t calculation."""
    # 1. Observed at start, then gap: [1, 1, 0, 0, 1, 0]
    mask_1 = np.array([1, 1, 0, 0, 1, 0], dtype=np.float32)
    delta_1 = compute_elapsed_time_since_observation(mask_1)
    expected_1 = np.array([0.0, 0.0, 1.0, 2.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(delta_1, expected_1)

    # 2. Missing at start, then observed: [0, 0, 1]
    mask_2 = np.array([0, 0, 1], dtype=np.float32)
    delta_2 = compute_elapsed_time_since_observation(mask_2)
    expected_2 = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(delta_2, expected_2)


def test_build_sequence_tensor_dimensions_and_channels() -> None:
    """Validate 72h historical sequence tensor shape (72, 6) and channel alignment."""
    t_iss = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    loc_id = 1001

    # 72 lookback hours: [T-72h, T)
    timestamps = pd.date_range("2026-10-30 00:00:00", periods=72, freq="1h", tz="UTC")

    obs_df = pd.DataFrame(
        {
            "location_id": np.full(72, loc_id),
            "ts_utc": timestamps,
            "pm25_ugm3": np.linspace(50.0, 200.0, 72),
            "available_at_utc": timestamps + timedelta(minutes=30),
        }
    )

    cams_df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "cams_pm25": np.linspace(40.0, 180.0, 72),
            "aod550": np.full(72, 0.8),
            "blh_m": np.full(72, 300.0),
            "u10_ms": np.full(72, 2.0),
            "v10_ms": np.full(72, -1.0),
            "t2m_c": np.full(72, 22.0),
            "d2m_c": np.full(72, 12.0),
            "available_at_utc": pd.Timestamp(t_iss) - timedelta(hours=2),
        }
    )

    fire_df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "exposure_e24": np.linspace(5.0, 50.0, 72),
            "available_at_utc": pd.Timestamp(t_iss) - timedelta(hours=1),
        }
    )

    seq = build_sequence_tensor(
        location_id=loc_id,
        issuance_utc=t_iss,
        obs_df=obs_df,
        cams_df=cams_df,
        fire_exposure_df=fire_df,
        lookback_hours=72,
    )

    assert seq.shape == (72, 6)
    assert seq.location_id == loc_id
    assert seq.channels == list(SEQUENCE_CHANNELS)
    assert len(seq.timestamps) == 72
    assert seq.values.shape == (72, 6)
    assert seq.mask.shape == (72, 6)
    assert seq.delta_t.shape == (72,)

    # Combined array has 6 columns matching C = 6 channels
    combined = seq.to_combined_array()
    assert combined.shape == (72, 6)

    # Verify channel semantic values
    # Channel 0: pm25_log1p_masked
    np.testing.assert_allclose(seq.values[:, 0], np.log1p(np.linspace(50.0, 200.0, 72)))
    # Channel 1: obs_minus_cams_pm25 (obs - cams)
    expected_diff = np.linspace(50.0, 200.0, 72) - np.linspace(40.0, 180.0, 72)
    np.testing.assert_allclose(seq.values[:, 1], expected_diff, rtol=1e-5)
    # Channel 4: pm25_mask (all valid)
    np.testing.assert_allclose(seq.values[:, 4], np.ones(72))
    # Channel 5: delta_t (all observed -> 0.0)
    np.testing.assert_allclose(seq.values[:, 5], np.zeros(72))


def test_sequence_tensor_point_in_time_rejection() -> None:
    """Enforce strict point-in-time correctness: reject inputs where available_at_utc > T."""
    t_iss = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    loc_id = 1001

    timestamps = pd.date_range("2026-10-30 00:00:00", periods=72, freq="1h", tz="UTC")

    # Inject lookahead leak into observations
    leaked_available = [pd.Timestamp(t) + timedelta(minutes=30) for t in timestamps]
    leaked_available[-1] = pd.Timestamp(t_iss) + timedelta(hours=1)  # Leaked into future!

    obs_df_leaked = pd.DataFrame(
        {
            "location_id": np.full(72, loc_id),
            "ts_utc": timestamps,
            "pm25_ugm3": np.full(72, 100.0),
            "available_at_utc": leaked_available,
        }
    )

    cams_df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "cams_pm25": np.full(72, 80.0),
            "available_at_utc": pd.Timestamp(t_iss) - timedelta(hours=1),
        }
    )

    with pytest.raises(ValueError, match="Point-in-time violation"):
        build_sequence_tensor(
            location_id=loc_id,
            issuance_utc=t_iss,
            obs_df=obs_df_leaked,
            cams_df=cams_df,
        )


def test_target_completeness_rule_and_leakage_assertion() -> None:
    """Test 18/24h completeness rule for block targets and target leakage assertion."""
    t_iss = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    loc_id = 1001

    # 72 forecast hours [T, T+72h)
    timestamps = pd.date_range("2026-11-02 00:00:00", periods=72, freq="1h", tz="UTC")

    # Block 1 (h=24): 20 valid hours (>= 18) -> valid
    # Block 2 (h=48): 12 valid hours (< 18) -> invalid (completeness failure)
    # Block 3 (h=72): 24 valid hours (>= 18) -> valid
    pm25_vals = np.full(72, 150.0)
    # Set 12 hours of block 2 to NaN
    pm25_vals[24:36] = np.nan

    future_obs = pd.DataFrame(
        {
            "location_id": np.full(72, loc_id),
            "ts_utc": timestamps,
            "pm25_ugm3": pm25_vals,
        }
    )

    targets_df = block_targets(future_obs, issuance_utc=t_iss, horizons=(24, 48, 72))

    assert len(targets_df) == 1
    row = targets_df.iloc[0]

    # Block 1: Valid
    assert row["valid_h24"] is True or row["valid_h24"] == 1
    assert row["y_h24"] == pytest.approx(150.0)

    # Block 2: Invalid due to completeness (< 18 hours)
    assert row["valid_h48"] is False or row["valid_h48"] == 0
    assert np.isnan(row["y_h48"])

    # Block 3: Valid
    assert row["valid_h72"] is True or row["valid_h72"] == 1
    assert row["y_h72"] == pytest.approx(150.0)

    # Leakage assertion: feature matrix with target column y_h24 must raise ValueError
    leaked_features = pd.DataFrame({"f_met_vc_mean_h24": [100.0], "y_h24": [150.0]})
    with pytest.raises(ValueError, match="Target leakage detected"):
        assert_no_target_leakage(leaked_features, issuance_utc=t_iss)


def test_build_feature_table_and_parquet_roundtrip(tmp_path: Path) -> None:
    """Test end-to-end feature store table construction, schema validation, and PyArrow metadata roundtrip."""
    t_iss = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    loc_id = 1001

    # 1. Historical observations (72h)
    obs_ts = pd.date_range("2026-10-30 00:00:00", periods=72, freq="1h", tz="UTC")
    obs_df = pd.DataFrame(
        {
            "location_id": np.full(72, loc_id),
            "ts_utc": obs_ts,
            "pm25_ugm3": np.linspace(60.0, 180.0, 72),
            "available_at_utc": obs_ts + timedelta(minutes=30),
        }
    )

    # 2. CAMS data: lookback (72h) + forecast (72h) = 144h
    cams_ts = pd.date_range("2026-10-30 00:00:00", periods=144, freq="1h", tz="UTC")
    cams_df = pd.DataFrame(
        {
            "ts_utc": cams_ts,
            "cams_pm25": np.linspace(50.0, 200.0, 144),
            "aod550": np.full(144, 0.75),
            "blh_m": np.full(144, 400.0),
            "u10_ms": np.full(144, -2.5),
            "v10_ms": np.full(144, 0.5),
            "t2m_c": np.full(144, 20.0),
            "d2m_c": np.full(144, 12.0),
            "sp_hpa": np.full(144, 1012.0),
            "available_at_utc": pd.Timestamp(t_iss) - timedelta(hours=3),
        }
    )

    # 3. Fire exposures
    fire_df = pd.DataFrame(
        {
            "ts_utc": cams_ts,
            "exposure_e24": np.full(144, 15.0),
            "exposure_e48": np.full(144, 15.0),
            "exposure_e72": np.full(144, 15.0),
            "available_at_utc": pd.Timestamp(t_iss) - timedelta(hours=2),
        }
    )

    # 4. Registry
    registry_df = pd.DataFrame(
        {
            "location_id": [loc_id],
            "latitude": [31.5204],
            "longitude": [74.3587],
            "elevation": [217.0],
            "is_reference": [True],
        }
    )

    # 5. Future observations for target generation
    future_ts = pd.date_range("2026-11-02 00:00:00", periods=72, freq="1h", tz="UTC")
    future_obs_df = pd.DataFrame(
        {
            "location_id": np.full(72, loc_id),
            "ts_utc": future_ts,
            "pm25_ugm3": np.full(72, 160.0),
        }
    )

    # Build feature table
    feat_table = build_feature_table(
        domain="lahore",
        issuance_utc=t_iss,
        obs_df=obs_df,
        cams_df=cams_df,
        fire_exposure_df=fire_df,
        registry_df=registry_df,
        future_obs_df=future_obs_df,
        horizons=(24, 48, 72),
    )

    assert len(feat_table) == 1
    assert "location_id" in feat_table.columns
    assert "domain" in feat_table.columns
    assert "f_met_vc_mean_h24" in feat_table.columns
    assert "f_stat_lat" in feat_table.columns
    assert "y_h24" in feat_table.columns
    assert "valid_h24" in feat_table.columns

    # Save to partitioned parquet
    saved_path = save_feature_table(
        df=feat_table,
        base_dir=tmp_path,
    )
    assert saved_path.exists()

    # Load back and verify schema and PyArrow metadata
    loaded_df, meta = load_feature_table(
        domain="lahore",
        issuance_date="2026-11-02",
        base_dir=tmp_path,
    )

    assert len(loaded_df) == 1
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["feature_set_version"] == DEFAULT_FEATURE_SET_VERSION
    assert meta["domain"] == "lahore"
    assert meta["config_hash"] == compute_config_hash()
