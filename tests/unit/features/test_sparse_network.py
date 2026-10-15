"""Unit and property-based tests for sparse-network simulation, twin-aware sampler, IDW history, and modality dropout."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from smogsense.features.sparse_network import (
    apply_modality_dropout,
    cluster_stations,
    compute_idw_history,
    compute_pairwise_geodesic_distances,
    create_sparse_network_task,
    sample_available_stations,
)


@pytest.fixture
def sample_stations_df() -> pd.DataFrame:
    """Fixture with isolated singletons and co-located pairs/triads:

    - Stations 101 & 102: Co-located pair in Lahore (< 50 m apart).
    - Station 103: Isolated singleton ~2 km away.
    - Station 104: Isolated singleton ~15 km away.
    - Stations 105, 106, 107: Co-located triad in Amritsar (< 40 m apart).
    - Station 108: Distant station > 30 km away.
    """
    return pd.DataFrame(
        {
            "location_id": [101, 102, 103, 104, 105, 106, 107, 108],
            # 101 and 102 are ~22 meters apart
            "latitude": [31.5204, 31.5206, 31.5400, 31.6200, 31.6340, 31.6342, 31.6341, 31.9000],
            "longitude": [74.3587, 74.3587, 74.3700, 74.4500, 74.8723, 74.8723, 74.8725, 75.2000],
            "colocated_group_id": [
                "lahore_ref_pair",
                "lahore_ref_pair",
                "",
                "",
                "amritsar_cluster",
                "amritsar_cluster",
                "amritsar_cluster",
                "",
            ],
        }
    )


def test_cluster_stations_grouping(sample_stations_df: pd.DataFrame) -> None:
    """Test that co-located stations form atomic clusters."""
    clusters = cluster_stations(sample_stations_df, max_colocation_dist_m=50.0)

    # Convert to set of frozensets
    cluster_sets = {frozenset(members) for members in clusters.values()}

    # Pair {101, 102} must be a cluster
    assert frozenset({101, 102}) in cluster_sets
    # Triad {105, 106, 107} must be a cluster
    assert frozenset({105, 106, 107}) in cluster_sets
    # Singletons {103}, {104}, {108}
    assert frozenset({103}) in cluster_sets
    assert frozenset({104}) in cluster_sets
    assert frozenset({108}) in cluster_sets

    # Total stations across clusters must equal 8
    all_clustered = [stn for members in clusters.values() for stn in members]
    assert len(all_clustered) == 8
    assert set(all_clustered) == set(sample_stations_df["location_id"])


def test_pairwise_geodesic_distances(sample_stations_df: pd.DataFrame) -> None:
    """Test symmetric all-pairs geodesic distance matrix in km."""
    dist_df = compute_pairwise_geodesic_distances(sample_stations_df)

    assert dist_df.shape == (8, 8)
    assert list(dist_df.index) == list(sample_stations_df["location_id"])
    assert list(dist_df.columns) == list(sample_stations_df["location_id"])

    # Diagonal must be 0.0
    for sid in sample_stations_df["location_id"]:
        assert dist_df.loc[sid, sid] == 0.0

    # Symmetric: dist(i, j) == dist(j, i)
    np.testing.assert_allclose(dist_df.values, dist_df.values.T)

    # Stations 101 and 102 are ~22 meters apart (< 0.05 km)
    d_101_102 = float(dist_df.to_numpy()[0, 1])
    assert d_101_102 < 0.05
    assert d_101_102 > 0.01

    # Station 108 is > 30 km away from station 101
    d_101_108 = float(dist_df.to_numpy()[0, 7])
    assert d_101_108 > 30.0


def test_sample_available_stations_twin_awareness(sample_stations_df: pd.DataFrame) -> None:
    """Critical Invariant: Co-located twins/clusters are NEVER split between A and B."""
    clusters_dict = cluster_stations(sample_stations_df)
    clusters = [set(m) for m in clusters_dict.values()]

    for target_n in [0, 1, 2, 3, 5, 8]:
        for seed in range(20):
            avail, held_out, sampled_n = sample_available_stations(
                sample_stations_df, n=target_n, rng=seed
            )

            assert sampled_n == target_n
            # Disjoint and exhaustive
            assert avail.isdisjoint(held_out)
            assert avail | held_out == set(sample_stations_df["location_id"])

            # Invariant: For every atomic cluster c, either c subset A or c subset B
            for c in clusters:
                in_a = c.issubset(avail)
                in_b = c.issubset(held_out)
                assert in_a or in_b, (
                    f"Cluster {c} was split! In A: {c & avail}, In B: {c & held_out}"
                )

            if target_n == 0:
                assert len(avail) == 0
                assert len(held_out) == len(sample_stations_df)
            if target_n >= len(sample_stations_df):
                assert len(avail) == len(sample_stations_df)
                assert len(held_out) == 0


def test_sample_available_stations_reproducibility(sample_stations_df: pd.DataFrame) -> None:
    """Sampling is strictly deterministic given the same integer seed."""
    a1, b1, n1 = sample_available_stations(sample_stations_df, n=3, rng=42)
    a2, b2, n2 = sample_available_stations(sample_stations_df, n=3, rng=42)
    assert a1 == a2
    assert b1 == b2
    assert n1 == n2


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=30)
def test_hypothesis_twin_aware_random_sampling(seed: int) -> None:
    """Property-based test: across arbitrary random seeds, no cluster is ever split."""
    # Synthetic network with a pair, a triplet, and two singletons
    df = pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4, 5, 6, 7],
            "latitude": [31.5000, 31.5002, 31.5500, 31.5501, 31.5502, 31.6000, 31.7000],
            "longitude": [74.3000, 74.3000, 74.3500, 74.3500, 74.3502, 74.4000, 74.5000],
        }
    )
    clusters = [set(m) for m in cluster_stations(df).values()]
    avail, held_out, _ = sample_available_stations(df, n=None, rng=seed)

    assert avail.isdisjoint(held_out)
    assert avail | held_out == set(df["location_id"])
    for c in clusters:
        assert c.issubset(avail) or c.issubset(held_out)


def test_idw_history_colocated_singularity() -> None:
    """When target station is in A (or co-located <= 10m), IDW returns exact station history."""
    target_id = 101
    stns_df = pd.DataFrame(
        {
            "location_id": [101, 103],
            "latitude": [31.5204, 31.5500],
            "longitude": [74.3587, 74.3800],
        }
    )

    timestamps = pd.date_range("2026-11-01 00:00:00", periods=5, freq="1h", tz="UTC")
    obs_df = pd.DataFrame(
        {
            "location_id": [101] * 5 + [103] * 5,
            "ts_utc": list(timestamps) + list(timestamps),
            "pm25_ugm3": [100.0, 110.0, 120.0, 130.0, 140.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        }
    )

    # 101 is in available set
    idw_hist, dist_nearest, mask = compute_idw_history(
        target_location_id=target_id,
        available_station_ids={101, 103},
        obs_df=obs_df,
        stations_df=stns_df,
    )

    assert dist_nearest == pytest.approx(0.0, abs=1e-3)
    assert mask.all()
    # History must match 101 exactly (100, 110, 120, 130, 140), ignoring 103
    np.testing.assert_allclose(
        idw_hist.to_numpy(dtype=np.float64),
        np.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype=np.float64),
    )


def test_idw_history_symmetric_equal_distance() -> None:
    """Symmetric stations at equal distances receive equal weight, returning unweighted mean."""
    # Target at (31.5000, 74.3000)
    # Station 201 is ~5.5 km East
    # Station 202 is ~5.5 km West
    stns_df = pd.DataFrame(
        {
            "location_id": [999, 201, 202],
            "latitude": [31.5000, 31.5000, 31.5000],
            "longitude": [74.3000, 74.3580, 74.2420],
        }
    )

    ts = pd.date_range("2026-11-01 00:00:00", periods=3, freq="1h", tz="UTC")
    obs_df = pd.DataFrame(
        {
            "location_id": [201] * 3 + [202] * 3,
            "ts_utc": list(ts) + list(ts),
            "pm25_ugm3": [100.0, 120.0, 140.0, 200.0, 220.0, 240.0],
        }
    )

    idw_hist, dist_nearest, mask = compute_idw_history(
        target_location_id=999,
        available_station_ids={201, 202},
        obs_df=obs_df,
        stations_df=stns_df,
    )

    assert mask.all()
    assert dist_nearest == pytest.approx(5.5, abs=0.2)
    # Equal distance implies equal weight: mean(100, 200) = 150, mean(120, 220) = 170, mean(140, 240) = 190
    np.testing.assert_allclose(
        idw_hist.to_numpy(dtype=np.float64),
        np.array([150.0, 170.0, 190.0], dtype=np.float64),
        rtol=1e-2,
    )


def test_idw_history_distance_cutoff_25km() -> None:
    """Stations beyond 25 km are strictly excluded from IDW weighting."""
    # Target at (31.5000, 74.3000)
    # Station 301 is ~10 km away (included)
    # Station 302 is ~40 km away (excluded)
    stns_df = pd.DataFrame(
        {
            "location_id": [999, 301, 302],
            "latitude": [31.5000, 31.5900, 31.9000],
            "longitude": [74.3000, 74.3000, 74.3000],
        }
    )

    ts = pd.date_range("2026-11-01 00:00:00", periods=2, freq="1h", tz="UTC")
    obs_df = pd.DataFrame(
        {
            "location_id": [301] * 2 + [302] * 2,
            "ts_utc": list(ts) + list(ts),
            "pm25_ugm3": [80.0, 90.0, 500.0, 500.0],  # 302 has huge values that must be excluded
        }
    )

    idw_hist, dist_nearest, mask = compute_idw_history(
        target_location_id=999,
        available_station_ids={301, 302},
        obs_df=obs_df,
        stations_df=stns_df,
        max_radius_km=25.0,
    )

    assert mask.all()
    assert dist_nearest == pytest.approx(10.0, abs=1.0)
    # Only 301 is used; 302's 500 ug/m3 is completely ignored
    np.testing.assert_allclose(
        idw_hist.to_numpy(dtype=np.float64), np.array([80.0, 90.0], dtype=np.float64)
    )


def test_idw_history_sensor_free_n0() -> None:
    """Sensor-free mode (N = 0 or no station in A within 25 km): values 0.0, mask False, distance 25.0 km."""
    stns_df = pd.DataFrame(
        {
            "location_id": [999, 401],
            "latitude": [31.5000, 32.5000],  # 401 is ~110 km away
            "longitude": [74.3000, 74.3000],
        }
    )
    ts = pd.date_range("2026-11-01 00:00:00", periods=3, freq="1h", tz="UTC")
    obs_df = pd.DataFrame(
        {"location_id": [401] * 3, "ts_utc": ts, "pm25_ugm3": [100.0, 100.0, 100.0]}
    )

    # 1. N = 0 (empty A)
    idw_n0, dist_n0, mask_n0 = compute_idw_history(
        target_location_id=999,
        available_station_ids=set(),
        obs_df=obs_df,
        stations_df=stns_df,
    )
    assert dist_n0 == 25.0
    assert not mask_n0.any()
    np.testing.assert_allclose(
        idw_n0.to_numpy(dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )

    # 2. Available station exists but > 25 km away
    idw_far, dist_far, mask_far = compute_idw_history(
        target_location_id=999,
        available_station_ids={401},
        obs_df=obs_df,
        stations_df=stns_df,
    )
    assert dist_far > 100.0
    assert not mask_far.any()
    np.testing.assert_allclose(
        idw_far.to_numpy(dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )


def test_modality_dropout_eval_mode_invariant() -> None:
    """In eval_mode=True, zero dropouts occur regardless of probabilities."""
    channels = ["pm25_log1p", "cams_pm25_log1p", "aod550", "fire_exposure_e24", "delta_t"]
    vals = np.ones((72, 5), dtype=np.float32)
    masks = np.ones((72, 5), dtype=np.float32)

    out_vals, out_masks, flags = apply_modality_dropout(
        feature_values=vals,
        feature_masks=masks,
        channel_names=channels,
        pm25_dropout_p=1.0,  # 100% dropout probability
        fire_dropout_p=1.0,
        aod_dropout_p=1.0,
        eval_mode=True,  # Evaluation mode disables dropout
        rng=42,
    )

    assert not any(flags.values())
    np.testing.assert_array_equal(out_vals, vals)
    np.testing.assert_array_equal(out_masks, masks)


def test_modality_dropout_channels_isolation() -> None:
    """PM2.5 dropout masks ground PM2.5 history but NEVER alters cams_pm25."""
    channels = ["pm25_log1p", "cams_pm25_log1p", "aod550", "fire_exposure_e24", "delta_t"]
    vals = np.full((10, 5), 5.0, dtype=np.float32)
    masks = np.ones((10, 5), dtype=np.float32)

    # Force PM2.5 dropout only
    out_vals, out_masks, flags = apply_modality_dropout(
        feature_values=vals,
        feature_masks=masks,
        channel_names=channels,
        pm25_dropout_p=1.0,
        fire_dropout_p=0.0,
        aod_dropout_p=0.0,
        eval_mode=False,
        rng=123,
    )

    assert flags["dropped_pm25"] is True
    assert flags["dropped_fire"] is False
    assert flags["dropped_aod"] is False

    # pm25_log1p (col 0) must be 0.0 and masked
    np.testing.assert_allclose(out_vals[:, 0], 0.0)
    np.testing.assert_allclose(out_masks[:, 0], 0.0)

    # cams_pm25_log1p (col 1) must remain 5.0 and unmasked!
    np.testing.assert_allclose(out_vals[:, 1], 5.0)
    np.testing.assert_allclose(out_masks[:, 1], 1.0)

    # aod550 (col 2) and fire (col 3) remain intact
    np.testing.assert_allclose(out_vals[:, 2], 5.0)
    np.testing.assert_allclose(out_vals[:, 3], 5.0)

    # delta_t (col 4) must be reset to linear ramp 1, 2, ..., 10
    expected_delta = np.arange(1, 11, dtype=np.float32)
    np.testing.assert_allclose(out_vals[:, 4], expected_delta)


def test_modality_dropout_fire_and_aod_groups() -> None:
    """Fire and AOD group dropouts correctly zero respective groups."""
    channels = ["pm25_log1p", "cams_pm25_log1p", "aod550", "fire_exposure_e24"]
    vals = np.full((5, 4), 10.0, dtype=np.float32)
    masks = np.ones((5, 4), dtype=np.float32)

    # Drop fire and AOD, keep PM2.5
    out_vals, out_masks, flags = apply_modality_dropout(
        feature_values=vals,
        feature_masks=masks,
        channel_names=channels,
        pm25_dropout_p=0.0,
        fire_dropout_p=1.0,
        aod_dropout_p=1.0,
        eval_mode=False,
    )

    assert flags["dropped_pm25"] is False
    assert flags["dropped_fire"] is True
    assert flags["dropped_aod"] is True

    # PM2.5 intact
    np.testing.assert_allclose(out_vals[:, 0], 10.0)
    # CAMS intact
    np.testing.assert_allclose(out_vals[:, 1], 10.0)
    # AOD and Fire zeroed out
    np.testing.assert_allclose(out_vals[:, 2], 0.0)
    np.testing.assert_allclose(out_vals[:, 3], 0.0)
    np.testing.assert_allclose(out_masks[:, 2], 0.0)
    np.testing.assert_allclose(out_masks[:, 3], 0.0)


def test_create_sparse_network_task_72h_embargo() -> None:
    """Test 72-hour embargo and window chronology in task construction."""
    stns_df = pd.DataFrame(
        {
            "location_id": [101, 102, 103],
            "latitude": [31.5, 31.6, 31.7],
            "longitude": [74.3, 74.4, 74.5],
        }
    )
    ref_time = datetime(2026, 11, 20, 0, 0, tzinfo=UTC)

    # 14 days support, 7 days query, 72h embargo
    task = create_sparse_network_task(
        domain="delhi",
        season="2022-2023",
        stations_df=stns_df,
        obs_df=pd.DataFrame(),
        n=1,
        support_window_days=14,
        query_window_days=7,
        embargo_hours=72,
        reference_time_utc=ref_time,
    )

    assert task.domain == "delhi"
    assert task.season == "2022-2023"
    assert task.query_window_end_utc == ref_time
    assert task.query_window_start_utc == ref_time - timedelta(days=7)

    # Embargo between support_end and query_start
    embargo_delta = task.query_window_start_utc - task.support_window_end_utc
    assert embargo_delta.total_seconds() / 3600.0 >= 72.0
    assert embargo_delta == timedelta(hours=72)
    assert task.support_window_start_utc == task.support_window_end_utc - timedelta(days=14)


def test_sparse_network_task_condition_c7_adaptation_status() -> None:
    """Validate Condition C7 adaptation status invariants:

    - N = 0 -> "skipped_n0"
    - N >= 1 with < 5 valid days -> "skipped_insufficient_support"
    - N >= 1 with >= 5 valid days -> "eligible"
    """
    stns_df = pd.DataFrame(
        {
            "location_id": [101, 102],
            "latitude": [31.5, 31.6],
            "longitude": [74.3, 74.4],
        }
    )
    ref_time = datetime(2026, 11, 25, 0, 0, tzinfo=UTC)

    # Case 1: N = 0
    task_n0 = create_sparse_network_task(
        domain="lahore",
        season="2026-2027",
        stations_df=stns_df,
        obs_df=pd.DataFrame(),
        n=0,
        reference_time_utc=ref_time,
    )
    assert task_n0.n_available == 0
    assert task_n0.adaptation_status == "skipped_n0"

    # Support window for ref_time (2026-11-25):
    # qry: 2026-11-18 to 2026-11-25
    # embargo: 2026-11-15 to 2026-11-18 (72h)
    # sup: 2026-11-01 to 2026-11-15 (14 days)

    # Case 2: N = 1, but only 2 days of observations (< 5 days required)
    obs_2days = pd.DataFrame(
        {
            "location_id": [101] * 48,
            "ts_utc": pd.date_range("2026-11-02 00:00:00", periods=48, freq="1h", tz="UTC"),
            "pm25_ugm3": np.full(48, 120.0),
        }
    )
    task_insufficient = create_sparse_network_task(
        domain="lahore",
        season="2026-2027",
        stations_df=stns_df,
        obs_df=obs_2days,
        n=1,
        min_support_days=5,
        reference_time_utc=ref_time,
        rng=0,
    )
    assert task_insufficient.n_available >= 1
    assert task_insufficient.adaptation_status == "skipped_insufficient_support"

    # Case 3: N = 1, with 8 days of observations (>= 5 days required)
    obs_8days = pd.DataFrame(
        {
            "location_id": [101] * 192,
            "ts_utc": pd.date_range("2026-11-02 00:00:00", periods=192, freq="1h", tz="UTC"),
            "pm25_ugm3": np.full(192, 120.0),
        }
    )
    task_eligible = create_sparse_network_task(
        domain="lahore",
        season="2026-2027",
        stations_df=stns_df,
        obs_df=obs_8days,
        n=1,
        min_support_days=5,
        reference_time_utc=ref_time,
        rng=0,
    )
    assert task_eligible.n_available >= 1
    assert task_eligible.adaptation_status == "eligible"
