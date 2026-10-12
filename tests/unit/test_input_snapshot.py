"""smogsense.tests.unit.test_input_snapshot — Unit tests for point-in-time input snapshot and replay.

Tests:
1. Roundtrip integrity: build -> save -> load preserves exact DataFrames and metadata.
2. Point-in-time leakage prevention: rejects observations, CAMS, or fire with available_at > T.
3. Partition path compliance: writes to .state/inputs/{domain}/dt=YYYY-MM-DD.parquet.
4. Degradation handling: handles missing CAMS and sparse observations gracefully.
5. Direct Parquet compatibility: file is readable by standard pandas with input_type column.
6. CLI execution: tests 'smogsense features build' and 'smogsense features snapshot'.

Specification:
- docs/system-architecture.md §2 & §4
- docs/data-engineering.md §12
- docs/evaluation-strategy.md §9
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.features.snapshot import (
    InputSnapshot,
    build_input_snapshot,
    load_input_snapshot,
    save_input_snapshot,
)

runner = CliRunner()


def _dummy_obs_df(t0: datetime) -> pd.DataFrame:
    times = [t0 - timedelta(hours=i) for i in range(1, 25)]
    return pd.DataFrame(
        {
            "location_id": [1001] * 24,
            "sensor_id": [5005] * 24,
            "domain": ["lahore"] * 24,
            "ts_utc": times,
            "pm25_ugm3": [85.5] * 24,
            "pm25_raw_ugm3": [85.5] * 24,
            "rh_pct": [60.0] * 24,
            "temp_c": [22.0] * 24,
            "qc_flags": np.array([0] * 24, dtype=np.int32),
            "imputed": [False] * 24,
            "is_reference": [True] * 24,
            "provider": ["openaq"] * 24,
            "source": ["openaq_api"] * 24,
            "available_at_utc": [t - timedelta(hours=1) for t in times],
        }
    )


def _dummy_cams_df(t0: datetime) -> pd.DataFrame:
    targets = [t0 + timedelta(hours=i) for i in range(72)]
    return pd.DataFrame(
        {
            "location_id": ["1001"] * 72,
            "domain": ["lahore"] * 72,
            "target_hour_utc": targets,
            "pm25_ugm3": [75.0] * 72,
            "aod550": [0.45] * 72,
            "bld_h": [800.0] * 72,
            "u10": [2.5] * 72,
            "v10": [-1.2] * 72,
            "t2m": [295.15] * 72,
            "d2m": [285.15] * 72,
            "grid_distance_km": [12.4] * 72,
            "available_at_utc": [t0 - timedelta(hours=2)] * 72,
        }
    )


def _dummy_fire_df(t0: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "domain": ["lahore"] * 3,
            "target_hour_utc": [t0 - timedelta(hours=1)] * 3,
            "ring_id": ["r0_50", "r50_150", "r150_300"],
            "sector_id": ["E", "SE", "NE"],
            "fire_count_norm": [12.0, 5.0, 2.0],
            "frp_norm": [120.0, 45.0, 15.0],
            "platforms_active": [2, 2, 2],
            "available_at_utc": [t0 - timedelta(hours=1)] * 3,
        }
    )


# --- 1. Roundtrip Integrity ---


def test_snapshot_roundtrip_integrity(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    obs = _dummy_obs_df(t0)
    cams = _dummy_cams_df(t0)
    fire = _dummy_fire_df(t0)
    cams_base = t0 - timedelta(hours=12)

    snapshot = build_input_snapshot(
        issuance_utc=t0,
        domain="lahore",
        obs_df=obs,
        cams_df=cams,
        fire_exposure_df=fire,
        cams_base_time_utc=cams_base,
        cams_lead_offset_h=12.0,
        run_id="run_20261015_0000",
        config_hash="sha_config_123",
        n_available_stations=1,
    )

    saved_file = save_input_snapshot(snapshot, state_dir=tmp_path)
    assert saved_file.exists()

    loaded = load_input_snapshot("lahore", "2026-10-15", state_dir=tmp_path)
    assert isinstance(loaded, InputSnapshot)
    assert loaded.issuance_utc == t0
    assert loaded.domain == "lahore"
    assert loaded.run_id == "run_20261015_0000"
    assert loaded.config_hash == "sha_config_123"
    assert loaded.cams_lead_offset_h == 12.0
    assert loaded.cams_base_time_utc == cams_base
    assert loaded.n_available_stations == 1

    # Exact DataFrame equality assertions
    pd.testing.assert_frame_equal(loaded.obs_df, obs)
    pd.testing.assert_frame_equal(loaded.cams_df, cams)
    assert loaded.fire_exposure_df is not None
    pd.testing.assert_frame_equal(loaded.fire_exposure_df, fire)


# --- 2. As-Of Point-In-Time Leakage Prevention ---


def test_leakage_rejected_when_obs_future() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    bad_obs = _dummy_obs_df(t0)
    # Inject record with available_at_utc 1 hour after issuance T0
    bad_obs.loc[0, "available_at_utc"] = t0 + timedelta(hours=1)

    with pytest.raises(ValueError, match=r"Point-in-time leakage.*available_at_utc"):
        build_input_snapshot(issuance_utc=t0, domain="lahore", obs_df=bad_obs)


def test_leakage_rejected_when_obs_ts_future() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    bad_obs = _dummy_obs_df(t0).drop(columns=["available_at_utc"])
    bad_obs.loc[0, "ts_utc"] = t0 + timedelta(hours=1)

    with pytest.raises(ValueError, match=r"Point-in-time leakage.*ts_utc"):
        build_input_snapshot(issuance_utc=t0, domain="lahore", obs_df=bad_obs)


def test_leakage_rejected_when_cams_cycle_unavailable() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    # Today's 00Z cycle is only guaranteed by 10:00 UTC (B + 10h > T0)
    unavailable_cams_base = t0

    with pytest.raises(ValueError, match=r"Point-in-time leakage.*CAMS cycle base"):
        build_input_snapshot(
            issuance_utc=t0,
            domain="lahore",
            cams_base_time_utc=unavailable_cams_base,
        )


def test_leakage_rejected_when_fire_future() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    bad_fire = _dummy_fire_df(t0)
    bad_fire.loc[0, "available_at_utc"] = t0 + timedelta(hours=2)

    with pytest.raises(ValueError, match=r"Point-in-time leakage.*fire_exposure_df"):
        build_input_snapshot(
            issuance_utc=t0,
            domain="lahore",
            fire_exposure_df=bad_fire,
        )


# --- 3. Partition Path Compliance ---


def test_partition_path_compliance(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 18, 0, 0, tzinfo=UTC)
    snapshot = build_input_snapshot(issuance_utc=t0, domain="lahore")

    out = save_input_snapshot(snapshot, state_dir=tmp_path)
    expected_path = tmp_path / "inputs" / "lahore" / "dt=2026-10-18.parquet"
    assert out == expected_path
    assert expected_path.exists()


# --- 4. Degradation Handling ---


def test_degradation_handling_missing_cams(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    obs = _dummy_obs_df(t0)

    # Missing CAMS (None)
    snapshot = build_input_snapshot(
        issuance_utc=t0,
        domain="lahore",
        obs_df=obs,
        cams_df=None,
    )
    saved = save_input_snapshot(snapshot, state_dir=tmp_path)
    assert saved.exists()

    loaded = load_input_snapshot("lahore", "2026-10-15", state_dir=tmp_path)
    assert loaded.cams_df.empty
    assert len(loaded.obs_df) == len(obs)


def test_degradation_handling_empty_observations(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    cams = _dummy_cams_df(t0)
    cams_base = t0 - timedelta(hours=12)

    # Sensor outage (empty obs)
    snapshot = build_input_snapshot(
        issuance_utc=t0,
        domain="lahore",
        obs_df=pd.DataFrame(),
        cams_df=cams,
        cams_base_time_utc=cams_base,
    )
    save_input_snapshot(snapshot, state_dir=tmp_path)

    loaded = load_input_snapshot("lahore", "2026-10-15", state_dir=tmp_path)
    assert loaded.obs_df.empty
    assert len(loaded.cams_df) == len(cams)
    assert loaded.n_available_stations == 0


# --- 5. Direct Parquet Query Compatibility ---


def test_direct_parquet_compatibility(tmp_path: Path) -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    obs = _dummy_obs_df(t0)
    cams = _dummy_cams_df(t0)
    cams_base = t0 - timedelta(hours=12)

    snapshot = build_input_snapshot(
        issuance_utc=t0,
        domain="lahore",
        obs_df=obs,
        cams_df=cams,
        cams_base_time_utc=cams_base,
    )
    saved_file = save_input_snapshot(snapshot, state_dir=tmp_path)

    # Reading with standard pandas read_parquet without custom helper
    direct_df = pd.read_parquet(saved_file)
    assert not direct_df.empty
    assert "input_type" in direct_df.columns
    assert set(direct_df["input_type"].unique()) == {"observation", "cams"}
    assert direct_df["snapshot_domain"].iloc[0] == "lahore"


# --- 6. CLI Execution ---


def test_cli_features_build_and_snapshot(tmp_path: Path) -> None:
    t0_str = "2026-10-15"

    # 1. Run smogsense features build
    res_build = runner.invoke(
        app,
        [
            "features",
            "build",
            "--issuance",
            t0_str,
            "--domain",
            "lahore",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert res_build.exit_code == 0
    assert "Features build completed for lahore" in res_build.stdout
    assert "Persisted canonical input snapshot to" in res_build.stdout

    # Verify partition file was written
    target_parquet = tmp_path / "inputs" / "lahore" / f"dt={t0_str}.parquet"
    assert target_parquet.exists()

    # 2. Run smogsense features snapshot
    res_snap = runner.invoke(
        app,
        [
            "features",
            "snapshot",
            "--issuance",
            t0_str,
            "--domain",
            "lahore",
            "--state-dir",
            str(tmp_path),
        ],
    )
    assert res_snap.exit_code == 0
    assert "Input Snapshot for LAHORE" in res_snap.stdout
    assert "Observations:" in res_snap.stdout
    assert "CAMS Series:" in res_snap.stdout


def test_input_snapshot_helpers_and_methods(tmp_path: Path) -> None:
    from smogsense.features.snapshot import _ensure_utc, _normalize_date_str

    # Test _normalize_date_str
    assert _normalize_date_str("2026-10-15T12:00:00Z") == "2026-10-15"
    assert _normalize_date_str("2026-10-15 12:00:00") == "2026-10-15"
    assert _normalize_date_str(date(2026, 10, 15)) == "2026-10-15"
    assert _normalize_date_str(datetime(2026, 10, 15, tzinfo=UTC)) == "2026-10-15"
    with pytest.raises(TypeError, match="Expected str, date, or datetime"):
        _normalize_date_str(12345)  # type: ignore[arg-type]

    # Test _ensure_utc
    naive_dt = datetime(2026, 10, 15, 0, 0)
    utc_dt = _ensure_utc(naive_dt)
    assert utc_dt.tzinfo == UTC

    # Test InputSnapshot dict methods
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    snap = build_input_snapshot(issuance_utc=t0, domain="lahore")
    assert snap["domain"] == "lahore"
    assert snap.get("domain") == "lahore"
    assert snap.get("nonexistent", "fallback") == "fallback"
    d = snap.to_dict()
    assert d["domain"] == "lahore"
    assert "issuance_utc" in d


def test_input_snapshot_leakage_checks_coverage() -> None:
    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

    # 1. obs_df with ts_utc > t_ref and no available_at_utc
    obs_viol = pd.DataFrame(
        {
            "location_id": [1001],
            "ts_utc": [t0 + timedelta(hours=1)],
        }
    )
    with pytest.raises(ValueError, match="obs_df contains records with ts_utc"):
        build_input_snapshot(t0, obs_df=obs_viol)

    # 2. cams_df with available_at_utc > t_ref
    cams_viol = pd.DataFrame(
        {
            "location_id": ["1001"],
            "available_at_utc": [t0 + timedelta(hours=1)],
        }
    )
    with pytest.raises(ValueError, match="cams_df contains records with available_at_utc"):
        build_input_snapshot(t0, cams_df=cams_viol)

    # 3. fire_exposure_df with available_at_utc > t_ref
    fire_viol = pd.DataFrame(
        {
            "domain": ["lahore"],
            "available_at_utc": [t0 + timedelta(hours=1)],
        }
    )
    with pytest.raises(ValueError, match="fire_exposure_df contains records with available_at_utc"):
        build_input_snapshot(t0, fire_exposure_df=fire_viol)

    # 4. fire_exposure_df with acq_datetime_utc + 3h > t_ref
    fire_acq_viol = pd.DataFrame(
        {
            "domain": ["lahore"],
            "acq_datetime_utc": [t0 - timedelta(hours=1)],  # acq + 3h = t0 + 2h > t0
        }
    )
    with pytest.raises(ValueError, match="fire_exposure_df contains detections available at"):
        build_input_snapshot(t0, fire_exposure_df=fire_acq_viol)

    # 5. cams_base_time_utc + 10h > t_ref
    with pytest.raises(ValueError, match=r"CAMS cycle base .* available at .* > issuance_utc"):
        build_input_snapshot(t0, cams_base_time_utc=t0 - timedelta(hours=5))


def test_input_snapshot_load_save_edge_cases(tmp_path: Path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    t0 = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    snap = build_input_snapshot(issuance_utc=t0, domain="lahore")

    # 1. Save with custom output_path
    custom_path = tmp_path / "custom_snap.parquet"
    saved_path = save_input_snapshot(snap, output_path=custom_path)
    assert saved_path == custom_path
    assert custom_path.exists()

    # 2. Load with custom snapshot_path
    loaded = load_input_snapshot("lahore", t0, snapshot_path=custom_path)
    assert loaded.domain == "lahore"
    assert loaded.issuance_utc == t0

    # 3. Load non-existent file
    with pytest.raises(FileNotFoundError, match="Input snapshot not found"):
        load_input_snapshot("lahore", t0, snapshot_path=tmp_path / "missing.parquet")

    # 4. Fallback table unpacking when metadata buffers are empty
    # Create parquet table with combined rows and empty buffer metadata
    df = pd.DataFrame(
        {
            "input_type": ["observation", "cams", "fire_exposure"],
            "location_id": ["1001", "1001", "none"],
            "pm25_ugm3": [45.0, 50.0, 0.0],
            "snapshot_issuance_utc": [t0, t0, t0],
            "snapshot_domain": ["lahore", "lahore", "lahore"],
        }
    )
    table = pa.Table.from_pandas(df)
    meta = {b"smogsense_meta": b'{"domain": "lahore", "run_id": "test"}'}
    table = table.replace_schema_metadata(meta)
    fallback_path = tmp_path / "fallback.parquet"
    pq.write_table(table, fallback_path)

    loaded_fb = load_input_snapshot("lahore", t0, snapshot_path=fallback_path)
    assert loaded_fb.domain == "lahore"
    assert not loaded_fb.obs_df.empty
    assert not loaded_fb.cams_df.empty
    assert loaded_fb.fire_exposure_df is not None and not loaded_fb.fire_exposure_df.empty
