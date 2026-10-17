"""tests.unit.features.test_backfill — Comprehensive unit and property tests for feature store backfill.

Validates:
1. Multi-season identification and interval discovery.
2. Chronological season partitioning into Blocks E / V / T (Delhi) and 'eval' (Lahore).
3. Season-blocked cross-fitting fold generation for Block E (Condition C1, C2).
4. 72-hour temporal embargo and boundary purging.
5. Deterministic OOF partition hashing.
6. Point-in-time correctness (Principle P3) during historical slicing.
7. End-to-end backfill execution, Parquet serialization with PyArrow metadata, and Pandera schema validation.
8. Typer CLI command invocation.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.features.backfill import (
    BackfillResult,
    compute_oof_partition_hash,
    get_season,
    get_season_date_range,
    get_seasons_for_interval,
    is_boundary_purged,
    partition_block_e_folds,
    partition_seasons,
    run_feature_backfill,
)
from smogsense.features.store import (
    DEFAULT_FEATURE_SET_VERSION,
    SCHEMA_VERSION,
    compute_config_hash,
    load_feature_table,
)


def test_get_season() -> None:
    """Test season string detection across various months and years."""
    # Autumn/Winter (smog season)
    assert get_season("2023-10-15") == "2023-2024"
    assert get_season("2023-11-01") == "2023-2024"
    assert get_season("2023-12-31") == "2023-2024"
    assert get_season("2024-01-01") == "2023-2024"
    assert get_season("2024-02-28") == "2023-2024"

    # Mid-year boundary
    assert get_season("2024-06-30") == "2023-2024"
    assert get_season("2024-07-01") == "2024-2025"

    # Earlier historical seasons
    assert get_season(date(2018, 10, 1)) == "2018-2019"
    assert get_season(datetime(2021, 1, 15, tzinfo=UTC)) == "2020-2021"


def test_get_season_date_range() -> None:
    """Test standard date range calculation including leap year handling."""
    # Non-leap year (2023)
    start_23, end_23 = get_season_date_range("2022-2023")
    assert start_23 == datetime(2022, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert end_23 == datetime(2023, 2, 28, 23, 59, 59, tzinfo=UTC)

    # Leap year (2024)
    start_24, end_24 = get_season_date_range("2023-2024")
    assert start_24 == datetime(2023, 10, 1, 0, 0, 0, tzinfo=UTC)
    assert end_24 == datetime(2024, 2, 29, 23, 59, 59, tzinfo=UTC)


def test_get_seasons_for_interval() -> None:
    """Test discovery of all seasons spanning an arbitrary date interval."""
    seasons = get_seasons_for_interval("2021-11-01", "2024-01-15")
    assert seasons == ["2021-2022", "2022-2023", "2023-2024"]

    # Single season span
    single = get_seasons_for_interval("2023-11-01", "2023-11-30")
    assert single == ["2023-2024"]


def test_partition_seasons_delhi() -> None:
    """Test Delhi chronological season partitioning into Blocks E / V / T (ADR-024)."""
    # 8 seasons (2018..2026)
    all_seasons = [
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023",
        "2023-2024",
        "2024-2025",
        "2025-2026",
    ]
    parts = partition_seasons(all_seasons, domain="delhi")

    # Oldest 6 in E, 2024-2025 in V, 2025-2026 in T
    for s in all_seasons[:6]:
        assert parts[s] == "E"
    assert parts["2024-2025"] == "V"
    assert parts["2025-2026"] == "T"

    # 4 seasons: 2 E, 1 V, 1 T
    four_seasons = ["2020-2021", "2021-2022", "2022-2023", "2023-2024"]
    p4 = partition_seasons(four_seasons, domain="delhi")
    assert p4["2020-2021"] == "E"
    assert p4["2021-2022"] == "E"
    assert p4["2022-2023"] == "V"
    assert p4["2023-2024"] == "T"

    # 3 seasons: 1 E, 1 V, 1 T
    three_seasons = ["2021-2022", "2022-2023", "2023-2024"]
    p3 = partition_seasons(three_seasons, domain="delhi")
    assert p3["2021-2022"] == "E"
    assert p3["2022-2023"] == "V"
    assert p3["2023-2024"] == "T"


def test_partition_seasons_lahore() -> None:
    """Test Lahore seasons mapped to 'eval' split."""
    lahore_seasons = ["2022-2023", "2023-2024", "2024-2025"]
    parts = partition_seasons(lahore_seasons, domain="lahore")
    assert all(sp == "eval" for sp in parts.values())


def test_partition_block_e_folds() -> None:
    """Test contiguous K-fold assignment along seasonal boundaries for Block E."""
    seasons_e = ["2018-2019", "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"]
    folds = partition_block_e_folds(seasons_e)

    assert len(folds) == 6
    # K = min(3, 6) = 3 folds
    fold_ids = set(folds.values())
    assert fold_ids == {0, 1, 2}

    # Verify contiguous partition
    assert folds["2018-2019"] == 0
    assert folds["2019-2020"] == 0
    assert folds["2020-2021"] == 1
    assert folds["2021-2022"] == 1
    assert folds["2022-2023"] == 2
    assert folds["2023-2024"] == 2


def test_embargo_and_boundary_purging() -> None:
    """Test 72-hour temporal embargo and boundary purging rules."""
    w_start = datetime(2023, 10, 1, 0, 0, 0, tzinfo=UTC)
    w_end = datetime(2024, 2, 29, 23, 59, 59, tzinfo=UTC)

    # 1. Inside first 72h -> PURGED (lookback would cross prior boundary)
    assert is_boundary_purged(w_start + timedelta(hours=12), w_start, w_end, embargo_hours=72.0)
    assert is_boundary_purged(w_start + timedelta(hours=71), w_start, w_end, embargo_hours=72.0)

    # 2. At or after 72h -> ELIGIBLE
    assert not is_boundary_purged(w_start + timedelta(hours=72), w_start, w_end, embargo_hours=72.0)
    assert not is_boundary_purged(w_start + timedelta(days=10), w_start, w_end, embargo_hours=72.0)

    # 3. Near the end within 72h of w_end -> PURGED (target horizon crosses subsequent boundary)
    assert is_boundary_purged(w_end - timedelta(hours=24), w_start, w_end, embargo_hours=72.0)


def test_oof_partition_hash_deterministic() -> None:
    """Test deterministic OOF partition hash calculation."""
    # Empty DataFrame gives deterministic empty hash
    empty_df = pd.DataFrame()
    h_empty = compute_oof_partition_hash(empty_df)
    assert h_empty == hashlib.sha256(b"none").hexdigest()

    # Create dummy fold assignments
    df1 = pd.DataFrame(
        {
            "location_id": [7001, 7002, 7001],
            "issuance_utc": [
                pd.Timestamp("2023-10-15 00:00:00", tz="UTC"),
                pd.Timestamp("2023-10-15 00:00:00", tz="UTC"),
                pd.Timestamp("2023-10-15 06:00:00", tz="UTC"),
            ],
            "fold_id": [0, 0, 0],
        }
    )

    # Permuted rows should produce identical hash
    df2 = df1.sample(frac=1.0, random_state=42).reset_index(drop=True)

    hash1 = compute_oof_partition_hash(df1)
    hash2 = compute_oof_partition_hash(df2)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Changing fold_id changes the hash
    df3 = df1.copy()
    df3.loc[0, "fold_id"] = 1
    hash3 = compute_oof_partition_hash(df3)
    assert hash3 != hash1


def test_point_in_time_guarantee_backfill() -> None:
    """Verify that backfill enforces available_at_utc <= T on past covariates."""
    now = datetime(2023, 11, 10, 0, 0, 0, tzinfo=UTC)

    # Leakage frame: observation with available_at_utc in the future
    obs_leaked = pd.DataFrame(
        {
            "location_id": [7001],
            "ts_utc": [now - timedelta(hours=2)],
            "available_at_utc": [now + timedelta(hours=1)],  # FUTURE LEAKAGE!
            "pm25_ugm3": [100.0],
        }
    )

    cams_clean = pd.DataFrame(
        {
            "point_id": ["stn_7001"],
            "location_id": [7001],
            "domain": ["delhi"],
            "base_time_utc": [now - timedelta(hours=12)],
            "valid_time_utc": [now - timedelta(hours=2)],
            "ts_utc": [now - timedelta(hours=2)],
            "available_at_utc": [now - timedelta(hours=2)],
            "lead_h": [10],
            "pm25_ugm3": [80.0],
            "cams_pm25": [80.0],
        }
    )

    with pytest.raises(ValueError, match="Point-in-time violation"):
        from smogsense.features.store import build_feature_table

        build_feature_table(
            domain="delhi",
            issuance_utc=now,
            obs_df=obs_leaked,
            cams_df=cams_clean,
        )


def test_run_feature_backfill_end_to_end(tmp_path: Path) -> None:
    """Test end-to-end multi-season feature store backfill execution and validation."""
    start_date = "2023-10-05"
    end_date = "2023-10-07"

    result: BackfillResult = run_feature_backfill(
        domain="delhi",
        start_date=start_date,
        end_date=end_date,
        split="E",
        out_dir=tmp_path,
        lattice_hours=(0, 12),
        embargo_hours=72.0,
        verify_schema=True,
        write_parquet=True,
    )

    assert result.domain == "delhi"
    assert "2023-2024" in result.seasons
    assert result.total_samples > 0
    assert len(result.partitions_written) > 0

    # Load back the saved partition and verify schema and PyArrow metadata
    first_path = result.partitions_written[0]
    assert first_path.exists()

    df_loaded, meta = load_feature_table(
        domain="delhi",
        issuance_date="2023-10-05",
        base_dir=tmp_path,
        season="2023-2024",
    )

    assert not df_loaded.empty
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["feature_set_version"] == DEFAULT_FEATURE_SET_VERSION
    assert meta["config_hash"] == compute_config_hash()
    assert meta["domain"] == "delhi"
    assert meta["source_split"] == "E"
    assert meta["season"] == "2023-2024"
    assert meta["oof_partition_hash"] == result.oof_partition_hash

    # Check required columns from schema
    assert "domain" in df_loaded.columns
    assert "location_id" in df_loaded.columns
    assert "issuance_utc" in df_loaded.columns
    assert "y_h24" in df_loaded.columns
    assert "valid_h24" in df_loaded.columns
    assert "f_cams_lead_offset_h" in df_loaded.columns
    assert "f_cams_bias_trailing_7d" in df_loaded.columns
    assert "fold_id" in df_loaded.columns


def test_cli_features_backfill(tmp_path: Path) -> None:
    """Test CLI command `smogsense features backfill`."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "features",
            "backfill",
            "--domain",
            "delhi",
            "--start",
            "2023-10-05",
            "--end",
            "2023-10-06",
            "--split",
            "E",
            "--out-dir",
            str(tmp_path),
            "--verify-schema",
        ],
    )

    assert result.exit_code == 0
    assert "Starting feature store backfill for domain: DELHI" in result.stdout
    assert "Feature backfill complete for DELHI" in result.stdout
    assert "OOF Partition Hash:" in result.stdout
