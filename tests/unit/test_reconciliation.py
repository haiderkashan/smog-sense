"""Unit tests for archive vs API reconciliation (Task P1-20).

Specification:
- docs/data-engineering.md §11 (Temporal alignment & provisional-vs-settled truth)
- docs/PRD.md FR-37 (Observation capture and revision policy)
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.pipeline.reconciliation import (
    reconcile_api_and_archive,
    run_reconciliation,
)
from smogsense.pipeline.seed_history import _load_archive_history

runner = CliRunner()


def test_reconcile_identical_observations() -> None:
    base_t = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_t + timedelta(hours=i) for i in range(10)]

    api_df = pd.DataFrame(
        {
            "sensor_id": [101] * 10,
            "ts_utc": timestamps,
            "pm25_ugm3": [50.0 + i for i in range(10)],
        }
    )
    archive_df = api_df.copy()

    stats = reconcile_api_and_archive(api_df, archive_df, tolerance=0.1)

    assert stats["status"] == "reconciled"
    assert stats["overlap_count"] == 10
    assert stats["api_rows"] == 10
    assert stats["archive_rows"] == 10
    assert stats["mean_abs_diff"] == 0.0
    assert stats["pct_identical"] == 100.0
    assert stats["revisions_count"] == 0
    assert stats["max_diff"] == 0.0
    assert stats["added_timestamps_count"] == 0
    assert stats["dropped_timestamps_count"] == 0


def test_reconcile_with_revisions() -> None:
    base_t = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_t + timedelta(hours=i) for i in range(5)]

    # API values: 50.0, 50.0, 50.0, 50.0, 50.0
    api_df = pd.DataFrame(
        {
            "location_id": [201] * 5,
            "ts_utc": timestamps,
            "pm25_ugm3": [50.0] * 5,
        }
    )
    # Archive values:
    # 0: 50.0 (identical, diff 0.0)
    # 1: 50.05 (within 0.1 tolerance, diff 0.05)
    # 2: 55.0 (revised, diff 5.0)
    # 3: 42.0 (revised, diff 8.0)
    # 4: 50.0 (identical, diff 0.0)
    archive_df = pd.DataFrame(
        {
            "location_id": [201] * 5,
            "ts_utc": timestamps,
            "pm25_ugm3": [50.0, 50.05, 55.0, 42.0, 50.0],
        }
    )

    stats = reconcile_api_and_archive(api_df, archive_df, tolerance=0.1)

    assert stats["status"] == "reconciled"
    assert stats["overlap_count"] == 5
    assert stats["revisions_count"] == 2
    assert stats["pct_identical"] == 60.0  # 3 out of 5
    assert stats["max_diff"] == 8.0
    # Mean abs diff = (0 + 0.05 + 5 + 8 + 0) / 5 = 13.05 / 5 = 2.61
    assert stats["mean_abs_diff"] == 2.61


def test_reconcile_disjoint_and_added_dropped() -> None:
    base_t = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)

    # API has hours 0, 1, 2
    api_df = pd.DataFrame(
        {
            "sensor_id": [101, 101, 101],
            "ts_utc": [base_t, base_t + timedelta(hours=1), base_t + timedelta(hours=2)],
            "pm25_ugm3": [40.0, 41.0, 42.0],
        }
    )
    # Archive has hours 1, 2, 3, 4
    archive_df = pd.DataFrame(
        {
            "sensor_id": [101, 101, 101, 101],
            "ts_utc": [
                base_t + timedelta(hours=1),
                base_t + timedelta(hours=2),
                base_t + timedelta(hours=3),
                base_t + timedelta(hours=4),
            ],
            "pm25_ugm3": [41.0, 42.0, 43.0, 44.0],
        }
    )

    stats = reconcile_api_and_archive(api_df, archive_df, tolerance=0.1)

    assert stats["overlap_count"] == 2  # hours 1 and 2
    assert stats["added_timestamps_count"] == 2  # hours 3 and 4
    assert stats["dropped_timestamps_count"] == 1  # hour 0
    assert stats["pct_identical"] == 100.0


def test_reconcile_empty_and_missing_keys() -> None:
    empty = pd.DataFrame()
    stats_both_empty = reconcile_api_and_archive(empty, empty)
    assert stats_both_empty["status"] == "empty_input"
    assert stats_both_empty["overlap_count"] == 0

    valid = pd.DataFrame({"sensor_id": [1], "ts_utc": [datetime.now(UTC)], "pm25_ugm3": [30.0]})
    stats_one_empty = reconcile_api_and_archive(valid, empty)
    assert stats_one_empty["status"] == "no_overlap"
    assert stats_one_empty["dropped_timestamps_count"] == 1

    no_keys_df = pd.DataFrame({"random_col": [1, 2, 3]})
    stats_bad_keys = reconcile_api_and_archive(no_keys_df, no_keys_df)
    assert stats_bad_keys["status"] == "missing_keys"


def test_run_reconciliation_with_files(tmp_path: Path) -> None:
    base_t = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_t + timedelta(hours=i) for i in range(4)]

    api_df = pd.DataFrame(
        {
            "sensor_id": [10] * 4,
            "ts_utc": timestamps,
            "pm25_ugm3": [40.0, 45.0, 50.0, 55.0],
        }
    )
    archive_df = pd.DataFrame(
        {
            "sensor_id": [10] * 4,
            "ts_utc": timestamps,
            "pm25_ugm3": [40.0, 45.0, 52.0, 55.0],
        }
    )

    api_path = tmp_path / "api_obs.parquet"
    archive_path = tmp_path / "archive_obs.parquet"
    report_path = tmp_path / "report.json"

    api_df.to_parquet(api_path, index=False)
    archive_df.to_parquet(archive_path, index=False)

    stats = run_reconciliation(
        domain="lahore",
        tolerance=0.1,
        api_path=api_path,
        archive_path=archive_path,
        output_report=report_path,
    )

    assert stats["status"] == "reconciled"
    assert stats["overlap_count"] == 4
    assert stats["revisions_count"] == 1
    assert report_path.exists()

    with report_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["revisions_count"] == 1
    assert loaded["domain"] == "lahore"


def test_cli_reconcile_archive_api(tmp_path: Path) -> None:
    base_t = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    timestamps = [base_t + timedelta(hours=i) for i in range(3)]

    api_df = pd.DataFrame(
        {
            "sensor_id": [55] * 3,
            "ts_utc": timestamps,
            "pm25_ugm3": [25.0, 26.0, 27.0],
        }
    )
    archive_df = api_df.copy()

    api_file = tmp_path / "api.parquet"
    archive_file = tmp_path / "archive.parquet"
    api_df.to_parquet(api_file, index=False)
    archive_df.to_parquet(archive_file, index=False)

    result = runner.invoke(
        app,
        [
            "reconcile",
            "archive-api",
            "--domain",
            "lahore",
            "--tolerance",
            "0.1",
            "--api-path",
            str(api_file),
            "--archive-path",
            str(archive_file),
        ],
    )
    assert result.exit_code == 0
    assert "Reconciliation (lahore): reconciled" in result.stdout
    assert "Overlapping observations: 3" in result.stdout
    assert "Identical (<= 0.1 ug/m3): 100.00%" in result.stdout


def test_seed_history_archive_loader(tmp_path: Path) -> None:
    start_utc = datetime(2026, 8, 20, 0, 0, tzinfo=UTC)
    cutoff = datetime(2026, 8, 25, 0, 0, tzinfo=UTC)

    # File with rows before, within, and after cutoff
    df = pd.DataFrame(
        {
            "location_id": [1, 1, 1],
            "sensor_id": [10, 10, 10],
            "ts_utc": [
                datetime(2026, 8, 19, 0, 0, tzinfo=UTC),  # Before start
                datetime(2026, 8, 22, 12, 0, tzinfo=UTC),  # Within window
                datetime(2026, 8, 26, 0, 0, tzinfo=UTC),  # After cutoff
            ],
            "pm25_ugm3": [30.0, 35.0, 40.0],
        }
    )
    parquet_path = tmp_path / "pilot.parquet"
    df.to_parquet(parquet_path, index=False)

    loaded = _load_archive_history("lahore", start_utc, cutoff, candidate_dirs=[tmp_path])
    assert len(loaded) == 1
    assert loaded.iloc[0]["pm25_ugm3"] == 35.0
