"""smogsense.tests.unit.test_backfill — Unit tests for Historical Backfill Engine (P1-13).

Specification:
- roadmap.md Task P1-13
- docs/data-engineering.md §4, §7 & §9
- .github/workflows/historical-backfill.yml
"""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import respx
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.pipeline.backfill import (
    backfill_cams_month,
    backfill_era5_month,
    backfill_firms_month,
    backfill_openaq_month,
    create_empty_cams_df,
    create_empty_fire_df,
    get_domain_locations,
    month_ranges,
    run_backfill,
)


def _make_sample_archive_gz() -> bytes:
    csv_text = (
        "location_id,sensors_id,datetime,parameter,value,provider,is_monitor\n"
        "1001,5001,2024-01-15T00:00:00Z,pm25,55.0,EPA_Punjab,true\n"
        "1001,5002,2024-01-15T00:00:00Z,relativehumidity,70.0,EPA_Punjab,true\n"
        "1001,5003,2024-01-15T00:00:00Z,temperature,18.0,EPA_Punjab,true\n"
    )
    return gzip.compress(csv_text.encode("utf-8"))


def test_month_ranges_single_month() -> None:
    ranges = month_ranges("2024-05-10", "2024-05-20")
    assert len(ranges) == 1
    year, month, start_d, end_d = ranges[0]
    assert year == 2024
    assert month == 5
    assert start_d == date(2024, 5, 10)
    assert end_d == date(2024, 5, 20)


def test_month_ranges_multi_month_cross_year() -> None:
    ranges = month_ranges("2023-11-15", "2024-02-10")
    assert len(ranges) == 4
    # Nov 2023
    assert ranges[0] == (2023, 11, date(2023, 11, 15), date(2023, 11, 30))
    # Dec 2023
    assert ranges[1] == (2023, 12, date(2023, 12, 1), date(2023, 12, 31))
    # Jan 2024
    assert ranges[2] == (2024, 1, date(2024, 1, 1), date(2024, 1, 31))
    # Feb 2024 (leap year)
    assert ranges[3] == (2024, 2, date(2024, 2, 1), date(2024, 2, 10))


def test_month_ranges_invalid_order() -> None:
    with pytest.raises(ValueError, match=r"start date .* must be <= end date"):
        month_ranges("2024-05-20", "2024-05-10")


def test_get_domain_locations() -> None:
    delhi_locs = get_domain_locations("delhi")
    assert isinstance(delhi_locs, list)
    assert len(delhi_locs) > 0

    lahore_locs = get_domain_locations("lahore")
    assert isinstance(lahore_locs, list)
    assert len(lahore_locs) > 0


def test_empty_schemas_validity() -> None:
    cams_empty = create_empty_cams_df("delhi")
    assert cams_empty.empty
    assert "point_id" in cams_empty.columns
    assert "pm25_ugm3" in cams_empty.columns

    fire_empty = create_empty_fire_df("lahore")
    assert fire_empty.empty
    assert "fire_count_r0_50" in fire_empty.columns
    assert "exposure_e24" in fire_empty.columns


@respx.mock
def test_backfill_openaq_month_with_data(tmp_path: Path) -> None:
    respx.get(url__regex=r"https://openaq-data-archive\.s3\.amazonaws\.com/.*").respond(
        status_code=200,
        content=_make_sample_archive_gz(),
    )

    dest_file = tmp_path / "openaq-archive_lahore_2024_01.parquet"
    res = backfill_openaq_month(
        domain="lahore",
        year=2024,
        month=1,
        start_d=date(2024, 1, 15),
        end_d=date(2024, 1, 15),
        dest_file=dest_file,
    )

    assert res["rows"] >= 1
    assert dest_file.exists()
    saved_df = pd.read_parquet(dest_file)
    assert len(saved_df) >= 1
    assert "pm25_ugm3" in saved_df.columns
    assert "sensor_id" in saved_df.columns


@respx.mock
def test_backfill_openaq_month_offline_404(tmp_path: Path) -> None:
    respx.get(url__regex=r"https://openaq-data-archive\.s3\.amazonaws\.com/.*").respond(
        status_code=404,
    )

    dest_file = tmp_path / "openaq-archive_lahore_2024_01.parquet"
    res = backfill_openaq_month(
        domain="lahore",
        year=2024,
        month=1,
        start_d=date(2024, 1, 1),
        end_d=date(2024, 1, 2),
        dest_file=dest_file,
    )

    assert res["rows"] == 0
    assert dest_file.exists()
    saved_df = pd.read_parquet(dest_file)
    assert len(saved_df) == 0


def test_backfill_cams_and_era5_month(tmp_path: Path) -> None:
    cams_file = tmp_path / "cams_lahore_2024_01.parquet"
    res_cams = backfill_cams_month(
        "lahore", 2024, 1, date(2024, 1, 1), date(2024, 1, 31), cams_file
    )
    assert cams_file.exists()
    assert res_cams["source"] == "cams"

    era5_file = tmp_path / "era5_delhi_2024_01.parquet"
    res_era5 = backfill_era5_month("delhi", 2024, 1, date(2024, 1, 1), date(2024, 1, 31), era5_file)
    assert era5_file.exists()
    assert res_era5["source"] == "era5"


def test_backfill_firms_month_empty(tmp_path: Path) -> None:
    firms_file = tmp_path / "firms_lahore_2024_01.parquet"
    res_firms = backfill_firms_month(
        "lahore", 2024, 1, date(2024, 1, 1), date(2024, 1, 5), firms_file
    )
    assert firms_file.exists()
    assert res_firms["source"] == "firms"
    df = pd.read_parquet(firms_file)
    assert "fire_count_r0_50" in df.columns


def test_run_backfill_resumption_and_idempotency(tmp_path: Path) -> None:
    out_dir = tmp_path / "backfill_out"
    out_dir.mkdir(parents=True)

    # Pre-create a partition
    pre_file = out_dir / "cams_lahore_2024_01.parquet"
    create_empty_cams_df("lahore").to_parquet(pre_file, index=False)

    res = run_backfill(
        source="cams",
        domain="lahore",
        start="2024-01-01",
        end="2024-02-15",
        output_dir=out_dir,
    )

    assert res["status"] == "complete"
    assert res["total_partitions"] == 2  # Jan and Feb
    assert res["partitions_skipped"] == 1  # Jan was skipped
    assert res["partitions_completed"] == 1  # Feb was created

    # Run again - both should now be skipped
    res2 = run_backfill(
        source="cams",
        domain="lahore",
        start="2024-01-01",
        end="2024-02-15",
        output_dir=out_dir,
    )
    assert res2["partitions_skipped"] == 2
    assert res2["partitions_completed"] == 0


def test_run_backfill_wall_clock_budget_protection(tmp_path: Path) -> None:
    out_dir = tmp_path / "budget_out"

    # Set elapsed time to exceed wall clock limit immediately
    with patch("time.monotonic", side_effect=[0.0, 400.0, 500.0, 600.0]):
        res = run_backfill(
            source="cams",
            domain="lahore",
            start="2024-01-01",
            end="2024-03-31",
            max_wall_minutes=5,  # 300 seconds budget; time.monotonic returns 400s
            output_dir=out_dir,
        )

    assert res["status"] == "stopped_early_wall_clock"
    assert res["partitions_completed"] == 0


def test_run_backfill_source_and_domain_validation() -> None:
    with pytest.raises(ValueError, match="Invalid source 'invalid_src'"):
        run_backfill(source="invalid_src")

    with pytest.raises(ValueError, match="Invalid domain 'invalid_dom'"):
        run_backfill(domain="invalid_dom")


@respx.mock
def test_run_backfill_all_sources_both_domains(tmp_path: Path) -> None:
    respx.get(url__regex=r"https://openaq-data-archive\.s3\.amazonaws\.com/.*").respond(
        status_code=404,
    )
    out_dir = tmp_path / "all_out"

    res = run_backfill(
        source="all",
        domain="both",
        start="2024-01-10",
        end="2024-01-20",
        output_dir=out_dir,
    )

    assert res["status"] == "complete"
    # 3 sources (openaq-archive, cams, firms) * 2 domains (delhi, lahore) * 1 month = 6 partitions
    assert res["total_partitions"] == 6
    assert res["partitions_completed"] == 6
    assert res["partitions_skipped"] == 0

    # Verify all 6 files exist
    expected_files = [
        "openaq-archive_delhi_2024_01.parquet",
        "openaq-archive_lahore_2024_01.parquet",
        "cams_delhi_2024_01.parquet",
        "cams_lahore_2024_01.parquet",
        "firms_delhi_2024_01.parquet",
        "firms_lahore_2024_01.parquet",
    ]
    for ef in expected_files:
        assert (out_dir / ef).exists()


def test_cli_backfill_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "backfill",
            "--source",
            "cams",
            "--domain",
            "lahore",
            "--start",
            "2024-01-01",
            "--end",
            "2024-01-31",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Historical backfill status: complete" in result.output
    assert (tmp_path / "cams_lahore_2024_01.parquet").exists()
