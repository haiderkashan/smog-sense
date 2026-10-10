"""Unit tests for OpenAQ AWS S3 archive reader (Task P1-19).

Specification:
- docs/data-engineering.md §4 (OpenAQ archive backfill)
- data/schemas/observations_hourly.schema.yaml
"""

import gzip
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from smogsense.data_ingestion.openaq_archive import (
    build_archive_url,
    fetch_archive_day,
    parse_archive_csv_bytes,
    run_backfill_pilot,
)


def _make_sample_archive_csv_gz() -> bytes:
    csv_text = (
        "location_id,sensors_id,datetime,parameter,value,provider,is_monitor\n"
        "1001,5001,2025-11-05T00:00:00Z,pm25,45.2,AirGradient,false\n"
        "1001,5002,2025-11-05T00:00:00Z,relativehumidity,65.0,AirGradient,false\n"
        "1001,5003,2025-11-05T00:00:00Z,temperature,22.0,AirGradient,false\n"
        "1001,5001,2025-11-05T01:00:00Z,pm25,50.1,AirGradient,false\n"
        "1001,5002,2025-11-05T01:00:00Z,relativehumidity,68.0,AirGradient,false\n"
        "1001,5003,2025-11-05T01:00:00Z,temperature,21.5,AirGradient,false\n"
    )
    return gzip.compress(csv_text.encode("utf-8"))


def test_build_archive_url() -> None:
    # Test datetime object
    dt = datetime(2025, 11, 5, 12, 0, tzinfo=UTC)
    url1 = build_archive_url(1001, dt)
    assert url1 == (
        "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/"
        "locationid=1001/year=2025/month=11/location-1001-20251105.csv.gz"
    )

    # Test YYYY-MM-DD string
    url2 = build_archive_url(2050, "2026-01-08")
    assert url2 == (
        "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/"
        "locationid=2050/year=2026/month=01/location-2050-20260108.csv.gz"
    )

    # Test YYYYMMDD string
    url3 = build_archive_url(300, "20240915")
    assert url3 == (
        "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/"
        "locationid=300/year=2024/month=09/location-300-20240915.csv.gz"
    )


def test_parse_archive_csv_bytes() -> None:
    gz_bytes = _make_sample_archive_csv_gz()
    df = parse_archive_csv_bytes(gz_bytes, location_id=1001, domain="lahore")

    assert len(df) == 2
    assert (df["location_id"] == 1001).all()
    assert (df["sensor_id"] == 5001).all()
    assert (df["domain"] == "lahore").all()
    assert (df["source"] == "openaq_archive").all()

    # Verify parameters extracted properly
    assert df.iloc[0]["pm25_raw_ugm3"] == 45.2
    assert df.iloc[0]["pm25_ugm3"] == 45.2
    assert df.iloc[0]["rh_pct"] == 65.0
    assert df.iloc[0]["temp_c"] == 22.0

    assert df.iloc[1]["pm25_raw_ugm3"] == 50.1
    assert df.iloc[1]["pm25_ugm3"] == 50.1
    assert df.iloc[1]["rh_pct"] == 68.0
    assert df.iloc[1]["temp_c"] == 21.5

    # Check available_at_utc is future (+72h)
    assert (df["available_at_utc"] > df["ts_utc"]).all()

    # Corrupt bytes return empty frame
    empty_df = parse_archive_csv_bytes(b"not_a_gzip", location_id=1001, domain="lahore")
    assert len(empty_df) == 0


@pytest.mark.anyio
@respx.mock
async def test_fetch_archive_day_200_and_404() -> None:
    dt = datetime(2025, 11, 5, tzinfo=UTC)
    url = build_archive_url(1001, dt)

    # Mock 200
    respx.get(url).mock(
        return_value=httpx.Response(
            200,
            content=_make_sample_archive_csv_gz(),
            headers={"Content-Type": "application/gzip"},
        )
    )

    df_200 = await fetch_archive_day(1001, dt, domain="lahore")
    assert len(df_200) == 2

    # Mock 404 for different location
    url_404 = build_archive_url(9999, dt)
    respx.get(url_404).mock(return_value=httpx.Response(404))

    df_404 = await fetch_archive_day(9999, dt, domain="lahore")
    assert len(df_404) == 0


@respx.mock
def test_run_backfill_pilot(tmp_path: Path) -> None:
    gz_bytes = _make_sample_archive_csv_gz()

    # Mock all archive URLs: return 200 for loc 1001, 404 otherwise
    def _mock_s3(request: httpx.Request) -> httpx.Response:
        if "locationid=1001" in str(request.url):
            return httpx.Response(200, content=gz_bytes)
        return httpx.Response(404)

    respx.get(url__regex=r"https://openaq-data-archive\.s3\.amazonaws\.com/.*").mock(
        side_effect=_mock_s3
    )

    stats = run_backfill_pilot(
        domain="lahore",
        max_files=4,
        shard_by="year",
        output_dir=tmp_path,
    )

    assert stats["files_attempted"] == 4
    assert stats["files_found"] >= 1
    assert stats["records_processed"] >= 1
    assert stats["elapsed_seconds"] >= 0.0
    assert stats["download_mb_per_s"] >= 0.0

    output_file = Path(stats["output_file"])
    assert output_file.exists()
    saved_df = pd.read_parquet(output_file)
    assert len(saved_df) == stats["records_processed"]


@respx.mock
def test_cli_archive_pilot() -> None:
    from typer.testing import CliRunner

    from smogsense.cli import app

    runner = CliRunner()
    gz_bytes = _make_sample_archive_csv_gz()

    respx.get(url__regex=r"https://openaq-data-archive\.s3\.amazonaws\.com/.*").mock(
        return_value=httpx.Response(200, content=gz_bytes)
    )

    result = runner.invoke(
        app,
        ["ingest", "archive-pilot", "--limit", "2", "--domain", "lahore"],
    )
    assert result.exit_code == 0
    assert "Archive pilot completed" in result.stdout
    assert "Throughput:" in result.stdout
