"""smogsense unit tests for NASA FIRMS fire ingestion (Task P1-09).

Tests:
- Day-range splitting into chunks <= 10 days
- Platform continuity policy (Suomi-NPP hard cut-off at 2026-11-01T13:00:00Z)
- Confidence filtering (keep nominal/high, drop low)
- Geodesic distance, azimuth, and bearing (Amritsar-Lahore worked example)
- Radial rings and compass sector binning
- Platform normalisation and Pandera schema validation (fire_exposure.schema.yaml)
- FirmsClient network queries and CLI command execution
"""

from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import respx
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.data_ingestion.firms import (
    DEFAULT_FIRMS_BASE_URL,
    DOMAIN_CENTROIDS,
    SUOMI_NPP_CUTOFF,
    FirmsClient,
    aggregate_fire_exposure,
    check_suomi_npp_cutoff,
    compute_geodesic_geometry,
    filter_confidence,
    parse_detections_csv,
    split_date_range,
)

runner = CliRunner()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# =====================================================================
# 1. Date Range Splitting
# =====================================================================


def test_split_date_range_single_day() -> None:
    chunks = split_date_range("2026-10-01", "2026-10-01", inclusive=True)
    assert len(chunks) == 1
    assert chunks[0] == (date(2026, 10, 1), 1)


def test_split_date_range_multi_week() -> None:
    # 25 days inclusive: 10, 10, 5
    chunks = split_date_range("2026-10-01", "2026-10-25", max_days=10, inclusive=True)
    assert len(chunks) == 3
    assert chunks[0] == (date(2026, 10, 1), 10)
    assert chunks[1] == (date(2026, 10, 11), 10)
    assert chunks[2] == (date(2026, 10, 21), 5)


def test_split_date_range_exclusive() -> None:
    # 24 days: 10, 10, 4
    chunks = split_date_range("2026-10-01", "2026-10-25", max_days=10, inclusive=False)
    assert len(chunks) == 3
    assert chunks[0] == (date(2026, 10, 1), 10)
    assert chunks[1] == (date(2026, 10, 11), 10)
    assert chunks[2] == (date(2026, 10, 21), 4)


def test_split_date_range_clamped_max_days() -> None:
    # Passing max_days=20 clamps to 10
    chunks = split_date_range(date(2026, 10, 1), date(2026, 10, 15), max_days=20)
    assert len(chunks) == 2
    assert chunks[0] == (date(2026, 10, 1), 10)
    assert chunks[1] == (date(2026, 10, 11), 5)


def test_split_date_range_invalid_arguments() -> None:
    with pytest.raises(ValueError, match=r"start_date .* must be <= end_date"):
        split_date_range("2026-10-15", "2026-10-01")

    with pytest.raises(ValueError, match="max_days must be >= 1"):
        split_date_range("2026-10-01", "2026-10-05", max_days=0)


# =====================================================================
# 2. Suomi-NPP Cutoff Policy (FR-03)
# =====================================================================


def test_suomi_npp_cutoff_allowed_before_boundary() -> None:
    # Before 2026-11-01 13:00 UTC
    check_suomi_npp_cutoff("VIIRS_SNPP_NRT", datetime(2026, 10, 15, 0, 0, tzinfo=UTC))
    check_suomi_npp_cutoff("VIIRS_SNPP_SP", datetime(2026, 11, 1, 12, 59, 59, tzinfo=UTC))
    check_suomi_npp_cutoff("VIIRS_SNPP_NRT", "2026-10-01T00:00:00Z")


def test_suomi_npp_cutoff_refused_on_and_after_boundary() -> None:
    # Exactly at 2026-11-01 13:00 UTC
    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_NRT", SUOMI_NPP_CUTOFF)

    # After boundary
    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_SP", datetime(2026, 11, 1, 14, 0, tzinfo=UTC))

    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_NRT", "2026-11-05T00:00:00Z")

    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_SP", date(2026, 11, 2))


def test_operational_platforms_not_restricted_by_cutoff() -> None:
    # NOAA-20 and NOAA-21 must continue post-cutoff
    check_suomi_npp_cutoff("VIIRS_NOAA21_NRT", datetime(2026, 12, 1, 0, 0, tzinfo=UTC))
    check_suomi_npp_cutoff("VIIRS_NOAA20_NRT", datetime(2026, 12, 1, 0, 0, tzinfo=UTC))


# =====================================================================
# 3. Confidence Filtering
# =====================================================================


def test_confidence_filtering() -> None:
    df = pd.DataFrame(
        {
            "confidence": ["n", "h", "nominal", "high", "l", "low", "85", "15", None],
            "frp": [10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 50.0, 2.0, 1.0],
        }
    )
    filtered = filter_confidence(df)
    kept_conf = filtered["confidence"].tolist()
    assert kept_conf == ["n", "h", "nominal", "high", "85"]


# =====================================================================
# 4. Parsing Detections CSV
# =====================================================================


def test_parse_detections_csv_empty_and_error_cases() -> None:
    assert parse_detections_csv("").empty
    assert parse_detections_csv("   \n").empty
    assert parse_detections_csv("No data found for this area").empty
    assert parse_detections_csv("Invalid MAP_KEY").empty

    # Header only
    header_only = "latitude,longitude,acq_date,acq_time,confidence,frp\n"
    assert parse_detections_csv(header_only).empty


def test_parse_detections_csv_valid_utc_and_availability() -> None:
    csv_text = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
31.6340,74.8723,325.4,0.4,0.4,2026-10-15,830,N21,VIIRS,n,2.0NRT,295.1,14.5,D
"""
    df = parse_detections_csv(csv_text)
    assert len(df) == 1
    assert df.loc[0, "latitude"] == 31.6340
    assert df.loc[0, "longitude"] == 74.8723
    assert df.loc[0, "frp"] == 14.5
    # Hour binning
    assert df.loc[0, "ts_utc"] == pd.Timestamp("2026-10-15T08:00:00Z")
    # Universal availability: acq_datetime (08:30) + 3 hours = 11:30
    assert df.loc[0, "available_at_utc"] == pd.Timestamp("2026-10-15T11:30:00Z")


# =====================================================================
# 5. Geodesic Geometry & Worked Example
# =====================================================================


def test_geodesic_geometry_amritsar_lahore_worked_example() -> None:
    """Validate against the worked example in docs/ml-architecture.md §3 and docs/data-engineering.md §9.

    Lahore: 31.5204°N, 74.3587°E
    Amritsar: 31.6340°N, 74.8723°E
    Expected:
      Distance d ≈ 50.4 km (in ring r50_150)
      Target->fire azimuth θ ≈ 75.4° - 75.5° (Sector E: [67.5°, 112.5°))
      Fire->target bearing β ≈ 255.4° - 255.5°
    """
    df = pd.DataFrame(
        [
            {
                "latitude": 31.6340,
                "longitude": 74.8723,
                "frp": 25.0,
                "ts_utc": pd.Timestamp("2026-10-15T08:00:00Z"),
            }
        ]
    )
    lahore_lat, lahore_lon = DOMAIN_CENTROIDS["lahore"]
    res = compute_geodesic_geometry(df, target_lat=lahore_lat, target_lon=lahore_lon)

    assert len(res) == 1
    dist_km = res.loc[0, "dist_km"]
    theta = res.loc[0, "azimuth_deg"]
    beta = res.loc[0, "bearing_deg"]
    ring = res.loc[0, "ring"]
    sector = res.loc[0, "sector"]

    assert 50.3 <= dist_km <= 50.5
    assert 75.2 <= theta <= 75.6
    assert 255.2 <= beta <= 255.6
    assert ring == "r50_150"
    assert sector == "E"


def test_geodesic_geometry_drops_beyond_600km() -> None:
    # Karachi: ~24.86°N, 67.00°E (~1,000 km from Lahore)
    df = pd.DataFrame(
        [
            {
                "latitude": 24.86,
                "longitude": 67.00,
                "frp": 100.0,
                "ts_utc": pd.Timestamp("2026-10-15T08:00:00Z"),
            }
        ]
    )
    lahore_lat, lahore_lon = DOMAIN_CENTROIDS["lahore"]
    res = compute_geodesic_geometry(df, target_lat=lahore_lat, target_lon=lahore_lon)
    # Filtered out because distance > 600 km
    assert res.empty


def test_all_radial_rings() -> None:
    lahore_lat, lahore_lon = DOMAIN_CENTROIDS["lahore"]
    # Point ~10 km away (r0_50)
    # Point ~80 km away (r50_150)
    # Point ~200 km away (r150_300)
    # Point ~450 km away (r300_600)
    df = pd.DataFrame(
        [
            {"latitude": 31.55, "longitude": 74.40, "frp": 10.0},
            {"latitude": 31.80, "longitude": 75.00, "frp": 20.0},
            {"latitude": 32.50, "longitude": 76.00, "frp": 30.0},
            {"latitude": 34.00, "longitude": 77.00, "frp": 40.0},
        ]
    )
    res = compute_geodesic_geometry(df, target_lat=lahore_lat, target_lon=lahore_lon)
    assert len(res) == 4
    rings = res["ring"].tolist()
    assert rings == ["r0_50", "r50_150", "r150_300", "r300_600"]


def test_all_eight_compass_sectors() -> None:
    lahore_lat, lahore_lon = DOMAIN_CENTROIDS["lahore"]
    # Synthesize detections around Lahore in N, NE, E, SE, S, SW, W, NW directions within 50 km
    delta = 0.3
    points = [
        {"latitude": lahore_lat + delta, "longitude": lahore_lon},  # N (~0°)
        {"latitude": lahore_lat + delta, "longitude": lahore_lon + delta},  # NE (~45°)
        {"latitude": lahore_lat, "longitude": lahore_lon + delta},  # E (~90°)
        {"latitude": lahore_lat - delta, "longitude": lahore_lon + delta},  # SE (~135°)
        {"latitude": lahore_lat - delta, "longitude": lahore_lon},  # S (~180°)
        {"latitude": lahore_lat - delta, "longitude": lahore_lon - delta},  # SW (~225°)
        {"latitude": lahore_lat, "longitude": lahore_lon - delta},  # W (~270°)
        {"latitude": lahore_lat + delta, "longitude": lahore_lon - delta},  # NW (~315°)
    ]
    df = pd.DataFrame(points)
    df["frp"] = 10.0
    res = compute_geodesic_geometry(df, target_lat=lahore_lat, target_lon=lahore_lon)
    sectors = res["sector"].tolist()
    assert sectors == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# =====================================================================
# 6. Aggregation & Platform Normalisation
# =====================================================================


def test_aggregate_fire_exposure_normalisation_and_schema() -> None:
    t0 = pd.Timestamp("2026-10-15T08:00:00Z")
    df = pd.DataFrame(
        [
            # 2 fires in r50_150, each FRP=50 (sum = 100)
            {
                "latitude": 31.6340,
                "longitude": 74.8723,
                "frp": 50.0,
                "confidence": "n",
                "ts_utc": t0,
            },
            {
                "latitude": 31.6340,
                "longitude": 74.8723,
                "frp": 50.0,
                "confidence": "n",
                "ts_utc": t0,
            },
        ]
    )
    # platforms_active = 2 -> fire_count = 2 / 2 = 1.0, frp = 100 / 2 = 50.0
    agg = aggregate_fire_exposure(
        detections_df=df,
        domain="lahore",
        start_utc=t0,
        end_utc=t0 + pd.Timedelta(hours=2),
        platforms_active=2,
    )

    assert len(agg) == 2  # 08:00 and 09:00
    # Hour 08:00
    row0 = agg.iloc[0]
    assert row0["domain"] == "lahore"
    assert row0["ts_utc"] == t0
    assert row0["platforms_active"] == 2
    assert row0["n_detections_total"] == 2
    assert row0["frp_sum_mw_total"] == 100.0
    assert row0["fire_count_r50_150"] == 1.0
    assert row0["frp_mw_r50_150"] == 50.0
    assert row0["exposure_e24"] == 0.0
    assert bool(row0["coverage_complete"]) is True

    # Hour 09:00 (no fires)
    row1 = agg.iloc[1]
    assert row1["n_detections_total"] == 0
    assert row1["frp_sum_mw_total"] == 0.0
    assert row1["fire_count_r50_150"] == 0.0


def test_aggregate_fire_exposure_empty_detections() -> None:
    t0 = pd.Timestamp("2026-10-15T00:00:00Z")
    agg = aggregate_fire_exposure(
        detections_df=pd.DataFrame(),
        domain="delhi",
        start_utc=t0,
        end_utc=t0 + pd.Timedelta(hours=3),
        platforms_active=2,
    )
    assert len(agg) == 3
    assert (agg["n_detections_total"] == 0).all()
    assert (agg["frp_sum_mw_total"] == 0.0).all()


def test_aggregate_fire_exposure_unknown_domain() -> None:
    with pytest.raises(ValueError, match="Unknown domain: 'karachi'"):
        aggregate_fire_exposure(
            detections_df=pd.DataFrame(),
            domain="karachi",
            start_utc="2026-10-15T00:00:00Z",
            end_utc="2026-10-15T01:00:00Z",
        )


# =====================================================================
# 7. FirmsClient Mocked Network Operations
# =====================================================================


def test_firms_client_url_construction() -> None:
    client = FirmsClient(map_key="TEST_KEY")
    bbox = {"west": 71.0, "south": 27.0, "east": 79.5, "north": 34.0}
    url = client.build_area_url("VIIRS_NOAA21_NRT", bbox, 10, "2026-10-01")
    expected = (
        f"{DEFAULT_FIRMS_BASE_URL}/TEST_KEY/VIIRS_NOAA21_NRT/71.0,27.0,79.5,34.0/10/2026-10-01"
    )
    assert url == expected


def test_firms_client_missing_key_raises() -> None:
    client = FirmsClient(map_key="")
    with pytest.raises(ValueError, match="FIRMS_MAP_KEY is required"):
        client.query_area_sync("VIIRS_NOAA21_NRT")


@pytest.mark.anyio
async def test_firms_client_query_mocked_with_respx() -> None:
    csv_mock = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
31.6340,74.8723,325.4,0.4,0.4,2026-10-01,830,N21,VIIRS,n,2.0NRT,295.1,15.0,D
"""
    client = FirmsClient(map_key="MOCK_MAP_KEY")
    with respx.mock(base_url="https://firms.modaps.eosdis.nasa.gov") as respx_mock:
        respx_mock.get(url__regex=r"/api/area/csv/.*").respond(
            status_code=200,
            text=csv_mock,
            headers={"Content-Type": "text/csv"},
        )

        df = await client.query_area(
            source="VIIRS_NOAA21_NRT",
            start_date="2026-10-01",
            end_date="2026-10-05",
        )
        assert len(df) == 1
        assert df.loc[0, "frp"] == 15.0


@pytest.mark.anyio
async def test_firms_client_data_availability_mocked() -> None:
    client = FirmsClient(map_key="MOCK_MAP_KEY")
    with respx.mock(base_url="https://firms.modaps.eosdis.nasa.gov") as respx_mock:
        respx_mock.get(url__regex=r"/api/data_availability/csv/.*").respond(
            status_code=200,
            text="data_source,min_date,max_date\nVIIRS_NOAA21_NRT,2023-01-01,2026-10-11\n",
        )
        res = await client.get_data_availability("VIIRS_NOAA21_NRT")
        assert res["source"] == "VIIRS_NOAA21_NRT"
        assert "2026-10-11" in res["raw_response"]


# =====================================================================
# 8. CLI Command Execution
# =====================================================================


def test_cli_ingest_firms_command(tmp_path: Path) -> None:
    out_file = tmp_path / "fire_exposure.parquet"
    mock_df = pd.DataFrame(
        [
            {
                "latitude": 31.6340,
                "longitude": 74.8723,
                "confidence": "n",
                "frp": 20.0,
                "acq_date": "2026-10-01",
                "acq_time": "0830",
                "acq_datetime_utc": pd.Timestamp("2026-10-01T08:30:00Z"),
                "ts_utc": pd.Timestamp("2026-10-01T08:00:00Z"),
                "available_at_utc": pd.Timestamp("2026-10-01T11:30:00Z"),
            }
        ]
    )

    with patch.object(FirmsClient, "query_area_sync", return_value=mock_df):
        result = runner.invoke(
            app,
            [
                "ingest",
                "firms",
                "--domain",
                "lahore",
                "--start",
                "2026-10-01T00:00:00Z",
                "--end",
                "2026-10-01T03:00:00Z",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert "Ingested 1 detections" in result.output
        assert "Saved fire exposure to" in result.output
        assert out_file.exists()

        loaded = pd.read_parquet(out_file)
        assert len(loaded) == 3  # 00:00, 01:00, 02:00
        assert "fire_count_r50_150" in loaded.columns
