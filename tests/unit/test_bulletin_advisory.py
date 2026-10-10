"""Unit tests for bulletin generation, AQI truncation, advisory escalation, and timezones."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from smogsense.publishing.bulletin import determine_aqi_category, generate_bulletin_json


def test_determine_aqi_category_truncation_boundaries() -> None:
    # 9.04 truncates to 9.0 (good)
    assert determine_aqi_category(9.04) == "good"
    assert determine_aqi_category(9.09) == "good"
    assert determine_aqi_category(9.10) == "moderate"

    # 35.45 truncates to 35.4 (moderate)
    assert determine_aqi_category(35.45) == "moderate"
    assert determine_aqi_category(35.49) == "moderate"
    assert determine_aqi_category(35.50) == "usg"

    # 55.45 truncates to 55.4 (usg)
    assert determine_aqi_category(55.45) == "usg"
    assert determine_aqi_category(55.49) == "usg"
    assert determine_aqi_category(55.50) == "unhealthy"

    # 125.45 truncates to 125.4 (unhealthy)
    assert determine_aqi_category(125.45) == "unhealthy"
    assert determine_aqi_category(125.49) == "unhealthy"
    assert determine_aqi_category(125.50) == "very_unhealthy"

    # 225.45 truncates to 225.4 (very_unhealthy)
    assert determine_aqi_category(225.45) == "very_unhealthy"
    assert determine_aqi_category(225.49) == "very_unhealthy"
    assert determine_aqi_category(225.50) == "hazardous"


def _sample_forecast_df(
    day1_pm25_median: float = 20.0,
    day1_q_upper: float = 25.0,
    method: str = "m1_cams_raw",
) -> pd.DataFrame:
    rows = []
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    for lead in [24, 48, 72]:
        # City row
        row_city: dict[str, Any] = {
            "issuance_utc": issuance,
            "generated_at_utc": issuance,
            "domain": "lahore",
            "level": "city",
            "point_id": "city:lahore",
            "horizon_h": lead,
            "window_start_utc": issuance + pd.Timedelta(hours=lead - 24),
            "window_end_utc": issuance + pd.Timedelta(hours=lead),
            "method": method,
            "model_version": "baseline",
            "mode": "full",
            "cams_base_time_utc": issuance,
            "data_cutoff_utc": issuance,
            "git_sha": "unknown",
            "config_hash": "abc",
            "calibrated": False,
            "run_id": "test_run",
            "is_rerun": False,
            "adaptation_status": "skipped_n0",
            "cams_lead_offset_h": 0.0,
            "n_available_stations": 1,
        }
        for q in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            val = (day1_pm25_median if q <= 50 else day1_q_upper) if lead == 24 else 20.0
            row_city[f"q{q:02d}"] = float(val)
        rows.append(row_city)

        # Station row
        row_st: dict[str, Any] = dict(row_city)
        row_st["level"] = "station"
        row_st["point_id"] = "station:42"
        rows.append(row_st)

    return pd.DataFrame(rows)


def test_generate_bulletin_advisory_escalation() -> None:
    # Construct a city forecast for lead 24 where median is moderate (~30 ug/m3)
    # but upper quantiles (>q60) are above 55.5 ug/m3 (unhealthy)
    # so P(USG) or P(next category) >= 0.35
    # Category order: good, moderate, usg, unhealthy, very_unhealthy, hazardous
    # Median = 30.0 (moderate). Next category is "usg" (35.5 - 55.4).
    # If q55..q95 are in [36.0, 50.0], that's 9 out of 19 quantiles => ~47% probability in USG!
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    df = _sample_forecast_df(day1_pm25_median=30.0, day1_q_upper=45.0)

    bulletin = generate_bulletin_json(
        forecast_df=df,
        issuance_utc=issuance,
        mode="full",
        model="m1_cams_raw",
        data_cutoff_utc=issuance,
        domain="lahore",
        sources_status={"openaq": {"status": "ok", "as_of_utc": issuance.isoformat()}},
        stations_metadata={
            "42": {"name": "Test Station", "lat": 31.5, "lon": 74.3, "is_reference": True}
        },
        git_sha="unknown",
    )

    # Day 1 median is moderate, but next category (USG) has prob >= 0.35, so it escalates to USG
    assert bulletin["horizons"][0]["category"]["median_q50"] == "moderate"
    assert bulletin["advisory"]["category"] == "usg"


def test_generate_bulletin_advisory_no_escalation() -> None:
    # When probability of next category is small (< 0.35)
    # Median is 20.0 (moderate), upper quantiles are 22.0 (also moderate)
    # P(usg) is 0.0 => advisory remains moderate
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    df = _sample_forecast_df(day1_pm25_median=20.0, day1_q_upper=22.0)

    bulletin = generate_bulletin_json(
        forecast_df=df,
        issuance_utc=issuance,
        mode="full",
        model="m1_cams_raw",
        data_cutoff_utc=issuance,
        domain="lahore",
        sources_status={"openaq": {"status": "ok", "as_of_utc": issuance.isoformat()}},
        stations_metadata={
            "42": {"name": "Test Station", "lat": 31.5, "lon": 74.3, "is_reference": True}
        },
        git_sha="unknown",
    )

    assert bulletin["horizons"][0]["category"]["median_q50"] == "moderate"
    assert bulletin["advisory"]["category"] == "moderate"


def test_generate_bulletin_local_time_and_station_lookup() -> None:
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    df = _sample_forecast_df()

    # stations_metadata has integer key 42
    bulletin = generate_bulletin_json(
        forecast_df=df,
        issuance_utc=issuance,
        mode="full",
        model="m1_cams_raw",
        data_cutoff_utc=issuance,
        domain="lahore",
        sources_status={"openaq": {"status": "ok", "as_of_utc": issuance.isoformat()}},
        stations_metadata={
            42: {"name": "Int Key Station", "lat": 31.52, "lon": 74.35, "is_reference": False}
        },  # type: ignore[dict-item]
        git_sha="unknown",
    )

    # Check local time has +05:00
    for h in bulletin["horizons"]:
        assert h["window_start_local"].endswith("+05:00")
        assert h["window_end_local"].endswith("+05:00")

    # Check station metadata was correctly resolved
    assert len(bulletin["stations"]) == 1
    st = bulletin["stations"][0]
    assert st["point_id"] == "station:42"
    assert st["name"] == "Int Key Station"
    assert st["lat"] == 31.52
    assert st["lon"] == 74.35
    assert not st["is_reference"]


def test_generate_bulletin_method_filter() -> None:
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    df_m1 = _sample_forecast_df(day1_pm25_median=10.0, method="m1_cams_raw")
    df_m0 = _sample_forecast_df(day1_pm25_median=50.0, method="m0_persistence")
    combined_df = pd.concat([df_m1, df_m0], ignore_index=True)

    bulletin = generate_bulletin_json(
        forecast_df=combined_df,
        issuance_utc=issuance,
        mode="full",
        model="m1_cams_raw",
        data_cutoff_utc=issuance,
        domain="lahore",
        sources_status={"openaq": {"status": "ok", "as_of_utc": issuance.isoformat()}},
        stations_metadata={"42": {"name": "Test", "lat": 31.5, "lon": 74.3, "is_reference": True}},
        git_sha="unknown",
    )

    # Should have filtered to m1_cams_raw (median 10.0 => moderate), not m0_persistence (50.0 => usg)
    assert bulletin["horizons"][0]["quantiles_ugm3"]["q50"] == 10.0


def test_generate_bulletin_schema_missing_raises() -> None:
    issuance = datetime(2026, 10, 8, 0, 0, tzinfo=UTC)
    df = _sample_forecast_df()

    with (
        patch("smogsense.publishing.bulletin.Path.exists", return_value=False),
        pytest.raises(FileNotFoundError, match="Bulletin schema not found"),
    ):
        generate_bulletin_json(
            forecast_df=df,
            issuance_utc=issuance,
            mode="full",
            model="m1_cams_raw",
            data_cutoff_utc=issuance,
            domain="lahore",
            sources_status={"openaq": {"status": "ok", "as_of_utc": issuance.isoformat()}},
            stations_metadata={},
            git_sha="unknown",
        )
