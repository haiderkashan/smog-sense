"""FIRMS CSV columns and limits contract test.

Specification: docs/data-engineering.md -> 'NASA FIRMS fire ingestion'
"""

from datetime import UTC, datetime

import pytest

from smogsense.data_ingestion.firms import (
    SUOMI_NPP_CUTOFF,
    check_suomi_npp_cutoff,
    parse_detections_csv,
    split_date_range,
)

pytestmark = pytest.mark.contract


def test_viirs_csv_parsing_contract() -> None:
    """VIIRS columns: latitude, longitude, bright_ti4, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_ti5, frp, daynight."""
    viirs_csv = """latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight
31.6340,74.8723,325.4,0.4,0.4,2026-10-15,830,N21,VIIRS,n,2.0NRT,295.1,14.5,D
31.7500,74.9000,310.2,0.5,0.4,2026-10-15,830,N21,VIIRS,l,2.0NRT,290.0,5.2,D
"""
    df = parse_detections_csv(viirs_csv)
    # Low confidence dropped; nominal kept
    assert len(df) == 1
    assert df.loc[0, "confidence"] == "n"
    assert df.loc[0, "frp"] == 14.5
    assert df.loc[0, "ts_utc"] == datetime(2026, 10, 15, 8, 0, tzinfo=UTC)
    assert df.loc[0, "available_at_utc"] == datetime(2026, 10, 15, 11, 30, tzinfo=UTC)


def test_modis_csv_parsing_contract() -> None:
    """MODIS columns: latitude, longitude, brightness, scan, track, acq_date, acq_time, satellite, instrument, confidence, version, bright_t31, frp, daynight."""
    modis_csv = """latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_t31,frp,daynight
31.6340,74.8723,325.4,1.0,1.0,2026-10-15,1030,T,MODIS,80,6.1NRT,295.1,20.0,D
31.7500,74.9000,310.2,1.0,1.0,2026-10-15,1030,T,MODIS,15,6.1NRT,290.0,4.0,D
"""
    df = parse_detections_csv(modis_csv)
    assert len(df) == 1
    assert df.loc[0, "frp"] == 20.0
    assert df.loc[0, "ts_utc"] == datetime(2026, 10, 15, 10, 0, tzinfo=UTC)


def test_day_range_clamp_at_ten_contract() -> None:
    """Single FIRMS query DAY_RANGE is clamped at 10."""
    chunks = split_date_range("2026-10-01", "2026-10-25", max_days=15)
    for _date, days in chunks:
        assert days <= 10
    total_days = sum(days for _date, days in chunks)
    assert total_days == 25


def test_suomi_npp_cutoff_contract() -> None:
    """Suomi-NPP product delivery ends 2026-11-01 13:00 UTC. Queries after this instant must be refused."""
    # Allowed before cutoff
    check_suomi_npp_cutoff("VIIRS_SNPP_NRT", datetime(2026, 10, 31, 23, 59, tzinfo=UTC))
    check_suomi_npp_cutoff("VIIRS_SNPP_SP", datetime(2026, 11, 1, 12, 59, tzinfo=UTC))

    # Refused on or after cutoff
    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_NRT", SUOMI_NPP_CUTOFF)

    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_NRT", datetime(2026, 11, 2, 0, 0, tzinfo=UTC))

    with pytest.raises(ValueError, match="Suomi-NPP product delivery ended 2026-11-01T13:00:00Z"):
        check_suomi_npp_cutoff("VIIRS_SNPP_SP", "2026-11-05T00:00:00Z")
