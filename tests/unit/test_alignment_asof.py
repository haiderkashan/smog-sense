from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from smogsense.preprocessing.alignment import asof_cams_run, stitch_cams_series


def test_asof_cams_run_lattice():
    """
    Test exactly the lattice defined in the prompt:
    00:00Z -> previous day 12:00Z
    06:00Z -> previous day 12:00Z
    12:00Z -> same day 00:00Z
    18:00Z -> same day 00:00Z
    """
    assert asof_cams_run(datetime(2026, 1, 2, 0, 0, tzinfo=UTC)) == datetime(
        2026, 1, 1, 12, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 1, 2, 6, 0, tzinfo=UTC)) == datetime(
        2026, 1, 1, 12, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 1, 2, 12, 0, tzinfo=UTC)) == datetime(
        2026, 1, 2, 0, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 1, 2, 18, 0, tzinfo=UTC)) == datetime(
        2026, 1, 2, 0, 0, tzinfo=UTC
    )


def test_asof_cams_run_boundaries():
    """
    Exact availability boundaries:
    09:59:59 -> previous 12Z
    10:00:00 -> current 00Z
    21:59:59 -> current 00Z
    22:00:00 -> current 12Z
    """
    assert asof_cams_run(datetime(2026, 1, 2, 9, 59, 59, tzinfo=UTC)) == datetime(
        2026, 1, 1, 12, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 1, 2, 10, 0, 0, tzinfo=UTC)) == datetime(
        2026, 1, 2, 0, 0, tzinfo=UTC
    )

    assert asof_cams_run(datetime(2026, 1, 1, 21, 59, 59, tzinfo=UTC)) == datetime(
        2026, 1, 1, 0, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 1, 1, 22, 0, 0, tzinfo=UTC)) == datetime(
        2026, 1, 1, 12, 0, tzinfo=UTC
    )


def test_no_future_run_selection():
    """Future CAMS run can never be selected."""
    res = asof_cams_run(datetime(2026, 1, 2, 10, 0, 0, tzinfo=UTC))
    assert res == datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    assert res <= datetime(2026, 1, 2, 10, 0, 0, tzinfo=UTC)


def test_timezone_normalization():
    """Test naive and non-UTC timezones."""
    dt_naive = datetime(2026, 1, 2, 0, 0)
    assert asof_cams_run(dt_naive) == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    tz_plus_5 = timezone(timedelta(hours=5))
    dt_aware = datetime(2026, 1, 2, 5, 0, tzinfo=tz_plus_5)  # == 00:00Z
    assert asof_cams_run(dt_aware) == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_calendar_boundaries():
    """Month and year rollovers."""
    assert asof_cams_run(datetime(2026, 1, 1, 0, 0, tzinfo=UTC)) == datetime(
        2025, 12, 31, 12, 0, tzinfo=UTC
    )
    assert asof_cams_run(datetime(2026, 3, 1, 0, 0, tzinfo=UTC)) == datetime(
        2026, 2, 28, 12, 0, tzinfo=UTC
    )


def test_stitch_cams_series_continuous():
    """
    Test Contract B — normal bitemporal stitching.
    No gaps, no <12h rule.
    09Z -> previous 12Z, lead 21h
    10Z -> current 00Z, lead 10h
    21Z -> current 00Z, lead 21h
    22Z -> current 12Z, lead 10h
    """
    df = stitch_cams_series(datetime(2026, 1, 2, 12, 0, tzinfo=UTC), window_hours=24)
    assert len(df) == 25
    assert df["target_hour_utc"].is_monotonic_increasing

    # Check t = 10Z (T=12Z)
    row_10 = df[df["target_hour_utc"] == datetime(2026, 1, 2, 10, 0, tzinfo=UTC)].iloc[0]
    assert row_10["cams_cycle_utc"] == datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    assert row_10["lead_time_hours"] == 10

    # Check t = 09Z
    row_9 = df[df["target_hour_utc"] == datetime(2026, 1, 2, 9, 0, tzinfo=UTC)].iloc[0]
    assert row_9["cams_cycle_utc"] == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert row_9["lead_time_hours"] == 21


def test_stitch_cams_invalid_window():
    with pytest.raises(ValueError, match="window_hours must be non-negative"):
        stitch_cams_series(datetime(2026, 1, 2, 12, 0, tzinfo=UTC), window_hours=-1)


def test_stitch_cams_window_zero():
    df = stitch_cams_series(datetime(2026, 1, 2, 12, 0, tzinfo=UTC), window_hours=0)
    assert len(df) == 1
    assert df.iloc[0]["target_hour_utc"] == datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
    assert df.iloc[0]["lead_time_hours"] == 12


def test_stitch_cams_schema():
    df = stitch_cams_series(datetime(2026, 1, 2, 12, 0, tzinfo=UTC), window_hours=24)
    assert df.index.name is None
    assert list(df.columns) == ["target_hour_utc", "cams_cycle_utc", "lead_time_hours"]
    assert pd.api.types.is_integer_dtype(df["lead_time_hours"])


def test_stitch_cams_series_invariant():
    """Mathematically prove temporal invariants over a large window."""
    issuance = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    window = 500
    df = stitch_cams_series(issuance, window_hours=window)

    assert len(df) == window + 1

    for _, row in df.iterrows():
        t = row["target_hour_utc"]
        b = row["cams_cycle_utc"]

        # 1. B in {00Z, 12Z}
        assert b.hour in (0, 12)
        assert b.minute == 0
        assert b.second == 0

        # 2. B + 10h <= t
        assert b + timedelta(hours=10) <= t

        # 3. t <= issuance
        assert t <= issuance

        # 4. B is the maximal eligible cycle
        # The next cycle would be b + 12h.
        next_cycle = b + timedelta(hours=12)
        assert next_cycle + timedelta(hours=10) > t


def test_stitch_cams_non_hour_issuance():
    """Test explicit flooring of non-hour issuance."""
    issuance = datetime(2026, 1, 2, 12, 59, 30, tzinfo=UTC)
    df = stitch_cams_series(issuance, window_hours=0)

    # It floors to 12:00:00Z
    assert len(df) == 1
    assert df.iloc[0]["target_hour_utc"] == datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    assert df.iloc[0]["cams_cycle_utc"] == datetime(2026, 1, 2, 0, 0, 0, tzinfo=UTC)
