"""Unit tests for boundary purging, embargo enforcement, and twin-aware splits (splits.py; T1-14)."""

import pandas as pd
import pytest

from smogsense.evaluation.splits import (
    create_purged_embargoed_splits,
    enforce_embargo,
    purge_split_boundaries,
    validate_no_split_overlap,
    validate_twin_aware_stations,
)


def test_enforce_embargo() -> None:
    # 72 hours exact gap
    split_a_end = "2026-11-01T00:00:00Z"
    split_b_start = "2026-11-04T00:00:00Z"
    assert enforce_embargo(split_a_end, split_b_start, min_embargo_h=72) is True

    # Less than 72 hours gap raises ValueError
    split_b_too_soon = "2026-11-03T12:00:00Z"  # 60 hours
    with pytest.raises(ValueError, match="Temporal embargo violation"):
        enforce_embargo(split_a_end, split_b_too_soon, min_embargo_h=72)


def test_purge_split_boundaries_forward_and_backward() -> None:
    # Hourly timestamps from 2026-11-01T00:00 to 2026-11-10T00:00 (217 hours)
    dates = pd.date_range("2026-11-01T00:00:00Z", "2026-11-10T00:00:00Z", freq="1h")
    df = pd.DataFrame({"issuance_utc": dates, "val": range(len(dates))})

    # Forward lookback purging (72 h) and backward target purging (72 h)
    # Split: 2026-11-01T00:00 to 2026-11-10T00:00
    # Minimum valid issuance: 2026-11-01 + 72 h = 2026-11-04T00:00
    # Maximum valid issuance: 2026-11-10 - 72 h = 2026-11-07T00:00
    purged = purge_split_boundaries(
        df,
        split_start="2026-11-01T00:00:00Z",
        split_end="2026-11-10T00:00:00Z",
        lookback_h=72,
        horizon_max_h=72,
        has_prior_split=True,
        has_subsequent_split=True,
    )
    assert not purged.empty
    assert purged["issuance_utc"].min() == pd.Timestamp("2026-11-04T00:00:00Z")
    assert purged["issuance_utc"].max() == pd.Timestamp("2026-11-07T00:00:00Z")

    # If first split in history (has_prior_split=False): no forward purge
    purged_no_prior = purge_split_boundaries(
        df,
        split_start="2026-11-01T00:00:00Z",
        split_end="2026-11-10T00:00:00Z",
        lookback_h=72,
        horizon_max_h=72,
        has_prior_split=False,
        has_subsequent_split=True,
    )
    assert purged_no_prior["issuance_utc"].min() == pd.Timestamp("2026-11-01T00:00:00Z")
    assert purged_no_prior["issuance_utc"].max() == pd.Timestamp("2026-11-07T00:00:00Z")


def test_validate_no_split_overlap() -> None:
    # Split A: 2026-11-01 to 2026-11-05
    # Split B: 2026-11-12 to 2026-11-15 (more than 72 h lookback and target clear)
    dates_a = pd.date_range("2026-11-01T00:00:00Z", "2026-11-05T00:00:00Z", freq="1D")
    dates_b = pd.date_range("2026-11-12T00:00:00Z", "2026-11-15T00:00:00Z", freq="1D")

    df_a = pd.DataFrame({"issuance_utc": dates_a})
    df_b = pd.DataFrame({"issuance_utc": dates_b})

    # Should pass without exception
    validate_no_split_overlap(df_a, df_b, lookback_h=72, horizon_max_h=72)

    # Overlapping case: Split A ends Nov 05, Split B starts Nov 06
    # Target of A reaches Nov 08 (Nov 05 + 72 h), while lookback of B reaches back to Nov 03 (Nov 06 - 72 h)
    dates_b_leak = pd.date_range("2026-11-06T00:00:00Z", "2026-11-10T00:00:00Z", freq="1D")
    df_b_leak = pd.DataFrame({"issuance_utc": dates_b_leak})

    with pytest.raises(ValueError, match="Temporal leakage detected"):
        validate_no_split_overlap(df_a, df_b_leak, lookback_h=72, horizon_max_h=72)


def test_validate_twin_aware_stations() -> None:
    # 8664 and 8665 are co-located twin stations
    twin_pairs = [("8664", "8665"), ("9001", "9002")]

    # Safe: both twins in train, other pair in test
    validate_twin_aware_stations(
        train_station_ids={"8664", "8665", "1000"},
        test_station_ids={"9001", "9002", "2000"},
        co_located_pairs=twin_pairs,
    )

    # Violation: 8664 in train, but 8665 in test
    with pytest.raises(ValueError, match="Twin-aware split violation"):
        validate_twin_aware_stations(
            train_station_ids={"8664", "1000"},
            test_station_ids={"8665", "2000"},
            co_located_pairs=twin_pairs,
        )


def test_create_purged_embargoed_splits_workflow() -> None:
    dates = pd.date_range("2026-10-01T00:00:00Z", "2026-12-01T00:00:00Z", freq="1D")
    df = pd.DataFrame({"issuance_utc": dates, "val": range(len(dates))})

    partitions = [
        ("train", "2026-10-01T00:00:00Z", "2026-10-25T00:00:00Z"),
        ("valid", "2026-10-29T00:00:00Z", "2026-11-10T00:00:00Z"),  # 4 days (96 h) embargo
        ("test", "2026-11-15T00:00:00Z", "2026-11-30T00:00:00Z"),  # 5 days (120 h) embargo
    ]

    splits = create_purged_embargoed_splits(
        df, partitions, lookback_h=72, horizon_max_h=72, min_embargo_h=72
    )

    assert set(splits.keys()) == {"train", "valid", "test"}
    assert not splits["train"].empty
    assert not splits["valid"].empty
    assert not splits["test"].empty

    # Train max date <= 2026-10-22 (25th - 72 h)
    assert splits["train"]["issuance_utc"].max() <= pd.Timestamp("2026-10-22T00:00:00Z")
    # Valid min date >= 2026-11-01 (29th + 72 h)
    assert splits["valid"]["issuance_utc"].min() >= pd.Timestamp("2026-11-01T00:00:00Z")
