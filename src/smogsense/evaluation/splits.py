"""smogsense.evaluation.splits — Purged, embargoed, blocked time-series splits.

72 h embargo between train/validation/test blocks; spatial hold-out by station; no sample whose
lookback or target window overlaps another split (López de Prado, 2018; T1-14).

Specification: docs/evaluation-strategy.md → 'Data splits and leakage control'
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd


def enforce_embargo(
    split_a_end: pd.Timestamp | datetime | str,
    split_b_start: pd.Timestamp | datetime | str,
    min_embargo_h: int = 72,
) -> bool:
    """Validate that split B starts at least min_embargo_h after split A ends.

    Args:
        split_a_end: Timestamp marking the end of the earlier split.
        split_b_start: Timestamp marking the start of the subsequent split.
        min_embargo_h: Minimum embargo duration in hours (default 72 h).

    Returns:
        True if embargo condition is satisfied.

    Raises:
        ValueError: If split_b_start is earlier than split_a_end + min_embargo_h.
    """
    t_a = pd.to_datetime(split_a_end, utc=True)
    t_b = pd.to_datetime(split_b_start, utc=True)

    required_gap = timedelta(hours=min_embargo_h)
    actual_gap = t_b - t_a

    if actual_gap < required_gap:
        raise ValueError(
            f"Temporal embargo violation: subsequent split starts at {t_b}, but prior split "
            f"ends at {t_a}. Required embargo gap is {min_embargo_h} h ({required_gap}), "
            f"but actual gap is only {actual_gap.total_seconds() / 3600.0:.2f} h."
        )
    return True


def purge_split_boundaries(
    df: pd.DataFrame,
    split_start: pd.Timestamp | datetime | str,
    split_end: pd.Timestamp | datetime | str,
    lookback_h: int = 72,
    horizon_max_h: int = 72,
    timestamp_col: str = "issuance_utc",
    has_prior_split: bool = True,
    has_subsequent_split: bool = True,
) -> pd.DataFrame:
    """Purge samples near split boundaries to prevent information leakage (López de Prado, 2018; T1-14).

    1. Forward lookback purging (if has_prior_split):
       Any issuance T with T < split_start + lookback_h has a lookback window [T - lookback_h, T]
       that extends into the preceding split, creating autocorrelation feature leakage.
    2. Backward target purging (if has_subsequent_split):
       Any issuance T with T + horizon_max_h > split_end has an evaluation target window
       [T, T + horizon_max_h] extending past the split boundary.

    Args:
        df: DataFrame containing issuance timestamps.
        split_start: Start timestamp of the split block.
        split_end: End timestamp of the split block.
        lookback_h: Duration of the feature history lookback in hours (default 72).
        horizon_max_h: Maximum target forecast horizon in hours (default 72).
        timestamp_col: Name of the timestamp column in df.
        has_prior_split: Whether an earlier split precedes this block (triggers lookback purge).
        has_subsequent_split: Whether a later split follows this block (triggers target purge).

    Returns:
        Purged DataFrame filtered to non-leaking issuance times.
    """
    if df.empty:
        return df.copy()

    t_start = pd.to_datetime(split_start, utc=True)
    t_end = pd.to_datetime(split_end, utc=True)

    series = pd.to_datetime(df[timestamp_col], utc=True)

    min_t = t_start + timedelta(hours=lookback_h) if has_prior_split else t_start
    max_t = t_end - timedelta(hours=horizon_max_h) if has_subsequent_split else t_end

    mask = (series >= min_t) & (series <= max_t)
    return df[mask].copy().reset_index(drop=True)


def validate_no_split_overlap(
    split_a_df: pd.DataFrame,
    split_b_df: pd.DataFrame,
    lookback_h: int = 72,
    horizon_max_h: int = 72,
    timestamp_col: str = "issuance_utc",
) -> None:
    """Assert that no sample's lookback or target window in split A overlaps with split B (T1-14).

    Args:
        split_a_df: DataFrame of split A issuances.
        split_b_df: DataFrame of split B issuances.
        lookback_h: Duration of lookback in hours.
        horizon_max_h: Maximum target horizon in hours.
        timestamp_col: Name of timestamp column.

    Raises:
        ValueError: If any temporal overlap between feature lookbacks and evaluation targets is detected.
    """
    if split_a_df.empty or split_b_df.empty:
        return

    ts_a = pd.to_datetime(split_a_df[timestamp_col], utc=True)
    ts_b = pd.to_datetime(split_b_df[timestamp_col], utc=True)

    # Determine chronological order
    min_a, max_a = ts_a.min(), ts_a.max()
    min_b, max_b = ts_b.min(), ts_b.max()

    if max_a <= min_b:
        earlier_ts, later_ts = ts_a, ts_b
        earlier_name, later_name = "split A", "split B"
    elif max_b <= min_a:
        earlier_ts, later_ts = ts_b, ts_a
        earlier_name, later_name = "split B", "split A"
    else:
        raise ValueError(
            f"Interleaved split violation: timestamps overlap directly. "
            f"Range A: [{min_a}, {max_a}], Range B: [{min_b}, {max_b}]."
        )

    # The latest possible target time for earlier split is max(earlier_ts) + horizon_max_h
    # The earliest possible lookback time for later split is min(later_ts) - lookback_h
    latest_target = earlier_ts.max() + timedelta(hours=horizon_max_h)
    earliest_lookback = later_ts.min() - timedelta(hours=lookback_h)

    if latest_target > earliest_lookback:
        overlap_h = (latest_target - earliest_lookback).total_seconds() / 3600.0
        raise ValueError(
            f"Temporal leakage detected between {earlier_name} and {later_name}: "
            f"Latest target of {earlier_name} reaches {latest_target}, but earliest "
            f"lookback of {later_name} reaches back to {earliest_lookback} "
            f"(overlap = {overlap_h:.1f} h)."
        )


def validate_twin_aware_stations(
    train_station_ids: set[str] | Sequence[str],
    test_station_ids: set[str] | Sequence[str],
    co_located_pairs: Sequence[tuple[str, str]] | pd.DataFrame,
) -> None:
    """Assert that co-located twin stations (distance <= 50 m) never straddle train and test splits (T1-14).

    Args:
        train_station_ids: Collection of station IDs in train split.
        test_station_ids: Collection of station IDs in test split.
        co_located_pairs: List of (station_1, station_2) tuples or DataFrame with columns station_a, station_b.

    Raises:
        ValueError: If any pair of twin stations has one station in train and the other in test.
    """
    train_set = {str(s) for s in train_station_ids}
    test_set = {str(s) for s in test_station_ids}

    pairs: list[tuple[str, str]] = []
    if isinstance(co_located_pairs, pd.DataFrame):
        for _, row in co_located_pairs.iterrows():
            col_a = (
                row.get("station_a")
                or row.get("location_id_1")
                or row.get("station_id_1")
                or row.iloc[0]
            )
            col_b = (
                row.get("station_b")
                or row.get("location_id_2")
                or row.get("station_id_2")
                or row.iloc[1]
            )
            pairs.append((str(col_a), str(col_b)))
    else:
        pairs = [(str(p[0]), str(p[1])) for p in co_located_pairs]

    violations: list[tuple[str, str]] = []
    for s1, s2 in pairs:
        in_train_1 = s1 in train_set
        in_train_2 = s2 in train_set
        in_test_1 = s1 in test_set
        in_test_2 = s2 in test_set

        # If s1 in train and s2 in test, or s2 in train and s1 in test
        if (in_train_1 and in_test_2) or (in_train_2 and in_test_1):
            violations.append((s1, s2))

    if violations:
        viol_str = ", ".join(f"({s1}, {s2})" for s1, s2 in violations)
        raise ValueError(
            f"Twin-aware split violation (T1-14): {len(violations)} co-located station pair(s) "
            f"straddle the train/test split boundary: [{viol_str}]. Co-located twins must "
            f"be assigned to the same partition."
        )


def create_purged_embargoed_splits(
    df: pd.DataFrame,
    partitions: Sequence[tuple[str, pd.Timestamp | datetime | str, pd.Timestamp | datetime | str]],
    lookback_h: int = 72,
    horizon_max_h: int = 72,
    min_embargo_h: int = 72,
    timestamp_col: str = "issuance_utc",
) -> dict[str, pd.DataFrame]:
    """Create a dictionary of cleanly purged and embargoed time-series partitions.

    Args:
        df: Input DataFrame containing timestamp_col.
        partitions: Sequence of (partition_name, start_time, end_time) in chronological order.
        lookback_h: Feature lookback duration.
        horizon_max_h: Maximum target forecast horizon.
        min_embargo_h: Required temporal embargo between partitions.
        timestamp_col: Name of issuance timestamp column.

    Returns:
        Dict mapping partition_name to purged DataFrame.
    """
    if len(partitions) == 0:
        return {}

    # Validate chronological order and embargoes
    for i in range(len(partitions) - 1):
        _, _, prev_end = partitions[i]
        _, next_start, _ = partitions[i + 1]
        enforce_embargo(prev_end, next_start, min_embargo_h=min_embargo_h)

    result: dict[str, pd.DataFrame] = {}
    for i, (name, start, end) in enumerate(partitions):
        has_prior = i > 0
        has_subsequent = i < (len(partitions) - 1)
        part_df = purge_split_boundaries(
            df,
            split_start=start,
            split_end=end,
            lookback_h=lookback_h,
            horizon_max_h=horizon_max_h,
            timestamp_col=timestamp_col,
            has_prior_split=has_prior,
            has_subsequent_split=has_subsequent,
        )
        result[name] = part_df

    # Verify cross-split isolation
    part_names = list(result.keys())
    for i in range(len(part_names) - 1):
        validate_no_split_overlap(
            result[part_names[i]],
            result[part_names[i + 1]],
            lookback_h=lookback_h,
            horizon_max_h=horizon_max_h,
            timestamp_col=timestamp_col,
        )

    return result


__all__ = [
    "create_purged_embargoed_splits",
    "enforce_embargo",
    "purge_split_boundaries",
    "validate_no_split_overlap",
    "validate_twin_aware_stations",
]
