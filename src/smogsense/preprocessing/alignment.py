"""smogsense.preprocessing.alignment - Bitemporal as-of joins.

Implements the availability rule B*(T) = max{B in {00Z, 12Z} : B + 10 h <= T} for CAMS and per-
source latency tables so that features at issuance T only use information that was knowable at
T.

Public contract (implemented in Phase 1):
- asof_cams_run(issuance) -> base_time
- stitch_cams_series(issuance, window) -> hourly series from the most recent knowable cycle for
  each valid hour

Specification: docs/data-engineering.md -> 'Temporal alignment'

Contract update: unified rule and corrected stitching.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd


def normalize_utc(dt: datetime) -> datetime:
    """Normalize datetime to UTC. If naive, assume UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def asof_cams_run(issuance: datetime) -> datetime:
    """
    Returns the most recent CAMS cycle B that is available at the issuance time T.
    Availability rule: B*(T) = max { B in {00Z, 12Z} : B + 10h <= T }
    """
    issuance = normalize_utc(issuance)

    available_at = issuance - timedelta(hours=10)

    date_part = available_at.date()
    cycle_12z = datetime(date_part.year, date_part.month, date_part.day, 12, tzinfo=UTC)
    cycle_00z = datetime(date_part.year, date_part.month, date_part.day, 0, tzinfo=UTC)

    if cycle_12z <= available_at:
        return cycle_12z
    if cycle_00z <= available_at:
        return cycle_00z

    raise ValueError("Unexpected error in asof_cams_run computation")


def stitch_cams_series(issuance: datetime, window_hours: int = 72) -> pd.DataFrame:
    """
    Stitch CAMS forecast information according to the temporal contract.
    Contract B — normal bitemporal stitching.
    Select: B = max { cycle : B + 10h <= t } with no <12h restriction.
    This produces a continuous series.
    """
    if window_hours < 0:
        raise ValueError("window_hours must be non-negative")

    issuance = normalize_utc(issuance)
    issuance = issuance.replace(minute=0, second=0, microsecond=0)

    records = []

    for i in range(window_hours, -1, -1):
        t = issuance - timedelta(hours=i)

        # B = max { cycle : B + 10h <= t }
        selected_b = asof_cams_run(t)

        records.append(
            {
                "target_hour_utc": t,
                "cams_cycle_utc": selected_b,
                "lead_time_hours": int((t - selected_b).total_seconds() / 3600.0),
            }
        )

    df = pd.DataFrame(records)
    df["lead_time_hours"] = df["lead_time_hours"].astype("Int64")
    return df
