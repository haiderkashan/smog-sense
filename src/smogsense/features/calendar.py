"""smogsense.features.calendar — Cyclical time encodings and event flags.

Implements:
- Cyclical sin/cos encodings for local hour-of-day, day-of-week, and day-of-year.
- Domain-specific event flags: Diwali flag using window around dates in
  data/external/static_covariates/diwali_dates.csv.
- Strict domain adaptation rule: The Diwali feature exists ONLY for Indian domains
  (delhi, amritsar, ludhiana) and is DROPPED (not zero-filled) when targeting Lahore.

Specification:
- docs/ml-architecture.md §2 ('Feature engineering (physics-informed)')
- configs/features.yaml ('calendar')
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Hardcoded fallback dates for Diwali in case CSV is missing
DEFAULT_DIWALI_DATES: dict[int, str] = {
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
    2021: "2021-11-04",
    2022: "2022-10-24",
    2023: "2023-11-12",
    2024: "2024-11-01",
    2025: "2025-10-20",
    2026: "2026-11-08",
    2027: "2027-10-29",
    2028: "2028-10-17",
    2029: "2029-11-05",
    2030: "2030-10-26",
}

INDIAN_DOMAINS: set[str] = {"delhi", "amritsar", "ludhiana"}

DOMAIN_TIMEZONES: dict[str, str] = {
    "lahore": "Asia/Karachi",
    "delhi": "Asia/Kolkata",
    "amritsar": "Asia/Kolkata",
    "ludhiana": "Asia/Kolkata",
}


def load_diwali_dates(csv_path: str | Path | None = None) -> set[date]:
    """Load Diwali calendar dates from CSV or fall back to verified defaults."""
    candidates = [
        Path("data/external/static_covariates/diwali_dates.csv"),
        Path("/app/data/external/static_covariates/diwali_dates.csv"),
        Path(__file__).resolve().parents[3] / "data/external/static_covariates/diwali_dates.csv",
    ]
    if csv_path is not None:
        candidates.insert(0, Path(csv_path))

    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "date" in df.columns:
                    return {pd.to_datetime(d).date() for d in df["date"].dropna()}
            except (OSError, pd.errors.EmptyDataError, ValueError):
                pass

    return {pd.to_datetime(d).date() for d in DEFAULT_DIWALI_DATES.values()}


def compute_cyclical_time_features(
    dt_utc: datetime | pd.Timestamp,
    timezone: str = "Asia/Karachi",
) -> dict[str, float]:
    """Compute local sin/cos encodings for hour-of-day, day-of-week, and day-of-year."""
    t_utc = pd.to_datetime(dt_utc, utc=True)
    tz = ZoneInfo(timezone)
    local_dt = t_utc.astimezone(tz)

    hour = local_dt.hour + (local_dt.minute / 60.0)
    dow = local_dt.weekday()  # Monday=0, Sunday=6
    doy = local_dt.timetuple().tm_yday

    sin_hour = float(np.sin(2.0 * np.pi * hour / 24.0))
    cos_hour = float(np.cos(2.0 * np.pi * hour / 24.0))

    sin_dow = float(np.sin(2.0 * np.pi * dow / 7.0))
    cos_dow = float(np.cos(2.0 * np.pi * dow / 7.0))

    # Use 365.25 for leap-year continuity
    sin_doy = float(np.sin(2.0 * np.pi * doy / 365.25))
    cos_doy = float(np.cos(2.0 * np.pi * doy / 365.25))

    return {
        "sin_hour": sin_hour,
        "cos_hour": cos_hour,
        "sin_dow": sin_dow,
        "cos_dow": cos_dow,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
    }


def is_diwali_window(
    dt_utc: datetime | pd.Timestamp,
    timezone: str = "Asia/Kolkata",
    window_days: int = 1,
    diwali_dates: set[date] | None = None,
) -> bool:
    """Return True if local date falls within [-window_days, +window_days] of Diwali."""
    t_utc = pd.to_datetime(dt_utc, utc=True)
    tz = ZoneInfo(timezone)
    local_date = t_utc.astimezone(tz).date()

    dates = diwali_dates if diwali_dates is not None else load_diwali_dates()
    return any(abs((local_date - d).days) <= window_days for d in dates)


def extract_block_calendar_features(
    issuance_utc: datetime | pd.Timestamp,
    domain: str = "lahore",
    horizons: Sequence[int] = (24, 48, 72),
    diwali_dates_file: str | Path | None = None,
) -> dict[str, float]:
    """Extract model-ready calendar features for forecast horizons.

    The Diwali flag is included ONLY for Indian domains (delhi, amritsar, ludhiana)
    and is strictly omitted (not zero-filled) for Lahore.
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    domain_clean = domain.lower().strip()
    tz_name = DOMAIN_TIMEZONES.get(domain_clean, "Asia/Karachi")

    features: dict[str, float] = {}

    # 1. Issuance reference time cyclical features
    iss_cyclical = compute_cyclical_time_features(t_iss, timezone=tz_name)
    for k, v in iss_cyclical.items():
        features[f"f_cal_{k}"] = v

    # 2. Horizon target block mid-points cyclical encodings
    is_indian = domain_clean in INDIAN_DOMAINS
    diwali_set = load_diwali_dates(diwali_dates_file) if is_indian else set()

    for h in horizons:
        # Mid-point of target block [T+h-24, T+h) is T+h-12
        mid_block = t_iss + timedelta(hours=int(h) - 12)
        mid_cyclical = compute_cyclical_time_features(mid_block, timezone=tz_name)

        features[f"f_cal_sin_dow_h{h}"] = mid_cyclical["sin_dow"]
        features[f"f_cal_cos_dow_h{h}"] = mid_cyclical["cos_dow"]
        features[f"f_cal_sin_doy_h{h}"] = mid_cyclical["sin_doy"]
        features[f"f_cal_cos_doy_h{h}"] = mid_cyclical["cos_doy"]

        # Diwali event flag: strictly omitted for Lahore
        if is_indian:
            in_diwali = is_diwali_window(
                mid_block, timezone=tz_name, window_days=1, diwali_dates=diwali_set
            )
            features[f"f_cal_diwali_h{h}"] = 1.0 if in_diwali else 0.0

    return features
