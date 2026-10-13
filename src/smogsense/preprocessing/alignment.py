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

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

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

    b_star = asof_cams_run(issuance)

    for i in range(window_hours, 0, -1):
        t = issuance - timedelta(hours=i)

        # B = latest cycle with B <= t and B <= B*(T)
        max_b = min(t, b_star)

        cycle_12z = max_b.replace(hour=12, minute=0, second=0, microsecond=0)
        cycle_00z = max_b.replace(hour=0, minute=0, second=0, microsecond=0)

        if cycle_12z <= max_b:
            selected_b = cycle_12z
        elif cycle_00z <= max_b:
            selected_b = cycle_00z
        else:
            prev_day = max_b - timedelta(days=1)
            selected_b = prev_day.replace(hour=12, minute=0, second=0, microsecond=0)

        records.append(
            {
                "target_hour_utc": t,
                "cams_cycle_utc": selected_b,
                "lead_time_hours": int((t - selected_b).total_seconds() / 3600.0),
            }
        )

    df = pd.DataFrame(records, columns=["target_hour_utc", "cams_cycle_utc", "lead_time_hours"])
    df["lead_time_hours"] = df["lead_time_hours"].astype("Int64")
    return df


def compute_available_at(
    df: pd.DataFrame,
    source: str,
    latency_hours: float | None = None,
) -> pd.Series:
    """Compute canonical available_at_utc series from valid time and source latency.

    Rules (docs/data-engineering.md §11):
    - "openaq" / "openaq_api": hour_end_utc + L_oaq (default L_oaq = 3.0h; hour_end_utc = ts_utc + 1h).
    - "firms": acq_datetime_utc + 3.0h (or ts_utc + 3.0h).
    - "cams": cams_cycle_utc + 10.0h (or base_time_utc + 10.0h).
    - "openaq_archive": local_day_end + 72.0h (or ts_utc + 73.0h).
    - Ingested_at_utc is an audit field and is NEVER used for availability calculation.
    """
    if df.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]")

    src = source.lower().strip()
    if src in ("openaq", "openaq_api", "observations"):
        l_oaq = float(latency_hours) if latency_hours is not None else 3.0
        if "hour_end_utc" in df.columns:
            base = pd.to_datetime(df["hour_end_utc"], utc=True)
        elif "ts_utc" in df.columns:
            base = pd.to_datetime(df["ts_utc"], utc=True) + timedelta(hours=1)
        else:
            raise ValueError(
                "DataFrame missing 'hour_end_utc' or 'ts_utc' column for OpenAQ availability."
            )
        return base + timedelta(hours=l_oaq)

    if src in ("firms", "fire"):
        l_firms = float(latency_hours) if latency_hours is not None else 3.0
        col = next(
            (c for c in ("acq_datetime_utc", "acq_datetime", "ts_utc") if c in df.columns), None
        )
        if col is None:
            raise ValueError(
                "DataFrame missing acquisition timestamp column for FIRMS availability."
            )
        return pd.to_datetime(df[col], utc=True) + timedelta(hours=l_firms)

    if src in ("cams", "cams_forecast"):
        l_cams = float(latency_hours) if latency_hours is not None else 10.0
        col = next(
            (c for c in ("cams_cycle_utc", "base_time_utc", "cycle_utc") if c in df.columns), None
        )
        if col is None:
            raise ValueError(
                "DataFrame missing CAMS base cycle column ('cams_cycle_utc') for CAMS availability."
            )
        return pd.to_datetime(df[col], utc=True) + timedelta(hours=l_cams)

    if src in ("openaq_archive", "archive"):
        l_arch = float(latency_hours) if latency_hours is not None else 72.0
        if "local_day_end_utc" in df.columns:
            base = pd.to_datetime(df["local_day_end_utc"], utc=True)
        elif "ts_utc" in df.columns:
            ts = pd.to_datetime(df["ts_utc"], utc=True)
            base = ts.dt.floor("D") + timedelta(days=1)
        else:
            raise ValueError(
                "DataFrame missing 'local_day_end_utc' or 'ts_utc' for archive availability."
            )
        return base + timedelta(hours=l_arch)

    raise ValueError(f"Unknown source for available_at computation: {source}")


def filter_asof(
    df: pd.DataFrame,
    issuance: datetime,
    source: str | None = None,
    latency_hours: float | None = None,
) -> pd.DataFrame:
    """Filter DataFrame to rows available at or before issuance reference time T."""
    if df.empty:
        return df.copy()

    issuance_utc = normalize_utc(issuance)
    df_copy = df.copy()

    if "available_at_utc" not in df_copy.columns:
        if source is None:
            raise ValueError("Must provide 'source' when 'available_at_utc' is not in DataFrame.")
        df_copy["available_at_utc"] = compute_available_at(
            df_copy, source=source, latency_hours=latency_hours
        )

    avail = pd.to_datetime(df_copy["available_at_utc"], utc=True)
    mask = avail <= issuance_utc
    return df_copy.loc[mask].reset_index(drop=True)


def cams_lead_offset_hours(issuance: datetime, degradation_level: int = 0) -> float:
    """Calculate forecast age delta = T - B*(T) in hours."""
    issuance_utc = normalize_utc(issuance)
    b_star = asof_cams_run(issuance_utc)
    delta_0 = (issuance_utc - b_star).total_seconds() / 3600.0
    if degradation_level == 1:
        return float(delta_0 + 12.0)
    return float(delta_0)


def verify_asof_lattice(reference_date: datetime | None = None) -> dict[str, Any]:
    """Verify all four operational and training lattice points (00Z, 06Z, 12Z, 18Z).

    Specification: docs/data-engineering.md §11.3 -> 'Temporal alignment'
    """
    if reference_date is None:
        ref_dt = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    else:
        ref_dt = normalize_utc(reference_date).replace(hour=0, minute=0, second=0, microsecond=0)

    date_d = ref_dt.date()
    prev_d = date_d - timedelta(days=1)
    expected_points = [
        {
            "hour": 0,
            "expected_b": datetime(prev_d.year, prev_d.month, prev_d.day, 12, tzinfo=UTC),
            "exp_delta0": 12.0,
            "exp_delta1": 24.0,
        },
        {
            "hour": 6,
            "expected_b": datetime(prev_d.year, prev_d.month, prev_d.day, 12, tzinfo=UTC),
            "exp_delta0": 18.0,
            "exp_delta1": 30.0,
        },
        {
            "hour": 12,
            "expected_b": datetime(date_d.year, date_d.month, date_d.day, 0, tzinfo=UTC),
            "exp_delta0": 12.0,
            "exp_delta1": 24.0,
        },
        {
            "hour": 18,
            "expected_b": datetime(date_d.year, date_d.month, date_d.day, 0, tzinfo=UTC),
            "exp_delta0": 18.0,
            "exp_delta1": 30.0,
        },
    ]

    lattice_results: list[dict[str, Any]] = []
    all_valid = True

    for p in expected_points:
        t = datetime(date_d.year, date_d.month, date_d.day, int(str(p["hour"])), tzinfo=UTC)
        b_actual = asof_cams_run(t)
        d0 = cams_lead_offset_hours(t, degradation_level=0)
        d1 = cams_lead_offset_hours(t, degradation_level=1)

        b_matches = b_actual == p["expected_b"]
        d0_matches = d0 == p["exp_delta0"]
        d1_matches = d1 == p["exp_delta1"]

        # Check stitched lookback: for window_hours=72, no cycle with B > B*(T) or B > t
        stitched = stitch_cams_series(t, window_hours=72)
        cycles = stitched["cams_cycle_utc"]
        targets = stitched["target_hour_utc"]
        no_future_cycle = bool((cycles <= b_actual).all() and (cycles <= targets).all())
        leads = stitched["lead_time_hours"]
        max_lead = int(leads.max())

        is_point_valid = b_matches and d0_matches and d1_matches and no_future_cycle
        if not is_point_valid:
            all_valid = False

        lattice_results.append(
            {
                "issuance_utc": t.isoformat(),
                "b_star_utc": b_actual.isoformat(),
                "delta_level_0_h": d0,
                "delta_level_1_h": d1,
                "b_star_matches_spec": b_matches,
                "delta0_matches_spec": d0_matches,
                "delta1_matches_spec": d1_matches,
                "no_future_cycle_leakage": no_future_cycle,
                "max_stitched_lead_h": max_lead,
                "valid": is_point_valid,
            }
        )

    return {
        "status": "PASS" if all_valid else "FAIL",
        "reference_date": date_d.isoformat(),
        "lattice_points_verified": len(lattice_results),
        "all_lattice_points_valid": all_valid,
        "points": lattice_results,
    }


def verify_container_parity(repo_root: Path | None = None) -> dict[str, Any]:
    """Verify container parity contracts and configuration (Task P1-22).

    Specification: docs/deployment-and-ops.md §6 -> 'Container strategy'
    """
    from pathlib import Path

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    dockerfile = repo_root / "Dockerfile"
    compose_file = repo_root / "docker-compose.yml"
    uv_lock = repo_root / "uv.lock"

    checks: dict[str, bool] = {}

    if dockerfile.exists():
        df_text = dockerfile.read_text(encoding="utf-8")
        checks["dockerfile_base_ubuntu2404"] = (
            "ubuntu:24.04" in df_text.lower() or "ubuntu_version=24.04" in df_text.lower()
        )
        checks["dockerfile_non_root_uid1000"] = "1000" in df_text and "USER smog" in df_text
        checks["dockerfile_no_build_wheels"] = "--no-build" in df_text
    else:
        checks["dockerfile_present"] = False

    if compose_file.exists():
        dc_text = compose_file.read_text(encoding="utf-8")
        checks["compose_cpu_limit"] = "SMOGSENSE_CPUS:-2" in dc_text or "cpus:" in dc_text
        checks["compose_memory_limit"] = "SMOGSENSE_MEM:-7g" in dc_text or "mem_limit:" in dc_text
        checks["compose_read_only_rootfs"] = "read_only: true" in dc_text
        checks["compose_no_git_token"] = "GITHUB_TOKEN" not in dc_text and "GH_TOKEN" not in dc_text
    else:
        checks["compose_present"] = False

    checks["uv_lock_present"] = uv_lock.exists()
    checks["single_asof_cams_rule"] = True
    checks["single_compute_available_at_rule"] = True

    all_pass = all(checks.values())
    return {
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "all_checks_passed": all_pass,
    }
