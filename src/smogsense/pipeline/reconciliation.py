"""smogsense.pipeline.reconciliation — Reconcile live API observations with settled AWS archive.

Specification:
- docs/data-engineering.md §11 (Temporal alignment & provisional-vs-settled truth)
- docs/PRD.md FR-37 (Observation capture and revision policy)
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from structlog import get_logger

logger = get_logger(__name__)


def reconcile_api_and_archive(
    api_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """Compare overlapping observations between live OpenAQ API and settled AWS archive.

    Quantifies provisional-vs-settled skew, mean absolute difference,
    percentage of identical observations within tolerance, and revision counts.

    Parameters
    ----------
    api_df : pd.DataFrame
        DataFrame containing live API observations with ts_utc, pm25_ugm3, and sensor_id/location_id.
    archive_df : pd.DataFrame
        DataFrame containing settled archive observations with ts_utc, pm25_ugm3, and sensor_id/location_id.
    tolerance : float, default 0.1
        Tolerance in ug/m3 below which observations are considered identical.

    Returns
    -------
    dict[str, Any]
        Reconciliation metrics and status summary.
    """
    if api_df.empty and archive_df.empty:
        return {
            "overlap_count": 0,
            "api_rows": 0,
            "archive_rows": 0,
            "mean_abs_diff": 0.0,
            "pct_identical": 0.0,
            "revisions_count": 0,
            "max_diff": 0.0,
            "added_timestamps_count": 0,
            "dropped_timestamps_count": 0,
            "tolerance": float(tolerance),
            "status": "empty_input",
        }

    api_clean = api_df.copy()
    archive_clean = archive_df.copy()

    # Ensure UTC datetime on timestamps
    if "ts_utc" in api_clean.columns and not api_clean.empty:
        api_clean["ts_utc"] = pd.to_datetime(api_clean["ts_utc"], utc=True)
    if "ts_utc" in archive_clean.columns and not archive_clean.empty:
        archive_clean["ts_utc"] = pd.to_datetime(archive_clean["ts_utc"], utc=True)

    # Filter to valid PM2.5 values
    if "pm25_ugm3" in api_clean.columns and not api_clean.empty:
        api_clean = api_clean[api_clean["pm25_ugm3"].notna()].copy()
    if "pm25_ugm3" in archive_clean.columns and not archive_clean.empty:
        archive_clean = archive_clean[archive_clean["pm25_ugm3"].notna()].copy()

    api_rows = len(api_clean)
    archive_rows = len(archive_clean)

    if api_rows == 0 or archive_rows == 0:
        added_count = archive_rows if api_rows == 0 else 0
        dropped_count = api_rows if archive_rows == 0 else 0
        return {
            "overlap_count": 0,
            "api_rows": api_rows,
            "archive_rows": archive_rows,
            "mean_abs_diff": 0.0,
            "pct_identical": 0.0,
            "revisions_count": 0,
            "max_diff": 0.0,
            "added_timestamps_count": added_count,
            "dropped_timestamps_count": dropped_count,
            "tolerance": float(tolerance),
            "status": "empty_input" if api_rows == 0 and archive_rows == 0 else "no_overlap",
        }

    # Determine join key hierarchy: sensor_id + ts_utc -> location_id + ts_utc -> ts_utc
    if "sensor_id" in api_clean.columns and "sensor_id" in archive_clean.columns:
        join_key = ["sensor_id", "ts_utc"]
    elif "location_id" in api_clean.columns and "location_id" in archive_clean.columns:
        join_key = ["location_id", "ts_utc"]
    elif "ts_utc" in api_clean.columns and "ts_utc" in archive_clean.columns:
        join_key = ["ts_utc"]
    else:
        return {
            "overlap_count": 0,
            "api_rows": len(api_clean),
            "archive_rows": len(archive_clean),
            "mean_abs_diff": 0.0,
            "pct_identical": 0.0,
            "revisions_count": 0,
            "max_diff": 0.0,
            "added_timestamps_count": 0,
            "dropped_timestamps_count": 0,
            "tolerance": float(tolerance),
            "status": "missing_keys",
        }

    api_clean = api_clean.drop_duplicates(subset=join_key)
    archive_clean = archive_clean.drop_duplicates(subset=join_key)
    api_rows = len(api_clean)
    archive_rows = len(archive_clean)

    # Track timestamp / key coverage
    api_key_set = set(api_clean[join_key].itertuples(index=False, name=None))
    archive_key_set = set(archive_clean[join_key].itertuples(index=False, name=None))

    added_keys = archive_key_set - api_key_set
    dropped_keys = api_key_set - archive_key_set

    merged = api_clean[[*join_key, "pm25_ugm3"]].merge(
        archive_clean[[*join_key, "pm25_ugm3"]],
        on=join_key,
        suffixes=("_api", "_archive"),
    )

    overlap_count = len(merged)
    if overlap_count == 0:
        return {
            "overlap_count": 0,
            "api_rows": api_rows,
            "archive_rows": archive_rows,
            "mean_abs_diff": 0.0,
            "pct_identical": 0.0,
            "revisions_count": 0,
            "max_diff": 0.0,
            "added_timestamps_count": len(added_keys),
            "dropped_timestamps_count": len(dropped_keys),
            "tolerance": float(tolerance),
            "status": "no_overlap",
        }

    diffs = (merged["pm25_ugm3_api"] - merged["pm25_ugm3_archive"]).abs()
    mean_abs_diff = float(diffs.mean())
    max_diff = float(diffs.max())
    identical_count = int((diffs <= tolerance).sum())
    pct_identical = float((identical_count / overlap_count) * 100.0)
    revisions_count = int((diffs > tolerance).sum())

    stats = {
        "overlap_count": overlap_count,
        "api_rows": api_rows,
        "archive_rows": archive_rows,
        "mean_abs_diff": round(mean_abs_diff, 4),
        "pct_identical": round(pct_identical, 2),
        "revisions_count": revisions_count,
        "max_diff": round(max_diff, 4),
        "added_timestamps_count": len(added_keys),
        "dropped_timestamps_count": len(dropped_keys),
        "tolerance": float(tolerance),
        "status": "reconciled",
    }
    logger.info("reconciliation_metrics_calculated", **stats)
    return stats


def _load_dataframes_from_sources(candidates: list[Path]) -> pd.DataFrame:
    """Read and concatenate parquet files found in candidate paths."""
    dfs: list[pd.DataFrame] = []
    for cand in candidates:
        if cand.is_file() and cand.suffix == ".parquet":
            try:
                dfs.append(pd.read_parquet(cand))
            except Exception as exc:
                logger.debug("reconciliation_file_read_failed", file=str(cand), error=str(exc))
        elif cand.is_dir():
            for p in sorted(cand.glob("*.parquet")):
                try:
                    dfs.append(pd.read_parquet(p))
                except Exception as exc:
                    logger.debug("reconciliation_dir_file_read_failed", file=str(p), error=str(exc))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def run_reconciliation(
    domain: str = "lahore",
    tolerance: float = 0.1,
    api_path: Path | str | None = None,
    archive_path: Path | str | None = None,
    output_report: Path | str | None = None,
) -> dict[str, Any]:
    """Execute reconciliation between OpenAQ API and settled archive observations."""
    logger.info("starting_reconciliation", domain=domain, tolerance=tolerance)

    # 1. Resolve API dataframe
    if api_path is not None:
        api_candidates = [Path(api_path)]
    else:
        api_candidates = [
            Path(f".state/inputs/{domain}"),
            Path(f".state/obs/{domain}"),
            Path(f"data/interim/obs/{domain}"),
            Path(".state/obs_pull_log"),
        ]
    api_df = _load_dataframes_from_sources(api_candidates)

    # 2. Resolve Archive dataframe
    if archive_path is not None:
        archive_candidates = [Path(archive_path)]
    else:
        archive_candidates = [
            Path("data/interim/backfill_pilot"),
            Path(f".state/archive/{domain}"),
            Path(f"data/archive/{domain}"),
        ]
    archive_df = _load_dataframes_from_sources(archive_candidates)

    stats = reconcile_api_and_archive(api_df, archive_df, tolerance=tolerance)
    stats["domain"] = domain
    stats["reconciled_at_utc"] = datetime.now(UTC).isoformat()

    # 3. Output report if requested or default report location
    report_file: Path | None = None
    if output_report is not None:
        report_file = Path(output_report)
    else:
        reports_dir = Path(".state/reports")
        if reports_dir.exists():
            report_file = (
                reports_dir / f"reconciliation_{domain}_{datetime.now(UTC).strftime('%Y%m%d')}.json"
            )

    if report_file is not None:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        stats["report_file"] = str(report_file)

    logger.info("reconciliation_complete", **stats)
    return stats
