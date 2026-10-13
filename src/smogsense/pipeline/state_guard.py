"""smogsense.pipeline.state_guard - State branch size guard and yearly rotation.

Enforces storage budgets on the orphan `state` branch (warn at 400 MB, alert at 700 MB),
verifies journal file integrity, blocks raw binary leaks, and supports yearly archival
rotation.

Specification: docs/deployment-and-ops.md §10 -> 'Cost and quota monitoring'
"""

from __future__ import annotations

import json
import os
import tarfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd
from structlog import get_logger

logger = get_logger(__name__)

FORBIDDEN_EXTENSIONS: tuple[str, ...] = (
    ".grib",
    ".grib2",
    ".idx",
    ".nc",
    ".nc4",
    ".bin",
)


def compute_directory_size_bytes(directory: Path) -> int:
    """Compute total size in bytes of all files in directory."""
    if not directory.exists():
        return 0
    total = 0
    for root, _, files in os.walk(directory):
        for f in files:
            p = Path(root) / f
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def verify_state_branch(
    state_dir: Path = Path(".state"),
    warn_mb: float = 400.0,
    alert_mb: float = 700.0,
) -> dict[str, Any]:
    """Verify size guard and journal integrity of state directory (Task P1-23).

    Checks:
    1. Size threshold against warn_mb (400 MB) and alert_mb (700 MB).
    2. Absence of forbidden raw binary formats (*.grib, *.nc, etc.).
    3. Structural integrity of Parquet and JSON journals.
    """
    state_path = Path(state_dir)
    if not state_path.exists():
        return {
            "status": "ok",
            "size_mb": 0.0,
            "warn_mb": warn_mb,
            "alert_mb": alert_mb,
            "forbidden_files": [],
            "readable_counts": {},
            "corrupt_files": [],
            "valid": True,
        }

    total_bytes = compute_directory_size_bytes(state_path)
    size_mb = round(total_bytes / (1024.0 * 1024.0), 3)

    # Check for forbidden binary files
    forbidden_files: list[str] = []
    for root, _, files in os.walk(state_path):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in FORBIDDEN_EXTENSIONS:
                rel = str(Path(root, f).relative_to(state_path))
                forbidden_files.append(rel)

    # Check journal files
    readable_counts: dict[str, int] = {}
    corrupt_files: list[str] = []

    for category in ("manifests", "forecasts", "scores", "inputs", "obs", "obs_pull_log"):
        cat_dir = state_path / category
        if not cat_dir.exists():
            continue
        count = 0
        for p in cat_dir.rglob("*"):
            if not p.is_file():
                continue
            rel_name = str(p.relative_to(state_path))
            if p.suffix.lower() == ".json":
                try:
                    with p.open("r", encoding="utf-8") as json_file:
                        json.load(json_file)
                    count += 1
                except Exception:
                    corrupt_files.append(rel_name)
            elif p.suffix.lower() == ".parquet":
                try:
                    pd.read_parquet(p)
                    count += 1
                except Exception:
                    corrupt_files.append(rel_name)
        readable_counts[category] = count

    # Determine status
    if size_mb >= alert_mb or bool(forbidden_files) or bool(corrupt_files):
        status = "alert"
    elif size_mb >= warn_mb:
        status = "warn"
    else:
        status = "ok"

    valid = status in ("ok", "warn") and not forbidden_files and not corrupt_files

    return {
        "status": status,
        "size_mb": size_mb,
        "warn_mb": warn_mb,
        "alert_mb": alert_mb,
        "forbidden_files": forbidden_files,
        "readable_counts": readable_counts,
        "corrupt_files": corrupt_files,
        "valid": valid,
    }


def rotate_state_year(
    state_dir: Path,
    archive_year: int,
    bundle_out_dir: Path,
) -> dict[str, Any]:
    """Archive and prune historical partitions for a given year (Task P1-23).

    Partitions matching dt=YYYY-* or YYYY-MM-DD for archive_year are archived
    into a compressed tar archive in bundle_out_dir and removed from the active
    state directory.
    """
    state_path = Path(state_dir)
    bundle_out = Path(bundle_out_dir)
    bundle_out.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        return {
            "status": "skipped",
            "archive_year": archive_year,
            "files_archived": 0,
            "files_pruned": 0,
            "bundle_path": None,
        }

    year_prefix = f"{archive_year}-"
    dt_prefix = f"dt={archive_year}-"

    archived_files: list[Path] = []
    # Collect files matching the archive year
    for p in state_path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(state_path)
        rel_str = str(rel)

        # Preserve root README.md and .gitignore
        if rel_str in ("README.md", ".gitignore", ".git"):
            continue

        if dt_prefix in rel_str or year_prefix in p.name or f"_{archive_year}" in p.name:
            archived_files.append(p)

    if not archived_files:
        return {
            "status": "ok",
            "archive_year": archive_year,
            "files_archived": 0,
            "files_pruned": 0,
            "bundle_path": None,
        }

    archive_tar = bundle_out / f"state_{archive_year}.tar.gz"
    with tarfile.open(archive_tar, "w:gz") as tar:
        for f in archived_files:
            rel = f.relative_to(state_path)
            tar.add(f, arcname=str(rel))

    # Prune archived files and clean up empty parent directories
    pruned_count = 0
    dirs_to_check: set[Path] = set()
    for f in archived_files:
        try:
            dirs_to_check.add(f.parent)
            f.unlink()
            pruned_count += 1
        except OSError:
            continue

    # Remove empty directories
    for d in sorted(dirs_to_check, key=lambda p: len(str(p)), reverse=True):
        if d.exists() and d != state_path:
            with suppress(OSError):
                if not any(d.iterdir()):
                    d.rmdir()

    return {
        "status": "ok",
        "archive_year": archive_year,
        "files_archived": len(archived_files),
        "files_pruned": pruned_count,
        "bundle_path": str(archive_tar),
    }
