"""Integration tests for fast offline daily pipeline execution (Task P1-14).

Verifies:
1. Fast offline daily pipeline runs in < 120 seconds using offline fixtures.
2. Generates canonical input snapshot (.state/inputs/lahore/dt=YYYY-MM-DD.parquet).
3. Generates schema-valid forecast log (.state/forecasts/forecast_*.parquet).
4. Generates dual-written manifest files (.state/manifests/run_*.json and YYYY-MM-DD.json).
5. Generates public bulletin and site (site/forecast/latest.json and site/index.html).
6. Enforces idempotency (exit code 11 on duplicate run without force).
7. Supports forced rerun (--force -> exit code 10, marks is_rerun=True on rerun manifest).
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.features.snapshot import load_input_snapshot
from smogsense.pipeline.orchestrator import load_pandera_schema, run_daily_pipeline
from smogsense.utils.io import validate_frame


@pytest.fixture
def clean_state() -> Generator[None, None, None]:
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
        Path(".state/inputs/lahore"),
        Path(".state/obs_pull_log"),
        Path("site/forecast"),
    ]
    files = [
        Path(".state/artifacts/seed_history.json"),
        Path("site/latest.json"),
        Path("site/index.html"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()
    # Also clean any site/run_*.json
    if Path("site").is_dir():
        for f in Path("site").glob("run_*.json"):
            with contextlib.suppress(Exception):
                f.unlink()

    yield

    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()
    if Path("site").is_dir():
        for f in Path("site").glob("run_*.json"):
            with contextlib.suppress(Exception):
                f.unlink()


def test_fast_offline_daily_pipeline_cli(clean_state: None) -> None:
    """Test fast offline daily pipeline run via CLI runner."""
    runner = CliRunner()
    issuance_str = "2026-10-15T00:00:00Z"
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    run_id = f"run_{issuance_dt.strftime('%Y%m%d_%H%M')}"
    canonical_date = issuance_dt.strftime("%Y-%m-%d")

    start_time = time.perf_counter()
    result = runner.invoke(
        app,
        [
            "run",
            "daily",
            "--mode",
            "fixtures",
            "--issuance",
            issuance_str,
        ],
    )
    duration = time.perf_counter() - start_time

    # Must complete in under 120 seconds (typically < 10s)
    assert duration < 120.0, f"Pipeline execution took {duration:.2f}s (budget: 120s)"
    assert result.exit_code == 10, f"Pipeline failed with output:\n{result.output}"

    # 1. Verify input snapshot
    snapshot_path = Path(f".state/inputs/lahore/dt={canonical_date}.parquet")
    assert snapshot_path.exists(), f"Snapshot missing at {snapshot_path}"
    snapshot = load_input_snapshot("lahore", issuance_dt, snapshot_path=snapshot_path)
    assert snapshot.domain == "lahore"
    assert snapshot.issuance_utc == issuance_dt
    assert not snapshot.obs_df.empty, "Snapshot observations DataFrame is empty"
    assert not snapshot.cams_df.empty, "Snapshot CAMS DataFrame is empty"
    assert snapshot.fire_exposure_df is None or not snapshot.fire_exposure_df.empty

    # 2. Verify forecast log schema and content
    forecast_path = Path(f".state/forecasts/forecast_{run_id}.parquet")
    assert forecast_path.exists(), f"Forecast log missing at {forecast_path}"
    forecast_df = pd.read_parquet(forecast_path)
    # Validate against forecast_log Pandera schema
    schema = load_pandera_schema("forecast_log.schema.yaml")
    validate_frame(forecast_df, schema)
    assert (forecast_df["method"] == "m1_cams_raw").all()
    assert (forecast_df["domain"] == "lahore").all()
    assert len(forecast_df) > 0

    # 3. Verify manifest dual-write
    run_manifest_path = Path(f".state/manifests/{run_id}.json")
    date_manifest_path = Path(f".state/manifests/{canonical_date}.json")
    assert run_manifest_path.exists(), f"Run manifest missing: {run_manifest_path}"
    assert date_manifest_path.exists(), f"Date manifest missing: {date_manifest_path}"

    with run_manifest_path.open("r", encoding="utf-8") as f:
        run_manifest = json.load(f)
    with date_manifest_path.open("r", encoding="utf-8") as f:
        date_manifest = json.load(f)

    assert run_manifest["published"] is True
    assert run_manifest["is_rerun"] is False
    assert run_manifest["exit_code"] == 10
    assert date_manifest["published"] is True
    assert date_manifest["run_id"] == run_id

    # 4. Verify bulletin and website outputs
    bulletin_path = Path("site/forecast/latest.json")
    assert bulletin_path.exists(), f"Bulletin missing at {bulletin_path}"
    with bulletin_path.open("r", encoding="utf-8") as f:
        bulletin = json.load(f)
    assert (
        bulletin["domain"]["id"] == "lahore"
        if isinstance(bulletin["domain"], dict)
        else bulletin["domain"] == "lahore"
    )
    assert bulletin["mode"] == "baseline_only"
    assert len(bulletin["horizons"]) == 3

    index_html_path = Path("site/index.html")
    assert index_html_path.exists(), "site/index.html was not generated"
    html_content = index_html_path.read_text(encoding="utf-8")
    assert len(html_content) > 100
    assert "<html" in html_content.lower()

    # 5. Verify idempotency: duplicate run without force returns exit code 11
    result_idempotent = runner.invoke(
        app,
        [
            "run",
            "daily",
            "--mode",
            "fixtures",
            "--issuance",
            issuance_str,
        ],
    )
    assert result_idempotent.exit_code == 11

    # 6. Verify forced rerun: running with --force returns exit code 10
    result_force = runner.invoke(
        app,
        [
            "run",
            "daily",
            "--mode",
            "fixtures",
            "--issuance",
            issuance_str,
            "--force",
        ],
    )
    assert result_force.exit_code == 10
    rerun_manifests = sorted(Path(".state/manifests").glob(f"{run_id}_rerun_*.json"))
    assert len(rerun_manifests) >= 1
    with rerun_manifests[0].open("r", encoding="utf-8") as f:
        rerun_m = json.load(f)
    assert rerun_m["is_rerun"] is True


def test_fast_offline_daily_pipeline_python_api(clean_state: None) -> None:
    """Test fast offline daily pipeline run directly via orchestrator API."""
    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)

    start_time = time.perf_counter()
    exit_code = run_daily_pipeline(issuance_dt, mode="fixtures")
    duration = time.perf_counter() - start_time

    assert duration < 120.0
    assert exit_code == 10

    # Verify idempotency
    exit_code_again = run_daily_pipeline(issuance_dt, mode="fixtures")
    assert exit_code_again == 11

    # Verify force rerun
    exit_code_force = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
    assert exit_code_force == 10
