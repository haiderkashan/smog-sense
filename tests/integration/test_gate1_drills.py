"""Operational drills for Gate 1 (Task P1-08).

Verifies the four mandatory Gate 1 operational drills ahead of scheduled heartbeat:
1. Drill 1: Initial publication, dual-manifest persistence, and valid_until (+30h).
2. Drill 2: Idempotent early exit (rc=11) on repeated runs without force.
3. Drill 3: Watchdog state verification (smogsense state exists returns 11 when published, 0 when absent).
4. Drill 4: Rerun isolation (distinct rerun manifest with is_rerun=true, preserving original run).

Specification:
- docs/system-architecture.md -> 'Component responsibilities and CLI contract'
- docs/deployment-and-ops.md -> 'Runbook 1: Scheduled daily pipeline'
- docs/test-specifications.md -> 'Gate 1 verification drills'
"""

import contextlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.pipeline.orchestrator import run_daily_pipeline

runner = CliRunner()


@pytest.fixture
def clean_state():
    """Ensure a spotless ledger and manifest directory before and after drills."""
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
        Path(".state/inputs/lahore"),
        Path(".state/obs_pull_log"),
        Path("site/forecast"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    yield
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()


def test_gate1_drill_1_initial_publication(clean_state: None) -> None:
    """Drill 1: Initial publication succeeds, writes dual manifests, and sets valid_until (+30h)."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # 1. Run pipeline in offline fixtures mode
    exit_code = run_daily_pipeline(issuance, force=False, mode="fixtures")
    assert exit_code in (0, 10), f"Expected 0 or 10, got {exit_code}"

    # 2. Check canonical daily manifest: 2026-10-19.json
    canonical_manifest_path = Path(".state/manifests/2026-10-19.json")
    assert canonical_manifest_path.exists(), "Canonical manifest 2026-10-19.json must exist"
    canonical_data = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    assert canonical_data.get("published") is True
    assert canonical_data.get("is_rerun") is False
    assert canonical_data.get("run_id") == "run_20261019_0017"

    # 3. Check timestamped run manifest: run_20261019_0017.json
    run_manifest_path = Path(".state/manifests/run_20261019_0017.json")
    assert run_manifest_path.exists(), "Timestamped run manifest run_20261019_0017.json must exist"
    run_data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_data.get("published") is True
    assert run_data.get("is_rerun") is False
    assert run_data.get("run_id") == "run_20261019_0017"

    # 4. Check bulletin JSON carries valid_until_utc == 2026-10-20T06:17:00Z (+30h)
    bulletin_path = Path("site/forecast/2026-10-19.json")
    assert bulletin_path.exists(), "Bulletin file site/forecast/2026-10-19.json must exist"
    bulletin = json.loads(bulletin_path.read_text(encoding="utf-8"))
    assert bulletin["valid_until_utc"] == "2026-10-20T06:17:00Z"
    assert "basis" in bulletin
    assert bulletin["basis"]["n_panel"] >= 1


def test_gate1_drill_2_idempotency_early_exit(clean_state: None) -> None:
    """Drill 2: Rerunning without force exits early with rc=11 and creates no rerun manifest."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # Initial publication
    rc1 = run_daily_pipeline(issuance, force=False, mode="fixtures")
    assert rc1 in (0, 10)

    # Immediate second run without force
    rc2 = run_daily_pipeline(issuance, force=False, mode="fixtures")
    assert rc2 == 11, f"Expected idempotency early exit 11, got {rc2}"

    # Verify no rerun manifest was created
    rerun_manifests = list(Path(".state/manifests").glob("*rerun*.json"))
    assert len(rerun_manifests) == 0, "No rerun manifest should be created on idempotent early exit"


def test_gate1_drill_3_watchdog_verification(clean_state: None) -> None:
    """Drill 3: smogsense state exists returns 0 when missing and 11 when published."""
    issuance_str = "2026-10-19T00:17:00Z"
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # In clean state: state exists must return 0 (not published -> watchdog triggers alert)
    res_clean = runner.invoke(app, ["state", "exists", "--issuance", issuance_str])
    assert res_clean.exit_code == 0, (
        f"Expected exit code 0 when unpublished, got {res_clean.exit_code}"
    )

    # Publish forecast
    rc = run_daily_pipeline(issuance, force=False, mode="fixtures")
    assert rc in (0, 10)

    # After publication: state exists must return 11 (published -> watchdog stands down)
    res_published = runner.invoke(app, ["state", "exists", "--issuance", issuance_str])
    assert res_published.exit_code == 11, (
        f"Expected exit code 11 when published, got {res_published.exit_code}"
    )

    # Also test --issuance latest with today's date
    now = datetime.now(UTC)
    today_str = now.strftime("%Y-%m-%d")
    today_manifest = Path(".state/manifests") / f"{today_str}.json"
    today_manifest.write_text(json.dumps({"published": True, "run_id": f"run_{today_str}_0017"}))

    res_latest = runner.invoke(app, ["state", "exists", "--issuance", "latest"])
    assert res_latest.exit_code == 11, (
        f"Expected exit code 11 for latest, got {res_latest.exit_code}"
    )


def test_gate1_drill_4_rerun_isolation(clean_state: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Drill 4: Forced rerun produces isolated rerun manifest, preserves original run, and respects SMOGSENSE_FORCE=1."""
    issuance = datetime(2026, 10, 19, 0, 17, tzinfo=UTC)

    # 1. Initial publication
    rc1 = run_daily_pipeline(issuance, force=False, mode="fixtures")
    assert rc1 in (0, 10)

    initial_manifest_path = Path(".state/manifests/run_20261019_0017.json")
    assert initial_manifest_path.exists()
    initial_manifest = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
    assert initial_manifest["is_rerun"] is False

    initial_log_path = Path(".state/forecasts/forecast_run_20261019_0017.parquet")
    assert initial_log_path.exists()
    df_initial = pd.read_parquet(initial_log_path)
    assert not df_initial.empty

    # 2. Rerun with force=True
    rc2 = run_daily_pipeline(issuance, force=True, mode="fixtures")
    assert rc2 in (0, 10)

    # Original manifest must remain preserved and unmodified
    initial_after = json.loads(initial_manifest_path.read_text(encoding="utf-8"))
    assert initial_after["is_rerun"] is False
    assert initial_after["run_id"] == "run_20261019_0017"

    # Original forecast log parquet must remain intact
    assert initial_log_path.exists()
    df_after = pd.read_parquet(initial_log_path)
    assert len(df_after) == len(df_initial)

    # Distinct rerun manifest must exist with is_rerun=True
    rerun_manifests = list(Path(".state/manifests").glob("run_20261019_0017_rerun_*.json"))
    assert len(rerun_manifests) == 1, "Expected exactly 1 rerun manifest"
    rerun_manifest = json.loads(rerun_manifests[0].read_text(encoding="utf-8"))
    assert rerun_manifest["is_rerun"] is True
    assert rerun_manifest["published"] is True
    assert rerun_manifest["run_id"] != "run_20261019_0017"

    # Canonical manifest 2026-10-19.json points to the rerun
    canonical = json.loads(Path(".state/manifests/2026-10-19.json").read_text(encoding="utf-8"))
    assert canonical["is_rerun"] is True
    assert canonical["run_id"] == rerun_manifest["run_id"]

    # 3. Test SMOGSENSE_FORCE=1 environment variable parity via CLI runner
    # Wait 1.05s so second rerun timestamp has distinct second
    time.sleep(1.05)
    monkeypatch.setenv("SMOGSENSE_FORCE", "1")
    res_cli = runner.invoke(
        app,
        ["run", "daily", "--issuance", "2026-10-19T00:17:00Z", "--mode", "fixtures"],
    )
    assert res_cli.exit_code in (0, 10), (
        f"Expected 0 or 10 with SMOGSENSE_FORCE=1, got {res_cli.exit_code}"
    )

    # Should have generated a second rerun manifest
    reruns_after_cli = list(Path(".state/manifests").glob("run_20261019_0017_rerun_*.json"))
    assert len(reruns_after_cli) == 2, "Expected 2 rerun manifests after CLI forced run"
