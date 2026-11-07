"""Unit tests for the shadow replay engine (Task P3-08).

Specifications:
- docs/evaluation-strategy.md (§9 Prospective shadow-run protocol, §10 Reporting artifacts)
- docs/system-architecture.md (§2, §4)
- configs/evaluation.yaml (calendar & shadow blocks)

Verifies:
1. test_discover_replay_dates: Correctly enumerates 49 daily dates between 19 Oct and 6 Dec 2026.
2. test_replay_single_day_snapshot_load: Loads point-in-time snapshot and generates schema-valid forecast log DataFrame.
3. test_replay_provenance_separation: Asserts run_id starts with 'replay_' and is_rerun = True.
4. test_replay_n0_sensor_free_fallback: Replays a day with zero local sensors, asserting status 'skipped_n0' and fallback to M5 base.
5. test_replay_scoring_and_report_generation: Verifies scoring replayed forecasts and serializing reports/replay_shadow.json and reports/replay_shadow.md.
6. test_replay_cli_command: Tests CLI flags and execution via CliRunner.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.features.snapshot import build_input_snapshot, save_input_snapshot
from smogsense.pipeline.orchestrator import load_pandera_schema
from smogsense.pipeline.replay import (
    discover_replay_dates,
    replay_single_day,
    run_shadow_replay,
)
from smogsense.pipeline.shadow import _load_score_log_schema
from smogsense.utils.io import validate_frame


def test_discover_replay_dates() -> None:
    """Verify discover_replay_dates correctly enumerates 49 daily dates between 19 Oct and 6 Dec 2026."""
    dates = discover_replay_dates(
        start_date="2026-10-19",
        end_date="2026-12-06",
        domain="lahore",
    )
    # Exactly 49 daily forecast issuance dates
    assert len(dates) == 49
    assert dates[0] == datetime(2026, 10, 19, 0, 0, tzinfo=UTC)
    assert dates[-1] == datetime(2026, 12, 6, 0, 0, tzinfo=UTC)

    # Monotonically increasing by exactly 1 day
    for i in range(len(dates) - 1):
        assert dates[i + 1] - dates[i] == timedelta(days=1)


def test_replay_single_day_snapshot_load(tmp_path: Path) -> None:
    """Verify loading point-in-time snapshot and generating schema-valid forecast log DataFrame."""
    issuance_dt = datetime(2026, 10, 25, 0, 0, tzinfo=UTC)
    state_dir = tmp_path / ".state"

    # 1. Build and save a canonical point-in-time input snapshot
    obs_df = pd.DataFrame(
        [
            {
                "location_id": "1001",
                "ts_utc": issuance_dt - timedelta(hours=i),
                "pm25_ugm3": 110.0 + i,
                "qc_flags": 0,
            }
            for i in range(1, 150)
        ]
    )
    cams_df = pd.DataFrame(
        [
            {
                "location_id": "1001",
                "target_hour_utc": issuance_dt + timedelta(hours=h),
                "pm25_ugm3": 95.0,
            }
            for h in range(72)
        ]
        + [
            {
                "location_id": "centroid",
                "target_hour_utc": issuance_dt + timedelta(hours=h),
                "pm25_ugm3": 105.0,
            }
            for h in range(72)
        ]
    )
    snapshot = build_input_snapshot(
        issuance_utc=issuance_dt,
        domain="lahore",
        obs_df=obs_df,
        cams_df=cams_df,
        cams_base_time_utc=issuance_dt - timedelta(hours=12),
        cams_lead_offset_h=12.0,
        run_id="run_20261025_0000",
        n_available_stations=1,
    )
    save_input_snapshot(snapshot, domain="lahore", issuance_date=issuance_dt, state_dir=state_dir)

    # 2. Execute single-day replay over the snapshot
    res = replay_single_day(
        issuance_utc=issuance_dt,
        domain="lahore",
        challenger_method="m7_hybrid_maml",
        state_dir=state_dir,
        fixture_fallback=False,
    )

    assert res["issuance_utc"] == issuance_dt
    assert res["run_id"] == "replay_20261025_0000"
    assert res["method"] == "m7_hybrid_maml"
    assert "forecast_df" in res

    forecast_df = res["forecast_df"]
    # Schema conformance
    schema = load_pandera_schema("forecast_log.schema.yaml")
    validated = validate_frame(forecast_df, schema)
    assert len(validated) == len(forecast_df)
    assert len(forecast_df) > 0


def test_replay_provenance_separation(tmp_path: Path) -> None:
    """Verify replay forecasts are explicitly tagged with run_id = replay_* and is_rerun = True."""
    issuance_dt = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    state_dir = tmp_path / ".state"

    res = replay_single_day(
        issuance_utc=issuance_dt,
        domain="lahore",
        challenger_method="m7_hybrid_maml",
        state_dir=state_dir,
        fixture_fallback=True,
    )

    run_id = res["run_id"]
    assert run_id.startswith("replay_")
    assert run_id == "replay_20261102_0000"

    df = res["forecast_df"]
    assert (df["run_id"] == run_id).all()
    assert (df["is_rerun"] == True).all()  # noqa: E712
    assert (df["method"] == "m7_hybrid_maml").all()

    manifest = res["manifest"]
    assert manifest["run_id"] == run_id
    assert manifest["is_rerun"] is True
    assert manifest["published"] is False
    assert manifest["mode"] == "replay"
    assert manifest["challenger"]["method"] == "m7_hybrid_maml"


def test_replay_n0_sensor_free_fallback(tmp_path: Path) -> None:
    """Verify replaying a day with zero local sensors enforces Condition C7 (skipped_n0, M5 base)."""
    issuance_dt = datetime(2026, 11, 10, 0, 0, tzinfo=UTC)
    state_dir = tmp_path / ".state"

    # Point-in-time snapshot with empty observations (N=0)
    cams_df = pd.DataFrame(
        [
            {
                "location_id": "centroid",
                "target_hour_utc": issuance_dt + timedelta(hours=h),
                "pm25_ugm3": 130.0,
            }
            for h in range(72)
        ]
    )
    snapshot = build_input_snapshot(
        issuance_utc=issuance_dt,
        domain="lahore",
        obs_df=pd.DataFrame(),
        cams_df=cams_df,
        cams_base_time_utc=issuance_dt - timedelta(hours=12),
        cams_lead_offset_h=12.0,
        run_id="run_20261110_0000",
        n_available_stations=0,
    )
    save_input_snapshot(snapshot, domain="lahore", issuance_date=issuance_dt, state_dir=state_dir)

    res = replay_single_day(
        issuance_utc=issuance_dt,
        domain="lahore",
        challenger_method="m7_hybrid_maml",
        state_dir=state_dir,
        fixture_fallback=False,
    )

    assert res["adaptation_status"] == "skipped_n0"
    df = res["forecast_df"]
    assert (df["adaptation_status"] == "skipped_n0").all()
    assert (df["n_available_stations"] == 0).all()


def test_replay_scoring_and_report_generation(tmp_path: Path) -> None:
    """Verify scoring replayed forecasts and serializing summary JSON and Markdown reports."""
    state_dir = tmp_path / ".state"
    output_dir = tmp_path / "reports"

    # Replay 2 days to verify multi-day aggregation
    report = run_shadow_replay(
        start_date="2026-10-19",
        end_date="2026-10-20",
        domain="lahore",
        challenger_method="m7_hybrid_maml",
        state_dir=state_dir,
        output_dir=output_dir,
        score=True,
        fixture_fallback=True,
    )

    assert report["total_days"] == 2
    assert report["method"] == "m7_hybrid_maml"
    assert report["total_forecast_rows"] > 0
    assert "adaptation_status_counts" in report
    assert "adaptation_status_shares" in report

    # 1. Verify score_log_replay.parquet
    score_file = state_dir / "scores" / "score_log_replay.parquet"
    assert score_file.exists(), f"Score log missing at {score_file}"
    score_df = pd.read_parquet(score_file)
    schema = _load_score_log_schema()
    validated_scores = validate_frame(score_df, schema)
    assert len(validated_scores) == len(score_df)
    assert (score_df["method"] == "m7_hybrid_maml").all()
    assert all(str(rid).startswith("replay_") for rid in score_df["forecast_run_id"])

    # 2. Verify replay_shadow.json
    json_path = output_dir / "replay_shadow.json"
    assert json_path.exists()
    with json_path.open("r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["total_days"] == 2
    assert loaded_json["domain"] == "lahore"

    # 3. Verify replay_shadow.md
    md_path = output_dir / "replay_shadow.md"
    assert md_path.exists()
    with md_path.open("rb") as f:
        raw_bytes = f.read()
    # Enforce strictly LF line endings (\r\n must not be present)
    assert b"\r\n" not in raw_bytes, "Report markdown contains CRLF line endings"
    content = raw_bytes.decode("utf-8")
    assert "# Replay Shadow Evaluation Report" in content
    assert "Adaptation Status Distribution" in content


def test_replay_cli_command(tmp_path: Path) -> None:
    """Verify Typer CLI replay shadow and run replay commands."""
    runner = CliRunner()
    state_dir = str(tmp_path / ".state")
    output_dir = str(tmp_path / "reports")

    # 1. Run smogsense replay shadow
    result = runner.invoke(
        app,
        [
            "replay",
            "shadow",
            "--start-date",
            "2026-10-19",
            "--end-date",
            "2026-10-19",
            "--domain",
            "lahore",
            "--method",
            "m7_hybrid_maml",
            "--state-dir",
            state_dir,
            "--output-dir",
            output_dir,
            "--score",
        ],
    )
    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert "Shadow Replay Completed for lahore" in result.output

    # 2. Run smogsense run replay alias
    result_run = runner.invoke(
        app,
        [
            "run",
            "replay",
            "--start-date",
            "2026-10-19",
            "--end-date",
            "2026-10-19",
            "--domain",
            "lahore",
            "--method",
            "m7_hybrid_maml",
            "--state-dir",
            state_dir,
            "--output-dir",
            output_dir,
            "--no-score",
        ],
    )
    assert result_run.exit_code == 0, f"Command failed: {result_run.output}"
    assert "Shadow Replay Completed for lahore" in result_run.output
