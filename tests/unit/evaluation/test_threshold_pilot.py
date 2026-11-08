"""tests/unit/evaluation/test_threshold_pilot.py — Unit tests for Threshold Pilot engine.

Specifications:
- docs/ml-architecture.md §8 ("Few-shot deployment adaptation", "Condition C6 Acceptance Gate")
- docs/test-specifications.md §T1-16 ("Adaptation outcomes")
- docs/PRD.md (FR-17, FR-18, FR-39)
- configs/model.yaml (adaptation & challenger blocks)
- decisions.md (ADR-024 Revision 3)

Verifies:
1. test_threshold_pilot_sweep_grid: Verifies simulation over parameter sweep grid.
2. test_min_support_days_boundary: Asserts support windows with < min_support_days yield skipped_insufficient_support.
3. test_residual_bound_rejection: Asserts residuals exceeding rho_99 yield rejected_residual_bound.
4. test_pinball_gain_rejection: Asserts residuals failing 2% / 60% check gain yield rejected_no_gain (including both-failing rule).
5. test_pilot_report_serialization: Verifies JSON roundtrip and Markdown generation with clean LF endings.
6. test_cli_threshold_pilot_command: Verifies CLI execution and exit codes via CliRunner for both main command and alias.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.threshold_pilot import (
    DEFAULT_CALIBRATED_RHO_99,
    DEFAULT_RECOMMENDED_MIN_DAYS_BETTER,
    DEFAULT_RECOMMENDED_MIN_IMPROVEMENT,
    DEFAULT_RECOMMENDED_MIN_SUPPORT_DAYS,
    ThresholdPilotReport,
    ThresholdSweepPoint,
    evaluate_task_acceptance_status,
    run_threshold_pilot,
)


def test_threshold_pilot_sweep_grid(tmp_path: Path) -> None:
    """Verify run_threshold_pilot executes across sweep points and populates report."""
    report = run_threshold_pilot(
        n_tasks=12,
        min_improvement_grid=[0.01, 0.02],
        min_days_better_grid=[0.50, 0.60],
        min_support_days_grid=[3, 5],
        rho_99=DEFAULT_CALIBRATED_RHO_99,
        output_dir=tmp_path / "reports",
        seed=101,
    )

    assert isinstance(report, ThresholdPilotReport)
    assert report.total_simulated_tasks == 12
    assert report.recommended_min_improvement == DEFAULT_RECOMMENDED_MIN_IMPROVEMENT
    assert report.recommended_min_days_better == DEFAULT_RECOMMENDED_MIN_DAYS_BETTER
    assert report.recommended_min_support_days == DEFAULT_RECOMMENDED_MIN_SUPPORT_DAYS
    assert report.calibrated_rho_99 == DEFAULT_CALIBRATED_RHO_99

    # Grid size: 2 * 2 * 2 = 8 sweep points
    assert len(report.sensitivity_grid) == 8

    for pt in report.sensitivity_grid:
        assert isinstance(pt, ThresholdSweepPoint)
        total_counts = sum(pt.status_counts.values())
        assert total_counts == 12
        total_pct = sum(pt.status_percentages.values())
        assert abs(total_pct - 100.0) < 1e-3
        assert pt.mean_residual_r >= 0.0

    # Verify report files written to output_dir
    json_path = tmp_path / "reports" / "threshold_pilot.json"
    md_path = tmp_path / "reports" / "threshold_pilot.md"
    assert json_path.is_file()
    assert md_path.is_file()


def test_min_support_days_boundary() -> None:
    """Verify min_support_days threshold separates insufficient support from viable tasks."""
    # 1. Under boundary (4.0 days < 5 days) -> skipped_insufficient_support
    st_under = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=4.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.04,
        days_better_fraction=0.75,
        residual_r=0.15,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_under == "skipped_insufficient_support"

    # 2. At boundary (5.0 days == 5 days) with all checks passing -> adapted
    st_pass = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=5.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.04,
        days_better_fraction=0.75,
        residual_r=0.15,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_pass == "adapted"

    # 3. Insufficient split window flag -> skipped_insufficient_support
    st_insuff = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=False,
        insufficient_window=True,
        rel_improvement=0.0,
        days_better_fraction=0.0,
        residual_r=0.0,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_insuff == "skipped_insufficient_support"

    # 4. Sensor-free task (N=0) -> skipped_n0
    st_n0 = evaluate_task_acceptance_status(
        is_n0=True,
        duration_days=14.0,
        min_support_days=5,
        has_fit=False,
        insufficient_window=False,
        rel_improvement=0.0,
        days_better_fraction=0.0,
        residual_r=0.0,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_n0 == "skipped_n0"


def test_residual_bound_rejection() -> None:
    """Verify residual statistic exceeding rho_99 triggers rejected_residual_bound."""
    rho_99 = 0.362452

    # Residual statistic R = 0.45 exceeds rho_99 even though gains are strong
    st_bound_fail = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.08,
        days_better_fraction=0.85,
        residual_r=0.45,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_bound_fail == "rejected_residual_bound"

    # Residual statistic R = 0.25 <= rho_99 passes
    st_bound_pass = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.08,
        days_better_fraction=0.85,
        residual_r=0.25,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_bound_pass == "adapted"


def test_pinball_gain_rejection() -> None:
    """Verify failure to meet 2% relative gain or 60% daily win fraction yields rejected_no_gain."""
    # 1. Gain below min_improvement (1.5% < 2%)
    st_gain_fail = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.015,
        days_better_fraction=0.70,
        residual_r=0.18,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_gain_fail == "rejected_no_gain"

    # 2. Daily win fraction below threshold (50% < 60%)
    st_win_fail = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.05,
        days_better_fraction=0.50,
        residual_r=0.18,
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_win_fail == "rejected_no_gain"

    # 3. Both tests fail: ADR-024 & T1-16 dictate rejected_no_gain
    st_both_fail = evaluate_task_acceptance_status(
        is_n0=False,
        duration_days=14.0,
        min_support_days=5,
        has_fit=True,
        insufficient_window=False,
        rel_improvement=0.01,
        days_better_fraction=0.40,
        residual_r=0.55,  # Exceeds rho_99 as well
        rho_99=0.362452,
        min_improvement=0.02,
        min_days_better=0.60,
    )
    assert st_both_fail == "rejected_no_gain"


def test_pilot_report_serialization(tmp_path: Path) -> None:
    """Verify JSON round-trip serialization and Markdown generation with clean LF endings."""
    sweep_pt = ThresholdSweepPoint(
        min_improvement=0.02,
        min_days_better=0.60,
        min_support_days=5,
        rho_99=0.362452,
        status_counts={
            "adapted": 8,
            "skipped_n0": 2,
            "skipped_insufficient_support": 1,
            "rejected_no_gain": 1,
            "rejected_residual_bound": 0,
            "rejected_base_binding": 0,
        },
        status_percentages={
            "adapted": 66.7,
            "skipped_n0": 16.7,
            "skipped_insufficient_support": 8.3,
            "rejected_no_gain": 8.3,
            "rejected_residual_bound": 0.0,
            "rejected_base_binding": 0.0,
        },
        mean_pinball_gain_check=0.038,
        mean_residual_r=0.195,
    )

    report = ThresholdPilotReport(
        created_at_utc="2026-11-08T12:00:00Z",
        block_evaluated="Delhi Block V + Lahore Dev",
        total_simulated_tasks=12,
        recommended_min_improvement=0.02,
        recommended_min_days_better=0.60,
        recommended_min_support_days=5,
        calibrated_rho_99=0.362452,
        empirical_status_distribution=sweep_pt.status_percentages,
        sensitivity_grid=[sweep_pt],
        metadata={"domain": "delhi", "seed": 42},
    )

    # 1. JSON Round-Trip
    json_path = tmp_path / "pilot.json"
    report.save(json_path)
    loaded = ThresholdPilotReport.from_file(json_path)

    assert loaded.total_simulated_tasks == 12
    assert loaded.recommended_min_improvement == 0.02
    assert len(loaded.sensitivity_grid) == 1
    assert loaded.sensitivity_grid[0].min_support_days == 5

    # 2. Markdown File Formatting & LF Line Endings
    md_path = tmp_path / "pilot.md"
    report.save_markdown(md_path)
    raw_bytes = md_path.read_bytes()

    assert b"\r\n" not in raw_bytes, "Found CRLF in generated markdown"
    assert raw_bytes.endswith(b"\n"), "Missing single trailing newline in markdown"

    md_text = raw_bytes.decode("utf-8")
    assert "# SmogSense Phase 3: Threshold Pilot & Adaptation Status Distribution Report" in md_text
    assert "0.362452" in md_text
    assert "`adapted`" in md_text


def test_cli_threshold_pilot_command(tmp_path: Path) -> None:
    """Verify CLI threshold-pilot command and pilot alias execute cleanly."""
    runner = CliRunner()
    out_dir = str(tmp_path / "cli_reports")

    # 1. Main command
    res = runner.invoke(
        app,
        [
            "eval",
            "threshold-pilot",
            "--n-tasks",
            "6",
            "--output-dir",
            out_dir,
            "--seed",
            "99",
        ],
    )
    assert res.exit_code == 0
    assert "THRESHOLD PILOT SUMMARY & INVARIANT CONFIRMATION" in res.output
    assert "Recommended min_improvement: 0.02 (2.0%)" in res.output

    # 2. Alias command
    res_alias = runner.invoke(
        app,
        [
            "eval",
            "pilot",
            "--n-tasks",
            "4",
            "--output-dir",
            out_dir,
            "--seed",
            "100",
        ],
    )
    assert res_alias.exit_code == 0
    assert "THRESHOLD PILOT SUMMARY & INVARIANT CONFIRMATION" in res_alias.output
