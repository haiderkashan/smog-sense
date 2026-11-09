"""Unit tests for human review gates engine, expert dossiers, and lay user comprehension protocol.

Task: Phase 4 Task P4-07
Specification: docs/dissemination-and-ui.md §11, docs/PRD.md (FR-22-FR-37), ADR-004
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.publishing.review_gates import (
    ComprehensionUserSession,
    GateAuditReport,
    GateID,
    audit_review_gates,
    evaluate_comprehension_sessions,
    format_review_summary_table,
    load_review_gates,
)


@pytest.fixture
def runner() -> CliRunner:
    """Provide Typer test CLI runner."""
    return CliRunner()


def test_load_and_validate_expert_review_dossiers() -> None:
    """Verify that G-SCI, G-LANG, and G-HLTH dossiers exist in docs/reviews and meet criteria."""
    reviews_dir = Path("docs/reviews")
    assert reviews_dir.is_dir()

    gates, _sessions = load_review_gates(reviews_dir)

    # 1. Check G-SCI (Scientific & Atmospheric Review)
    assert GateID.G_SCI in gates
    g_sci = gates[GateID.G_SCI]
    assert g_sci.passed is True
    assert "scientist" in g_sci.reviewer_role.lower()
    assert len(g_sci.observations) >= 3
    assert any("pinball" in obs.lower() or "quantile" in obs.lower() for obs in g_sci.observations)

    # 2. Check G-LANG (Bilingual Urdu Editorial Review)
    assert GateID.G_LANG in gates
    g_lang = gates[GateID.G_LANG]
    assert g_lang.passed is True
    assert "urdu" in g_lang.reviewer_role.lower() or "editor" in g_lang.reviewer_role.lower()
    assert len(g_lang.observations) >= 3
    assert any("parity" in obs.lower() or "calque" in obs.lower() for obs in g_lang.observations)

    # 3. Check G-HLTH (Public Health Medical Review)
    assert GateID.G_HLTH in gates
    g_hlth = gates[GateID.G_HLTH]
    assert g_hlth.passed is True
    assert "physician" in g_hlth.reviewer_role.lower() or "health" in g_hlth.reviewer_role.lower()
    assert len(g_hlth.observations) >= 3
    assert any("mask" in obs.lower() or "respirator" in obs.lower() for obs in g_hlth.observations)

    # 4. Check G-UX (Comprehension Evaluation Dossier)
    assert GateID.G_UX in gates
    g_ux = gates[GateID.G_UX]
    assert g_ux.passed is True
    assert "journalist" in g_ux.reviewer_role.lower() or "ux" in g_ux.reviewer_role.lower()


def test_lay_user_comprehension_protocol_thresholds() -> None:
    """Verify lay user comprehension sessions achieve >= 80% on Q1 and >= 80% on Q2-Q5."""
    reviews_dir = Path("docs/reviews")
    _, sessions = load_review_gates(reviews_dir)

    assert len(sessions) >= 5, "Protocol requires minimum 5 lay participants in Lahore"
    assert len(sessions) == 6

    # Verify participant IDs and personas
    p_ids = [s.participant_id for s in sessions]
    assert p_ids == ["P1", "P2", "P3", "P4", "P5", "P6"]

    # Compute summary
    summary = evaluate_comprehension_sessions(sessions)

    # Invariants per docs/dissemination-and-ui.md §11
    assert summary.total_participants == 6
    assert summary.percent_protective_q1 >= 80.0, "Target >= 80% protective action on Q1"
    assert summary.percent_protective_q1 == 100.0

    assert summary.percent_correct_q2_q5 >= 80.0, "Target >= 80% correct on Questions 2-5"
    assert summary.percent_correct_q2_q5 == 91.7

    assert summary.q2_percent_correct == 83.3  # 5/6 answered '1'
    assert summary.q3_percent_correct == 100.0  # 6/6 understood upper is not ceiling
    assert summary.q4_percent_correct == 100.0  # 6/6 understood purple is worse than orange
    assert summary.q5_percent_correct == 83.3  # 5/6 identified validity window

    assert summary.mean_trust_score >= 4.0
    assert summary.mean_clarity_score >= 4.0
    assert summary.passed is True


def test_review_gate_engine_rejection_on_missing_or_failed_gate(tmp_path: Path) -> None:
    """Verify that audit_review_gates correctly detects missing, failed, or unqualified gates."""
    # 1. Empty directory -> All gates missing
    report_empty = audit_review_gates(tmp_path)
    assert report_empty.all_passed is False
    assert len(report_empty.missing_gates) == 4
    assert len(report_empty.diagnostics) >= 4

    # 2. Directory with one failed gate
    fake_sci = tmp_path / "g_sci_review.md"
    fake_sci.write_text(
        "---\n"
        "gate_id: g_sci\n"
        "reviewer_role: Environmental scientist\n"
        "date: '2026-11-09'\n"
        "passed: false\n"
        "scope: 'Methodology'\n"
        "observations: ['Scientific disagreement']\n"
        "---\n# G-SCI",
        encoding="utf-8",
    )

    report_partial = audit_review_gates(tmp_path)
    assert report_partial.all_passed is False
    assert "g_sci" in report_partial.failed_gates
    assert "g_lang" in report_partial.missing_gates

    # 3. Failing comprehension session (< 80%)
    failing_sessions = [
        ComprehensionUserSession(
            participant_id=f"P{i}",
            persona=f"User {i}",
            q1_response="Ignore",
            q1_protective=False,  # Fails Q1 threshold
            q2_response=5,
            q2_correct=False,
            q3_response="Yes",
            q3_correct=False,
            q4_response="Orange",
            q4_correct=False,
            q5_response="Forever",
            q5_correct=False,
            q6_feedback="",
            trust_rating=2,
            clarity_rating=2,
        )
        for i in range(5)
    ]
    summary_failing = evaluate_comprehension_sessions(failing_sessions)
    assert summary_failing.passed is False
    assert summary_failing.percent_protective_q1 == 0.0
    assert summary_failing.percent_correct_q2_q5 == 0.0


def test_cli_review_gates_audit_command(runner: CliRunner) -> None:
    """Verify smogsense review-gates audit and smogsense review check CLI commands."""
    # 1. Standard table format
    res_table = runner.invoke(app, ["review-gates", "audit", "--reviews-dir", "docs/reviews"])
    assert res_table.exit_code == 0
    assert "SmogSense Pre-Launch Human Review Gates Audit" in res_table.output
    assert "G-SCI" in res_table.output
    assert "G-LANG" in res_table.output
    assert "G-HLTH" in res_table.output
    assert "G-UX" in res_table.output
    assert "ALL GATES APPROVED" in res_table.output

    # 2. JSON format
    res_json = runner.invoke(
        app, ["review", "check", "--reviews-dir", "docs/reviews", "--format", "json"]
    )
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["all_passed"] is True
    assert "g_sci" in data["gates"]
    assert "g_lang" in data["gates"]
    assert "g_hlth" in data["gates"]
    assert "g_ux" in data["gates"]
    assert data["comprehension_summary"]["passed"] is True
    assert data["comprehension_summary"]["total_participants"] == 6

    # 3. Failure on non-existent directory -> Exit code 30
    res_fail = runner.invoke(
        app, ["review-gates", "audit", "--reviews-dir", "non_existent_dir_12345"]
    )
    assert res_fail.exit_code == 30
    assert "GATES AUDIT FAILED" in res_fail.output


def test_format_review_summary_table_formatting() -> None:
    """Verify format_review_summary_table handles both passed and failed states."""
    passed_report = audit_review_gates("docs/reviews")
    tbl_passed = format_review_summary_table(passed_report)
    assert "ALL GATES APPROVED (READY FOR LAUNCH)" in tbl_passed

    failed_report = GateAuditReport(
        all_passed=False,
        gates={},
        missing_gates=["g_sci", "g_lang", "g_hlth", "g_ux"],
        failed_gates=[],
        diagnostics=["Missing all review gate dossiers"],
    )
    tbl_failed = format_review_summary_table(failed_report)
    assert "OVERALL STATUS: GATES AUDIT FAILED (ACTION REQUIRED)" in tbl_failed
    assert "Missing all review gate dossiers" in tbl_failed
