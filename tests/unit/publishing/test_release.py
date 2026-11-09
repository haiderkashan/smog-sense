"""Unit tests for pre-launch verification checklist, release notes automation, and Gate G6 closure.

Task: Phase 4 Task P4-08
Specification: docs/dissemination-and-ui.md §11, docs/PRD.md (FR-22-FR-37), roadmap.md Gate G6
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.publishing.release import (
    PreLaunchChecklist,
    format_prelaunch_audit_table,
    generate_release_notes,
)


@pytest.fixture
def runner() -> CliRunner:
    """Provide Typer test CLI runner."""
    return CliRunner()


def test_pre_launch_checklist_full_pass() -> None:
    """Verify that PreLaunchChecklist passes all checks against production artifacts."""
    checklist = PreLaunchChecklist(
        reviews_dir="docs/reviews",
        site_dir="site",
        bulletin_path="site/forecast/latest.json",
        i18n_dir="web/i18n",
    )

    result = checklist.run_all_checks(check_git=False)

    assert result.review_gates_passed is True
    assert result.site_budgets_passed is True
    assert result.bilingual_parity_passed is True
    assert result.bulletin_schema_passed is True
    assert result.git_cleanliness_passed is True
    assert result.all_passed is True
    assert len(result.diagnostics) == 0

    # Test formatted table
    tbl = format_prelaunch_audit_table(result)
    assert "Gate G6 Pre-Launch Verification Audit" in tbl
    assert "OVERALL AUDIT: PASSED (GATE G6 READY FOR RELEASE)" in tbl


def test_pre_launch_checklist_detects_review_gate_failure(tmp_path: Path) -> None:
    """Verify that PreLaunchChecklist fails if review gates are missing or unapproved."""
    # Empty reviews directory
    checklist = PreLaunchChecklist(
        reviews_dir=tmp_path,
        site_dir="site",
        bulletin_path="site/forecast/latest.json",
        i18n_dir="web/i18n",
    )
    result = checklist.run_all_checks(check_git=False)
    assert result.all_passed is False
    assert result.review_gates_passed is False
    assert any("Review Gates:" in d for d in result.diagnostics)


def test_pre_launch_checklist_detects_budget_breach(tmp_path: Path) -> None:
    """Verify that PreLaunchChecklist detects oversized files or budget breaches."""
    # Copy minimal site and inject an oversized asset
    fake_site = tmp_path / "site"
    fake_site.mkdir()
    shutil.copytree("site", fake_site, dirs_exist_ok=True)

    # Inject an oversized file breaching the 60 KB HTML budget
    oversized_html = fake_site / "index.html"
    oversized_html.write_text("<!DOCTYPE html><html>" + ("A" * 70000) + "</html>", encoding="utf-8")

    checklist = PreLaunchChecklist(
        reviews_dir="docs/reviews",
        site_dir=fake_site,
        bulletin_path=fake_site / "forecast" / "latest.json",
        i18n_dir="web/i18n",
    )
    result = checklist.run_all_checks(check_git=False)
    assert result.all_passed is False
    assert result.site_budgets_passed is False
    assert any("Site Budgets:" in d for d in result.diagnostics)


def test_generate_release_notes_synthesis() -> None:
    """Verify synthesis of release notes markdown containing metadata, tables, and scientific attributions."""
    notes = generate_release_notes(
        version="v1.0.0-rc1",
        target_date="2026-12-15",
        site_dir="site",
        reviews_dir="docs/reviews",
    )

    assert "# SmogSense Release Notes - v1.0.0-rc1" in notes
    assert "* **Release Tag:** `v1.0.0-rc1`" in notes
    assert "* **Release Date:** 2026-12-15" in notes
    assert "$0.00 Lifetime Budget" in notes
    assert "Gate G6 Status:** **CLOSED / APPROVED" in notes

    # Scientific highlights
    assert "Physics-Residual Meta-Quantile Learning" in notes
    assert "Model M5" in notes
    assert "Model M7" in notes
    assert "19 evaluation quantiles" in notes

    # Review gate table
    assert "G-SCI" in notes
    assert "G-LANG" in notes
    assert "G-HLTH" in notes
    assert "G-UX" in notes
    assert "Environmental scientist" in notes
    assert "Native Urdu editor" in notes
    assert "Public health physician" in notes

    # Comprehension scorecard
    assert "Empirical Lay User Comprehension Scorecard" in notes
    assert "100.0%" in notes
    assert "91.7%" in notes


def test_cli_release_audit_command(runner: CliRunner) -> None:
    """Verify smogsense release audit CLI command in table and JSON formats."""
    # 1. JSON output
    res_json = runner.invoke(app, ["release", "audit", "--json"])
    data = json.loads(res_json.output)
    assert "all_passed" in data
    assert "check_details" in data
    assert data["review_gates_passed"] is True
    assert data["site_budgets_passed"] is True
    assert data["bilingual_parity_passed"] is True
    assert data["bulletin_schema_passed"] is True

    # 2. Failure with non-existent directory -> Exit code 31
    res_fail = runner.invoke(app, ["release", "audit", "--reviews-dir", "non_existent_reviews_dir"])
    assert res_fail.exit_code == 31
    assert "OVERALL AUDIT: FAILED" in res_fail.output


def test_cli_release_notes_and_tag_candidate(runner: CliRunner, tmp_path: Path) -> None:
    """Verify release notes file export and tag-candidate dry-run command."""
    # 1. Notes generation to file
    out_file = tmp_path / "RELEASE_v1.0.0-rc1.md"
    res_notes = runner.invoke(
        app,
        [
            "release",
            "notes",
            "--version",
            "v1.0.0-rc1",
            "--date",
            "2026-12-15",
            "--out",
            str(out_file),
        ],
    )
    assert res_notes.exit_code == 0
    assert out_file.is_file()
    assert "SmogSense Release Notes - v1.0.0-rc1" in out_file.read_text(encoding="utf-8")

    # 2. Tag-candidate dry run
    res_tag = runner.invoke(
        app, ["release", "tag-candidate", "--version", "v1.0.0-rc1", "--dry-run"]
    )
    # If working tree is not yet committed, it appropriately blocks or displays result
    if res_tag.exit_code == 0:
        assert "[DRY-RUN]" in res_tag.output
        assert "git tag -a v1.0.0-rc1" in res_tag.output
    else:
        assert res_tag.exit_code == 31
        assert "[BLOCKED]" in res_tag.output


def test_publish_ghpages_script_dry_run() -> None:
    """Verify scripts/publish_ghpages.sh runs in dry-run mode without requiring GITHUB_TOKEN."""
    bash_bin = shutil.which("bash") or "C:\\Program Files\\Git\\bin\\bash.exe"
    if not Path(bash_bin).is_file():
        pytest.skip("Bash executable not available for testing publish_ghpages.sh")

    script_path = Path("scripts/publish_ghpages.sh").resolve()
    assert script_path.is_file()

    env = dict(os.environ)
    env["DRY_RUN"] = "1"

    proc = subprocess.run(  # noqa: S603
        [bash_bin, str(script_path), "site"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"Script failed with stderr: {proc.stderr}"
    assert "gh-pages dry-run:" in proc.stdout
    assert "validated site" in proc.stdout
    assert "staged to temporary directory with .nojekyll" in proc.stdout
