"""Unit and integration tests for scheduler-independent expiry notice and watchdog staleness banner.

Specification: docs/dissemination-and-ui.md -> 'Static site architecture'
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.publishing.bulletin import generate_bulletin_json
from smogsense.publishing.deploy import validate_site
from smogsense.publishing.site import build_site, render_expired_site


@pytest.fixture
def sample_bulletin() -> dict[str, Any]:
    """Load canonical test fixture bulletin."""
    fixture_path = Path("tests/fixtures/bulletin_valid_example.json")
    with fixture_path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _sample_forecast_df() -> pd.DataFrame:
    """Create minimal valid forecast DataFrame for bulletin generation with all 19 quantiles."""
    rows = []
    issuance = datetime(2026, 11, 5, 0, 0, tzinfo=UTC)
    for lead in [24, 48, 72]:
        row_city: dict[str, Any] = {
            "issuance_utc": issuance,
            "generated_at_utc": issuance,
            "domain": "lahore",
            "level": "city",
            "point_id": "city:lahore",
            "horizon_h": lead,
            "window_start_utc": issuance + pd.Timedelta(hours=lead - 24),
            "window_end_utc": issuance + pd.Timedelta(hours=lead),
            "method": "m2_cams_bc",
            "model_version": "baseline",
            "mode": "full",
            "cams_base_time_utc": issuance,
            "data_cutoff_utc": issuance,
            "git_sha": "unknown",
            "config_hash": "abc",
            "calibrated": False,
            "run_id": "test_run",
            "is_rerun": False,
            "adaptation_status": "skipped_n0",
            "cams_lead_offset_h": 0.0,
            "n_available_stations": 1,
            "forecast_ugm3": 180.0,
        }
        for q in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
            row_city[f"q{q:02d}"] = 100.0 + q * 2.0
        rows.append(row_city)

        # Station row
        row_st = dict(row_city)
        row_st["level"] = "station"
        row_st["point_id"] = "station:42"
        rows.append(row_st)

    return pd.DataFrame(rows)


def test_valid_until_exact_30h_offset() -> None:
    """Verify that valid_until_utc is computed exactly as issuance_utc + 30h across varied times."""
    df = _sample_forecast_df()
    meta = {"42": {"name": "Ref Station", "lat": 31.5, "lon": 74.3, "is_reference": True}}
    src_status = {"openaq": {"status": "ok", "as_of_utc": "2026-11-09T00:00:00Z"}}

    test_cases = [
        datetime(2026, 11, 5, 0, 0, tzinfo=UTC),
        datetime(2026, 10, 15, 12, 30, tzinfo=UTC),
        datetime(2026, 1, 1, 23, 45, tzinfo=UTC),
        datetime(2026, 12, 31, 6, 0),  # Naive datetime
    ]

    for issuance in test_cases:
        bulletin = generate_bulletin_json(
            forecast_df=df,
            issuance_utc=issuance,
            mode="full",
            model="m2_cams_bc",
            data_cutoff_utc=issuance,
            domain="lahore",
            sources_status=src_status,
            stations_metadata=meta,
            git_sha="abcdef1234567890",
        )

        issuance_clean = bulletin["issuance_utc"].replace("Z", "+00:00")
        valid_until_clean = bulletin["valid_until_utc"].replace("Z", "+00:00")

        dt_issuance = datetime.fromisoformat(issuance_clean)
        dt_valid = datetime.fromisoformat(valid_until_clean)

        diff = dt_valid - dt_issuance
        assert diff == timedelta(hours=30), (
            f"Expected 30h offset for issuance {issuance}, got {diff}"
        )


def test_static_expiry_line_rendered_en_and_ur(
    tmp_path: Path, sample_bulletin: dict[str, Any]
) -> None:
    """Verify localized 'Valid until' line is statically rendered in English and Urdu HTML."""
    site_dir = tmp_path / "site"
    build_site(
        bulletin=sample_bulletin,
        output_dir=site_dir,
        configs_dir="configs",
        templates_dir="web/templates",
        static_dir="web/static",
        validate=False,
    )

    # 1. English page check
    index_en = (site_dir / "index.html").read_text(encoding="utf-8")
    assert 'class="forecast-metadata__valid-until"' in index_en
    assert "Valid until" in index_en
    assert "(Pakistan time)" in index_en

    # 2. Urdu page check
    index_ur = (site_dir / "ur" / "index.html").read_text(encoding="utf-8")
    assert 'class="forecast-metadata__valid-until"' in index_ur
    assert "تک درست" in index_ur
    assert "(پاکستانی وقت)" in index_ur


def test_expiry_banner_dom_attributes_and_accessibility(
    tmp_path: Path, sample_bulletin: dict[str, Any]
) -> None:
    """Verify DOM attributes, accessibility roles, and hidden state on active forecast page."""
    site_dir = tmp_path / "site"
    build_site(
        bulletin=sample_bulletin,
        output_dir=site_dir,
        configs_dir="configs",
        templates_dir="web/templates",
        static_dir="web/static",
        expired=False,
        validate=True,
    )

    index_en = (site_dir / "index.html").read_text(encoding="utf-8")

    # Banner element invariants
    match = re.search(r'<div\s+id="expiry-banner"[^>]*>', index_en)
    assert match is not None, "Expiry banner div missing in HTML"
    tag_str = match.group(0)

    assert 'class="expiry-banner"' in tag_str
    assert 'role="alert"' in tag_str
    assert 'aria-live="assertive"' in tag_str
    assert f'data-valid-until="{sample_bulletin["valid_until_utc"]}"' in tag_str
    assert 'data-expired="false"' in tag_str
    assert "hidden" in tag_str, "Active forecast must have hidden attribute on expiry banner"

    # Accessible content inside banner
    assert "⏰" in index_en
    assert "This forecast has expired. Wait for the next update" in index_en

    # Urdu page verification
    index_ur = (site_dir / "ur" / "index.html").read_text(encoding="utf-8")
    match_ur = re.search(r'<div\s+id="expiry-banner"[^>]*>', index_ur)
    assert match_ur is not None
    tag_ur = match_ur.group(0)
    assert "hidden" in tag_ur
    assert "یہ پیش گوئی اب میعاد سے باہر ہے" in index_ur


def test_pre_rendered_expired_site_unhides_banner(
    tmp_path: Path, sample_bulletin: dict[str, Any]
) -> None:
    """Verify render_expired_site and build_site(..., expired=True) statically unhides banner."""
    site_dir = tmp_path / "expired_site"

    summary = render_expired_site(
        state_dir=tmp_path / ".state",
        out_dir=site_dir,
        bulletin=sample_bulletin,
        validate=True,
    )
    assert summary["status"] == "ok"
    assert summary["expired"] is True

    index_en = (site_dir / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<div\s+id="expiry-banner"[^>]*>', index_en)
    assert match is not None
    tag_str = match.group(0)

    # When pre-rendered as expired:
    # 1. hidden attribute must NOT be present
    assert "hidden" not in tag_str, "Expired site must not have hidden attribute on expiry banner"
    # 2. data-expired must be 'true'
    assert 'data-expired="true"' in tag_str

    # Urdu page
    index_ur = (site_dir / "ur" / "index.html").read_text(encoding="utf-8")
    match_ur = re.search(r'<div\s+id="expiry-banner"[^>]*>', index_ur)
    assert match_ur is not None
    tag_ur = match_ur.group(0)
    assert "hidden" not in tag_ur
    assert 'data-expired="true"' in tag_ur


def test_e2e_stale_bulletin_pipeline_execution(
    tmp_path: Path, sample_bulletin: dict[str, Any]
) -> None:
    """Simulate a bulletin generated 32h in the past and assert static degradation and CSP compliance."""
    stale_bulletin = dict(sample_bulletin)

    # Set issuance 32 hours ago (so valid_until was 2 hours ago)
    past_issuance = datetime.now(UTC) - timedelta(hours=32)
    past_valid_until = past_issuance + timedelta(hours=30)

    stale_bulletin["issuance_utc"] = past_issuance.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_bulletin["valid_until_utc"] = past_valid_until.strftime("%Y-%m-%dT%H:%M:%SZ")

    site_dir = tmp_path / "stale_e2e_site"

    # 1. Build site with live settings (simulating cached page before client JS runs)
    build_site(
        bulletin=stale_bulletin,
        output_dir=site_dir,
        configs_dir="configs",
        validate=False,
    )

    # 2. Run validator to verify CSP and performance budgets pass cleanly
    violations = validate_site(site_dir=site_dir, configs_dir="configs", raise_on_breach=True)
    assert violations == []

    # 3. Assert client-side script is present and uncorrupted
    js_file = site_dir / "assets" / "js" / "dashboard.js"
    assert js_file.is_file()
    js_content = js_file.read_text(encoding="utf-8")
    assert "checkForecastExpiry" in js_content
    assert "smogsense:expired" in js_content
    assert "removeAttribute('hidden')" in js_content

    # 4. Assert HTML markup has data-valid-until in the past
    index_html = (site_dir / "index.html").read_text(encoding="utf-8")
    assert f'data-valid-until="{stale_bulletin["valid_until_utc"]}"' in index_html


def test_cli_site_build_stale_flag(tmp_path: Path, sample_bulletin: dict[str, Any]) -> None:
    """Verify CLI site build --stale triggers static unhidden banner rendering."""
    runner = CliRunner()
    bulletin_file = tmp_path / "bulletin.json"
    with bulletin_file.open("w", encoding="utf-8") as f:
        json.dump(sample_bulletin, f)

    out_dir = tmp_path / "cli_stale_site"

    result = runner.invoke(
        app,
        [
            "site",
            "build",
            "--stale",
            "--bulletin",
            str(bulletin_file),
            "--out-dir",
            str(out_dir),
            "--validate",
        ],
    )

    assert result.exit_code == 0
    assert "Rendered expired static site" in result.output

    # Check generated index.html
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<div\s+id="expiry-banner"[^>]*>', index_html)
    assert match is not None
    tag_str = match.group(0)

    assert "hidden" not in tag_str
    assert 'data-expired="true"' in tag_str

    # Run validation on the generated site
    val_result = runner.invoke(
        app,
        [
            "site",
            "validate",
            "--site-dir",
            str(out_dir),
            "--configs-dir",
            "configs",
        ],
    )
    assert val_result.exit_code == 0
    assert "Site validation passed" in val_result.output
