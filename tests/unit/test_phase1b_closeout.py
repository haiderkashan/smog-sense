"""tests.unit.test_phase1b_closeout - Test suite for Phase 1b closeout tasks P1-21, P1-22, P1-23.

Verifies:
- Task P1-21: audit_truth_basis, 50m co-location hierarchy, provider pairing, and CLI audit truth-basis.
- Task P1-22: compute_available_at, filter_asof, cams_lead_offset_hours, verify_asof_lattice, verify_container_parity, and CLI audit asof-parity.
- Task P1-23: verify_state_branch, rotate_state_year, generate_site expiry banner and JS check, render_expired_site, and CLI commands.
- Showcase notebook 02_baselines_m0_m3b.ipynb integrity and completeness.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from smogsense.analysis.station_audit import audit_truth_basis
from smogsense.cli import app
from smogsense.pipeline.state_guard import rotate_state_year, verify_state_branch
from smogsense.preprocessing.alignment import (
    cams_lead_offset_hours,
    compute_available_at,
    filter_asof,
    verify_asof_lattice,
    verify_container_parity,
)
from smogsense.publishing.site import generate_site, render_expired_site

runner = CliRunner()


# =====================================================================
# Task P1-21: Co-location Hierarchy & Truth-Basis Audit
# =====================================================================


def test_audit_truth_basis_colocation_and_hierarchy() -> None:
    """Test geodesic 50m grouping, reference priority, and provider pairs."""
    locations_df = pd.DataFrame(
        [
            # Group A: 1001 (reference) + 1002 (low-cost), distance ~15m
            {
                "location_id": 1001,
                "lat": 31.55010,
                "lon": 74.33020,
                "is_reference": True,
                "provider": "AirNow",
                "eligible": True,
            },
            {
                "location_id": 1002,
                "lat": 31.55020,
                "lon": 74.33025,
                "is_reference": False,
                "provider": "PurpleAir",
                "eligible": True,
            },
            # Group B: 2001 (low-cost) + 2002 (low-cost), distance ~12m
            {
                "location_id": 2001,
                "lat": 31.52000,
                "lon": 74.35000,
                "is_reference": False,
                "provider": "Clarity",
                "eligible": True,
            },
            {
                "location_id": 2002,
                "lat": 31.52008,
                "lon": 74.35005,
                "is_reference": False,
                "provider": "PurpleAir",
                "eligible": True,
            },
            # Group C: 3001 (isolated low-cost)
            {
                "location_id": 3001,
                "lat": 31.45000,
                "lon": 74.25000,
                "is_reference": False,
                "provider": "PurpleAir",
                "eligible": True,
            },
        ]
    )

    t0 = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    obs_rows = []
    # 24 hours for each station
    for h in range(24):
        ts = t0 + timedelta(hours=h)
        for lid, val in [(1001, 80.0), (1002, 120.0), (2001, 70.0), (2002, 75.0), (3001, 95.0)]:
            obs_rows.append({"location_id": lid, "ts_utc": ts, "pm25_ugm3": val, "imputed": False})

    obs_df = pd.DataFrame(obs_rows)

    rep = audit_truth_basis(locations_df, observations_df=obs_df, domain="lahore", radius_m=50.0)

    assert rep["domain"] == "lahore"
    assert rep["total_stations"] == 5
    assert rep["total_groups"] == 3  # A, B, C
    assert len(rep["colocated_multi_station_groups"]) == 2  # A and B

    # Group A verification
    grp_a = next(g for g in rep["colocated_multi_station_groups"] if 1001 in g["location_ids"])
    assert grp_a["station_count"] == 2
    assert grp_a["has_reference_monitor"] is True
    assert grp_a["group_basis"] == "reference"
    assert "AirNow" in grp_a["providers"] and "PurpleAir" in grp_a["providers"]
    assert len(grp_a["pairwise_distances_m"]) == 1
    assert grp_a["pairwise_distances_m"][0]["distance_m"] < 50.0

    # Group B verification
    grp_b = next(g for g in rep["colocated_multi_station_groups"] if 2001 in g["location_ids"])
    assert grp_b["has_reference_monitor"] is False
    assert grp_b["group_basis"] == "low_cost"

    # Overall panel viability
    assert rep["panel_viable"] is True
    assert rep["n_reference_groups"] == 1
    assert rep["n_lowcost_groups"] == 2
    assert rep["overall_truth_basis"] == "mixed"
    assert len(rep["daily_truth_series"]) == 1
    daily_res = rep["daily_truth_series"][0]
    assert daily_res["valid"] is True
    assert daily_res["n_valid_stations"] == 3


def test_audit_truth_basis_insufficient_stations() -> None:
    """Test truth basis audit when fewer than 3 valid groups exist."""
    locations_df = pd.DataFrame(
        [
            {"location_id": 1, "lat": 31.5, "lon": 74.3, "is_reference": False, "eligible": True},
            {"location_id": 2, "lat": 31.6, "lon": 74.4, "is_reference": False, "eligible": True},
        ]
    )
    rep = audit_truth_basis(locations_df, min_panel_stations=3)
    assert rep["total_stations"] == 2
    assert rep["panel_viable"] is False
    assert rep["overall_truth_basis"] == "insufficient"


def test_cli_audit_truth_basis(tmp_path: Path) -> None:
    """Test CLI subcommand `smogsense audit truth-basis`."""
    out_json = tmp_path / "truth_report.json"
    result = runner.invoke(
        app, ["audit", "truth-basis", "--domain", "lahore", "--output", str(out_json)]
    )
    assert result.exit_code == 0
    assert "Truth-Basis Audit for LAHORE:" in result.stdout
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "overall_truth_basis" in data
    assert "panel_candidates" in data


# =====================================================================
# Task P1-22: As-Of Lattice Verification & Container Parity
# =====================================================================


def test_compute_available_at_sources_and_pit_isolation() -> None:
    """Test canonical available_at_utc computation across all data streams."""
    t0 = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

    # 1. OpenAQ: hour_end + 3h (i.e. ts + 4h)
    df_oaq = pd.DataFrame({"ts_utc": [t0], "ingested_at_utc": [t0 + timedelta(hours=10)]})
    avail_oaq = compute_available_at(df_oaq, source="openaq")
    assert avail_oaq.iloc[0] == t0 + timedelta(hours=4)

    # 2. FIRMS: acq_datetime + 3h
    df_firms = pd.DataFrame({"acq_datetime_utc": [t0]})
    avail_firms = compute_available_at(df_firms, source="firms")
    assert avail_firms.iloc[0] == t0 + timedelta(hours=3)

    # 3. CAMS: base_time + 10h
    df_cams = pd.DataFrame({"cams_cycle_utc": [t0]})
    avail_cams = compute_available_at(df_cams, source="cams")
    assert avail_cams.iloc[0] == t0 + timedelta(hours=10)

    # 4. OpenAQ Archive: day_end + 72h
    df_arch = pd.DataFrame({"ts_utc": [datetime(2026, 10, 20, 5, 0, tzinfo=UTC)]})
    avail_arch = compute_available_at(df_arch, source="openaq_archive")
    expected_arch = datetime(2026, 10, 21, 0, 0, tzinfo=UTC) + timedelta(hours=72)
    assert avail_arch.iloc[0] == expected_arch


def test_filter_asof_ignores_ingested_at_and_filters_future() -> None:
    """Test that filter_asof excludes rows available after T and ignores ingested_at."""
    t_issuance = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)

    df = pd.DataFrame(
        [
            # Row 1: available before T (valid time 19:00 previous day + 4h = 23:00 < T) -> KEEP
            {
                "ts_utc": t_issuance - timedelta(hours=5),
                "available_at_utc": t_issuance - timedelta(hours=1),
                "ingested_at_utc": t_issuance + timedelta(hours=10),
                "val": 10.0,
            },
            # Row 2: available after T (available at T + 2h) -> EXCLUDE despite past ts_utc
            {
                "ts_utc": t_issuance - timedelta(hours=2),
                "available_at_utc": t_issuance + timedelta(hours=2),
                "ingested_at_utc": t_issuance - timedelta(hours=1),
                "val": 20.0,
            },
        ]
    )

    filtered = filter_asof(df, issuance=t_issuance)
    assert len(filtered) == 1
    assert filtered.iloc[0]["val"] == 10.0


def test_cams_lead_offset_hours() -> None:
    """Test delta calculation at levels 0 and 1 across lattice points."""
    d00z = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)
    d06z = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)

    assert cams_lead_offset_hours(d00z, degradation_level=0) == 12.0
    assert cams_lead_offset_hours(d00z, degradation_level=1) == 24.0

    assert cams_lead_offset_hours(d06z, degradation_level=0) == 18.0
    assert cams_lead_offset_hours(d06z, degradation_level=1) == 30.0


def test_verify_asof_lattice_all_four_points() -> None:
    """Verify all four lattice points against docs/data-engineering.md §11.3."""
    rep = verify_asof_lattice()
    assert rep["status"] == "PASS"
    assert rep["all_lattice_points_valid"] is True
    assert rep["lattice_points_verified"] == 4

    # Verify continuity and lead bounds
    for pt in rep["points"]:
        assert pt["b_star_matches_spec"] is True
        assert pt["delta0_matches_spec"] is True
        assert pt["delta1_matches_spec"] is True
        assert pt["no_future_cycle_leakage"] is True
        assert pt["max_stitched_lead_h"] <= 18


def test_verify_container_parity() -> None:
    """Verify container parity checks on repository contracts."""
    rep = verify_container_parity()
    assert rep["status"] == "PASS"
    assert rep["all_checks_passed"] is True
    assert rep["checks"]["dockerfile_base_ubuntu2404"] is True
    assert rep["checks"]["compose_cpu_limit"] is True
    assert rep["checks"]["compose_read_only_rootfs"] is True
    assert rep["checks"]["compose_no_git_token"] is True


def test_cli_audit_asof_parity() -> None:
    """Test CLI subcommand `smogsense audit asof-parity`."""
    result = runner.invoke(app, ["audit", "asof-parity"])
    assert result.exit_code == 0
    assert "As-Of Lattice Verification" in result.stdout
    assert "Container Parity Audit: PASS" in result.stdout


# =====================================================================
# Task P1-23: Daily Workflow Enhancements & State Protection
# =====================================================================


def test_verify_state_branch_budgets_and_integrity(tmp_path: Path) -> None:
    """Test state size guard (ok, warn, alert) and forbidden binary detection."""
    st_dir = tmp_path / ".state"
    st_dir.mkdir(parents=True)

    # 1. Clean empty state -> ok
    rep = verify_state_branch(st_dir, warn_mb=10.0, alert_mb=20.0)
    assert rep["status"] == "ok"
    assert rep["valid"] is True

    # 2. Add valid parquet file
    forecast_dir = st_dir / "forecasts"
    forecast_dir.mkdir(parents=True)
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(forecast_dir / "forecast.parquet")
    rep = verify_state_branch(st_dir, warn_mb=10.0, alert_mb=20.0)
    assert rep["status"] == "ok"
    assert rep["readable_counts"]["forecasts"] == 1

    # 3. Add forbidden binary -> alert
    (st_dir / "test.grib").write_bytes(b"GRIB1234")
    rep_bin = verify_state_branch(st_dir, warn_mb=10.0, alert_mb=20.0)
    assert rep_bin["status"] == "alert"
    assert "test.grib" in rep_bin["forbidden_files"]
    assert rep_bin["valid"] is False
    (st_dir / "test.grib").unlink()

    # 4. Warn threshold test
    rep_warn = verify_state_branch(st_dir, warn_mb=0.000001, alert_mb=20.0)
    assert rep_warn["status"] == "warn"
    assert rep_warn["valid"] is True


def test_rotate_state_year(tmp_path: Path) -> None:
    """Test yearly archival of state partitions and working tree pruning."""
    st_dir = tmp_path / ".state"
    st_dir.mkdir(parents=True)
    bundle_out = tmp_path / "bundles"

    # Create root files that MUST be preserved
    (st_dir / "README.md").write_text("state readme", encoding="utf-8")
    (st_dir / ".gitignore").write_text("*.grib\n", encoding="utf-8")

    # Create 2024 partitions (to archive) and 2026 partitions (to preserve)
    p2024 = st_dir / "obs" / "lahore" / "dt=2024-11-01"
    p2024.mkdir(parents=True)
    (p2024 / "data.parquet").write_bytes(b"2024data")

    p2026 = st_dir / "obs" / "lahore" / "dt=2026-10-20"
    p2026.mkdir(parents=True)
    (p2026 / "data.parquet").write_bytes(b"2026data")

    res = rotate_state_year(st_dir, archive_year=2024, bundle_out_dir=bundle_out)
    assert res["status"] == "ok"
    assert res["files_archived"] == 1
    assert res["files_pruned"] == 1
    assert Path(res["bundle_path"]).exists()

    # 2024 file should be removed
    assert not (p2024 / "data.parquet").exists()
    # 2026 file and root files must be intact
    assert (p2026 / "data.parquet").exists()
    assert (st_dir / "README.md").exists()
    assert (st_dir / ".gitignore").exists()


def test_site_expiry_cues_and_render_expired(tmp_path: Path) -> None:
    """Test valid_until static line, JS expiry script, and render_expired_site."""
    out_dir = tmp_path / "site"
    bulletin = {
        "run_id": "test_run_1",
        "issuance_utc": "2026-10-20T00:00:00Z",
        "valid_until_utc": "2026-10-21T06:00:00Z",
        "mode": "full",
        "horizons": [
            {"lead_h": 24, "quantiles_ugm3": {"q50": 85.0}, "category": {"median_q50": "unhealthy"}}
        ],
    }

    # 1. Normal site generation: index.html has static valid_until and client script
    generate_site(bulletin, out_dir=out_dir, expired=False)
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Valid until:" in index_html
    assert "2026-10-21T06:00:00Z" in index_html
    assert 'id="expiry-banner"' in index_html
    assert 'data-expired="false"' in index_html
    assert "new Date() > validUntil" in index_html

    # 2. Render expired site: index.html has static EXPIRED FORECAST banner and data-expired="true"
    rep = render_expired_site(state_dir=tmp_path / "nonexistent", out_dir=out_dir)
    assert rep["status"] == "ok"
    assert rep["expired"] is True

    expired_html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "EXPIRED FORECAST" in expired_html
    assert 'data-expired="true"' in expired_html

    latest_json = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest_json.get("expired") is True


def test_cli_state_verify_and_site_build(tmp_path: Path) -> None:
    """Test CLI commands `smogsense state verify` and `smogsense site build --stale`."""
    st_dir = tmp_path / ".state"
    st_dir.mkdir(parents=True)
    res_verify = runner.invoke(app, ["state", "verify", "--state-dir", str(st_dir)])
    assert res_verify.exit_code == 0
    assert "State Branch Verification: status=OK" in res_verify.stdout

    site_dir = tmp_path / "site"
    res_site = runner.invoke(
        app, ["site", "build", "--stale", "--state-dir", str(st_dir), "--out-dir", str(site_dir)]
    )
    assert res_site.exit_code == 0
    assert "Rendered expired static site" in res_site.stdout
    assert (site_dir / "index.html").exists()


# =====================================================================
# Showcase Notebook Verification
# =====================================================================


def test_showcase_notebook_valid_json_and_covers_all_baselines() -> None:
    """Verify notebooks/02_baselines_m0_m3b.ipynb is valid JSON and covers all baselines."""
    nb_path = Path("notebooks/02_baselines_m0_m3b.ipynb")
    assert nb_path.exists(), "Showcase notebook missing"

    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 10

    # Ensure all cells have no unstripped output
    for c in data["cells"]:
        if c.get("cell_type") == "code":
            assert c.get("outputs") == [], f"Unstripped output found in cell {c.get('id')}"

    full_text = nb_path.read_text(encoding="utf-8")
    for baseline in (
        "M0Persistence",
        "M1Cams",
        "M2CamsBiasCorrected",
        "M3Climatology",
        "M3bLinearQuantileRegression",
    ):
        assert baseline in full_text, f"Missing baseline {baseline} in showcase notebook"

    assert "compute_city_panel_target" in full_text
    assert "google.colab" in full_text
