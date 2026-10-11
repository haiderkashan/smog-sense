"""tests.unit.test_station_audit - Comprehensive test suite for station audit engine.

Verifies:
1. Acceptance criteria and decision rules from docs/data-engineering.md §13.
2. Station eligibility and sparsity curve held-out count (N_max = eligible - 3).
3. Regulatory reference monitor presence check.
4. Winter-day completeness calculation (>= 18 valid hours).
5. Co-location hierarchy and truth basis assignment (50m radius).
6. Report persistence (JSON + Parquet) and CLI command surface.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.analysis.station_audit import (
    audit_domain_stations,
    compute_station_uptimes,
    compute_winter_day_completeness,
    evaluate_decision_rules,
    run_station_audit,
)
from smogsense.cli import app


def test_evaluate_decision_rules_lahore_pass():
    """Verify decision rule evaluations when criteria are met."""
    rules = evaluate_decision_rules(
        domain="lahore",
        eligible_count=7,
        reference_count=2,
        winter_completeness_pct=75.0,
    )

    assert rules["target_family_sparsity"].status == "PASS"
    assert "N_max = 4" in rules["target_family_sparsity"].observed

    assert rules["reference_monitor_anchor"].status == "PASS"
    assert rules["winter_completeness"].status == "PASS"


def test_evaluate_decision_rules_sparsity_warn():
    """Verify decision rule warning when eligible stations < 4."""
    rules = evaluate_decision_rules(
        domain="lahore",
        eligible_count=2,
        reference_count=1,
        winter_completeness_pct=60.0,
    )

    assert rules["target_family_sparsity"].status == "WARN"
    assert "Widen target family" in rules["target_family_sparsity"].recommendation


def test_evaluate_decision_rules_delhi_transfer():
    """Verify Delhi source domain transfer feasibility rule."""
    # Delhi with >= 20 stations and >= 5 seasons passes
    delhi_pass = evaluate_decision_rules(
        domain="delhi",
        eligible_count=25,
        reference_count=10,
        winter_completeness_pct=85.0,
        delhi_seasons_count=6,
    )
    assert delhi_pass["delhi_transfer_feasibility"].status == "PASS"

    # Delhi with < 20 stations triggers recommendation to enable amritsar/ludhiana
    delhi_warn = evaluate_decision_rules(
        domain="delhi",
        eligible_count=12,
        reference_count=4,
        winter_completeness_pct=80.0,
        delhi_seasons_count=3,
    )
    assert delhi_warn["delhi_transfer_feasibility"].status == "WARN"
    assert "amritsar and ludhiana" in delhi_warn["delhi_transfer_feasibility"].recommendation


def test_compute_station_uptimes():
    """Verify station uptime calculation over a rolling window."""
    t_end = pd.Timestamp("2026-01-10 00:00:00", tz="UTC")
    # 10 days of data for two stations
    dates = pd.date_range(t_end - pd.Timedelta(days=10), t_end, freq="h")

    rows = [{"location_id": 1, "ts_utc": dt, "pm25_ugm3": 50.0} for dt in dates]
    rows.extend([{"location_id": 2, "ts_utc": dt, "pm25_ugm3": 45.0} for dt in dates[::2]])

    obs_df = pd.DataFrame(rows)
    uptimes = compute_station_uptimes(obs_df, window_days=10, end_time=t_end)

    assert 1 in uptimes.index
    assert 2 in uptimes.index
    assert uptimes[1] == 1.0  # 100% of expected 240 hours
    assert uptimes[2] == pytest.approx(0.5, abs=0.05)


def test_compute_winter_day_completeness():
    """Verify winter-day completeness percentage (days with >= 18 valid hours)."""
    # Create 4 winter days (October)
    # Days 1, 2 have 24 valid hours (complete)
    # Day 3 has 12 valid hours (incomplete)
    # Day 4 has 20 valid hours (complete)
    # Result: 3/4 complete = 75.0%
    d1 = pd.date_range("2026-10-01 00:00", periods=24, freq="h", tz="UTC")
    d2 = pd.date_range("2026-10-02 00:00", periods=24, freq="h", tz="UTC")
    d3 = pd.date_range("2026-10-03 00:00", periods=12, freq="h", tz="UTC")
    d4 = pd.date_range("2026-10-04 00:00", periods=20, freq="h", tz="UTC")

    all_dates = list(d1) + list(d2) + list(d3) + list(d4)
    obs_df = pd.DataFrame(
        {
            "location_id": [101] * len(all_dates),
            "ts_utc": all_dates,
            "pm25_ugm3": [50.0] * len(all_dates),
            "imputed": [False] * len(all_dates),
        }
    )

    completeness = compute_winter_day_completeness(obs_df, min_valid_hours=18)
    assert completeness == 75.0


def test_audit_domain_stations_with_colocation():
    """Verify full audit including co-location hierarchy and truth basis assignment."""
    now = pd.Timestamp.now(tz="UTC")
    locations_df = pd.DataFrame(
        [
            {
                "location_id": 1001,
                "name": "US Consulate",
                "lat": 31.5500,
                "lon": 74.3300,
                "is_reference": True,
                "provider": "State Department",
                "first_datetime": now - pd.Timedelta(days=200),
                "last_datetime": now,
                "pm25_uptime_90d": 0.95,
            },
            {
                "location_id": 1002,
                "name": "Co-located Lowcost",
                "lat": 31.5501,  # ~15m away
                "lon": 74.3301,
                "is_reference": False,
                "provider": "AirGradient",
                "first_datetime": now - pd.Timedelta(days=100),
                "last_datetime": now,
                "pm25_uptime_90d": 0.90,
            },
            {
                "location_id": 1003,
                "name": "Isolated Lowcost 1",
                "lat": 31.4800,
                "lon": 74.3200,
                "is_reference": False,
                "provider": "PurpleAir",
                "first_datetime": now - pd.Timedelta(days=80),
                "last_datetime": now,
                "pm25_uptime_90d": 0.85,
            },
            {
                "location_id": 1004,
                "name": "Isolated Lowcost 2",
                "lat": 31.5000,
                "lon": 74.3400,
                "is_reference": False,
                "provider": "AirVisual",
                "first_datetime": now - pd.Timedelta(days=50),
                "last_datetime": now,
                "pm25_uptime_90d": 0.80,
            },
        ]
    )

    summary, _reg_df = audit_domain_stations(
        domain="lahore",
        locations_df=locations_df,
        min_uptime=0.50,
    )

    assert summary.domain == "lahore"
    assert summary.total_locations == 4
    assert summary.reference_station_count == 1
    assert summary.lowcost_station_count == 3
    assert summary.eligible_station_count == 4
    assert summary.n_max_held_out == 1  # 4 - 3 = 1

    # Check co-location
    assert summary.colocated_groups_count == 1
    coloc_grp = summary.colocated_groups[0]
    assert coloc_grp["station_count"] == 2
    assert coloc_grp["has_reference_monitor"] is True
    assert coloc_grp["truth_anchor"] == "reference_monitor"


def test_run_station_audit_outputs(tmp_path: Path):
    """Verify run_station_audit writes JSON summary and Parquet registry."""
    summary, _reg_df = run_station_audit(domain="lahore", output_dir=tmp_path)

    json_file = tmp_path / "audit_report_lahore.json"
    parquet_file = tmp_path / "eligible_stations_lahore.parquet"

    assert json_file.exists()
    assert parquet_file.exists()

    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["domain"] == "lahore"
    assert "decision_rules" in data
    assert "target_family_sparsity" in data["decision_rules"]
    assert "reference_monitor_anchor" in data["decision_rules"]

    saved_df = pd.read_parquet(parquet_file)
    assert len(saved_df) == summary.total_locations
    assert "eligible" in saved_df.columns
    assert "colocated_group_id" in saved_df.columns


def test_station_audit_cli(tmp_path: Path):
    """Verify smogsense audit stations CLI execution."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["audit", "stations", "--domain", "lahore", "--output", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "Station Audit Complete for LAHORE:" in result.stdout
    assert "Decision Rules Status:" in result.stdout
    assert (tmp_path / "audit_report_lahore.json").exists()
    assert (tmp_path / "eligible_stations_lahore.parquet").exists()


def test_station_audit_notebook_hygiene():
    """Verify 01_station_audit_lahore.ipynb is valid JSON and stripped of outputs."""
    nb_path = Path("notebooks/01_station_audit_lahore.ipynb")
    assert nb_path.exists()

    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 3

    for cell in cells:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs") == [], "Notebook contains unstripped outputs!"
            assert cell.get("execution_count") is None, (
                "Notebook contains unstripped execution_count!"
            )
