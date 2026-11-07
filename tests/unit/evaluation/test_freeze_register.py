"""tests.unit.evaluation.test_freeze_register — Unit tests for Phase 3 Task P3-10.

Tests:
1. test_create_freeze_register_structure_and_hashes
2. test_city_panel_eligibility_and_minimum_count (asserts >= 3 valid stations, FR-38)
3. test_hypotheses_extraction_matches_evaluation_yaml
4. test_verify_freeze_register_success
5. test_verify_detects_config_drift
6. test_verify_detects_bundle_tamper
7. test_verify_detects_station_panel_tamper
8. test_freeze_register_serialization_json_and_md (LF line endings & single newline)
9. test_cli_freeze_register_command

Specification:
- docs/evaluation-strategy.md §1, §9, §10
- docs/ml-architecture.md §10
- docs/PRD.md (FR-21, FR-38, NFR-16)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.config import Settings
from smogsense.evaluation.freeze import (
    FreezeRegister,
    FreezeVerificationError,
    create_freeze_register,
    load_city_panel_stations,
    parse_hypotheses_from_settings,
    verify_freeze_register,
)

runner = CliRunner()


# =============================================================================
# Unit Tests
# =============================================================================
def test_create_freeze_register_structure_and_hashes(tmp_path: Path) -> None:
    """Verify create_freeze_register builds expected structure and valid SHA-256 digests."""
    out_json = tmp_path / "freeze_register.json"
    out_md = tmp_path / "freeze_register.md"

    reg = create_freeze_register(
        output_path=out_json,
        report_path=out_md,
        git_tag="v0.3.0-gate-g4",
        write_files=True,
    )

    assert reg.schema_version == 1
    assert reg.freeze_date == "2026-12-04"
    assert reg.confirmatory_window_start == "2026-12-07"
    assert reg.confirmatory_window_end == "2027-01-31"
    assert reg.git_tag == "v0.3.0-gate-g4"
    assert len(reg.git_commit_sha) >= 7

    # Check config hash
    assert len(reg.config_hash) == 64
    int(reg.config_hash, 16)  # must be valid hex

    # Check digests
    assert len(reg.bundle_sha256) == 64
    assert len(reg.base_sha256) == 64

    # Check city panel and hypotheses count
    assert len(reg.city_panel) >= 3
    assert len(reg.hypotheses) == 6

    # Invariants
    assert "rho_99" in reg.invariants
    assert reg.invariants["inner_steps_j"] == 3
    assert reg.invariants["grad_clip"] == 1.0
    assert reg.invariants["min_valid_hours"] == 18
    assert reg.invariants["embargo_hours"] == 72
    assert reg.invariants["quantiles_count"] == 19
    assert reg.invariants["calibrator_params_count"] == 47577
    assert reg.invariants["metasgd_step_sizes_count"] == 14

    assert out_json.is_file()
    assert out_md.is_file()


def test_city_panel_eligibility_and_minimum_count(tmp_path: Path) -> None:
    """Verify city panel enforces >= 3 eligible stations (FR-38) and raises if insufficient."""
    # Test loading actual registry
    stations = load_city_panel_stations("data/registry/registry_lahore.parquet", min_stations=3)
    assert len(stations) >= 3
    for s in stations:
        assert isinstance(s.location_id, int)
        assert isinstance(s.name, str)
        assert isinstance(s.is_reference, bool)
        assert isinstance(s.provider, str)
        assert isinstance(s.lat, float)
        assert isinstance(s.lon, float)

    # Test synthetic registry with only 2 stations fails
    sub_df = pd.DataFrame(
        [
            {
                "location_id": 1,
                "name": "Station 1",
                "is_reference": True,
                "provider": "Test",
                "lat": 31.5,
                "lon": 74.3,
                "elevation_m": 210.0,
                "sensor_model": "RefModel",
                "colocated_group_id": None,
                "eligible": True,
            },
            {
                "location_id": 2,
                "name": "Station 2",
                "is_reference": False,
                "provider": "Test",
                "lat": 31.6,
                "lon": 74.4,
                "elevation_m": 215.0,
                "sensor_model": "LowCostModel",
                "colocated_group_id": None,
                "eligible": True,
            },
        ]
    )
    insufficient_parquet = tmp_path / "registry_insufficient.parquet"
    sub_df.to_parquet(insufficient_parquet)

    with pytest.raises(ValueError, match="Insufficient eligible stations in city panel"):
        load_city_panel_stations(insufficient_parquet, min_stations=3)


def test_hypotheses_extraction_matches_evaluation_yaml() -> None:
    """Verify pre-registered hypotheses H1-H6 match evaluation configuration."""
    settings = Settings.load("configs")
    hypotheses = parse_hypotheses_from_settings(settings)

    assert len(hypotheses) == 6
    h_map = {h.hypothesis_id: h for h in hypotheses}

    # H1
    assert "H1" in h_map
    assert h_map["H1"].metric == "crpss_vs_m2"
    assert h_map["H1"].horizon_h == 24
    assert h_map["H1"].n_available == 5
    assert h_map["H1"].target_threshold == 0.15
    assert h_map["H1"].comparison_op == ">="

    # H2
    assert "H2" in h_map
    assert h_map["H2"].metric == "crpss_vs_m2"
    assert h_map["H2"].horizon_h == 72
    assert h_map["H2"].n_available == 5
    assert h_map["H2"].target_threshold == 0.1
    assert h_map["H2"].comparison_op == ">="

    # H3
    assert "H3" in h_map
    assert h_map["H3"].metric == "crpss_vs_m1"
    assert h_map["H3"].horizon_h == "all"
    assert h_map["H3"].target_threshold == 0.25
    assert h_map["H3"].comparison_op == ">="

    # H4
    assert "H4" in h_map
    assert h_map["H4"].metric == "crpss_vs_m2"
    assert h_map["H4"].horizon_h == 24
    assert h_map["H4"].n_available == 0
    assert h_map["H4"].target_threshold == 0.05
    assert h_map["H4"].comparison_op == ">="

    # H5
    assert "H5" in h_map
    assert h_map["H5"].metric == "crps_ratio_m7_over_m6"
    assert h_map["H5"].horizon_h == "all"
    assert h_map["H5"].n_available == [1, 2]
    assert h_map["H5"].target_threshold == 0.97
    assert h_map["H5"].comparison_op == "<="

    # H6
    assert "H6" in h_map
    assert h_map["H6"].metric == "coverage_80"
    assert h_map["H6"].target_threshold == 0.8


def test_verify_freeze_register_success(tmp_path: Path) -> None:
    """Verify verify_freeze_register succeeds on freshly generated freeze register."""
    out_json = tmp_path / "freeze_register.json"
    out_md = tmp_path / "freeze_register.md"

    create_freeze_register(
        output_path=out_json,
        report_path=out_md,
        write_files=True,
    )

    is_valid, drift_messages = verify_freeze_register(
        register_path=out_json,
        strict=True,
    )
    assert is_valid is True
    assert len(drift_messages) == 0


def test_verify_detects_config_drift(tmp_path: Path) -> None:
    """Verify verify_freeze_register detects configuration drift and raises FreezeVerificationError."""
    # 1. Create original register
    out_json = tmp_path / "freeze_register.json"
    out_md = tmp_path / "freeze_register.md"
    create_freeze_register(
        output_path=out_json,
        report_path=out_md,
        write_files=True,
    )

    # 2. Create mutated configs directory
    mutated_configs = tmp_path / "configs"
    shutil.copytree("configs", mutated_configs)

    # Mutate evaluation.yaml threshold
    eval_yaml = mutated_configs / "evaluation.yaml"
    content = eval_yaml.read_text(encoding="utf-8")
    mutated_content = content.replace("min_value: 0.15", "min_value: 0.20")
    eval_yaml.write_text(mutated_content, encoding="utf-8")

    # 3. Verify drift is caught
    is_valid, drift_messages = verify_freeze_register(
        register_path=out_json,
        strict=False,
        configs_dir=mutated_configs,
    )
    assert is_valid is False
    assert any("Configuration hash drift" in msg for msg in drift_messages)
    assert any("Hypothesis 'H1' threshold drift" in msg for msg in drift_messages)

    # 4. Strict mode raises FreezeVerificationError
    with pytest.raises(
        FreezeVerificationError, match="Gate G4 Freeze Register verification failed"
    ):
        verify_freeze_register(
            register_path=out_json,
            strict=True,
            configs_dir=mutated_configs,
        )


def test_verify_detects_bundle_tamper(tmp_path: Path) -> None:
    """Verify verify_freeze_register detects model bundle digest tampering."""
    out_json = tmp_path / "freeze_register.json"
    create_freeze_register(
        output_path=out_json,
        bundle_sha256="1111111111111111111111111111111111111111111111111111111111111111",
        base_sha256="2222222222222222222222222222222222222222222222222222222222222222",
        write_files=True,
    )

    # Create dummy artifact files with mismatched hash
    tampered_cal = tmp_path / "tampered_calibrator.pt"
    tampered_cal.write_bytes(b"tampered content")

    is_valid, drift_messages = verify_freeze_register(
        register_path=out_json,
        calibrator_path=tampered_cal,
        strict=False,
    )
    assert is_valid is False
    assert any("Candidate calibrator SHA-256 mismatch" in msg for msg in drift_messages)


def test_verify_detects_station_panel_tamper(tmp_path: Path) -> None:
    """Verify verify_freeze_register detects city panel station modification or removal."""
    out_json = tmp_path / "freeze_register.json"
    create_freeze_register(
        output_path=out_json,
        write_files=True,
    )

    # Load real registry and drop one station
    df = pd.read_parquet("data/registry/registry_lahore.parquet")
    tampered_df = df.iloc[:3].copy()  # drop remaining stations
    tampered_parquet = tmp_path / "tampered_registry.parquet"
    tampered_df.to_parquet(tampered_parquet)

    is_valid, drift_messages = verify_freeze_register(
        register_path=out_json,
        registry_path=tampered_parquet,
        strict=False,
    )
    assert is_valid is False
    assert any("City panel station set drift" in msg for msg in drift_messages)


def test_freeze_register_serialization_json_and_md(tmp_path: Path) -> None:
    """Verify JSON and Markdown serialization, schema completeness, and line ending hygiene."""
    out_json = tmp_path / "freeze_register.json"
    out_md = tmp_path / "freeze_register.md"

    reg = create_freeze_register(
        output_path=out_json,
        report_path=out_md,
        write_files=True,
    )

    # 1. Roundtrip from dict and JSON string
    d = reg.to_dict()
    assert isinstance(d, dict)
    reg_rebuilt = FreezeRegister.from_dict(d)
    assert reg_rebuilt.config_hash == reg.config_hash
    assert reg_rebuilt.git_tag == reg.git_tag
    assert len(reg_rebuilt.city_panel) == len(reg.city_panel)

    json_str = reg.to_json()
    reg_json = FreezeRegister.from_json(json_str)
    assert reg_json.bundle_sha256 == reg.bundle_sha256

    # 2. Roundtrip from file
    reg_loaded = FreezeRegister.load(out_json)
    assert reg_loaded.config_hash == reg.config_hash

    # 3. Check line endings and single trailing newline (hygiene contract)
    raw_json = out_json.read_bytes()
    assert b"\r\n" not in raw_json
    assert raw_json.endswith(b"\n")
    assert not raw_json.endswith(b"\n\n")

    raw_md = out_md.read_bytes()
    assert b"\r\n" not in raw_md
    assert raw_md.endswith(b"\n")
    assert not raw_md.endswith(b"\n\n")

    # 4. Markdown content verification
    md_text = out_md.read_text(encoding="utf-8")
    assert "# SmogSense Gate G4 Freeze Register & Invariant Lockdown" in md_text
    assert "Pre-Registered Confirmatory Hypotheses (H1-H6)" in md_text
    assert "Frozen City Panel Station Network" in md_text
    assert "Frozen Architectural Invariants & Hyperparameters" in md_text
    assert "Invariant Lockdown & Scientific Integrity Attestation" in md_text


def test_cli_freeze_register_command(tmp_path: Path) -> None:
    """Verify smogsense eval freeze-register and alias freeze execute cleanly via Typer CLI."""
    out_dir = tmp_path / "state"
    rep_dir = tmp_path / "reports"

    # 1. Create command
    res_create = runner.invoke(
        app,
        [
            "eval",
            "freeze-register",
            "--create",
            "--git-tag",
            "v0.3.0-gate-g4",
            "--output-dir",
            str(out_dir),
            "--report-dir",
            str(rep_dir),
        ],
    )
    assert res_create.exit_code == 0
    assert "FREEZE REGISTER SUMMARY" in res_create.stdout
    assert (out_dir / "freeze_register.json").is_file()
    assert (rep_dir / "freeze_register.md").is_file()

    # 2. Verify command on created register
    res_verify = runner.invoke(
        app,
        [
            "eval",
            "freeze-register",
            "--verify",
            "--output-dir",
            str(out_dir),
            "--report-dir",
            str(rep_dir),
        ],
    )
    assert res_verify.exit_code == 0
    assert "Gate G4 Freeze Register verification PASSED" in res_verify.stdout

    # 3. Test alias command `freeze`
    out_dir_alias = tmp_path / "state_alias"
    rep_dir_alias = tmp_path / "reports_alias"
    res_alias = runner.invoke(
        app,
        [
            "eval",
            "freeze",
            "--create",
            "--output-dir",
            str(out_dir_alias),
            "--report-dir",
            str(rep_dir_alias),
        ],
    )
    assert res_alias.exit_code == 0
    assert (out_dir_alias / "freeze_register.json").is_file()
