"""tests.unit.evaluation.test_freeze_pre_registration — Gate G4 Freeze Register & Pre-Registration Audit.

Verifies Task P5-01:
1. test_freeze_register_creation_and_roundtrip: Asserts .state/freeze_register.json and reports/freeze_register.md match schema and model digests.
2. test_freeze_register_detects_config_tampering: Injects temporary config modification and asserts exit 30 / FreezeVerificationError.
3. test_pre_registration_creation_and_verification: Asserts .state/pre_registration.json and docs/pre-registration.md concordance and zero drift.
4. test_pre_registration_detects_hypothesis_tampering: Modifies an H1-H6 threshold in test state and asserts rejection.
5. test_cli_eval_audit_command: End-to-end execution of smogsense eval audit via Typer CliRunner.

Specification:
- docs/evaluation-strategy.md §1, §6, §8, §9, §10
- docs/ml-architecture.md §10
- docs/PRD.md (FR-21, FR-38, NFR-16)
- decisions.md (ADR-002, ADR-016, ADR-019, ADR-024)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.freeze import (
    FreezeRegister,
    FreezeVerificationError,
    audit_confirmatory_lockdown,
    create_freeze_register,
    format_confirmatory_audit_table,
    verify_freeze_register,
)
from smogsense.evaluation.registration import (
    PreRegistrationVerificationError,
    create_pre_registration_document,
    verify_pre_registration,
)

runner = CliRunner()
pytestmark = pytest.mark.unit


def test_freeze_register_creation_and_roundtrip(tmp_path: Path) -> None:
    """Assert freeze register creation, model digest bindings, schema conformity, and roundtrip."""
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

    # Cryptographic digests
    assert len(reg.config_hash) == 64
    assert len(reg.bundle_sha256) == 64
    assert len(reg.base_sha256) == 64
    assert reg.base_sha256.startswith("be0f18fb")
    assert reg.bundle_sha256.startswith("cda76200")

    # City panel and hypotheses
    assert len(reg.city_panel) >= 3
    assert len(reg.hypotheses) == 6
    h_ids = {h.hypothesis_id for h in reg.hypotheses}
    assert h_ids == {"H1", "H2", "H3", "H4", "H5", "H6"}

    # Invariants
    assert "rho_99" in reg.invariants
    assert reg.invariants["inner_steps_j"] == 3
    assert reg.invariants["grad_clip"] == 1.0

    # Serialization roundtrip
    loaded = FreezeRegister.load(out_json)
    assert loaded.config_hash == reg.config_hash
    assert loaded.bundle_sha256 == reg.bundle_sha256
    assert loaded.base_sha256 == reg.base_sha256
    assert len(loaded.city_panel) == len(reg.city_panel)
    assert len(loaded.hypotheses) == len(reg.hypotheses)

    # Markdown report existence and single trailing newline
    assert out_md.is_file()
    md_content = out_md.read_text(encoding="utf-8")
    assert md_content.endswith("\n")
    assert not md_content.endswith("\n\n")
    assert "SmogSense Gate G4 Freeze Register" in md_content
    assert "be0f18fb" in md_content
    assert "cda76200" in md_content


def test_freeze_register_detects_config_tampering(tmp_path: Path) -> None:
    """Assert freeze register detects config tampering and raises FreezeVerificationError."""
    # 1. Create reference freeze register
    ref_reg_path = tmp_path / "freeze_register.json"
    ref_md_path = tmp_path / "freeze_register.md"
    create_freeze_register(
        output_path=ref_reg_path,
        report_path=ref_md_path,
        write_files=True,
    )

    # 2. Duplicate configs directory and modify a parameter
    custom_configs = tmp_path / "configs"
    shutil.copytree("configs", custom_configs)
    model_yaml = custom_configs / "model.yaml"
    content = model_yaml.read_text(encoding="utf-8")
    tampered_content = content.replace("steps: 3", "steps: 5")
    model_yaml.write_text(tampered_content, encoding="utf-8")

    # 3. Verify with strict=False
    is_valid, drift_messages = verify_freeze_register(
        register_path=ref_reg_path,
        configs_dir=custom_configs,
        strict=False,
    )
    assert is_valid is False
    assert any("Configuration hash drift" in m for m in drift_messages)

    # 4. Verify with strict=True raises FreezeVerificationError with exit_code=30
    with pytest.raises(FreezeVerificationError) as exc_info:
        verify_freeze_register(
            register_path=ref_reg_path,
            configs_dir=custom_configs,
            strict=True,
        )
    assert exc_info.value.exit_code == 30


def test_pre_registration_creation_and_verification(tmp_path: Path) -> None:
    """Assert pre-registration creation, documentation concordance, and zero drift."""
    out_state = tmp_path / "pre_registration.json"
    out_doc = tmp_path / "pre-registration.md"

    doc = create_pre_registration_document(
        configs_dir="configs",
        freeze_register_path=".state/freeze_register.json",
        output_state_path=out_state,
        output_doc_path=out_doc,
        git_tag="v0.3.0-gate-g4",
        write_files=True,
    )

    assert doc.schema_version == 1
    assert doc.registration_date == "2026-12-04"
    assert doc.git_release_tag == "v0.3.0-gate-g4"
    assert doc.shared_physics_base_sha256.startswith("be0f18fb")
    assert doc.candidate_calibrator_sha256.startswith("cda76200")
    assert doc.headline_model == "m7_hybrid_maml"
    assert len(doc.hypotheses) == 6

    # Verify zero drift against active state
    is_valid, drift_messages = verify_pre_registration(
        pre_registration_path=out_state,
        freeze_register_path=".state/freeze_register.json",
        configs_dir="configs",
        strict=False,
    )
    assert is_valid is True
    assert drift_messages == []

    # Markdown document verification
    assert out_doc.is_file()
    doc_text = out_doc.read_text(encoding="utf-8")
    assert doc_text.endswith("\n")
    assert not doc_text.endswith("\n\n")
    assert "SmogSense Gate G4 Confirmatory Pre-Registration" in doc_text
    assert "Tier 1 Powered" in doc_text
    assert "Tier 1 Indicative" in doc_text
    assert "Tier 2 Descriptive" in doc_text
    assert "2026-12-07 to 2027-01-31" in doc_text


def test_pre_registration_detects_hypothesis_tampering(tmp_path: Path) -> None:
    """Assert modification of pre-registered hypothesis thresholds triggers verification rejection."""
    out_state = tmp_path / "pre_registration.json"
    out_doc = tmp_path / "pre-registration.md"

    create_pre_registration_document(
        configs_dir="configs",
        freeze_register_path=".state/freeze_register.json",
        output_state_path=out_state,
        output_doc_path=out_doc,
        write_files=True,
    )

    # Mutate H5 threshold from 0.97 to 0.90
    data = json.loads(out_state.read_text(encoding="utf-8"))
    for h in data["hypotheses"]:
        if h["hypothesis_id"] == "H5":
            h["threshold_value"] = 0.90
    out_state.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # Non-strict check detects mutation
    is_valid, drift_messages = verify_pre_registration(
        pre_registration_path=out_state,
        freeze_register_path=".state/freeze_register.json",
        configs_dir="configs",
        strict=False,
    )
    assert is_valid is False
    assert any(
        "threshold altered" in m.lower() or "mutation" in m.lower() or "drift" in m.lower()
        for m in drift_messages
    )

    # Strict check raises PreRegistrationVerificationError with exit_code=30
    with pytest.raises(PreRegistrationVerificationError) as exc_info:
        verify_pre_registration(
            pre_registration_path=out_state,
            freeze_register_path=".state/freeze_register.json",
            configs_dir="configs",
            strict=True,
        )
    assert exc_info.value.exit_code == 30


def test_cli_eval_audit_command(tmp_path: Path) -> None:
    """Test smogsense eval audit and verify-all CLI commands with table and JSON outputs."""
    # 1. Standard audit command passing
    res = runner.invoke(app, ["eval", "audit"])
    assert res.exit_code == 0
    assert "SmogSense Gate G4 Freeze & Pre-Registration Verification Audit" in res.output
    assert "Freeze Register" in res.output
    assert "Pre-Registration Spec" in res.output
    assert "PASS" in res.output
    assert "OVERALL AUDIT: PASSED" in res.output

    # 2. JSON output format
    res_json = runner.invoke(app, ["eval", "audit", "--json"])
    assert res_json.exit_code == 0
    payload = json.loads(res_json.output)
    assert payload["is_valid"] is True
    assert payload["freeze_register"]["valid"] is True
    assert payload["pre_registration"]["valid"] is True
    assert payload["base_sha256"].startswith("be0f18fb")
    assert payload["bundle_sha256"].startswith("cda76200")

    # 3. Alias verify-all command
    res_alias = runner.invoke(app, ["eval", "verify-all"])
    assert res_alias.exit_code == 0
    assert "OVERALL AUDIT: PASSED" in res_alias.output

    # 4. Direct audit_confirmatory_lockdown Python API
    audit_res = audit_confirmatory_lockdown(strict=True)
    assert audit_res.is_valid is True
    assert audit_res.freeze_valid is True
    assert audit_res.prereg_valid is True
    table = format_confirmatory_audit_table(audit_res)
    assert "OVERALL AUDIT: PASSED" in table

    # 5. Drift detection via CLI with missing freeze register
    missing_path = tmp_path / "absent_freeze_register.json"
    res_fail = runner.invoke(app, ["eval", "audit", "-f", str(missing_path)])
    assert res_fail.exit_code == 30
