"""Unit tests for smogsense.evaluation.registration - Gate G4 Pre-Registration Engine.

Verifies:
1. Pre-registration document structure, hashes, and calendar specifications.
2. Hypothesis tiers (Tier 1 Powered, Tier 1 Indicative, Tier 2 Descriptive) and multiplicity mapping.
3. Statistical power note concordance with empirical pilot parameters (P2-14).
4. Anti-drift verification engine (clean verification, config hash drift, threshold mutation).
5. Serialization to JSON and Markdown hygiene (LF endings, table structure, no AI context mentions).
6. CLI pre-register and register commands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.registration import (
    HypothesisTier,
    PreRegistrationDocument,
    PreRegistrationVerificationError,
    StatisticalPowerNote,
    create_pre_registration_document,
    get_canonical_hypotheses,
    get_canonical_multiplicity_protocol,
    verify_pre_registration,
)

pytestmark = pytest.mark.unit


def test_pre_registration_doc_structure_and_hashes(tmp_path: Path) -> None:
    """Validate all fields, hashes, calendar dates, and invariants in PreRegistrationDocument."""
    doc = create_pre_registration_document(
        configs_dir="configs",
        freeze_register_path=".state/freeze_register.json",
        output_state_path=tmp_path / "pre_reg.json",
        output_doc_path=tmp_path / "pre_reg.md",
        git_tag="v0.3.0-gate-g4",
        write_files=False,
    )

    assert doc.schema_version == 1
    assert "Gate G4" in doc.title
    assert doc.registration_date == "2026-12-04"
    assert doc.git_release_tag == "v0.3.0-gate-g4"
    assert len(doc.git_commit_sha) >= 7
    assert len(doc.config_hash) == 64
    assert len(doc.shared_physics_base_sha256) == 64
    assert len(doc.candidate_calibrator_sha256) == 64
    assert doc.headline_model in {
        "m7_hybrid_maml",
        "m6_finetune_transfer",
        "m5_lgbm_source_zero_shot",
    }

    # Calendar specification
    cal = doc.evaluation_calendar
    assert cal.dev_history_end == "2026-10-18"
    assert "2026-10-19" in cal.dev_replay_window
    assert cal.gate_g4_freeze_date == "2026-12-04"
    assert cal.confirmatory_shadow_window == "2026-12-07 to 2027-01-31"
    assert cal.settlement_delay_hours == 96
    assert cal.analysis_freeze_gate_g7 == "2027-02-08"

    # Data integrity invariants
    invs = doc.data_integrity_invariants
    assert invs["min_valid_hours"] == 18
    assert invs["settlement_delay_hours"] == 96
    assert invs["no_retroactive_rescoring"] is True
    assert invs["inner_steps_j"] == 3
    assert invs["embargo_hours"] == 72


def test_hypothesis_tiers_and_multiplicity_mapping() -> None:
    """Assert H5 is Tier 1 Powered, H1-H4 are Tier 1 Indicative, and H6 is Tier 2 Descriptive."""
    hypotheses = get_canonical_hypotheses()
    by_id = {h.hypothesis_id: h for h in hypotheses}

    assert set(by_id.keys()) == {"H1", "H2", "H3", "H4", "H5", "H6"}

    # H1 - H4: Tier 1 Indicative
    for hid in ["H1", "H2", "H3", "H4"]:
        h = by_id[hid]
        assert h.tier == HypothesisTier.TIER_1_INDICATIVE
        assert h.domain == "lahore"
        assert h.multiplicity_family == "family_1_lahore"
        assert h.descriptive_if_days_under == 35

    # H5: Tier 1 Powered
    h5 = by_id["H5"]
    assert h5.tier == HypothesisTier.TIER_1_POWERED
    assert h5.domain == "delhi"
    assert h5.multiplicity_family == "family_2_delhi_h5"
    assert h5.metric == "crps_ratio_m7_over_m6"
    assert h5.threshold_value == 0.97
    assert h5.comparison_operator == "<="

    # H6: Tier 2 Descriptive
    h6 = by_id["H6"]
    assert h6.tier == HypothesisTier.TIER_2_DESCRIPTIVE
    assert h6.domain == "lahore"
    assert h6.multiplicity_family == "unadjusted_h6"
    assert h6.metric == "coverage_80"
    assert h6.threshold_value == 0.80

    # Multiplicity Protocol dictionary
    protocol = get_canonical_multiplicity_protocol()
    assert "family_1_lahore" in protocol
    assert protocol["family_1_lahore"]["method"] == "holm_bonferroni"
    assert protocol["family_1_lahore"]["fwer_alpha"] == 0.05

    assert "family_2_delhi_h5" in protocol
    assert protocol["family_2_delhi_h5"]["method"] == "single_hypothesis_unadjusted"
    assert protocol["family_2_delhi_h5"]["fwer_alpha"] == 0.05

    assert "unadjusted_h6" in protocol
    assert protocol["unadjusted_h6"]["fwer_alpha"] is None


def test_power_note_parameters_match_pilot() -> None:
    """Assert statistical power parameters match Task P2-14 pilot report exactly."""
    power_note = StatisticalPowerNote()

    assert power_note.h5_domain == "delhi"
    assert power_note.season == "2025-2026"
    assert power_note.eligible_stations == 49
    assert power_note.observed_log_ratio_sd == 0.0004
    assert power_note.effective_sample_size == 8.00
    assert power_note.statistical_power == 100.0
    assert "56 days" in power_note.lahore_field_power_note
    assert (
        "rho_1" in power_note.lahore_field_power_note or "rho" in power_note.lahore_field_power_note
    )


def test_verify_pre_registration_success_and_drift_detection(tmp_path: Path) -> None:
    """Test clean verification and detection of configuration drift or threshold tampering."""
    state_file = tmp_path / "pre_reg.json"
    doc_file = tmp_path / "pre_reg.md"

    doc = create_pre_registration_document(
        configs_dir="configs",
        freeze_register_path=".state/freeze_register.json",
        output_state_path=state_file,
        output_doc_path=doc_file,
        write_files=True,
    )

    # 1. Clean verification succeeds
    is_valid, drift = verify_pre_registration(
        pre_registration_path=state_file,
        freeze_register_path=".state/freeze_register.json",
        configs_dir="configs",
        strict=True,
    )
    assert is_valid is True
    assert len(drift) == 0

    # 2. Tampered config hash
    tampered_data = doc.to_dict()
    tampered_data["config_hash"] = "0" * 64
    state_file.write_text(json.dumps(tampered_data, indent=2), encoding="utf-8")

    is_valid, drift = verify_pre_registration(
        pre_registration_path=state_file,
        freeze_register_path=".state/freeze_register.json",
        configs_dir="configs",
        strict=False,
    )
    assert is_valid is False
    assert any("Configuration hash drift" in msg for msg in drift)

    with pytest.raises(PreRegistrationVerificationError) as exc_info:
        verify_pre_registration(
            pre_registration_path=state_file,
            freeze_register_path=".state/freeze_register.json",
            configs_dir="configs",
            strict=True,
        )
    assert exc_info.value.exit_code == 30

    # 3. Tampered hypothesis threshold
    tampered_data2 = doc.to_dict()
    tampered_data2["hypotheses"][0]["threshold_value"] = 0.99  # H1 threshold changed from 0.15
    state_file.write_text(json.dumps(tampered_data2, indent=2), encoding="utf-8")

    is_valid, drift = verify_pre_registration(
        pre_registration_path=state_file,
        freeze_register_path=".state/freeze_register.json",
        configs_dir="configs",
        strict=False,
    )
    assert is_valid is False
    assert any("Hypothesis H1 threshold altered" in msg for msg in drift)


def test_pre_registration_serialization_json_and_md(tmp_path: Path) -> None:
    """Validate serialization roundtrip to JSON and Markdown documentation hygiene."""
    doc = create_pre_registration_document(
        configs_dir="configs",
        freeze_register_path=".state/freeze_register.json",
        output_state_path=tmp_path / "state.json",
        output_doc_path=tmp_path / "doc.md",
        write_files=True,
    )

    # JSON roundtrip
    d = doc.to_dict()
    restored = PreRegistrationDocument.from_dict(d)
    assert restored.title == doc.title
    assert restored.config_hash == doc.config_hash
    assert len(restored.hypotheses) == len(doc.hypotheses)

    json_str = doc.to_json()
    from_json_doc = PreRegistrationDocument.from_json(json_str)
    assert from_json_doc.freeze_register_digest == doc.freeze_register_digest

    # Markdown generation
    md = doc.to_markdown()
    assert "# SmogSense Gate G4 Confirmatory Pre-Registration" in md
    assert "RQ1" in md
    assert "RQ2" in md
    assert "RQ3" in md
    assert "RQ4" in md
    assert "H1" in md
    assert "H5" in md
    assert "Holm-Bonferroni" in md
    assert "0.0004" in md
    assert "100.0%" in md

    # Ensure no git-ignored context files are referenced
    forbidden = [
        "memory.md",
        "decisions.md",
        "roadmap.md",
        "studyguide.md",
        "AGENTS.md",
        "CLAUDE.md",
    ]
    for f in forbidden:
        assert f not in md, f"Forbidden reference to {f} found in generated markdown"

    # Ensure LF endings and single trailing newline
    raw_md = (tmp_path / "doc.md").read_bytes()
    assert b"\r\n" not in raw_md
    assert raw_md.endswith(b"\n")
    assert not raw_md.endswith(b"\n\n")


def test_cli_pre_register_command(tmp_path: Path) -> None:
    """Test CLI pre-register and register commands with create and verify flags."""
    runner = CliRunner()
    doc_out = tmp_path / "cli_pre_reg.md"
    state_out = tmp_path / "cli_pre_reg.json"

    # 1. CLI create
    result = runner.invoke(
        app,
        [
            "eval",
            "pre-register",
            "--create",
            "--output-doc",
            str(doc_out),
            "--output-state",
            str(state_out),
        ],
    )
    assert result.exit_code == 0
    assert "PRE-REGISTRATION SPECIFICATION SUMMARY" in result.stdout
    assert state_out.is_file()
    assert doc_out.is_file()

    # 2. CLI verify
    result_v = runner.invoke(
        app,
        [
            "eval",
            "pre-register",
            "--verify",
            "--output-state",
            str(state_out),
        ],
    )
    assert result_v.exit_code == 0
    assert "[OK] Gate G4 Pre-Registration verification PASSED" in result_v.stdout

    # 3. CLI alias register verify
    result_alias = runner.invoke(
        app,
        [
            "eval",
            "register",
            "--verify",
            "--output-state",
            str(state_out),
        ],
    )
    assert result_alias.exit_code == 0
    assert "[OK] Gate G4 Pre-Registration verification PASSED" in result_alias.stdout
