"""tests.unit.evaluation.test_promotion — Unit test suite for Gate G5 promotion decision engine.

Tests:
1. test_promotion_rule_passes: Significant improvement at >= 2 horizons with non-worsened coverage passes.
2. test_promotion_rule_fails_insufficient_horizons: Significant at only 1 horizon fails promotion.
3. test_promotion_rule_fails_worse_coverage: Significant improvement but degraded coverage fails promotion.
4. test_headline_model_resolution_m7_wins: M7 beating M6 and M5 resolves to M7.
5. test_headline_model_resolution_m6_wins_on_tie: M7 tying M6 resolves to M6 (simpler rung).
6. test_headline_model_resolution_m5_fallback: M7/M6 failing over M5 resolves to M5 (physics base fallback).
7. test_guardrail_audit_checklist: Verifies all 7 guardrails pass and failures are flagged.
8. test_report_serialization_json_and_md: Verifies valid JSON and Markdown PR description generation with strictly LF line endings.
9. test_cli_promotion_command: Tests CLI flags and execution via CliRunner.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.evaluation.promotion import (
    PromotionDecisionResult,
    RungComparison,
    apply_promotion_to_config,
    audit_guardrails,
    evaluate_promotion_rule,
    generate_promotion_decision,
    resolve_headline_model,
)

runner = CliRunner()


def test_promotion_rule_passes() -> None:
    """Significant improvement at >= 2 horizons with non-worsened coverage passes."""
    cand_crps = {24: 20.0, 48: 20.0, 72: 20.0}
    base_crps = {24: 25.0, 48: 25.0, 72: 25.0}  # improvements +5.0 each
    cand_cov80 = {24: 0.81, 48: 0.80, 72: 0.79}  # errors <= 0.01
    base_cov80 = {24: 0.80, 48: 0.80, 72: 0.80}

    # 95% CIs strictly exclude 0 at all 3 horizons
    ci_95 = {24: (2.0, 8.0), 48: (1.5, 8.5), 72: (1.0, 9.0)}

    comp = evaluate_promotion_rule(
        candidate_rung="M7",
        baseline_rung="M6",
        candidate_crps=cand_crps,
        baseline_crps=base_crps,
        candidate_cov80=cand_cov80,
        baseline_cov80=base_cov80,
        crps_ci_95=ci_95,
    )

    assert comp.passes_promotion_rule is True
    assert comp.num_horizons_significant == 3
    assert comp.coverage_not_worse is True
    assert comp.candidate_rung == "M7"
    assert comp.baseline_rung == "M6"


def test_promotion_rule_fails_insufficient_horizons() -> None:
    """Significant at only 1 horizon fails promotion (requires >= 2 of 3)."""
    cand_crps = {24: 20.0, 48: 24.5, 72: 25.0}
    base_crps = {24: 25.0, 48: 25.0, 72: 25.0}
    cand_cov80 = {24: 0.80, 48: 0.80, 72: 0.80}
    base_cov80 = {24: 0.80, 48: 0.80, 72: 0.80}

    # Only 24h lower bound excludes 0
    ci_95 = {24: (2.0, 8.0), 48: (-0.5, 1.5), 72: (-2.0, 2.0)}

    comp = evaluate_promotion_rule(
        candidate_rung="M7",
        baseline_rung="M6",
        candidate_crps=cand_crps,
        baseline_crps=base_crps,
        candidate_cov80=cand_cov80,
        baseline_cov80=base_cov80,
        crps_ci_95=ci_95,
    )

    assert comp.num_horizons_significant == 1
    assert comp.horizons_significant == [24]
    assert comp.coverage_not_worse is True
    assert comp.passes_promotion_rule is False


def test_promotion_rule_fails_worse_coverage() -> None:
    """Significant CRPS improvement but degraded coverage (> 0.02) fails promotion."""
    cand_crps = {24: 20.0, 48: 20.0, 72: 20.0}
    base_crps = {24: 25.0, 48: 25.0, 72: 25.0}

    # Candidate coverage degrades by 0.10 (error 0.10 vs baseline 0.00)
    cand_cov80 = {24: 0.70, 48: 0.80, 72: 0.80}
    base_cov80 = {24: 0.80, 48: 0.80, 72: 0.80}
    ci_95 = {24: (2.0, 8.0), 48: (1.5, 8.5), 72: (1.0, 9.0)}

    comp = evaluate_promotion_rule(
        candidate_rung="M7",
        baseline_rung="M6",
        candidate_crps=cand_crps,
        baseline_crps=base_crps,
        candidate_cov80=cand_cov80,
        baseline_cov80=base_cov80,
        crps_ci_95=ci_95,
    )

    assert comp.num_horizons_significant == 3
    assert comp.coverage_not_worse is False
    assert comp.passes_promotion_rule is False


def test_headline_model_resolution_m7_wins() -> None:
    """M7 beating M6 and M5 resolves to M7."""
    comp_7_6 = RungComparison(
        candidate_rung="M7",
        baseline_rung="M6",
        crps_improvements={24: 5.0, 48: 5.0, 72: 5.0},
        crps_ci_95={24: (1.0, 9.0), 48: (1.0, 9.0), 72: (1.0, 9.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )
    comp_7_5 = RungComparison(
        candidate_rung="M7",
        baseline_rung="M5",
        crps_improvements={24: 10.0, 48: 10.0, 72: 10.0},
        crps_ci_95={24: (5.0, 15.0), 48: (5.0, 15.0), 72: (5.0, 15.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )
    comp_7_2 = RungComparison(
        candidate_rung="M7",
        baseline_rung="M2",
        crps_improvements={24: 2.0, 48: 2.0, 72: 2.0},
        crps_ci_95={24: (0.5, 3.5), 48: (0.5, 3.5), 72: (0.5, 3.5)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )

    hm, act, bndl, just = resolve_headline_model(
        [comp_7_6, comp_7_5, comp_7_2],
        all_guardrails_passed=True,
        default_bundle_tag="models-v0.2.0",
    )

    assert hm == "M7"
    assert act == "promote_m7"
    assert bndl == "models-v0.2.0"
    assert "Model M7" in just


def test_headline_model_resolution_m6_wins_on_tie() -> None:
    """M7 tying or failing against M6 resolves to M6 (simpler rung)."""
    comp_7_6 = RungComparison(
        candidate_rung="M7",
        baseline_rung="M6",
        crps_improvements={24: 0.0, 48: 0.0, 72: 0.0},
        crps_ci_95={24: (-2.0, 2.0), 48: (-2.0, 2.0), 72: (-2.0, 2.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[],
        num_horizons_significant=0,
        coverage_not_worse=True,
        passes_promotion_rule=False,  # Tie or failure
    )
    comp_6_5 = RungComparison(
        candidate_rung="M6",
        baseline_rung="M5",
        crps_improvements={24: 10.0, 48: 10.0, 72: 10.0},
        crps_ci_95={24: (5.0, 15.0), 48: (5.0, 15.0), 72: (5.0, 15.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )
    comp_6_2 = RungComparison(
        candidate_rung="M6",
        baseline_rung="M2",
        crps_improvements={24: 2.0, 48: 2.0, 72: 2.0},
        crps_ci_95={24: (0.5, 3.5), 48: (0.5, 3.5), 72: (0.5, 3.5)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )

    hm, act, bndl, just = resolve_headline_model(
        [comp_7_6, comp_6_5, comp_6_2],
        all_guardrails_passed=True,
        default_bundle_tag="models-v0.2.0",
    )

    assert hm == "M6"
    assert act == "promote_m6"
    assert bndl == "models-v0.2.0-m6"
    assert "simpler rung M6" in just


def test_headline_model_resolution_m5_fallback() -> None:
    """M7 and M6 failing over M5 falls back to robust physics base M5."""
    comp_7_6 = RungComparison(
        candidate_rung="M7",
        baseline_rung="M6",
        crps_improvements={24: 0.0, 48: 0.0, 72: 0.0},
        crps_ci_95={24: (-2.0, 2.0), 48: (-2.0, 2.0), 72: (-2.0, 2.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[],
        num_horizons_significant=0,
        coverage_not_worse=True,
        passes_promotion_rule=False,
    )
    comp_6_5 = RungComparison(
        candidate_rung="M6",
        baseline_rung="M5",
        crps_improvements={24: -1.0, 48: -1.0, 72: -1.0},
        crps_ci_95={24: (-3.0, 1.0), 48: (-3.0, 1.0), 72: (-3.0, 1.0)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[],
        num_horizons_significant=0,
        coverage_not_worse=True,
        passes_promotion_rule=False,
    )
    comp_5_2 = RungComparison(
        candidate_rung="M5",
        baseline_rung="M2",
        crps_improvements={24: 1.0, 48: 1.0, 72: 1.0},
        crps_ci_95={24: (0.2, 1.8), 48: (0.2, 1.8), 72: (0.2, 1.8)},
        coverage_deltas={24: 0.0, 48: 0.0, 72: 0.0},
        horizons_significant=[24, 48, 72],
        num_horizons_significant=3,
        coverage_not_worse=True,
        passes_promotion_rule=True,
    )

    hm, act, bndl, just = resolve_headline_model(
        [comp_7_6, comp_6_5, comp_5_2],
        all_guardrails_passed=True,
        default_bundle_tag="models-v0.2.0",
    )

    assert hm == "M5"
    assert act == "promote_m5"
    assert bndl == "models-v0.2.0-base"
    assert "Model M5" in just


def test_guardrail_audit_checklist() -> None:
    """Verifies all 7 guardrails pass and failures are flagged."""
    items = audit_guardrails()
    assert len(items) == 7
    assert all(item.passed for item in items)

    rule_ids = {item.rule_id for item in items}
    expected_ids = {
        "G-BUDGET",
        "G-CONTAINER",
        "G-PIT",
        "G-RESIDUAL-BOUND",
        "G-FALLBACK",
        "G-TEST-ISOLATION",
        "G-REPRODUCIBILITY",
    }
    assert rule_ids == expected_ids

    # Test override failure
    failing_items = audit_guardrails(override_rules={"G-BUDGET": False})
    budget_item = next(i for i in failing_items if i.rule_id == "G-BUDGET")
    assert budget_item.passed is False

    # Blocked promotion when guardrails fail
    hm, act, bndl, just = resolve_headline_model([], all_guardrails_passed=False)
    assert hm == "M2"
    assert act == "maintain_baseline"
    assert bndl == "none"
    assert "guardrails failed" in just


def test_report_serialization_json_and_md(tmp_path: Path) -> None:
    """Verifies valid JSON and Markdown PR description generation with strictly LF line endings."""
    # Write a test ladder report
    ladder_data = {
        "timestamp_utc": "2026-11-04T09:31:25.955522+00:00",
        "domain": "lahore",
        "eval_location_id": 8664,
        "n_eval_rows": 4648,
        "results_by_horizon": {
            "24": {
                "M0": {"crps": 7.10, "coverage_80": 0.80},
                "M1": {"crps": 29.03, "coverage_80": 0.80},
                "M2": {"crps": 7.26, "coverage_80": 0.80},
                "M5": {"crps": 36.50, "coverage_80": 0.12},
                "M6": {"crps": 24.50, "coverage_80": 0.12},
                "M7": {"crps": 27.00, "coverage_80": 0.86},
            },
            "48": {
                "M0": {"crps": 7.10, "coverage_80": 0.80},
                "M1": {"crps": 29.03, "coverage_80": 0.80},
                "M2": {"crps": 7.26, "coverage_80": 0.80},
                "M5": {"crps": 36.50, "coverage_80": 0.12},
                "M6": {"crps": 24.50, "coverage_80": 0.12},
                "M7": {"crps": 27.00, "coverage_80": 0.86},
            },
            "72": {
                "M0": {"crps": 7.10, "coverage_80": 0.80},
                "M1": {"crps": 29.03, "coverage_80": 0.80},
                "M2": {"crps": 7.26, "coverage_80": 0.80},
                "M5": {"crps": 36.50, "coverage_80": 0.12},
                "M6": {"crps": 24.50, "coverage_80": 0.12},
                "M7": {"crps": 27.00, "coverage_80": 0.86},
            },
        },
    }
    ladder_file = tmp_path / "ladder.json"
    ladder_file.write_text(json.dumps(ladder_data), encoding="utf-8")

    out_dir = tmp_path / "reports"
    decision = generate_promotion_decision(
        ladder_path=ladder_file,
        output_dir=out_dir,
    )

    assert isinstance(decision, PromotionDecisionResult)
    json_path = out_dir / "promotion_decision.json"
    md_path = out_dir / "promotion_decision.md"

    assert json_path.exists()
    assert md_path.exists()

    # Verify JSON round-trip
    loaded_data = json.loads(json_path.read_text(encoding="utf-8"))
    loaded_decision = PromotionDecisionResult.from_dict(loaded_data)
    assert loaded_decision.headline_model == decision.headline_model
    assert loaded_decision.recommended_action == decision.recommended_action
    assert len(loaded_decision.guardrails) == 7

    # Verify Markdown LF line endings
    raw_md_bytes = md_path.read_bytes()
    assert b"\r\n" not in raw_md_bytes, "Found CRLF line endings in promotion_decision.md"
    assert b"\n" in raw_md_bytes

    md_text = md_path.read_text(encoding="utf-8")
    assert "# promotion:" in md_text
    assert "Complete Model Ladder Scorecard" in md_text
    assert "Mandatory Guardrail Checklist" in md_text
    assert "[x] **G-BUDGET**" in md_text
    assert "```diff" in md_text


def test_cli_promotion_command(tmp_path: Path) -> None:
    """Tests CLI flags and execution via CliRunner."""
    out_dir = tmp_path / "reports"
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("promoted_bundle: none\nversion: 1\n", encoding="utf-8")

    # Run CLI command with existing reports/ladder.json
    res = runner.invoke(
        app,
        [
            "eval",
            "promotion",
            "--ladder-report",
            "reports/ladder.json",
            "--output-dir",
            str(out_dir),
            "--no-apply",
        ],
    )
    assert res.exit_code == 0
    assert "GATE G5 MODEL PROMOTION REVIEW" in res.output
    assert "PROMOTION DECISION SUMMARY" in res.output
    assert (out_dir / "promotion_decision.json").exists()
    assert (out_dir / "promotion_decision.md").exists()

    # Test apply_promotion_to_config directly
    applied = apply_promotion_to_config(
        model_config_path=cfg_file,
        promoted_bundle_value="models-v0.2.0-test",
    )
    assert applied is True
    updated_cfg = cfg_file.read_text(encoding="utf-8")
    assert "promoted_bundle: models-v0.2.0-test" in updated_cfg
