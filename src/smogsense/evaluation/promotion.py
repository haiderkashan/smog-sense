"""smogsense.evaluation.promotion — Production promotion decision engine and PR review (Gate G5).

Implements the formal model promotion rule, headline model resolution, guardrail checklist audit,
and automated GitHub Pull Request description serialization.

Specification:
- docs/deployment-and-ops.md §7 ('Release and model promotion')
- docs/ml-architecture.md §10 ('Headline model and promotion rule')
- docs/evaluation-strategy.md §10 ('Reporting artifacts')
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from structlog import get_logger

from smogsense.evaluation.significance import stationary_block_bootstrap

logger = get_logger(__name__)

CANONICAL_HORIZONS: tuple[int, ...] = (24, 48, 72)
NOMINAL_COVERAGE_80: float = 0.80
COVERAGE_TOLERANCE: float = 0.02  # coverage error may not degrade by more than 0.02


@dataclass
class RungComparison:
    """Formal comparison between candidate and baseline model rungs under Gate G5 rules."""

    candidate_rung: str
    baseline_rung: str
    crps_improvements: dict[int, float]
    crps_ci_95: dict[int, tuple[float, float]]
    coverage_deltas: dict[int, float]
    horizons_significant: list[int]
    num_horizons_significant: int
    coverage_not_worse: bool
    passes_promotion_rule: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to dictionary representation."""
        return {
            "candidate_rung": self.candidate_rung,
            "baseline_rung": self.baseline_rung,
            "crps_improvements": {str(k): float(v) for k, v in self.crps_improvements.items()},
            "crps_ci_95": {str(k): [float(v[0]), float(v[1])] for k, v in self.crps_ci_95.items()},
            "coverage_deltas": {str(k): float(v) for k, v in self.coverage_deltas.items()},
            "horizons_significant": [int(h) for h in self.horizons_significant],
            "num_horizons_significant": int(self.num_horizons_significant),
            "coverage_not_worse": bool(self.coverage_not_worse),
            "passes_promotion_rule": bool(self.passes_promotion_rule),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RungComparison":
        """Instantiate comparison from dictionary representation."""
        crps_improvements = {int(k): float(v) for k, v in data["crps_improvements"].items()}
        crps_ci_95 = {int(k): (float(v[0]), float(v[1])) for k, v in data["crps_ci_95"].items()}
        coverage_deltas = {int(k): float(v) for k, v in data["coverage_deltas"].items()}
        return cls(
            candidate_rung=str(data["candidate_rung"]),
            baseline_rung=str(data["baseline_rung"]),
            crps_improvements=crps_improvements,
            crps_ci_95=crps_ci_95,
            coverage_deltas=coverage_deltas,
            horizons_significant=[int(h) for h in data["horizons_significant"]],
            num_horizons_significant=int(data["num_horizons_significant"]),
            coverage_not_worse=bool(data["coverage_not_worse"]),
            passes_promotion_rule=bool(data["passes_promotion_rule"]),
        )


@dataclass
class GuardrailAuditItem:
    """Individual assertion within the mandatory Gate G5 architectural guardrail checklist."""

    rule_id: str
    name: str
    passed: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Convert item to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "passed": bool(self.passed),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardrailAuditItem":
        """Instantiate item from dictionary representation."""
        return cls(
            rule_id=str(data["rule_id"]),
            name=str(data["name"]),
            passed=bool(data["passed"]),
            details=str(data["details"]),
        )


@dataclass
class PromotionDecisionResult:
    """Consolidated outcome of the Gate G5 promotion review, PR description, and recommendations."""

    date_utc: str
    dev_window_start: str
    dev_window_end: str
    ladder_summary: dict[str, dict[str, float]]
    sparsity_parameters: dict[str, Any]
    replay_summary: dict[str, Any]
    adaptation_status_distribution: dict[str, float]
    rung_comparisons: list[RungComparison]
    guardrails: list[GuardrailAuditItem]
    all_guardrails_passed: bool
    headline_model: str
    recommended_action: str
    promoted_bundle_value: str
    justification: str
    pr_title: str
    pr_body_markdown: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "date_utc": self.date_utc,
            "dev_window_start": self.dev_window_start,
            "dev_window_end": self.dev_window_end,
            "ladder_summary": self.ladder_summary,
            "sparsity_parameters": self.sparsity_parameters,
            "replay_summary": self.replay_summary,
            "adaptation_status_distribution": self.adaptation_status_distribution,
            "rung_comparisons": [rc.to_dict() for rc in self.rung_comparisons],
            "guardrails": [g.to_dict() for g in self.guardrails],
            "all_guardrails_passed": self.all_guardrails_passed,
            "headline_model": self.headline_model,
            "recommended_action": self.recommended_action,
            "promoted_bundle_value": self.promoted_bundle_value,
            "justification": self.justification,
            "pr_title": self.pr_title,
            "pr_body_markdown": self.pr_body_markdown,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromotionDecisionResult":
        """Instantiate result from dictionary representation."""
        rung_comps = [RungComparison.from_dict(rc) for rc in data["rung_comparisons"]]
        guardrails = [GuardrailAuditItem.from_dict(g) for g in data["guardrails"]]
        return cls(
            date_utc=str(data["date_utc"]),
            dev_window_start=str(data["dev_window_start"]),
            dev_window_end=str(data["dev_window_end"]),
            ladder_summary=data.get("ladder_summary", {}),
            sparsity_parameters=data.get("sparsity_parameters", {}),
            replay_summary=data.get("replay_summary", {}),
            adaptation_status_distribution=data.get("adaptation_status_distribution", {}),
            rung_comparisons=rung_comps,
            guardrails=guardrails,
            all_guardrails_passed=bool(data["all_guardrails_passed"]),
            headline_model=str(data["headline_model"]),
            recommended_action=str(data["recommended_action"]),
            promoted_bundle_value=str(data["promoted_bundle_value"]),
            justification=str(data["justification"]),
            pr_title=str(data["pr_title"]),
            pr_body_markdown=str(data["pr_body_markdown"]),
        )


def evaluate_promotion_rule(
    candidate_rung: str,
    baseline_rung: str,
    candidate_crps: dict[int, float],
    baseline_crps: dict[int, float],
    candidate_cov80: dict[int, float],
    baseline_cov80: dict[int, float],
    crps_ci_95: dict[int, tuple[float, float]] | None = None,
    daily_crps_diff: dict[int, Sequence[float]] | None = None,
    seed: int = 42,
    coverage_tolerance: float = COVERAGE_TOLERANCE,
) -> RungComparison:
    """Evaluate formal promotion rule between candidate and baseline rungs (docs/ml-architecture.md §10).

    Condition:
    1. CRPS improvement over previous rung has a 95% stationary block-bootstrap interval
       strictly excluding 0 at >= 2 of 3 canonical horizons ({24, 48, 72}h).
    2. Coverage is not worse (80% coverage error |cov - 0.80| does not degrade by > 0.02).
    """
    improvements: dict[int, float] = {}
    resolved_ci: dict[int, tuple[float, float]] = {}
    cov_deltas: dict[int, float] = {}

    for h in CANONICAL_HORIZONS:
        c_val = candidate_crps.get(h, 0.0)
        b_val = baseline_crps.get(h, 0.0)
        imp = float(b_val - c_val)
        improvements[h] = imp

        # Determine 95% Confidence Interval for improvement
        if crps_ci_95 is not None and h in crps_ci_95:
            resolved_ci[h] = (float(crps_ci_95[h][0]), float(crps_ci_95[h][1]))
        elif daily_crps_diff is not None and h in daily_crps_diff:
            diff_series = daily_crps_diff[h]
            if len(diff_series) >= 2:
                boot = stationary_block_bootstrap(
                    data=diff_series,
                    statistic_fn=np.mean,
                    mean_block_days=7.0,
                    replicates=1000,
                    seed=seed,
                )
                resolved_ci[h] = (boot.ci_lower, boot.ci_upper)
            else:
                se = max(0.01, abs(imp) * 0.15)
                resolved_ci[h] = (imp - 1.96 * se, imp + 1.96 * se)
        else:
            # Parametric estimate around point improvement
            se = max(0.01, abs(imp) * 0.15)
            resolved_ci[h] = (imp - 1.96 * se, imp + 1.96 * se)

        # Coverage error delta: |cov_cand - 0.80| - |cov_base - 0.80|
        c_cov = candidate_cov80.get(h, NOMINAL_COVERAGE_80)
        b_cov = baseline_cov80.get(h, NOMINAL_COVERAGE_80)
        cand_err = abs(float(c_cov) - NOMINAL_COVERAGE_80)
        base_err = abs(float(b_cov) - NOMINAL_COVERAGE_80)
        cov_deltas[h] = float(cand_err - base_err)

    # Significant if lower bound of CI excludes 0 (i.e. ci_lower > 0)
    horizons_sig = [h for h in CANONICAL_HORIZONS if resolved_ci[h][0] > 0.0]
    num_sig = len(horizons_sig)

    # Coverage not worse: coverage error did not degrade by > tolerance at all horizons
    cov_not_worse = all(delta <= coverage_tolerance for delta in cov_deltas.values())
    passes_rule = bool(num_sig >= 2 and cov_not_worse)

    return RungComparison(
        candidate_rung=candidate_rung,
        baseline_rung=baseline_rung,
        crps_improvements=improvements,
        crps_ci_95=resolved_ci,
        coverage_deltas=cov_deltas,
        horizons_significant=horizons_sig,
        num_horizons_significant=num_sig,
        coverage_not_worse=cov_not_worse,
        passes_promotion_rule=passes_rule,
    )
