"""smogsense.evaluation.registration - Confirmatory Pre-Registration Engine (Gate G4).

Formalizes the pre-registration record for SmogSense ahead of the prospective
confirmatory shadow evaluation window (7 Dec 2026 - 31 Jan 2027):
- Strongly-typed hypothesis tier specifications (H1-H6: Powered, Indicative, Descriptive).
- Multiplicity control protocol (Holm-Bonferroni families and FWER control).
- Statistical power note incorporating empirical pilot parameters from Task P2-14.
- Confirmatory evaluation calendar and data integrity invariants.
- Anti-drift verification engine detecting configuration changes, threshold alterations,
  or cryptographic digest tampering.

Specification:
- docs/evaluation-strategy.md §1, §6, §8, §9, §10
- docs/ml-architecture.md §10
- docs/PRD.md (FR-21, FR-38, NFR-16)
- configs/evaluation.yaml (hypotheses, calendar, shadow)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from structlog import get_logger

from smogsense.config import Settings
from smogsense.errors import SmogSenseError
from smogsense.evaluation.freeze import (
    DEFAULT_FREEZE_DATE,
    DEFAULT_GIT_TAG,
    FreezeRegister,
    resolve_git_commit_sha,
    resolve_model_digests,
)
from smogsense.models.registry import compute_file_sha256

logger = get_logger(__name__)

DEFAULT_PRE_REGISTRATION_STATE = ".state/pre_registration.json"
DEFAULT_PRE_REGISTRATION_DOC = "docs/pre-registration.md"


# =============================================================================
# Exceptions
# =============================================================================
class PreRegistrationVerificationError(SmogSenseError, ValueError):
    """Raised when pre-registration verification detects configuration drift, threshold mutation, or tampering."""

    def __init__(self, message: str, exit_code: int = 30) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# =============================================================================
# Enums & Data Structures
# =============================================================================
class HypothesisTier(StrEnum):
    """Scientific hypothesis verification tiers (docs/evaluation-strategy.md §1)."""

    TIER_1_POWERED = "tier_1_powered"
    TIER_1_INDICATIVE = "tier_1_indicative"
    TIER_2_DESCRIPTIVE = "tier_2_descriptive"


@dataclass
class HypothesisSpec:
    """Pre-registered evaluation hypothesis specification with tier and multiplicity assignment."""

    hypothesis_id: str
    statement: str
    tier: HypothesisTier
    domain: str
    target_population: str
    metric: str
    horizon_h: str | int
    n_available: list[int] | int
    threshold_value: float | dict[str, float] | str
    comparison_operator: str
    multiplicity_family: str
    descriptive_if_days_under: int = 35

    def to_dict(self) -> dict[str, Any]:
        """Convert hypothesis specification to serializable dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "tier": self.tier.value if isinstance(self.tier, HypothesisTier) else str(self.tier),
            "domain": self.domain,
            "target_population": self.target_population,
            "metric": self.metric,
            "horizon_h": self.horizon_h,
            "n_available": self.n_available,
            "threshold_value": self.threshold_value,
            "comparison_operator": self.comparison_operator,
            "multiplicity_family": self.multiplicity_family,
            "descriptive_if_days_under": self.descriptive_if_days_under,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HypothesisSpec:
        """Deserialize hypothesis specification from dictionary."""
        tier_val = d.get("tier", HypothesisTier.TIER_1_INDICATIVE.value)
        tier = (
            HypothesisTier(tier_val)
            if isinstance(tier_val, str) and tier_val in [t.value for t in HypothesisTier]
            else HypothesisTier.TIER_1_INDICATIVE
        )
        return cls(
            hypothesis_id=str(d["hypothesis_id"]),
            statement=str(d["statement"]),
            tier=tier,
            domain=str(d["domain"]),
            target_population=str(d["target_population"]),
            metric=str(d["metric"]),
            horizon_h=d["horizon_h"],
            n_available=d["n_available"],
            threshold_value=d["threshold_value"],
            comparison_operator=str(d["comparison_operator"]),
            multiplicity_family=str(d["multiplicity_family"]),
            descriptive_if_days_under=int(d.get("descriptive_if_days_under", 35)),
        )


@dataclass
class StatisticalPowerNote:
    """Empirical statistical power parameters derived from Task P2-14 pilot study."""

    h5_domain: str = "delhi"
    season: str = "2025-2026"
    eligible_stations: int = 49
    observed_log_ratio_sd: float = 0.0004
    effective_sample_size: float = 8.00
    statistical_power: float = 100.0
    lahore_field_power_note: str = (
        "Confirmatory window spans 56 days (7 Dec 2026 - 31 Jan 2027). Under AR(1) temporal autocorrelation "
        "rho_1 ≈ 0.5, effective sample size is n_eff ≈ 19 - 28. Tests H1-H4 are evaluated as indicative with "
        "Holm-Bonferroni FWER control, and degrade to descriptive if fewer than 35 valid days are observed or n_eff < 25."
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert power note to serializable dictionary."""
        return {
            "h5_domain": self.h5_domain,
            "season": self.season,
            "eligible_stations": self.eligible_stations,
            "observed_log_ratio_sd": self.observed_log_ratio_sd,
            "effective_sample_size": self.effective_sample_size,
            "statistical_power": self.statistical_power,
            "lahore_field_power_note": self.lahore_field_power_note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StatisticalPowerNote:
        """Deserialize power note from dictionary."""
        return cls(
            h5_domain=str(d.get("h5_domain", "delhi")),
            season=str(d.get("season", "2025-2026")),
            eligible_stations=int(d.get("eligible_stations", 49)),
            observed_log_ratio_sd=float(d.get("observed_log_ratio_sd", 0.0004)),
            effective_sample_size=float(d.get("effective_sample_size", 8.00)),
            statistical_power=float(d.get("statistical_power", 100.0)),
            lahore_field_power_note=str(d.get("lahore_field_power_note", "")),
        )


@dataclass
class EvaluationCalendarSpec:
    """Evaluation timeline and data settlement rules (docs/evaluation-strategy.md §6)."""

    dev_history_end: str = "2026-10-18"
    dev_replay_window: str = "2026-10-19 to 2026-12-06"
    gate_g4_freeze_date: str = "2026-12-04"
    confirmatory_shadow_window: str = "2026-12-07 to 2027-01-31"
    settlement_delay_hours: int = 96
    analysis_freeze_gate_g7: str = "2027-02-08"

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation calendar to serializable dictionary."""
        return {
            "dev_history_end": self.dev_history_end,
            "dev_replay_window": self.dev_replay_window,
            "gate_g4_freeze_date": self.gate_g4_freeze_date,
            "confirmatory_shadow_window": self.confirmatory_shadow_window,
            "settlement_delay_hours": self.settlement_delay_hours,
            "analysis_freeze_gate_g7": self.analysis_freeze_gate_g7,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationCalendarSpec:
        """Deserialize evaluation calendar from dictionary."""
        return cls(
            dev_history_end=str(d.get("dev_history_end", "2026-10-18")),
            dev_replay_window=str(d.get("dev_replay_window", "2026-10-19 to 2026-12-06")),
            gate_g4_freeze_date=str(d.get("gate_g4_freeze_date", "2026-12-04")),
            confirmatory_shadow_window=str(
                d.get("confirmatory_shadow_window", "2026-12-07 to 2027-01-31")
            ),
            settlement_delay_hours=int(d.get("settlement_delay_hours", 96)),
            analysis_freeze_gate_g7=str(d.get("analysis_freeze_gate_g7", "2027-02-08")),
        )


# =============================================================================
# Canonical Hypotheses & Invariants
# =============================================================================
def get_canonical_hypotheses() -> list[HypothesisSpec]:
    """Return canonical pre-registered hypotheses H1-H6 with tier assignments."""
    return [
        HypothesisSpec(
            hypothesis_id="H1",
            statement="M7 beats CAMS-BC at 24 h with N=5 available stations",
            tier=HypothesisTier.TIER_1_INDICATIVE,
            domain="lahore",
            target_population="Lahore City Panel (leave-stations-out, N=5)",
            metric="crpss_vs_m2",
            horizon_h=24,
            n_available=5,
            threshold_value=0.15,
            comparison_operator=">=",
            multiplicity_family="family_1_lahore",
            descriptive_if_days_under=35,
        ),
        HypothesisSpec(
            hypothesis_id="H2",
            statement="M7 beats CAMS-BC at 72 h with N=5 available stations",
            tier=HypothesisTier.TIER_1_INDICATIVE,
            domain="lahore",
            target_population="Lahore City Panel (leave-stations-out, N=5)",
            metric="crpss_vs_m2",
            horizon_h=72,
            n_available=5,
            threshold_value=0.10,
            comparison_operator=">=",
            multiplicity_family="family_1_lahore",
            descriptive_if_days_under=35,
        ),
        HypothesisSpec(
            hypothesis_id="H3",
            statement="M7 beats raw CAMS at every horizon",
            tier=HypothesisTier.TIER_1_INDICATIVE,
            domain="lahore",
            target_population="Lahore City Panel (leave-stations-out, N=5)",
            metric="crpss_vs_m1",
            horizon_h="all",
            n_available=5,
            threshold_value=0.25,
            comparison_operator=">=",
            multiplicity_family="family_1_lahore",
            descriptive_if_days_under=35,
        ),
        HypothesisSpec(
            hypothesis_id="H4",
            statement="Transfer alone (N=0) still adds skill over CAMS-BC; otherwise report negative transfer",
            tier=HypothesisTier.TIER_1_INDICATIVE,
            domain="lahore",
            target_population="Lahore City Panel (pure zero-shot / sensor-free mode, N=0)",
            metric="crpss_vs_m2",
            horizon_h=24,
            n_available=0,
            threshold_value=0.05,
            comparison_operator=">=",
            multiplicity_family="family_1_lahore",
            descriptive_if_days_under=35,
        ),
        HypothesisSpec(
            hypothesis_id="H5",
            statement="Meta-learning adds value over plain fine-tuning at N in {1,2}",
            tier=HypothesisTier.TIER_1_POWERED,
            domain="delhi",
            target_population="Delhi Season T Regulatory Monitors (>= 20 stations)",
            metric="crps_ratio_m7_over_m6",
            horizon_h="all",
            n_available=[1, 2],
            threshold_value=0.97,
            comparison_operator="<=",
            multiplicity_family="family_2_delhi_h5",
            descriptive_if_days_under=0,
        ),
        HypothesisSpec(
            hypothesis_id="H6",
            statement="Calibration: pooled coverage of [q10,q90] within +/-0.05 of 0.80; P(y<=q50) within 0.07 of 0.50",
            tier=HypothesisTier.TIER_2_DESCRIPTIVE,
            domain="lahore",
            target_population="Pooled Spatial-Temporal City Panel",
            metric="coverage_80",
            horizon_h="all",
            n_available=0,
            threshold_value=0.80,
            comparison_operator="+/-0.05",
            multiplicity_family="unadjusted_h6",
            descriptive_if_days_under=0,
        ),
    ]


def get_canonical_multiplicity_protocol() -> dict[str, Any]:
    """Return canonical multiplicity control protocol dictionary."""
    return {
        "family_1_lahore": {
            "description": "Step-down Holm-Bonferroni correction over Lahore confirmatory hypotheses",
            "hypotheses": ["H1", "H2", "H3(24)", "H3(48)", "H3(72)", "H4"],
            "fwer_alpha": 0.05,
            "method": "holm_bonferroni",
        },
        "family_2_delhi_h5": {
            "description": "Independent confirmatory test on Delhi Season T regulatory network",
            "hypotheses": ["H5"],
            "fwer_alpha": 0.05,
            "method": "single_hypothesis_unadjusted",
        },
        "unadjusted_h6": {
            "description": "Descriptive calibration diagnostics without hypothesis rejection penalties",
            "hypotheses": ["H6"],
            "fwer_alpha": None,
            "method": "descriptive_coverage_interval",
        },
    }


def get_canonical_data_integrity_invariants() -> dict[str, Any]:
    """Return canonical data integrity invariants."""
    return {
        "min_valid_hours": 18,
        "settlement_delay_hours": 96,
        "no_retroactive_rescoring": True,
        "point_in_time_enforcement": "available_at <= T",
        "residual_bound_rho_99": 0.36245196908712385,
        "inner_steps_j": 3,
        "grad_clip": 1.0,
        "embargo_hours": 72,
        "cams_lead_offset_h": 12,
    }


# =============================================================================
# PreRegistrationDocument
# =============================================================================
@dataclass
class PreRegistrationDocument:
    """Complete Gate G4 Pre-Registration Document."""

    schema_version: int
    title: str
    created_at_utc: str
    registration_date: str
    git_release_tag: str
    git_commit_sha: str
    config_hash: str
    freeze_register_digest: str
    candidate_calibrator_sha256: str
    shared_physics_base_sha256: str
    headline_model: str
    hypotheses: list[HypothesisSpec]
    statistical_power_note: StatisticalPowerNote
    evaluation_calendar: EvaluationCalendarSpec
    multiplicity_protocol: dict[str, Any]
    data_integrity_invariants: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert pre-registration document to serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "created_at_utc": self.created_at_utc,
            "registration_date": self.registration_date,
            "git_release_tag": self.git_release_tag,
            "git_commit_sha": self.git_commit_sha,
            "config_hash": self.config_hash,
            "freeze_register_digest": self.freeze_register_digest,
            "candidate_calibrator_sha256": self.candidate_calibrator_sha256,
            "shared_physics_base_sha256": self.shared_physics_base_sha256,
            "headline_model": self.headline_model,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "statistical_power_note": self.statistical_power_note.to_dict(),
            "evaluation_calendar": self.evaluation_calendar.to_dict(),
            "multiplicity_protocol": self.multiplicity_protocol,
            "data_integrity_invariants": self.data_integrity_invariants,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize document to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path | str) -> Path:
        """Write pre-registration document to JSON file with LF line endings."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json(indent=2) + "\n"
        with p.open("wb") as f:
            f.write(content.encode("utf-8"))
        return p

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreRegistrationDocument:
        """Deserialize pre-registration document from dictionary."""
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            title=str(d.get("title", "SmogSense Gate G4 Pre-Registration")),
            created_at_utc=str(d["created_at_utc"]),
            registration_date=str(d.get("registration_date", DEFAULT_FREEZE_DATE)),
            git_release_tag=str(d.get("git_release_tag", DEFAULT_GIT_TAG)),
            git_commit_sha=str(d["git_commit_sha"]),
            config_hash=str(d["config_hash"]),
            freeze_register_digest=str(d["freeze_register_digest"]),
            candidate_calibrator_sha256=str(d["candidate_calibrator_sha256"]),
            shared_physics_base_sha256=str(d["shared_physics_base_sha256"]),
            headline_model=str(d.get("headline_model", "m7_hybrid_maml")),
            hypotheses=[HypothesisSpec.from_dict(h) for h in d.get("hypotheses", [])],
            statistical_power_note=StatisticalPowerNote.from_dict(
                d.get("statistical_power_note", {})
            ),
            evaluation_calendar=EvaluationCalendarSpec.from_dict(d.get("evaluation_calendar", {})),
            multiplicity_protocol=dict(d.get("multiplicity_protocol", {})),
            data_integrity_invariants=dict(d.get("data_integrity_invariants", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> PreRegistrationDocument:
        """Deserialize pre-registration document from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path | str) -> PreRegistrationDocument:
        """Load pre-registration document from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Pre-registration document not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_markdown(self) -> str:
        """Generate formal public pre-registration Markdown specification.

        Produces GitHub Flavored Markdown with strict table alignments,
        pure LF line endings, and single trailing newline.
        Contains zero references to git-ignored AI context files.
        """
        lines: list[str] = [
            f"# {self.title}",
            "",
            "## Cryptographic Provenance & Freeze Register Binding",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Document Title | {self.title} |",
            f"| Registration Date | {self.registration_date} (Gate G4 Freeze) |",
            f"| Created At (UTC) | `{self.created_at_utc}` |",
            f"| Git Release Tag | `{self.git_release_tag}` |",
            f"| Git Commit SHA | `{self.git_commit_sha}` |",
            f"| Deterministic Config Hash | `{self.config_hash}` |",
            f"| Freeze Register Digest | `{self.freeze_register_digest}` |",
            f"| Candidate Calibrator (M7) SHA-256 | `{self.candidate_calibrator_sha256}` |",
            f"| Shared Physics Base (M5) SHA-256 | `{self.shared_physics_base_sha256}` |",
            f"| Headline Candidate Model | `{self.headline_model}` |",
            "",
            "---",
            "",
            "## 1. Context & Research Questions",
            "",
            "Ahead of Gate G4 (4 December 2026) and the prospective confirmatory shadow evaluation window "
            "(7 December 2026 - 31 January 2027), SmogSense commits this immutable pre-registration record. "
            "In accordance with modern empirical open science, scores are never recomputed retroactively, "
            "and candidate models are evaluated against pre-registered criteria.",
            "",
            "| RQ | Core Research Question | Primary Rung Comparison | Evaluation Metric |",
            "|---|---|---|---|",
            "| **RQ1** | Does the learned system beat the fair physics baseline (bias-corrected CAMS) for Lahore at 24, 48, and 72 h? | M7 vs M2 (CAMS-BC) | Continuous Ranked Probability Score (CRPS) |",
            "| **RQ2** | How does probabilistic forecast skill degrade as available ground stations $N \\to 0$ (the Sensor-Sparsity Curve)? | M7, M6, M5, M4, M3b | $\\mathrm{CRPS}(N) = c_\\infty + (c_0 - c_\\infty) e^{-N/\\nu}$ |",
            "| **RQ3** | Does meta-learning (Meta-SGD) add value over plain fine-tuning of a Delhi-pretrained base in sparse settings? | M7 vs M6 | CRPS Ratio (M7 / M6) on Delhi Season T |",
            "| **RQ4** | Are 80% central prediction intervals honest and calibrated under prospective operational conditions? | Headline Model | 80% Interval Coverage and Median PIT |",
            "",
            "---",
            "",
            "## 2. Pre-Registered Hypotheses & Scientific Tiers",
            "",
            "Hypotheses are structured into strict scientific tiers to separate confirmatory power from exploratory diagnostics:",
            "- **Tier 1, Powered (H5)**: Delhi Season T contains 49 regulatory monitors and multi-year data, providing adequate sample size ($100.0\\%$ power at $\\alpha = 0.05$) to detect a $3\\%$ skill gain.",
            "- **Tier 1, Indicative (H1-H4)**: Evaluated prospectively on the Lahore City Panel. Given single-season meteorological correlation ($n_{\\text{eff}} \\approx 19\\text{--}28$), tests are reported with confidence intervals and Holm-Bonferroni corrections. Tests degrade to descriptive if $< 35$ confirmatory days are observed.",
            "- **Tier 2, Descriptive (H6)**: Evaluated as descriptive calibration diagnostics without hypothesis rejection penalties.",
            "",
            "| ID | Tier | Statement | Domain | Population | Metric | Horizon | $N$ | Threshold / Operator | Family |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for h in self.hypotheses:
            h_n = str(h.n_available)
            tier_label = h.tier.value.replace("_", " ").title()
            op_str = f"{h.comparison_operator} {h.threshold_value}"
            lines.append(
                f"| `{h.hypothesis_id}` | {tier_label} | {h.statement} | `{h.domain}` | {h.target_population} | `{h.metric}` | `{h.horizon_h}` | `{h_n}` | `{op_str}` | `{h.multiplicity_family}` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. Multiplicity Control Protocol",
                "",
                "To control family-wise error rate (FWER) without compromising statistical discovery:",
                "1. **Family 1 (Lahore Confirmatory Hypotheses)**: Step-down Holm-Bonferroni correction applied over the family $\\{H_1, H_2, H_3(24), H_3(48), H_3(72), H_4\\}$ at $\\alpha = 0.05$.",
                "2. **Family 2 (Delhi Transfer Power Test)**: Independent confirmatory test for $H_5$ evaluated at $\\alpha = 0.05$ one-sided.",
                "3. **Family 3 (Calibration Diagnostics)**: Descriptive evaluation of $H_6$ coverage (target $80\\% \\pm 5\\%$) and median coverage (target $50\\% \\pm 7\\%$) without rejection penalty.",
                "",
                "---",
                "",
                "## 4. Statistical Power & Sample Size Note",
                "",
                "Empirical parameters established during the Task P2-14 pilot study:",
                "",
                "| Parameter | Empirical Value | Specification Note |",
                "|---|---|---|",
                f"| H5 Test Domain | `{self.statistical_power_note.h5_domain}` | Delhi airshed regulatory monitoring network |",
                f"| Evaluated Split Season | `{self.statistical_power_note.season}` | Confirmatory Season T split |",
                f"| Eligible Reference Monitors | **{self.statistical_power_note.eligible_stations} / 49** | Exceeds $\\ge 20$ monitor threshold requirement |",
                f"| Daily Log-Ratio SD ($\\sigma$) | **{self.statistical_power_note.observed_log_ratio_sd}** | Measured variance across pilot evaluation |",
                f"| Effective Sample Size ($n_{{\\text{{eff}}}}$) | **{self.statistical_power_note.effective_sample_size:.2f}** | From 25 pilot days, lag-1 autocorrelation $\\rho_1 = +0.515$ |",
                f"| Analytical Statistical Power | **{self.statistical_power_note.statistical_power:.1f}%** | At $\\alpha = 0.05$ one-sided for target gain $\\delta = 0.03046$ |",
                "",
                f"> [!NOTE]\n> **Lahore Confirmatory Power Note**: {self.statistical_power_note.lahore_field_power_note}",
                "",
                "---",
                "",
                "## 5. Confirmatory Evaluation Calendar & Invariants",
                "",
                "| Milestone | Target Dates | Protocol Description |",
                "|---|---|---|",
                f"| Development History End | `{self.evaluation_calendar.dev_history_end}` | Historical baseline data cut-off date |",
                f"| Dev Replay Window | `{self.evaluation_calendar.dev_replay_window}` | 49-day replay shadow window over point-in-time logged inputs |",
                f"| Gate G4 Freeze Date | `{self.evaluation_calendar.gate_g4_freeze_date}` | Model weights, configurations, and hypotheses permanently frozen |",
                f"| Confirmatory Shadow Window | `{self.evaluation_calendar.confirmatory_shadow_window}` | 56-day prospective confirmatory evaluation period |",
                f"| Settled Truth Archive Delay | `{self.evaluation_calendar.settlement_delay_hours} hours` | Settlement delay for OpenAQ AWS historical archive validation |",
                f"| Analysis Freeze (Gate G7) | `{self.evaluation_calendar.analysis_freeze_gate_g7}` | Final scoring freeze and thesis analysis compilation |",
                "",
                "### Invariant Data Integrity Rules",
                "- **18/24h Completeness Rule**: Observations require $\\ge 18/24$ valid, non-imputed hourly readings to form a valid block mean. Imputed values never count towards completeness.",
                "- **Point-in-Time Correctness**: All features at forecast issuance $T$ use only observations satisfying $\\text{available\\_at} \\le T$.",
                "- **Zero Retroactive Rescoring**: Prospective forecasts logged to `.state/forecast_log/` are permanently immutable.",
                "- **Cryptographic Binding**: Candidate residual calibrator must declare matching `base_sha256` matching the frozen physics base.",
                "",
                "---",
                "",
                "## 6. Negative Results & Occam's Razor Policy",
                "",
                "SmogSense adheres to a strict hierarchical degradation and negative-results reporting policy:",
                "1. If Model M7 (Meta-SGD) does not demonstrate statistically significant gain over Model M6 (fine-tuning) on Delhi Season T ($H_5$ not met), the simpler fine-tuning architecture (Model M6) is favored.",
                "2. If neither calibrator demonstrates superior skill over the frozen physics base (Model M5) under the deployment acceptance gate, Model M5 serves as the headline model, demonstrating the bounds of pure zero-shot physics transfer.",
                "3. If transfer skill falls below bias-corrected CAMS ($H_4$ not met), negative transfer is formally documented.",
                "4. All negative, null, or confirmatory results are reported with identical scientific prominence in the final capstone thesis.",
                "",
            ]
        )
        return "\n".join(lines)


# =============================================================================
# Creation & Verification Engines
# =============================================================================
def create_pre_registration_document(
    configs_dir: Path | str = "configs",
    freeze_register_path: Path | str = DEFAULT_PRE_REGISTRATION_STATE,
    output_state_path: Path | str = DEFAULT_PRE_REGISTRATION_STATE,
    output_doc_path: Path | str = DEFAULT_PRE_REGISTRATION_DOC,
    git_tag: str = DEFAULT_GIT_TAG,
    git_commit_sha: str | None = None,
    write_files: bool = True,
) -> PreRegistrationDocument:
    """Create and lock the Gate G4 Pre-Registration specification."""
    settings = Settings.load(configs_dir)
    config_hash = settings.hash()
    git_sha = resolve_git_commit_sha(git_commit_sha)

    # 1. Resolve Freeze Register details
    f_path = Path(".state/freeze_register.json")
    if f_path.is_file():
        freeze_reg = FreezeRegister.load(f_path)
        freeze_digest = compute_file_sha256(f_path)
        base_sha = freeze_reg.base_sha256
        bundle_sha = freeze_reg.bundle_sha256
        headline_mod = freeze_reg.headline_model
    else:
        # Fallback digest resolution if freeze register not yet saved to disk
        bundle_sha, base_sha = resolve_model_digests("data/processed/models")
        freeze_digest = "0" * 64
        headline_mod = "m7_hybrid_maml"

    # 2. Build Sub-Components
    hypotheses = get_canonical_hypotheses()
    power_note = StatisticalPowerNote()
    calendar_spec = EvaluationCalendarSpec()
    multiplicity_protocol = get_canonical_multiplicity_protocol()
    data_invariants = get_canonical_data_integrity_invariants()

    doc = PreRegistrationDocument(
        schema_version=1,
        title="SmogSense Gate G4 Confirmatory Pre-Registration: Probabilistic 72-Hour PM2.5 Transfer Forecasting",
        created_at_utc=datetime.now(UTC).isoformat(),
        registration_date=DEFAULT_FREEZE_DATE,
        git_release_tag=git_tag,
        git_commit_sha=git_sha,
        config_hash=config_hash,
        freeze_register_digest=freeze_digest,
        candidate_calibrator_sha256=bundle_sha,
        shared_physics_base_sha256=base_sha,
        headline_model=headline_mod,
        hypotheses=hypotheses,
        statistical_power_note=power_note,
        evaluation_calendar=calendar_spec,
        multiplicity_protocol=multiplicity_protocol,
        data_integrity_invariants=data_invariants,
    )

    if write_files:
        doc.save(output_state_path)
        # Write markdown documentation with LF line endings
        md_content = doc.to_markdown()
        out_md = Path(output_doc_path)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with out_md.open("wb") as f:
            f.write(md_content.encode("utf-8"))

    return doc


def verify_pre_registration(
    pre_registration_path: Path | str = DEFAULT_PRE_REGISTRATION_STATE,
    freeze_register_path: Path | str = ".state/freeze_register.json",
    configs_dir: Path | str = "configs",
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Audit active state against the Gate G4 pre-registration specification."""
    p_path = Path(pre_registration_path)
    if str(pre_registration_path) == str(DEFAULT_PRE_REGISTRATION_STATE) and not p_path.is_file() and (Path(configs_dir) / "pre_registration.json").is_file():
        p_path = Path(configs_dir) / "pre_registration.json"
    if not p_path.is_file():
        msg = f"Pre-registration document not found: {p_path}"
        if strict:
            raise PreRegistrationVerificationError(msg, exit_code=30)
        return False, [msg]

    doc = PreRegistrationDocument.load(p_path)
    drift_messages: list[str] = []

    # 1. Configuration Drift
    settings = Settings.load(configs_dir)
    current_config_hash = settings.hash()
    if current_config_hash != doc.config_hash:
        drift_messages.append(
            f"Configuration hash drift detected: pre-registered {doc.config_hash[:16]}... "
            f"!= active {current_config_hash[:16]}..."
        )

    # 2. Freeze Register Digest
    f_path = Path(freeze_register_path)
    if str(freeze_register_path) == ".state/freeze_register.json" and not f_path.is_file() and (Path(configs_dir) / "freeze_register.json").is_file():
        f_path = Path(configs_dir) / "freeze_register.json"
    if f_path.is_file():
        current_freeze_digest = compute_file_sha256(f_path)
        if (
            doc.freeze_register_digest != "0" * 64
            and current_freeze_digest != doc.freeze_register_digest
        ):
            drift_messages.append(
                f"Freeze register digest mismatch: pre-registered {doc.freeze_register_digest[:16]}... "
                f"!= active {current_freeze_digest[:16]}..."
            )

        freeze_reg = FreezeRegister.load(f_path)
        if freeze_reg.base_sha256 != doc.shared_physics_base_sha256:
            drift_messages.append(
                f"Base SHA-256 mismatch: pre-registered {doc.shared_physics_base_sha256[:16]}... "
                f"!= freeze register {freeze_reg.base_sha256[:16]}..."
            )
        if freeze_reg.bundle_sha256 != doc.candidate_calibrator_sha256:
            drift_messages.append(
                f"Calibrator bundle SHA-256 mismatch: pre-registered {doc.candidate_calibrator_sha256[:16]}... "
                f"!= freeze register {freeze_reg.bundle_sha256[:16]}..."
            )

    # 3. Hypotheses Threshold Drift
    eval_cfg = settings.evaluation if hasattr(settings, "evaluation") else {}
    hypotheses_cfg = eval_cfg.get("hypotheses", {})
    for h in doc.hypotheses:
        cfg_h = hypotheses_cfg.get(h.hypothesis_id)
        if not cfg_h:
            drift_messages.append(
                f"Pre-registered hypothesis {h.hypothesis_id} missing from configs/evaluation.yaml"
            )
            continue
        # Compare threshold values
        thresh_val: float | None = None
        if isinstance(h.threshold_value, (int, float)):
            thresh_val = float(h.threshold_value)
        elif isinstance(h.threshold_value, str):
            try:
                thresh_val = float(h.threshold_value)
            except ValueError:
                thresh_val = None

        if thresh_val is not None:
            if "min_value" in cfg_h and float(cfg_h["min_value"]) != thresh_val:
                drift_messages.append(
                    f"Hypothesis {h.hypothesis_id} threshold altered: pre-registered {h.threshold_value} "
                    f"!= active config {cfg_h['min_value']}"
                )
            elif "max_value" in cfg_h and float(cfg_h["max_value"]) != thresh_val:
                drift_messages.append(
                    f"Hypothesis {h.hypothesis_id} threshold altered: pre-registered {h.threshold_value} "
                    f"!= active config {cfg_h['max_value']}"
                )

    is_valid = len(drift_messages) == 0
    if not is_valid and strict:
        raise PreRegistrationVerificationError(
            f"Pre-registration verification failed with {len(drift_messages)} issue(s):\n"
            + "\n".join(f"  - {m}" for m in drift_messages),
            exit_code=30,
        )

    return is_valid, drift_messages
