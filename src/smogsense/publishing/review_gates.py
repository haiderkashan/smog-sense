"""smogsense.publishing.review_gates — Human review gates and lay comprehension verification engine.

Verifies independent expert sign-offs across Gate G-SCI (Scientific & Atmospheric),
Gate G-LANG (Bilingual Urdu Editorial), Gate G-HLTH (Public Health Medical), and
Gate G-UX (Lay User Comprehension Protocol in Lahore).

Specification: docs/dissemination-and-ui.md §11, docs/PRD.md (FR-22-FR-37), ADR-004
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from smogsense.errors import SmogSenseError

logger = logging.getLogger(__name__)


class GateID(StrEnum):
    """Identifier for the 4 independent human review gates."""

    G_SCI = "g_sci"
    G_LANG = "g_lang"
    G_HLTH = "g_hlth"
    G_UX = "g_ux"


class ReviewGateError(SmogSenseError, ValueError):
    """Raised when one or more human review gates fail or are missing."""

    exit_code: int = 30


@dataclass(frozen=True)
class GateSignoff:
    """Formal sign-off dossier for an expert review gate."""

    gate_id: GateID
    reviewer_role: str
    date: str
    scope: str
    passed: bool
    observations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert sign-off record to dictionary."""
        data = asdict(self)
        data["gate_id"] = self.gate_id.value
        return data


@dataclass(frozen=True)
class ComprehensionUserSession:
    """Think-aloud interview session record for a single lay participant."""

    participant_id: str
    persona: str
    q1_response: str
    q1_protective: bool
    q2_response: int | str
    q2_correct: bool
    q3_response: str
    q3_correct: bool
    q4_response: str
    q4_correct: bool
    q5_response: str
    q5_correct: bool
    q6_feedback: str
    trust_rating: int
    clarity_rating: int

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ComprehensionSummary:
    """Statistical summary of the lay user comprehension protocol."""

    total_participants: int
    percent_protective_q1: float
    percent_correct_q2_q5: float
    q2_percent_correct: float
    q3_percent_correct: float
    q4_percent_correct: float
    q5_percent_correct: float
    mean_trust_score: float
    mean_clarity_score: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary."""
        return asdict(self)


@dataclass
class GateAuditReport:
    """Comprehensive audit report across all four review gates."""

    all_passed: bool
    gates: dict[str, GateSignoff]
    comprehension_summary: ComprehensionSummary | None = None
    missing_gates: list[str] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert audit report to dictionary."""
        return {
            "all_passed": self.all_passed,
            "gates": {k: v.to_dict() for k, v in self.gates.items()},
            "comprehension_summary": self.comprehension_summary.to_dict()
            if self.comprehension_summary
            else None,
            "missing_gates": self.missing_gates,
            "failed_gates": self.failed_gates,
            "diagnostics": self.diagnostics,
        }


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content if present."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_str = parts[1]
            body = parts[2]
            try:
                data = yaml.safe_load(fm_str)
                if isinstance(data, dict):
                    return data, body.strip()
            except Exception as e:
                logger.warning("Failed to parse YAML frontmatter: %s", e)
    return {}, content.strip()


def load_review_gates(
    reviews_dir: Path | str = "docs/reviews",
) -> tuple[dict[GateID, GateSignoff], list[ComprehensionUserSession]]:
    """Discover and parse review gate dossiers and comprehension sessions.

    Scans the specified directory for markdown dossiers with YAML frontmatter
    and JSON session archives.
    """
    r_dir = Path(reviews_dir)
    gates: dict[GateID, GateSignoff] = {}
    sessions: list[ComprehensionUserSession] = []

    if not r_dir.is_dir():
        logger.warning("Reviews directory not found at: %s", r_dir)
        return gates, sessions

    # 1. Parse markdown files with YAML frontmatter
    for md_file in sorted(r_dir.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            meta, _ = parse_frontmatter(content)
            if not meta:
                continue

            gate_raw = meta.get("gate_id")
            if not gate_raw:
                continue

            try:
                gid = GateID(str(gate_raw).lower())
            except ValueError:
                logger.warning("Unrecognized gate_id '%s' in %s", gate_raw, md_file.name)
                continue

            signoff = GateSignoff(
                gate_id=gid,
                reviewer_role=str(meta.get("reviewer_role", "Unknown")),
                date=str(meta.get("date", "")),
                scope=str(meta.get("scope", "")),
                passed=bool(meta.get("passed", False)),
                observations=list(meta.get("observations", [])),
                recommendations=list(meta.get("recommendations", [])),
            )
            gates[gid] = signoff
        except Exception as e:
            logger.warning("Error reading review file %s: %s", md_file, e)

    # 2. Parse comprehension sessions JSON
    sessions_file = r_dir / "comprehension_sessions.json"
    if sessions_file.is_file():
        try:
            with sessions_file.open("r", encoding="utf-8") as f:
                raw_sessions = json.load(f)
            if isinstance(raw_sessions, list):
                sessions.extend(
                    ComprehensionUserSession(
                        participant_id=str(s.get("participant_id", "")),
                        persona=str(s.get("persona", "")),
                        q1_response=str(s.get("q1_response", "")),
                        q1_protective=bool(s.get("q1_protective", False)),
                        q2_response=s.get("q2_response", 0),
                        q2_correct=bool(s.get("q2_correct", False)),
                        q3_response=str(s.get("q3_response", "")),
                        q3_correct=bool(s.get("q3_correct", False)),
                        q4_response=str(s.get("q4_response", "")),
                        q4_correct=bool(s.get("q4_correct", False)),
                        q5_response=str(s.get("q5_response", "")),
                        q5_correct=bool(s.get("q5_correct", False)),
                        q6_feedback=str(s.get("q6_feedback", "")),
                        trust_rating=int(s.get("trust_rating", 0)),
                        clarity_rating=int(s.get("clarity_rating", 0)),
                    )
                    for s in raw_sessions
                )
        except Exception as e:
            logger.warning("Error reading comprehension sessions from %s: %s", sessions_file, e)

    return gates, sessions


def evaluate_comprehension_sessions(
    sessions: list[ComprehensionUserSession],
) -> ComprehensionSummary:
    """Compute statistical performance across comprehension test interviews.

    Success Criteria per docs/dissemination-and-ui.md §11:
    - >= 80% choose protective action on Question 1.
    - >= 80% correct on Questions 2-5.
    - Minimum 5 lay participants.
    """
    total = len(sessions)
    if total == 0:
        return ComprehensionSummary(
            total_participants=0,
            percent_protective_q1=0.0,
            percent_correct_q2_q5=0.0,
            q2_percent_correct=0.0,
            q3_percent_correct=0.0,
            q4_percent_correct=0.0,
            q5_percent_correct=0.0,
            mean_trust_score=0.0,
            mean_clarity_score=0.0,
            passed=False,
        )

    prot_q1 = sum(1 for s in sessions if s.q1_protective) / total * 100.0
    c_q2 = sum(1 for s in sessions if s.q2_correct) / total * 100.0
    c_q3 = sum(1 for s in sessions if s.q3_correct) / total * 100.0
    c_q4 = sum(1 for s in sessions if s.q4_correct) / total * 100.0
    c_q5 = sum(1 for s in sessions if s.q5_correct) / total * 100.0

    mean_q2_q5 = (c_q2 + c_q3 + c_q4 + c_q5) / 4.0
    mean_trust = sum(s.trust_rating for s in sessions) / float(total)
    mean_clarity = sum(s.clarity_rating for s in sessions) / float(total)

    passed = (total >= 5) and (prot_q1 >= 80.0) and (mean_q2_q5 >= 80.0)

    return ComprehensionSummary(
        total_participants=total,
        percent_protective_q1=round(prot_q1, 1),
        percent_correct_q2_q5=round(mean_q2_q5, 1),
        q2_percent_correct=round(c_q2, 1),
        q3_percent_correct=round(c_q3, 1),
        q4_percent_correct=round(c_q4, 1),
        q5_percent_correct=round(c_q5, 1),
        mean_trust_score=round(mean_trust, 2),
        mean_clarity_score=round(mean_clarity, 2),
        passed=passed,
    )


def audit_review_gates(
    reviews_dir: Path | str = "docs/reviews",
) -> GateAuditReport:
    """Execute complete verification audit across all four expert review gates."""
    gates, sessions = load_review_gates(reviews_dir)
    comp_summary: ComprehensionSummary | None = None

    if sessions:
        comp_summary = evaluate_comprehension_sessions(sessions)

    required_gates = [GateID.G_SCI, GateID.G_LANG, GateID.G_HLTH, GateID.G_UX]
    missing_gates: list[str] = []
    failed_gates: list[str] = []
    diagnostics: list[str] = []

    role_patterns = {
        GateID.G_SCI: ["scientist", "environmental"],
        GateID.G_LANG: ["urdu", "editor", "linguist"],
        GateID.G_HLTH: ["physician", "doctor", "health"],
        GateID.G_UX: ["journalist", "evaluator", "ux", "researcher"],
    }

    for req_gid in required_gates:
        if req_gid not in gates:
            missing_gates.append(req_gid.value)
            diagnostics.append(f"Missing sign-off dossier for gate: {req_gid.value.upper()}")
            continue

        signoff = gates[req_gid]

        # Check approval
        if not signoff.passed:
            failed_gates.append(req_gid.value)
            diagnostics.append(f"Gate {req_gid.value.upper()} sign-off is marked NOT approved.")

        # Check role qualification
        role_lower = signoff.reviewer_role.lower()
        patterns = role_patterns.get(req_gid, [])
        if not any(p in role_lower for p in patterns):
            diagnostics.append(
                f"Gate {req_gid.value.upper()} reviewer role '{signoff.reviewer_role}' "
                f"does not match expected role qualifications ({', '.join(patterns)})."
            )

    # For G-UX: if sessions were evaluated, verify statistical success criteria
    if GateID.G_UX in gates and comp_summary is not None and not comp_summary.passed:
        if GateID.G_UX.value not in failed_gates:
            failed_gates.append(GateID.G_UX.value)
        diagnostics.append(
            f"Comprehension test did not achieve target thresholds: "
            f"Q1 Protective={comp_summary.percent_protective_q1}% (target >= 80%), "
            f"Q2-Q5 Mean={comp_summary.percent_correct_q2_q5}% (target >= 80%), "
            f"Participants={comp_summary.total_participants} (target >= 5)."
        )

    all_passed = (len(missing_gates) == 0) and (len(failed_gates) == 0)

    return GateAuditReport(
        all_passed=all_passed,
        gates={k.value: v for k, v in gates.items()},
        comprehension_summary=comp_summary,
        missing_gates=missing_gates,
        failed_gates=failed_gates,
        diagnostics=diagnostics,
    )


def format_review_summary_table(report: GateAuditReport) -> str:
    """Format structured review gate results into a terminal table."""
    lines: list[str] = []
    w = 88
    lines.append("=" * w)
    lines.append(f"{'SmogSense Pre-Launch Human Review Gates Audit':^{w}}")
    lines.append("=" * w)
    lines.append(
        f" {'Gate':<8} {'Status':<8} {'Reviewer Role':<28} {'Date':<12} {'Scope / Criteria':<28}"
    )
    lines.append("-" * w)

    gate_names = {
        "g_sci": "G-SCI",
        "g_lang": "G-LANG",
        "g_hlth": "G-HLTH",
        "g_ux": "G-UX",
    }

    for gid_key, display_name in gate_names.items():
        if gid_key in report.gates:
            g = report.gates[gid_key]
            status_str = "PASS" if g.passed and gid_key not in report.failed_gates else "FAIL"
            role_trunc = g.reviewer_role[:26]
            scope_trunc = g.scope[:26] + "..." if len(g.scope) > 26 else g.scope
            lines.append(
                f" {display_name:<8} {status_str:<8} {role_trunc:<28} {g.date:<12} {scope_trunc:<28}"
            )
        else:
            lines.append(
                f" {display_name:<8} {'MISSING':<8} {'[No Signoff Recorded]':<28} {'—':<12} {'—':<28}"
            )

    lines.append("-" * w)

    if report.comprehension_summary:
        s = report.comprehension_summary
        lines.append(
            f" Comprehension Protocol: {s.total_participants} participants | Q1 Protective: {s.percent_protective_q1}% | Q2-Q5 Mean: {s.percent_correct_q2_q5}%"
        )
        p_str = "PASS" if s.passed else "FAIL"
        lines.append(
            f" Mean Trust: {s.mean_trust_score:.2f}/5.0 | Mean Clarity: {s.mean_clarity_score:.2f}/5.0 | Outcome: {p_str}"
        )
        lines.append("-" * w)

    if report.all_passed:
        lines.append(f"{'OVERALL STATUS: ALL GATES APPROVED (READY FOR LAUNCH)':^{w}}")
    else:
        lines.append(f"{'OVERALL STATUS: GATES AUDIT FAILED (ACTION REQUIRED)':^{w}}")
        if report.diagnostics:
            lines.append("-" * w)
            lines.append(" Diagnostics:")
            lines.extend(f"  • {d}" for d in report.diagnostics)

    lines.append("=" * w)
    return "\n".join(lines)
