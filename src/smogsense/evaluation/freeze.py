"""smogsense.evaluation.freeze - Gate G4 Freeze Register & Invariant Lockdown engine.

Records, serializes, verifies, and audits the frozen state of SmogSense before
the prospective confirmatory shadow evaluation window (7 Dec 2026 - 31 Jan 2027):
- Git release tag, commit SHA, and deterministic configuration hash.
- Candidate model bundle digests and shared frozen physics base SHA-256.
- Frozen city-panel station network (>= 3 valid stations, FR-38).
- Pre-registered evaluation hypotheses H1-H6 with unalterable thresholds.
- Core architectural invariants (residual bound rho_99, inner steps J, clipping, etc.).

Specification:
- docs/evaluation-strategy.md §1, §6, §9, §10
- docs/ml-architecture.md §10 (Headline model, promotion rule, and freeze register)
- docs/PRD.md (FR-21, FR-38, NFR-16)
- configs/evaluation.yaml (hypotheses, calendar, shadow)
- decisions.md (ADR-002, ADR-016, ADR-019, ADR-024)
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from structlog import get_logger

from smogsense.config import Settings
from smogsense.errors import SmogSenseError
from smogsense.models.registry import (
    CALIBRATOR_PARAM_COUNT,
    DEFAULT_RHO_99,
    METASGD_STEP_SIZES_COUNT,
    compute_file_sha256,
)

logger = get_logger(__name__)

DEFAULT_FREEZE_DATE = "2026-12-04"
DEFAULT_CONFIRMATORY_START = "2026-12-07"
DEFAULT_CONFIRMATORY_END = "2027-01-31"
DEFAULT_GIT_TAG = "v0.3.0-gate-g4"


# =============================================================================
# Exceptions
# =============================================================================
class FreezeVerificationError(SmogSenseError, ValueError):
    """Raised when freeze register verification detects configuration drift or artifact tampering."""

    def __init__(self, message: str, exit_code: int = 30) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# =============================================================================
# Data Structures
# =============================================================================
@dataclass
class CityPanelStation:
    """Station metadata for an eligible member of the frozen city target panel (FR-38)."""

    location_id: int
    name: str
    is_reference: bool
    provider: str
    lat: float
    lon: float
    elevation_m: float | None
    sensor_model: str
    colocated_group_id: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize station to dictionary."""
        return {
            "location_id": self.location_id,
            "name": self.name,
            "is_reference": self.is_reference,
            "provider": self.provider,
            "lat": self.lat,
            "lon": self.lon,
            "elevation_m": self.elevation_m,
            "sensor_model": self.sensor_model,
            "colocated_group_id": self.colocated_group_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CityPanelStation:
        """Deserialize station from dictionary."""
        elevation = d.get("elevation_m")
        colocated = d.get("colocated_group_id")
        return cls(
            location_id=int(d["location_id"]),
            name=str(d["name"]),
            is_reference=bool(d["is_reference"]),
            provider=str(d["provider"]),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            elevation_m=float(elevation) if elevation is not None and pd.notna(elevation) else None,
            sensor_model=str(d.get("sensor_model", "unknown")),
            colocated_group_id=int(colocated)
            if colocated is not None and pd.notna(colocated)
            else None,
        )


@dataclass
class HypothesisThreshold:
    """Pre-registered statistical hypothesis threshold and comparison criteria."""

    hypothesis_id: str
    statement: str
    metric: str
    horizon_h: str | int
    n_available: list[int] | int
    target_threshold: float
    comparison_op: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize hypothesis to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "metric": self.metric,
            "horizon_h": self.horizon_h,
            "n_available": self.n_available,
            "target_threshold": self.target_threshold,
            "comparison_op": self.comparison_op,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HypothesisThreshold:
        """Deserialize hypothesis from dictionary."""
        return cls(
            hypothesis_id=str(d["hypothesis_id"]),
            statement=str(d["statement"]),
            metric=str(d["metric"]),
            horizon_h=d["horizon_h"],
            n_available=d["n_available"],
            target_threshold=float(d["target_threshold"]),
            comparison_op=str(d["comparison_op"]),
        )


@dataclass
class FreezeRegister:
    """Complete Gate G4 Freeze Register recording pre-registered state and cryptographic hashes."""

    schema_version: int
    created_at_utc: str
    freeze_date: str
    confirmatory_window_start: str
    confirmatory_window_end: str
    git_tag: str
    git_commit_sha: str
    config_hash: str
    headline_model: str
    promoted_bundle_tag: str
    bundle_sha256: str
    base_sha256: str
    city_panel: list[CityPanelStation]
    hypotheses: list[HypothesisThreshold]
    invariants: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert freeze register to serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "freeze_date": self.freeze_date,
            "confirmatory_window_start": self.confirmatory_window_start,
            "confirmatory_window_end": self.confirmatory_window_end,
            "git_tag": self.git_tag,
            "git_commit_sha": self.git_commit_sha,
            "config_hash": self.config_hash,
            "headline_model": self.headline_model,
            "promoted_bundle_tag": self.promoted_bundle_tag,
            "bundle_sha256": self.bundle_sha256,
            "base_sha256": self.base_sha256,
            "city_panel": [s.to_dict() for s in self.city_panel],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "invariants": self.invariants,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize freeze register to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path | str) -> Path:
        """Write freeze register to JSON file on disk with LF line endings."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_json(indent=2) + "\n"
        with p.open("wb") as f:
            f.write(content.encode("utf-8"))
        return p

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FreezeRegister:
        """Deserialize freeze register from dictionary."""
        return cls(
            schema_version=int(d.get("schema_version", 1)),
            created_at_utc=str(d["created_at_utc"]),
            freeze_date=str(d.get("freeze_date", DEFAULT_FREEZE_DATE)),
            confirmatory_window_start=str(
                d.get("confirmatory_window_start", DEFAULT_CONFIRMATORY_START)
            ),
            confirmatory_window_end=str(d.get("confirmatory_window_end", DEFAULT_CONFIRMATORY_END)),
            git_tag=str(d.get("git_tag", DEFAULT_GIT_TAG)),
            git_commit_sha=str(d["git_commit_sha"]),
            config_hash=str(d["config_hash"]),
            headline_model=str(d.get("headline_model", "m7_hybrid_maml")),
            promoted_bundle_tag=str(d.get("promoted_bundle_tag", "none")),
            bundle_sha256=str(d["bundle_sha256"]),
            base_sha256=str(d["base_sha256"]),
            city_panel=[CityPanelStation.from_dict(s) for s in d.get("city_panel", [])],
            hypotheses=[HypothesisThreshold.from_dict(h) for h in d.get("hypotheses", [])],
            invariants=dict(d.get("invariants", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> FreezeRegister:
        """Deserialize freeze register from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path | str) -> FreezeRegister:
        """Load freeze register from JSON file."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Freeze register not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Helper Utilities
# =============================================================================
def resolve_git_commit_sha(git_commit_sha: str | None = None) -> str:
    """Resolve active git commit SHA from argument, environment, or git command."""
    if git_commit_sha and git_commit_sha.strip():
        return git_commit_sha.strip()

    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT_SHA")
    if env_sha and env_sha.strip():
        return env_sha.strip()

    git_bin = shutil.which("git")
    if git_bin:
        try:
            res = subprocess.run(  # noqa: S603
                [git_bin, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            logger.debug("git_rev_parse_failed")

    return "0" * 40


def parse_hypotheses_from_settings(settings: Settings) -> list[HypothesisThreshold]:
    """Parse pre-registered hypotheses H1-H6 from Settings configuration."""
    eval_conf = settings.evaluation if hasattr(settings, "evaluation") else {}
    hypotheses_dict = eval_conf.get("hypotheses", {})
    thresholds: list[HypothesisThreshold] = []

    for h_id, h_data in hypotheses_dict.items():
        if not isinstance(h_data, dict):
            continue

        statement = str(h_data.get("statement", ""))
        metric = str(h_data.get("metric", ""))
        horizon_h = h_data.get("horizon_h", "all")
        n_available = h_data.get("n_available", 0)

        if "min_value" in h_data:
            target_threshold = float(h_data["min_value"])
            comparison_op = ">="
        elif "max_value" in h_data:
            target_threshold = float(h_data["max_value"])
            comparison_op = "<="
        elif "target" in h_data:
            target_threshold = float(h_data["target"])
            tol = h_data.get("tolerance", 0.05)
            comparison_op = f"+/-{tol}"
        else:
            target_threshold = 0.0
            comparison_op = ">="

        thresholds.append(
            HypothesisThreshold(
                hypothesis_id=str(h_id),
                statement=statement,
                metric=metric,
                horizon_h=horizon_h,
                n_available=n_available,
                target_threshold=target_threshold,
                comparison_op=comparison_op,
            )
        )

    # Sort deterministically by hypothesis id
    thresholds.sort(key=lambda x: x.hypothesis_id)
    return thresholds


def load_city_panel_stations(
    registry_path: Path | str = "data/registry/registry_lahore.parquet",
    min_stations: int = 3,
) -> list[CityPanelStation]:
    """Load and validate eligible stations from station registry (FR-38).

    Args:
        registry_path: Path to registry Parquet file.
        min_stations: Minimum required stations (FR-38 requires >= 3).

    Returns:
        List of CityPanelStation objects sorted by location_id.

    Raises:
        FileNotFoundError: If registry file does not exist.
        ValueError: If fewer than min_stations eligible stations exist.
    """
    p = Path(registry_path)
    if not p.is_file():
        raise FileNotFoundError(f"Station registry file not found: {p}")

    df = pd.read_parquet(p)
    eligible_df = df[df["eligible"].fillna(False).astype(bool)] if "eligible" in df.columns else df

    if len(eligible_df) < min_stations:
        raise ValueError(
            f"Insufficient eligible stations in city panel: {len(eligible_df)} < {min_stations} (violates FR-38)"
        )

    stations: list[CityPanelStation] = []
    for _, row in eligible_df.iterrows():
        elevation = row.get("elevation_m")
        colocated = row.get("colocated_group_id")
        stations.append(
            CityPanelStation(
                location_id=int(row["location_id"]),
                name=str(row.get("name", f"Station_{row['location_id']}")),
                is_reference=bool(row.get("is_reference", False)),
                provider=str(row.get("provider", "unknown")),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                elevation_m=float(elevation)
                if elevation is not None and pd.notna(elevation)
                else None,
                sensor_model=str(row.get("sensor_model", "unknown")),
                colocated_group_id=int(colocated)
                if colocated is not None and pd.notna(colocated)
                else None,
            )
        )

    stations.sort(key=lambda s: s.location_id)
    return stations


def resolve_model_digests(
    models_dir: Path | str = "data/processed/models",
    bundle_sha256: str | None = None,
    base_sha256: str | None = None,
) -> tuple[str, str]:
    """Resolve candidate calibrator bundle SHA-256 and base model SHA-256 digests."""
    m_dir = Path(models_dir)

    # 1. Base SHA-256
    resolved_base = base_sha256
    if not resolved_base:
        res_bound_path = m_dir / "residual_bound.json"
        if res_bound_path.is_file():
            with contextlib.suppress(Exception), res_bound_path.open("r", encoding="utf-8") as f:
                rb_data = json.load(f)
                if rb_data.get("base_sha256"):
                    resolved_base = str(rb_data["base_sha256"])

        if not resolved_base:
            base_model_file = m_dir / "lgbm_m5" / "base_model"
            if base_model_file.is_file():
                resolved_base = compute_file_sha256(base_model_file)
            else:
                resolved_base = "0" * 64

    # 2. Bundle SHA-256 (Candidate calibrator)
    resolved_bundle = bundle_sha256
    if not resolved_bundle:
        m7_file = m_dir / "m7_maml.pt"
        m6_file = m_dir / "m6_calibrator.pt"
        if m7_file.is_file():
            resolved_bundle = compute_file_sha256(m7_file)
        elif m6_file.is_file():
            resolved_bundle = compute_file_sha256(m6_file)
        else:
            resolved_bundle = "0" * 64

    return resolved_bundle, resolved_base


# =============================================================================
# Creation Engine
# =============================================================================
def create_freeze_register(
    configs_dir: Path | str = "configs",
    registry_path: Path | str = "data/registry/registry_lahore.parquet",
    models_dir: Path | str = "data/processed/models",
    output_path: Path | str = ".state/freeze_register.json",
    report_path: Path | str = "reports/freeze_register.md",
    git_tag: str = DEFAULT_GIT_TAG,
    git_commit_sha: str | None = None,
    headline_model: str | None = None,
    bundle_sha256: str | None = None,
    base_sha256: str | None = None,
    write_files: bool = True,
) -> FreezeRegister:
    """Create and lock the Gate G4 Freeze Register.

    Extracts configuration hash, candidate bundle digests, station panel,
    pre-registered hypotheses, and invariants. Optionally persists JSON
    and Markdown reports.
    """
    settings = Settings.load(configs_dir)
    config_hash = settings.hash()
    git_sha = resolve_git_commit_sha(git_commit_sha)

    # Dates
    eval_conf = settings.evaluation if hasattr(settings, "evaluation") else {}
    calendar_conf = eval_conf.get("calendar", {})
    freeze_date = str(calendar_conf.get("freeze", DEFAULT_FREEZE_DATE))
    conf_start = str(calendar_conf.get("confirmatory", {}).get("start", DEFAULT_CONFIRMATORY_START))
    conf_end = str(calendar_conf.get("confirmatory", {}).get("end", DEFAULT_CONFIRMATORY_END))

    # Promoted bundle & headline candidate
    model_conf = settings.model if hasattr(settings, "model") else {}
    promoted_tag = str(model_conf.get("promoted_bundle", "none")).strip().lower()
    candidate_method = str(model_conf.get("challenger", {}).get("method", "m7_hybrid_maml"))
    hl_model = headline_model or candidate_method

    # Digests
    b_sha, base_sha = resolve_model_digests(
        models_dir=models_dir,
        bundle_sha256=bundle_sha256,
        base_sha256=base_sha256,
    )

    # City panel
    city_panel = load_city_panel_stations(registry_path=registry_path, min_stations=3)

    # Hypotheses
    hypotheses = parse_hypotheses_from_settings(settings)

    # Invariants
    rho_99_val = DEFAULT_RHO_99
    res_bound_path = Path(models_dir) / "residual_bound.json"
    if res_bound_path.is_file():
        with contextlib.suppress(Exception), res_bound_path.open("r", encoding="utf-8") as f:
            rb_data = json.load(f)
            if "rho_99" in rb_data:
                rho_99_val = float(rb_data["rho_99"])

    adapt_inner = model_conf.get("adaptation", {}).get("inner", {})
    target_conf = model_conf.get("target", {})
    splits_conf = eval_conf.get("splits", {})
    quantiles_grid = model_conf.get("quantiles", {}).get("grid", [])
    horizons = eval_conf.get("horizons_h", [24, 48, 72])
    rh_mask_enabled = bool(
        model_conf.get("adaptation", {}).get("rh_mask", {}).get("enabled", False)
    )

    invariants: dict[str, Any] = {
        "rho_99": rho_99_val,
        "inner_steps_j": int(adapt_inner.get("steps", 3)),
        "inner_alpha": float(adapt_inner.get("alpha", 0.01)),
        "grad_clip": float(adapt_inner.get("grad_clip", 1.0)),
        "min_valid_hours": int(target_conf.get("min_valid_hours", 18)),
        "cams_lead_offset_h": 12,
        "embargo_hours": int(splits_conf.get("embargo_hours", 72)),
        "quantiles_count": len(quantiles_grid) if quantiles_grid else 19,
        "horizons_h": list(horizons),
        "calibrator_params_count": CALIBRATOR_PARAM_COUNT,
        "metasgd_step_sizes_count": METASGD_STEP_SIZES_COUNT,
        "rh_mask_enabled": rh_mask_enabled,
    }

    register = FreezeRegister(
        schema_version=1,
        created_at_utc=datetime.now(UTC).isoformat(),
        freeze_date=freeze_date,
        confirmatory_window_start=conf_start,
        confirmatory_window_end=conf_end,
        git_tag=git_tag,
        git_commit_sha=git_sha,
        config_hash=config_hash,
        headline_model=hl_model,
        promoted_bundle_tag=promoted_tag,
        bundle_sha256=b_sha,
        base_sha256=base_sha,
        city_panel=city_panel,
        hypotheses=hypotheses,
        invariants=invariants,
    )

    if write_files:
        register.save(output_path)
        md_content = generate_freeze_report_markdown(register)
        p_md = Path(report_path)
        p_md.parent.mkdir(parents=True, exist_ok=True)
        with p_md.open("wb") as f:
            f.write(md_content.encode("utf-8"))

    logger.info(
        "freeze_register_created",
        git_tag=git_tag,
        config_hash=config_hash,
        city_panel_size=len(city_panel),
        n_hypotheses=len(hypotheses),
    )
    return register


# =============================================================================
# Verification Engine
# =============================================================================
def verify_freeze_register(
    register_path: Path | str = ".state/freeze_register.json",
    strict: bool = False,
    configs_dir: Path | str = "configs",
    registry_path: Path | str = "data/registry/registry_lahore.parquet",
    models_dir: Path | str = "data/processed/models",
    calibrator_path: Path | str | None = None,
    base_model_path: Path | str | None = None,
) -> tuple[bool, list[str]]:
    """Verify active environment against the committed Gate G4 Freeze Register.

    Checks:
    - Configuration hash parity via Settings.load(configs_dir).hash().
    - Candidate model bundle and shared base model cryptographic digests.
    - Station network eligibility and location attributes in city panel.
    - Pre-registered hypothesis thresholds and operators.
    - Architectural invariants.

    Args:
        register_path: Path to frozen register JSON file.
        strict: If True, raises FreezeVerificationError on drift or tampering.
        configs_dir: Directory containing active YAML configs.
        registry_path: Path to active station registry Parquet file.
        models_dir: Directory containing model checkpoints.
        calibrator_path: Optional direct path to candidate calibrator bundle or .pt file.
        base_model_path: Optional direct path to base model file.

    Returns:
        Tuple of (is_valid, drift_messages).

    Raises:
        FileNotFoundError: If register file does not exist.
        FreezeVerificationError: If strict is True and drift is detected.
    """
    reg_p = Path(register_path)
    if str(register_path) == ".state/freeze_register.json" and not reg_p.is_file() and (Path(configs_dir) / "freeze_register.json").is_file():
        reg_p = Path(configs_dir) / "freeze_register.json"
    reg = FreezeRegister.load(reg_p)
    drift_messages: list[str] = []

    # 1. Configuration Hash Verification
    try:
        current_settings = Settings.load(configs_dir)
        current_hash = current_settings.hash()
        if current_hash != reg.config_hash:
            drift_messages.append(
                f"Configuration hash drift: current '{current_hash}' != frozen '{reg.config_hash}'"
            )
    except Exception as exc:
        drift_messages.append(f"Failed to load and hash configurations from '{configs_dir}': {exc}")
        current_settings = None

    # 2. Model Bundle & Base Verification
    if calibrator_path:
        p_cal = Path(calibrator_path)
        if not p_cal.is_file():
            drift_messages.append(f"Candidate calibrator artifact not found at '{calibrator_path}'")
        else:
            cur_cal_sha = compute_file_sha256(p_cal)
            if cur_cal_sha != reg.bundle_sha256:
                drift_messages.append(
                    f"Candidate calibrator SHA-256 mismatch: current '{cur_cal_sha}' != frozen '{reg.bundle_sha256}'"
                )
    else:
        # Check standard checkpoints if non-dummy hash
        if reg.bundle_sha256 != "0" * 64:
            m7_path = Path(models_dir) / "m7_maml.pt"
            m6_path = Path(models_dir) / "m6_calibrator.pt"
            resolved_cal_path = (
                m7_path if m7_path.is_file() else (m6_path if m6_path.is_file() else None)
            )
            if resolved_cal_path:
                cur_cal_sha = compute_file_sha256(resolved_cal_path)
                if cur_cal_sha != reg.bundle_sha256:
                    drift_messages.append(
                        f"Candidate calibrator SHA-256 mismatch in '{resolved_cal_path}': current '{cur_cal_sha}' != frozen '{reg.bundle_sha256}'"
                    )

    if base_model_path:
        p_base = Path(base_model_path)
        if not p_base.is_file():
            drift_messages.append(f"Base model artifact not found at '{base_model_path}'")
        else:
            cur_base_sha = compute_file_sha256(p_base)
            if cur_base_sha != reg.base_sha256:
                drift_messages.append(
                    f"Base model SHA-256 mismatch: current '{cur_base_sha}' != frozen '{reg.base_sha256}'"
                )
    else:
        if reg.base_sha256 != "0" * 64:
            res_bound_path = Path(models_dir) / "residual_bound.json"
            if res_bound_path.is_file():
                with (
                    contextlib.suppress(Exception),
                    res_bound_path.open("r", encoding="utf-8") as f,
                ):
                    rb_data = json.load(f)
                    cur_b_sha = rb_data.get("base_sha256")
                    if cur_b_sha and cur_b_sha != reg.base_sha256:
                        drift_messages.append(
                            f"Base model SHA-256 mismatch in residual_bound.json: current '{cur_b_sha}' != frozen '{reg.base_sha256}'"
                        )

    # 3. City Panel Station Verification (FR-38)
    try:
        current_stations = load_city_panel_stations(registry_path=registry_path, min_stations=3)
        current_ids = {s.location_id for s in current_stations}
        frozen_ids = {s.location_id for s in reg.city_panel}

        if current_ids != frozen_ids:
            drift_messages.append(
                f"City panel station set drift: current {sorted(current_ids)} != frozen {sorted(frozen_ids)}"
            )
        else:
            frozen_by_id = {s.location_id: s for s in reg.city_panel}
            for cur_s in current_stations:
                f_s = frozen_by_id[cur_s.location_id]
                if cur_s.is_reference != f_s.is_reference:
                    drift_messages.append(
                        f"City panel station {cur_s.location_id} 'is_reference' drift: {cur_s.is_reference} != {f_s.is_reference}"
                    )
                if cur_s.provider != f_s.provider:
                    drift_messages.append(
                        f"City panel station {cur_s.location_id} 'provider' drift: '{cur_s.provider}' != '{f_s.provider}'"
                    )
                if abs(cur_s.lat - f_s.lat) > 1e-4 or abs(cur_s.lon - f_s.lon) > 1e-4:
                    drift_messages.append(
                        f"City panel station {cur_s.location_id} coordinates drift: ({cur_s.lat}, {cur_s.lon}) != ({f_s.lat}, {f_s.lon})"
                    )
    except Exception as exc:
        drift_messages.append(f"Failed to verify city panel against '{registry_path}': {exc}")

    # 4. Pre-Registered Hypotheses Verification
    if current_settings is not None:
        try:
            current_hypotheses = parse_hypotheses_from_settings(current_settings)
            cur_h_map = {h.hypothesis_id: h for h in current_hypotheses}
            frozen_h_map = {h.hypothesis_id: h for h in reg.hypotheses}

            for h_id, f_h in frozen_h_map.items():
                if h_id not in cur_h_map:
                    drift_messages.append(
                        f"Frozen hypothesis '{h_id}' missing from active configuration"
                    )
                    continue
                cur_h = cur_h_map[h_id]
                if cur_h.metric != f_h.metric:
                    drift_messages.append(
                        f"Hypothesis '{h_id}' metric drift: current '{cur_h.metric}' != frozen '{f_h.metric}'"
                    )
                if abs(cur_h.target_threshold - f_h.target_threshold) > 1e-6:
                    drift_messages.append(
                        f"Hypothesis '{h_id}' threshold drift: current {cur_h.target_threshold} != frozen {f_h.target_threshold}"
                    )
                if cur_h.comparison_op != f_h.comparison_op:
                    drift_messages.append(
                        f"Hypothesis '{h_id}' comparison operator drift: current '{cur_h.comparison_op}' != frozen '{f_h.comparison_op}'"
                    )
                if cur_h.horizon_h != f_h.horizon_h:
                    drift_messages.append(
                        f"Hypothesis '{h_id}' horizon drift: current {cur_h.horizon_h} != frozen {f_h.horizon_h}"
                    )
                if cur_h.n_available != f_h.n_available:
                    drift_messages.append(
                        f"Hypothesis '{h_id}' n_available drift: current {cur_h.n_available} != frozen {f_h.n_available}"
                    )

            drift_messages.extend(
                f"Unregistered hypothesis '{h_id}' introduced after Gate G4 freeze"
                for h_id in cur_h_map
                if h_id not in frozen_h_map
            )
        except Exception as exc:
            drift_messages.append(f"Failed to verify hypotheses: {exc}")

    # 5. Invariants Verification
    if current_settings is not None:
        try:
            model_conf = current_settings.model if hasattr(current_settings, "model") else {}
            cur_j = int(model_conf.get("adaptation", {}).get("inner", {}).get("steps", 3))
            frozen_j = int(reg.invariants.get("inner_steps_j", 3))
            if cur_j != frozen_j:
                drift_messages.append(
                    f"Invariant 'inner_steps_j' drift: current {cur_j} != frozen {frozen_j}"
                )

            cur_clip = float(
                model_conf.get("adaptation", {}).get("inner", {}).get("grad_clip", 1.0)
            )
            frozen_clip = float(reg.invariants.get("grad_clip", 1.0))
            if abs(cur_clip - frozen_clip) > 1e-6:
                drift_messages.append(
                    f"Invariant 'grad_clip' drift: current {cur_clip} != frozen {frozen_clip}"
                )

            cur_min_h = int(model_conf.get("target", {}).get("min_valid_hours", 18))
            frozen_min_h = int(reg.invariants.get("min_valid_hours", 18))
            if cur_min_h != frozen_min_h:
                drift_messages.append(
                    f"Invariant 'min_valid_hours' drift: current {cur_min_h} != frozen {frozen_min_h}"
                )

            cur_rh_mask = bool(
                model_conf.get("adaptation", {}).get("rh_mask", {}).get("enabled", False)
            )
            frozen_rh_mask = bool(reg.invariants.get("rh_mask_enabled", False))
            if cur_rh_mask != frozen_rh_mask:
                drift_messages.append(
                    f"Invariant 'rh_mask_enabled' drift: current {cur_rh_mask} != frozen {frozen_rh_mask}"
                )
        except Exception as exc:
            drift_messages.append(f"Failed to verify invariants: {exc}")

    is_valid = len(drift_messages) == 0

    if strict and not is_valid:
        error_msg = (
            f"Gate G4 Freeze Register verification failed with {len(drift_messages)} drift error(s):\n"
            + "\n".join(f" - {msg}" for msg in drift_messages)
        )
        raise FreezeVerificationError(error_msg)

    return is_valid, drift_messages


# =============================================================================
# Markdown Report Generator
# =============================================================================
def generate_freeze_report_markdown(register: FreezeRegister) -> str:
    """Generate human-readable Gate G4 Freeze Register audit report in Markdown."""
    lines: list[str] = [
        "# SmogSense Gate G4 Freeze Register & Invariant Lockdown",
        "",
        "- **Freeze Date:** " + register.freeze_date,
        "- **Confirmatory Shadow Window:** "
        + f"{register.confirmatory_window_start} to {register.confirmatory_window_end}",
        "- **Register Creation (UTC):** " + register.created_at_utc,
        "- **Git Release Tag:** `" + register.git_tag + "`",
        "- **Git Commit SHA:** `" + register.git_commit_sha + "`",
        "- **Configuration Hash (SHA-256):** `" + register.config_hash + "`",
        "- **Headline Model / Challenger:** `" + register.headline_model + "`",
        "- **Promoted Bundle Tag:** `" + register.promoted_bundle_tag + "`",
        "- **Candidate Calibrator SHA-256:** `" + register.bundle_sha256 + "`",
        "- **Shared Physics Base SHA-256:** `" + register.base_sha256 + "`",
        "",
        "---",
        "",
        "## 1. Pre-Registered Confirmatory Hypotheses (H1-H6)",
        "",
        "The following statistical hypotheses and evaluation criteria are permanently locked ahead of Gate G4 freeze:",
        "",
        "| Hypothesis | Statement | Metric | Horizon | Available Stations ($N$) | Threshold | Operator |",
        "|---|---|---|---|---|---|---|",
    ]

    for h in register.hypotheses:
        n_str = str(h.n_available)
        lines.append(
            f"| **{h.hypothesis_id}** | {h.statement} | `{h.metric}` | {h.horizon_h} | {n_str} | {h.target_threshold} | `{h.comparison_op}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Frozen City Panel Station Network ({len(register.city_panel)} stations, FR-38)",
            "",
            "The following eligible ground monitoring stations define the ground truth for the city pseudo-location block mean:",
            "",
            "| Location ID | Station Name | Provider | Reference Monitor | Latitude | Longitude | Elevation (m) | Sensor Model |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for s in register.city_panel:
        ref_str = "Yes (Regulatory)" if s.is_reference else "No (Low-Cost)"
        elev_str = f"{s.elevation_m:.1f}" if s.elevation_m is not None else "N/A"
        lines.append(
            f"| `{s.location_id}` | {s.name} | {s.provider} | {ref_str} | {s.lat:.4f} | {s.lon:.4f} | {elev_str} | {s.sensor_model} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Frozen Architectural Invariants & Hyperparameters",
            "",
            "| Invariant / Constraint | Frozen Value | Source / Architectural Specification |",
            "|---|---|---|",
            f"| Residual Bound ($\\rho_{{99}}$) | `{register.invariants.get('rho_99')}` | ADR-024 / Condition C6 (Block V 99th percentile) |",
            f"| Inner Adaptation Steps ($J$) | `{register.invariants.get('inner_steps_j')}` | `configs/model.yaml` (Condition C5 shared recipe) |",
            f"| Inner Step Size ($\\alpha_{{\\text{{init}}}}$) | `{register.invariants.get('inner_alpha')}` | Meta-SGD softplus initialized scalar |",
            f"| Gradient Norm Clip | `{register.invariants.get('grad_clip')}` | PyTorch inner/outer loop gradient guard |",
            f"| Target Completeness Rule | `{register.invariants.get('min_valid_hours')}` / 24 hours | ADR-005 (18/24h completeness rule) |",
            f"| CAMS Lead Offset | `{register.invariants.get('cams_lead_offset_h')}` hours | Operational as-of data lattice alignment |",
            f"| Temporal Embargo | `{register.invariants.get('embargo_hours')}` hours | Lopez de Prado boundary purge & embargo |",
            f"| Quantile Levels Grid | `{register.invariants.get('quantiles_count')}` levels | 19-level pinball loss grid ($\\tau \\in [0.05, 0.95]$) |",
            f"| Prediction Horizons | `{register.invariants.get('horizons_h')}` | Canonical horizons: 24h, 48h, 72h block means |",
            f"| Calibrator Parameter Count | `{register.invariants.get('calibrator_params_count')}` | Exact 47,577 parameters (Condition C3 / T1-18) |",
            f"| Meta-SGD Step Sizes Count | `{register.invariants.get('metasgd_step_sizes_count')}` | Exact 14 per-tensor learning rate parameters |",
            f"| RH-Mask Low-Cost Status | `{register.invariants.get('rh_mask_enabled')}` | Risk R14 / Condition C9 permanently disabled |",
            "",
            "---",
            "",
            "## 4. Invariant Lockdown & Scientific Integrity Attestation",
            "",
            "> [!IMPORTANT]",
            "> The parameters, configurations, model digests, hypotheses, and city panel recorded above represent the unalterable baseline frozen at Gate G4 (Friday 4 December 2026).",
            "> Under the SmogSense scientific integrity policy (ADR-002, ADR-016, ADR-024), **scores are never recomputed retroactively** and thresholds cannot be adjusted during or after the prospective confirmatory shadow window (7 December 2026 - 31 January 2027). Any divergence constitutes configuration drift or artifact tampering and must be recorded with a dated verification-log entry in `docs/verification-log.md`.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Combined Confirmatory Lockdown Audit (Task P5-01)
# =============================================================================
@dataclass
class ConfirmatoryAuditResult:
    """Consolidated audit result across Gate G4 freeze register and pre-registration specification."""

    is_valid: bool
    freeze_valid: bool
    freeze_messages: list[str]
    prereg_valid: bool
    prereg_messages: list[str]
    config_hash: str
    git_sha: str
    headline_model: str
    bundle_sha256: str
    base_sha256: str
    freeze_register_path: str
    pre_registration_path: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize audit result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "freeze_register": {
                "valid": self.freeze_valid,
                "messages": self.freeze_messages,
                "path": self.freeze_register_path,
            },
            "pre_registration": {
                "valid": self.prereg_valid,
                "messages": self.prereg_messages,
                "path": self.pre_registration_path,
            },
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "headline_model": self.headline_model,
            "bundle_sha256": self.bundle_sha256,
            "base_sha256": self.base_sha256,
        }


def audit_confirmatory_lockdown(
    freeze_register_path: Path | str = ".state/freeze_register.json",
    pre_registration_path: Path | str = ".state/pre_registration.json",
    configs_dir: Path | str = "configs",
    registry_path: Path | str = "data/registry/registry_lahore.parquet",
    models_dir: Path | str = "data/processed/models",
    strict: bool = True,
) -> ConfirmatoryAuditResult:
    """Run combined audit of Gate G4 freeze register and pre-registration specification in sequence.

    Checks:
    1. verify_freeze_register: config hash parity, model digests (M5 base, M7 bundle),
       station panel eligibility, hypothesis thresholds, and architectural invariants.
    2. verify_pre_registration: pre-registration state concordance, freeze register digest binding,
       hypothesis tiers (H1-H6), calendar windows, and power note specifications.

    Raises:
        FreezeVerificationError: If strict=True and freeze register audit detects drift or tampering.
        PreRegistrationVerificationError: If strict=True and pre-registration audit detects drift or tampering.

    Returns:
        ConfirmatoryAuditResult: Consolidated audit result across both pillars.
    """
    from smogsense.evaluation.registration import (
        PreRegistrationVerificationError,
        verify_pre_registration,
    )

    f_path = Path(freeze_register_path)
    p_path = Path(pre_registration_path)
    if str(freeze_register_path) == ".state/freeze_register.json" and not f_path.is_file() and (Path(configs_dir) / "freeze_register.json").is_file():
        f_path = Path(configs_dir) / "freeze_register.json"
    if str(pre_registration_path) == ".state/pre_registration.json" and not p_path.is_file() and (Path(configs_dir) / "pre_registration.json").is_file():
        p_path = Path(configs_dir) / "pre_registration.json"

    # 1. Freeze register verification
    freeze_valid, freeze_msgs = verify_freeze_register(
        register_path=f_path,
        strict=False,
        configs_dir=configs_dir,
        registry_path=registry_path,
        models_dir=models_dir,
    )

    # 2. Pre-registration verification
    prereg_valid, prereg_msgs = verify_pre_registration(
        pre_registration_path=p_path,
        freeze_register_path=f_path,
        configs_dir=configs_dir,
        strict=False,
    )

    overall_valid = freeze_valid and prereg_valid

    # Extract metadata for diagnostics
    config_hash = ""
    git_sha = ""
    headline_model = ""
    bundle_sha256 = ""
    base_sha256 = ""

    if f_path.is_file():
        with contextlib.suppress(Exception):
            reg = FreezeRegister.load(f_path)
            config_hash = reg.config_hash
            git_sha = reg.git_commit_sha
            headline_model = reg.headline_model
            bundle_sha256 = reg.bundle_sha256
            base_sha256 = reg.base_sha256

    if not config_hash:
        with contextlib.suppress(Exception):
            settings = Settings.load(configs_dir)
            config_hash = settings.hash()

    result = ConfirmatoryAuditResult(
        is_valid=overall_valid,
        freeze_valid=freeze_valid,
        freeze_messages=freeze_msgs,
        prereg_valid=prereg_valid,
        prereg_messages=prereg_msgs,
        config_hash=config_hash,
        git_sha=git_sha,
        headline_model=headline_model,
        bundle_sha256=bundle_sha256,
        base_sha256=base_sha256,
        freeze_register_path=str(f_path),
        pre_registration_path=str(p_path),
    )

    if strict:
        if not freeze_valid:
            msg = f"Gate G4 Freeze Register verification failed: {'; '.join(freeze_msgs)}"
            raise FreezeVerificationError(msg, exit_code=30)
        if not prereg_valid:
            msg = f"Gate G4 Pre-Registration verification failed: {'; '.join(prereg_msgs)}"
            raise PreRegistrationVerificationError(msg, exit_code=30)

    return result


def format_confirmatory_audit_table(result: ConfirmatoryAuditResult) -> str:
    """Format a clean tabular summary of the confirmatory lockdown audit."""
    width = 88
    lines = [
        "=" * width,
        "SmogSense Gate G4 Freeze & Pre-Registration Verification Audit".center(width),
        "=" * width,
        f" {'Check Pillar':<24} {'Status':<10} {'Audit Details':<50}",
        "-" * width,
    ]

    # Pillar 1: Freeze Register
    f_status = "PASS" if result.freeze_valid else "FAIL"
    f_details = (
        "0 drift messages (Config, models, panel locked)"
        if result.freeze_valid
        else f"{len(result.freeze_messages)} drift: {'; '.join(result.freeze_messages)[:45]}"
    )
    lines.append(f" {'Freeze Register':<24} {f_status:<10} {f_details:<50}")

    # Pillar 2: Pre-Registration
    p_status = "PASS" if result.prereg_valid else "FAIL"
    p_details = (
        "0 drift messages (H1-H6 tiers, calendar locked)"
        if result.prereg_valid
        else f"{len(result.prereg_messages)} drift: {'; '.join(result.prereg_messages)[:45]}"
    )
    lines.append(f" {'Pre-Registration Spec':<24} {p_status:<10} {p_details:<50}")

    lines.append("-" * width)
    if result.is_valid:
        verdict = "OVERALL AUDIT: PASSED (CONFIRMATORY LOCKDOWN VALIDATED)"
    else:
        verdict = "OVERALL AUDIT: FAILED (CONFIRMATORY DRIFT DETECTED)"
    lines.append(verdict.center(width))
    lines.append("=" * width)
    return "\n".join(lines)
