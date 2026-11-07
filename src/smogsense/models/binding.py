"""smogsense.models.binding — Base / Calibrator cryptographic binding engine.

Enforces cryptographic binding between neural residual calibrators (Models M6 & M7)
and the single, shared, frozen LightGBM physics base (Model M5) per Condition C3
(ADR-024 Revision 3) and test specification T1-15:
- M6 (fine-tuned) and M7 (Meta-SGD) manifests must declare base_sha256 matching
  the active M5 physics base digest.
- Mismatched or tampered bundles raise BaseModelBindingError and force automated
  pipeline degradation to Level 2 (Model M2: CAMS-BC).

Specification:
- docs/ml-architecture.md §6 ("Binding C3"), §10 ("Training protocol and determinism", "Fallbacks")
- docs/test-specifications.md §T1-15 ("Bundle integrity and binding")
- docs/PRD.md (FR-21, NFR-16)
- decisions.md (ADR-024)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from structlog import get_logger

from smogsense.models.registry import (
    BaseModelBindingError,
    ModelBundleManifest,
    ModelRegistry,
    compute_file_sha256,
)

logger = get_logger(__name__)


def resolve_base_digest(
    active_base_manifest_or_path: ModelBundleManifest | Path | str,
    models_dir: Path | str = "data/processed/models",
) -> str:
    """Resolve active base model SHA-256 digest from a manifest, archive, file, or directory.

    Args:
        active_base_manifest_or_path: Base manifest instance, archive, directory, or model file.
        models_dir: Fallback directory to inspect for residual_bound.json or base models.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        BaseModelBindingError: If base model digest cannot be determined.
    """
    # 0. Already a valid 64-char hex SHA-256 digest string
    if isinstance(active_base_manifest_or_path, str):
        candidate = active_base_manifest_or_path.strip()
        if candidate.lower().startswith("sha256:"):
            candidate = candidate[7:].strip()
        if len(candidate) == 64 and all(c in "0123456789abcdefABCDEF" for c in candidate):
            return candidate.lower()

    # 1. Direct ModelBundleManifest object
    if isinstance(active_base_manifest_or_path, ModelBundleManifest):
        manifest = active_base_manifest_or_path
        if manifest.metadata and "base_sha256" in manifest.metadata:
            return str(manifest.metadata["base_sha256"]).strip()
        if "model.joblib" in manifest.files:
            return str(manifest.files["model.joblib"]).strip()
        if "m5_physics_base.joblib" in manifest.files:
            return str(manifest.files["m5_physics_base.joblib"]).strip()
        if manifest.base_sha256:
            return str(manifest.base_sha256).strip()
        if manifest.files:
            combined = ":".join(f"{k}={v}" for k, v in sorted(manifest.files.items()))
            return hashlib.sha256(combined.encode("utf-8")).hexdigest()
        raise BaseModelBindingError(
            f"Cannot resolve base_sha256 from manifest '{manifest.model_name}' without file hashes."
        )

    # 2. Path or str
    p = Path(active_base_manifest_or_path)

    # 2a. Direct file
    if p.is_file():
        if p.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(p, "r:gz") as tar:
                manifest_member = None
                for member in tar.getmembers():
                    if member.name.endswith("manifest.json"):
                        manifest_member = member
                        break
                if manifest_member is not None:
                    extracted = tar.extractfile(manifest_member)
                    if extracted is not None:
                        data = json.load(extracted)
                        m = ModelBundleManifest.from_dict(data)
                        return resolve_base_digest(m, models_dir=models_dir)
            # Fall back to archive hash if manifest unresolvable
            return compute_file_sha256(p)

        if p.name == "residual_bound.json":
            with p.open("r", encoding="utf-8") as f:
                rb_data = json.load(f)
                if rb_data.get("base_sha256"):
                    return str(rb_data["base_sha256"]).strip()

        if p.name == "manifest.json":
            m = ModelBundleManifest.from_file(p)
            return resolve_base_digest(m, models_dir=models_dir)

        # Standalone model file (e.g. model.joblib or m5_physics_base.joblib)
        return compute_file_sha256(p)

    # 2b. Directory
    if p.is_dir():
        manifest_file = p / "manifest.json"
        if manifest_file.is_file():
            m = ModelBundleManifest.from_file(manifest_file)
            return resolve_base_digest(m, models_dir=models_dir)

        rb_file = p / "residual_bound.json"
        if rb_file.is_file():
            with rb_file.open("r", encoding="utf-8") as f:
                rb_data = json.load(f)
                if rb_data.get("base_sha256"):
                    return str(rb_data["base_sha256"]).strip()

        for cand in ["model.joblib", "m5_physics_base.joblib"]:
            cand_p = p / cand
            if cand_p.is_file():
                return compute_file_sha256(cand_p)

        base_model_dir = p / "base_model"
        if base_model_dir.is_dir():
            boosters = sorted(f for f in base_model_dir.iterdir() if f.is_file())
            if boosters:
                hashes = [f"{b.name}:{compute_file_sha256(b)}" for b in boosters]
                return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()

    # 2c. Check models_dir fallback
    fallback_rb = Path(models_dir) / "residual_bound.json"
    if fallback_rb.is_file():
        with contextlib.suppress(Exception), fallback_rb.open("r", encoding="utf-8") as f:
            rb_data = json.load(f)
            if rb_data.get("base_sha256"):
                return str(rb_data["base_sha256"]).strip()

    raise BaseModelBindingError(
        f"Unable to resolve base model SHA-256 digest from '{active_base_manifest_or_path}'"
    )


def verify_base_calibrator_binding(
    calibrator_manifest: ModelBundleManifest,
    active_base_manifest_or_path: ModelBundleManifest | Path | str,
    strict: bool = True,
) -> bool:
    """Verify that a calibrator bundle is cryptographically bound to the active base model (Condition C3 / T1-15).

    Args:
        calibrator_manifest: Calibrator bundle manifest (Models M6 or M7).
        active_base_manifest_or_path: Active base model manifest, archive, directory, or file.
        strict: If True, raises BaseModelBindingError on mismatch. If False, returns False.

    Returns:
        True if cryptographic binding is valid.

    Raises:
        BaseModelBindingError: If base_sha256 is missing or differs from active base digest.
    """
    if calibrator_manifest.model_type != "residual_calibrator":
        logger.debug(
            "skipping_binding_check_non_calibrator",
            model_name=calibrator_manifest.model_name,
            model_type=calibrator_manifest.model_type,
        )
        return True

    calibrator_base_sha = calibrator_manifest.base_sha256
    if not calibrator_base_sha or not str(calibrator_base_sha).strip():
        msg = (
            f"Calibrator bundle '{calibrator_manifest.model_name}' manifest is missing required "
            f"'base_sha256' cryptographic binding (Condition C3 / T1-15)."
        )
        logger.warning("calibrator_missing_base_sha256", model=calibrator_manifest.model_name)
        if strict:
            raise BaseModelBindingError(msg)
        return False

    expected_base_sha = resolve_base_digest(active_base_manifest_or_path).strip()
    calibrator_base_sha = str(calibrator_base_sha).strip()

    if calibrator_base_sha != expected_base_sha:
        msg = (
            f"Base model binding verification failed: calibrator bundle '{calibrator_manifest.model_name}' "
            f"references base_sha256='{calibrator_base_sha}', but active physics base digest is "
            f"'{expected_base_sha}' (Condition C3 / T1-15)."
        )
        logger.warning(
            "base_model_binding_mismatch",
            model=calibrator_manifest.model_name,
            calibrator_base_sha=calibrator_base_sha,
            expected_base_sha=expected_base_sha,
        )
        if strict:
            raise BaseModelBindingError(msg)
        return False

    logger.info(
        "base_model_binding_verified",
        calibrator_model=calibrator_manifest.model_name,
        calibrator_version=calibrator_manifest.version,
        base_sha256=expected_base_sha,
    )
    return True


def audit_calibrator_binding(
    calibrator_bundle_path_or_tag: str | Path,
    base_bundle_path_or_tag: str | Path,
    cache_dir: Path | str = "data/processed/models",
) -> dict[str, Any]:
    """Audit base/calibrator binding between two bundles or paths and return structured card.

    Args:
        calibrator_bundle_path_or_tag: Calibrator bundle archive, directory, or tag.
        base_bundle_path_or_tag: Shared base bundle archive, directory, or tag.
        cache_dir: Bundle cache directory.

    Returns:
        Structured audit dictionary.
    """
    registry = ModelRegistry(cache_dir=cache_dir)

    # 1. Resolve and verify shared base
    base_path = Path(base_bundle_path_or_tag)
    base_manifest: ModelBundleManifest | None = None
    if base_path.exists():
        if base_path.is_dir() or base_path.name.endswith((".tar.gz", ".tgz")):
            with contextlib.suppress(Exception):
                base_manifest = registry.verify_bundle(base_path, verify_binding=False)
    else:
        with contextlib.suppress(Exception):
            _, base_manifest = registry.load_bundle(str(base_bundle_path_or_tag), verify=True)

    base_sha = resolve_base_digest(base_manifest if base_manifest is not None else base_path)

    # 2. Resolve and verify calibrator bundle
    cal_path = Path(calibrator_bundle_path_or_tag)
    if cal_path.exists():
        cal_manifest = registry.verify_bundle(
            cal_path,
            active_base_sha256=base_sha,
            verify_binding=False,
        )
    else:
        _, cal_manifest = registry.load_bundle(
            str(calibrator_bundle_path_or_tag),
            active_base_sha256=base_sha,
            verify=False,
        )

    # 3. Check binding
    is_bound = verify_base_calibrator_binding(cal_manifest, base_sha, strict=False)

    return {
        "status": "verified" if is_bound else "failed",
        "is_bound": is_bound,
        "calibrator_model": cal_manifest.model_name,
        "calibrator_version": cal_manifest.version,
        "calibrator_base_sha256": cal_manifest.base_sha256,
        "active_base_sha256": base_sha,
        "oof_partition_hash": cal_manifest.oof_partition_hash,
        "feature_set_version": cal_manifest.feature_set_version,
        "parameter_count": cal_manifest.parameter_count,
        "step_sizes_count": cal_manifest.step_sizes_count,
    }
