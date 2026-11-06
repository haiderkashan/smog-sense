"""smogsense.models.registry — Model artifact registry and bundle packaging.

Packages, resolves, verifies, and extracts production model release bundles
(GitHub Releases tag 'models-v*'). Enforces cryptographic provenance:
- SHA-256 hash verification for every file in the bundle (T1-15).
- Base model cryptographic binding (base_sha256) for residual calibrator bundles (ADR-024 Condition C3).
- Runtime configuration and feature set version compatibility checks.

Specification:
- docs/deployment-and-ops.md §Release and model promotion
- docs/ml-architecture.md §6 (Binding C3) & §10 (Training protocol and determinism)
- docs/test-specifications.md §T1-15 (Bundle Integrity and Binding)
- decisions.md (ADR-008, ADR-024)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from structlog import get_logger

from smogsense.errors import SmogSenseError

logger = get_logger(__name__)

# Canonical parameters
CALIBRATOR_PARAM_COUNT = 47577
METASGD_STEP_SIZES_COUNT = 14
DEFAULT_RHO_99 = 0.362452
DEFAULT_FEATURE_SET_VERSION = "fs1"
DEFAULT_RELEASE_REPO = "haiderkashan/smog-sense"


# =============================================================================
# Registry Exceptions
# =============================================================================
class BundleError(SmogSenseError, ValueError):
    """Base exception for model bundle registry operations."""

    def __init__(self, message: str, exit_code: int = 50) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class BundleTamperedError(BundleError):
    """Raised when bundle files fail cryptographic SHA-256 hash verification (T1-15)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=30)


class BaseModelBindingError(BundleError):
    """Raised when calibrator bundle references a different base_sha256 than active base (C3)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=30)


class BundleIncompatibleError(BundleError):
    """Raised when bundle feature_set_version or code is incompatible with runtime environment."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=30)


class BundleNotFoundError(BundleError, FileNotFoundError):
    """Raised when model bundle cannot be resolved locally or remotely."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=20)


# =============================================================================
# Cryptographic & Provenance Helpers
# =============================================================================
def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 checksum of a file in 64 KiB chunks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_directory_hashes(
    dir_path: Path | str,
    exclude_filenames: set[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Scan directory recursively and compute deterministic SHA-256 for all regular files.

    Returns:
        tuple (files_map, entries_list):
            files_map: mapping of relative posix path to SHA-256 hash.
            entries_list: list of dicts with {"path", "sha256", "bytes"}.
    """
    root = Path(dir_path)
    if not root.is_dir():
        raise NotADirectoryError(f"Directory not found: {root}")

    excluded = exclude_filenames or {"manifest.json", ".DS_Store"}
    files_map: dict[str, str] = {}
    entries: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rel_path = path.relative_to(root).as_posix()
            sha = compute_file_sha256(path)
            size = path.stat().st_size
            files_map[rel_path] = sha
            entries.append(
                {
                    "path": rel_path,
                    "sha256": sha,
                    "bytes": size,
                }
            )

    return files_map, entries


def compute_config_hash(
    model_config_path: Path | str = "configs/model.yaml",
    features_config_path: Path | str = "configs/features.yaml",
) -> str:
    """Compute SHA-256 hash across active configuration files for provenance tracking."""
    hasher = hashlib.sha256()
    for cfg in [Path(model_config_path), Path(features_config_path)]:
        if cfg.is_file():
            hasher.update(cfg.read_bytes())
        else:
            hasher.update(b"")
    return hasher.hexdigest()


def get_git_sha() -> str:
    """Query current repository commit SHA or return fallback."""
    # Check environment variable (common in CI / Docker)
    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha.strip()

    git_bin = shutil.which("git")
    if not git_bin:
        return "unknown"

    try:
        res = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception as e:
        logger.debug("git_rev_parse_failed", error=str(e))
        return "unknown"


def get_runtime_feature_set_version(
    features_config_path: Path | str = "configs/features.yaml",
) -> str:
    """Extract active feature_set_version from features.yaml."""
    p = Path(features_config_path)
    if p.is_file():
        try:
            with p.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict) and "feature_set_version" in data:
                    return str(data["feature_set_version"])
        except Exception as e:
            logger.debug("read_feature_set_version_failed", path=str(p), error=str(e))
    return DEFAULT_FEATURE_SET_VERSION


# =============================================================================
# Model Bundle Manifest
# =============================================================================
@dataclass
class ModelBundleManifest:
    """Standardized cryptographic provenance manifest for model bundles."""

    manifest_version: str
    model_name: str
    model_type: str
    version: str
    git_sha: str
    config_hash: str
    feature_set_version: str
    created_at_utc: str
    files: dict[str, str]
    file_entries: list[dict[str, Any]] = field(default_factory=list)
    base_sha256: str | None = None
    oof_partition_hash: str | None = None
    parameter_count: int | None = None
    step_sizes_count: int | None = None
    acceptance_thresholds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def save(self, path: Path | str) -> None:
        """Write manifest.json to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(self.to_json())
            f.write("\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelBundleManifest:
        """Deserialize manifest from dictionary."""
        # Handle files if presented as list of entries
        files_map: dict[str, str] = {}
        file_entries: list[dict[str, Any]] = []

        if isinstance(data.get("files"), dict):
            files_map = {str(k): str(v) for k, v in data["files"].items()}
        elif isinstance(data.get("files"), list):
            for item in data["files"]:
                if isinstance(item, dict) and "path" in item and "sha256" in item:
                    files_map[str(item["path"])] = str(item["sha256"])
                    file_entries.append(item)

        if "file_entries" in data and isinstance(data["file_entries"], list):
            file_entries = data["file_entries"]

        return cls(
            manifest_version=str(data.get("manifest_version", "1.0")),
            model_name=str(data.get("model_name", "")),
            model_type=str(data.get("model_type", "base")),
            version=str(data.get("version", "v1.0.0")),
            git_sha=str(data.get("git_sha", "unknown")),
            config_hash=str(data.get("config_hash", "")),
            feature_set_version=str(data.get("feature_set_version", DEFAULT_FEATURE_SET_VERSION)),
            created_at_utc=str(data.get("created_at_utc", datetime.now(UTC).isoformat())),
            files=files_map,
            file_entries=file_entries,
            base_sha256=data.get("base_sha256"),
            oof_partition_hash=data.get("oof_partition_hash"),
            parameter_count=data.get("parameter_count"),
            step_sizes_count=data.get("step_sizes_count"),
            acceptance_thresholds=data.get("acceptance_thresholds", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> ModelBundleManifest:
        """Parse manifest from JSON string."""
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: Path | str) -> ModelBundleManifest:
        """Load manifest from JSON file on disk."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return cls.from_json(f.read())


# =============================================================================
# Model Registry Engine
# =============================================================================
class ModelRegistry:
    """Production release bundle registry, packaging, and cryptographic verification engine.

    Coordinates bundle packaging, deterministic manifest hashing, local and remote
    resolution, and runtime binding integrity checks (Condition C3 / T1-15).
    """

    def __init__(
        self,
        cache_dir: Path | str = "data/processed/models",
        github_repo: str = DEFAULT_RELEASE_REPO,
        release_tag_prefix: str = "models-",
        model_config_path: Path | str = "configs/model.yaml",
        features_config_path: Path | str = "configs/features.yaml",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.github_repo = github_repo
        self.release_tag_prefix = release_tag_prefix
        self.model_config_path = Path(model_config_path)
        self.features_config_path = Path(features_config_path)

    def create_bundle(
        self,
        model_name: str,
        source_dir: Path | str,
        output_dir: Path | str | None = None,
        version: str = "v1.0.0",
        model_type: str | None = None,
        base_sha256: str | None = None,
        oof_partition_hash: str | None = None,
        archive: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Package a trained model directory into a signed release bundle with manifest.json.

        Args:
            model_name: Canonical name e.g. "m5_physics_base", "m6_supervised_calibrator", "m7_meta_calibrator".
            source_dir: Directory containing model artifacts (weights, metadata, etc.).
            output_dir: Destination directory (defaults to self.cache_dir / "bundles").
            version: Semantic version tag e.g. "v1.0.0".
            model_type: "base" or "residual_calibrator" (inferred if None).
            base_sha256: Required for calibrator bundles; active physics base SHA-256.
            oof_partition_hash: SHA-256 partition hash of Block E cross-fit folds.
            archive: If True, packages bundle as .tar.gz archive; if False, leaves as directory.
            extra_metadata: Additional descriptive metadata dictionary.

        Returns:
            Path to the created bundle (.tar.gz archive or bundle directory).
        """
        src = Path(source_dir)
        if not src.is_dir():
            raise NotADirectoryError(f"Model source directory not found: {src}")

        dest_root = Path(output_dir) if output_dir is not None else self.cache_dir / "bundles"
        dest_root.mkdir(parents=True, exist_ok=True)

        bundle_folder_name = f"smogsense-model-{model_name}-{version}"
        staging_dir = dest_root / bundle_folder_name
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Copy all model source files to staging
        for item in src.iterdir():
            if item.name == "manifest.json":
                continue
            dest_item = staging_dir / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)

        # Infer model type
        if model_type is None:
            if "m5" in model_name.lower() or "base" in model_name.lower():
                m_type = "base"
            else:
                m_type = "residual_calibrator"
        else:
            m_type = model_type.lower()

        # Compute file checksums across all staged files
        files_map, entries = compute_directory_hashes(staging_dir)

        # For base model, extract base_sha256 if not provided
        resolved_base_sha = base_sha256
        if m_type == "base" and resolved_base_sha is None:
            # Check if metadata.json in staging contains base_sha256
            meta_path = staging_dir / "metadata.json"
            if meta_path.is_file():
                try:
                    with meta_path.open(encoding="utf-8") as f:
                        meta_data = json.load(f)
                        resolved_base_sha = meta_data.get("base_sha256")
                except Exception as e:
                    logger.debug("read_base_sha_from_metadata_failed", error=str(e))

        # For calibrator models, enforce base_sha256 requirement (Condition C3)
        if m_type == "residual_calibrator" and not resolved_base_sha:
            raise ValueError(
                f"Calibrator bundle '{model_name}' requires explicit base_sha256 binding "
                f"(Condition C3 / T1-15)."
            )

        # Extract OOF partition hash if present
        resolved_oof_hash = oof_partition_hash
        if resolved_oof_hash is None:
            meta_path = staging_dir / "metadata.json"
            if meta_path.is_file():
                try:
                    with meta_path.open(encoding="utf-8") as f:
                        meta_data = json.load(f)
                        resolved_oof_hash = meta_data.get("oof_partition_hash")
                except Exception as e:
                    logger.debug("read_oof_hash_from_metadata_failed", error=str(e))

        # Parameter counts
        param_count: int | None = None
        step_sizes_count: int | None = None
        if m_type == "residual_calibrator":
            param_count = CALIBRATOR_PARAM_COUNT
            step_sizes_count = METASGD_STEP_SIZES_COUNT if "m7" in model_name.lower() else 0

        acceptance_thresholds: dict[str, Any] = {}
        if m_type == "residual_calibrator":
            acceptance_thresholds = {
                "rho_99": DEFAULT_RHO_99,
                "min_improvement": 0.02,
                "min_days_better_fraction": 0.60,
                "embargo_hours": 72,
            }

        # Build Manifest
        manifest = ModelBundleManifest(
            manifest_version="1.0",
            model_name=model_name,
            model_type=m_type,
            version=version,
            git_sha=get_git_sha(),
            config_hash=compute_config_hash(self.model_config_path, self.features_config_path),
            feature_set_version=get_runtime_feature_set_version(self.features_config_path),
            created_at_utc=datetime.now(UTC).isoformat(),
            files=files_map,
            file_entries=entries,
            base_sha256=resolved_base_sha,
            oof_partition_hash=resolved_oof_hash,
            parameter_count=param_count,
            step_sizes_count=step_sizes_count,
            acceptance_thresholds=acceptance_thresholds,
            metadata=extra_metadata or {},
        )

        manifest_path = staging_dir / "manifest.json"
        manifest.save(manifest_path)

        logger.info(
            "model_bundle_manifest_created",
            model_name=model_name,
            version=version,
            n_files=len(files_map),
            base_sha256=resolved_base_sha,
        )

        if not archive:
            return staging_dir

        # Package as .tar.gz archive
        archive_path = dest_root / f"{bundle_folder_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(staging_dir, arcname=bundle_folder_name)

        # Cleanup staging directory after successful archive
        shutil.rmtree(staging_dir)
        logger.info("model_bundle_archived", path=str(archive_path))
        return archive_path

    def verify_bundle(
        self,
        bundle_path_or_dir: Path | str,
        active_base_sha256: str | None = None,
        verify_binding: bool = True,
        verify_compatibility: bool = True,
    ) -> ModelBundleManifest:
        """Audit bundle for SHA-256 integrity, cryptographic binding, and compatibility (T1-15).

        Args:
            bundle_path_or_dir: Path to .tar.gz bundle archive or extracted bundle directory.
            active_base_sha256: Expected base_sha256 for calibrator binding check.
            verify_binding: Whether to verify base_sha256 binding for calibrators.
            verify_compatibility: Whether to verify feature_set_version matches runtime.

        Returns:
            Verified ModelBundleManifest instance.

        Raises:
            BundleTamperedError: If any file checksum does not match manifest.
            BaseModelBindingError: If calibrator base_sha256 does not match active_base_sha256.
            BundleIncompatibleError: If feature_set_version mismatches runtime code.
        """
        p = Path(bundle_path_or_dir)
        temp_extracted_dir: Path | None = None

        if not p.exists():
            raise BundleNotFoundError(f"Model bundle not found: {p}")

        if p.is_file() and p.name.endswith((".tar.gz", ".tgz")):
            # Extract to temporary directory for auditing
            temp_extracted_dir = (
                self.cache_dir / "tmp_audit" / f"audit_{datetime.now(UTC).timestamp()}"
            )
            temp_extracted_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(p, "r:gz") as tar:
                tar.extractall(path=temp_extracted_dir, filter="data")

            # Check if contents are nested inside top-level folder
            extracted_items = list(temp_extracted_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                audit_dir = extracted_items[0]
            else:
                audit_dir = temp_extracted_dir
        elif p.is_dir():
            audit_dir = p
        else:
            raise ValueError(f"Unsupported bundle format: {p}")

        try:
            # 1. Load and parse manifest.json
            manifest_path = audit_dir / "manifest.json"
            if not manifest_path.is_file():
                raise BundleTamperedError(
                    f"Bundle is corrupted or incomplete: 'manifest.json' missing from {audit_dir}"
                )

            manifest = ModelBundleManifest.from_file(manifest_path)

            # 2. Recompute and verify SHA-256 for every listed file (T1-15)
            for rel_file_path, expected_hash in manifest.files.items():
                target_file = audit_dir / rel_file_path
                if not target_file.is_file():
                    raise BundleTamperedError(
                        f"Bundle tampering detected: missing file '{rel_file_path}' "
                        f"declared in manifest."
                    )
                actual_hash = compute_file_sha256(target_file)
                if actual_hash != expected_hash:
                    raise BundleTamperedError(
                        f"Bundle tampering detected: file '{rel_file_path}' SHA-256 mismatch "
                        f"(expected '{expected_hash}', got '{actual_hash}')."
                    )

            # 3. Cryptographic base_sha256 binding verification (Condition C3 / T1-15)
            if (
                verify_binding
                and manifest.model_type == "residual_calibrator"
                and active_base_sha256 is not None
                and manifest.base_sha256 != active_base_sha256
            ):
                raise BaseModelBindingError(
                    f"Base model binding verification failed: calibrator bundle references "
                    f"base_sha256='{manifest.base_sha256}', but active physics base is "
                    f"'{active_base_sha256}' (Condition C3)."
                )

            # 4. Feature set version compatibility check
            if verify_compatibility:
                runtime_fs = get_runtime_feature_set_version(self.features_config_path)
                if manifest.feature_set_version != runtime_fs:
                    raise BundleIncompatibleError(
                        f"Bundle feature_set_version '{manifest.feature_set_version}' is "
                        f"incompatible with active runtime '{runtime_fs}'."
                    )

            logger.info(
                "model_bundle_verified",
                model_name=manifest.model_name,
                version=manifest.version,
                model_type=manifest.model_type,
                status="OK",
            )
            return manifest

        finally:
            if temp_extracted_dir is not None and temp_extracted_dir.exists():
                shutil.rmtree(temp_extracted_dir, ignore_errors=True)

    def resolve_bundle(
        self,
        bundle_name_or_tag: str,
        auto_download: bool = True,
    ) -> Path:
        """Locate model bundle on disk or download from GitHub Releases.

        Args:
            bundle_name_or_tag: Local file/dir path, bundle name, or release tag.
            auto_download: Whether to attempt download if missing locally.

        Returns:
            Path to resolved bundle archive or directory.
        """
        # 1. Direct local path check
        p = Path(bundle_name_or_tag)
        if p.exists():
            return p

        # 2. Check cache_dir
        cached_candidates = [
            self.cache_dir / bundle_name_or_tag,
            self.cache_dir / f"{bundle_name_or_tag}.tar.gz",
            self.cache_dir / "bundles" / bundle_name_or_tag,
            self.cache_dir / "bundles" / f"{bundle_name_or_tag}.tar.gz",
            self.cache_dir / f"smogsense-model-{bundle_name_or_tag}.tar.gz",
        ]
        for cand in cached_candidates:
            if cand.exists():
                return cand

        if not auto_download:
            raise BundleNotFoundError(
                f"Bundle '{bundle_name_or_tag}' not found locally in cache {self.cache_dir}"
            )

        # 3. Attempt download from GitHub Releases
        tag = (
            bundle_name_or_tag
            if bundle_name_or_tag.startswith(self.release_tag_prefix)
            else f"{self.release_tag_prefix}{bundle_name_or_tag}"
        )
        asset_name = f"smogsense-model-{bundle_name_or_tag}.tar.gz"
        url = f"https://github.com/{self.github_repo}/releases/download/{tag}/{asset_name}"
        dest_path = self.cache_dir / asset_name

        logger.info(
            "downloading_release_bundle",
            repo=self.github_repo,
            tag=tag,
            asset=asset_name,
            url=url,
        )

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url)
                if resp.status_code == 404:
                    raise BundleNotFoundError(
                        f"Release bundle asset '{asset_name}' not found on GitHub at {url} (HTTP 404)"
                    )
                resp.raise_for_status()
                with dest_path.open("wb") as f:
                    f.write(resp.content)
            logger.info("release_bundle_downloaded", path=str(dest_path))
            return dest_path
        except Exception as e:
            if isinstance(e, BundleNotFoundError):
                raise
            raise BundleNotFoundError(
                f"Failed to download bundle '{bundle_name_or_tag}' from {url}: {e}"
            ) from e

    def load_bundle(
        self,
        bundle_name_or_tag: str,
        active_base_sha256: str | None = None,
        verify: bool = True,
    ) -> tuple[Path, ModelBundleManifest]:
        """Resolve, extract (if archive), audit, and return unpacked bundle directory and manifest.

        Args:
            bundle_name_or_tag: Local file/dir path or release tag.
            active_base_sha256: Expected base model SHA-256 for binding check.
            verify: Whether to enforce full SHA-256 verification.

        Returns:
            tuple (unpacked_bundle_dir, manifest)
        """
        bundle_path = self.resolve_bundle(bundle_name_or_tag)

        # If already a directory
        if bundle_path.is_dir():
            manifest = self.verify_bundle(
                bundle_path,
                active_base_sha256=active_base_sha256,
                verify_binding=verify,
                verify_compatibility=verify,
            )
            return bundle_path, manifest

        # If .tar.gz archive, extract into persistent extracted cache
        extract_root = self.cache_dir / "unpacked" / bundle_path.stem
        if bundle_path.name.endswith(".tar.gz"):
            extract_root = self.cache_dir / "unpacked" / bundle_path.name[:-7]

        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(path=extract_root, filter="data")

        # Resolve top-level folder if nested
        extracted_items = list(extract_root.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            final_dir = extracted_items[0]
        else:
            final_dir = extract_root

        manifest = self.verify_bundle(
            final_dir,
            active_base_sha256=active_base_sha256,
            verify_binding=verify,
            verify_compatibility=verify,
        )
        return final_dir, manifest
