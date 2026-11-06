"""tests/unit/models/test_registry.py — Unit tests for model registry and bundle packaging.

Specification:
- docs/deployment-and-ops.md §Release and model promotion
- docs/ml-architecture.md §6 (Binding C3) & §10 (Training protocol and determinism)
- docs/test-specifications.md §T1-15 (Bundle Integrity and Binding)
- decisions.md (ADR-008, ADR-024)

Verifies:
1. Bundle creation, manifest serialization, and SHA-256 file hashing.
2. Tamper detection (T1-15): modifying file content inside a bundle raises BundleTamperedError.
3. Missing declared file inside a bundle raises BundleTamperedError.
4. Cryptographic base_sha256 binding: calibrator bundle with mismatched base_sha256
   raises BaseModelBindingError when verified against active base model.
5. Incompatible feature_set_version raises BundleIncompatibleError.
6. Tarball archive packaging and round-trip extraction/verification.
7. Local resolution and bundle not found error handling.
8. CLI command integration:
   - 'smogsense registry create-bundle' packages files and writes manifest.
   - 'smogsense registry verify' audits bundle and exits with code 0 on success.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.registry import (
    BaseModelBindingError,
    BundleIncompatibleError,
    BundleNotFoundError,
    BundleTamperedError,
    ModelBundleManifest,
    ModelRegistry,
    compute_file_sha256,
)


@pytest.fixture
def sample_model_dir(tmp_path: Path) -> Path:
    """Create a synthetic model directory mimicking a base or calibrator artifact."""
    src = tmp_path / "sample_model_src"
    src.mkdir(parents=True, exist_ok=True)

    # Write dummy model files
    (src / "weights.pt").write_bytes(b"dummy_neural_weights_data_content_12345")
    (src / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "base",
                "base_sha256": hashlib.sha256(b"dummy_base_model").hexdigest(),
                "oof_partition_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    (src / "booster_01.txt").write_text("booster_tree_parameters_h24_q50", encoding="utf-8")
    return src


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify compute_file_sha256 computes expected SHA-256 and handles missing files."""
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"smogsense_cryptographic_test")
    expected = hashlib.sha256(b"smogsense_cryptographic_test").hexdigest()
    assert compute_file_sha256(test_file) == expected

    with pytest.raises(FileNotFoundError):
        compute_file_sha256(tmp_path / "non_existent.txt")


def test_create_and_verify_bundle_directory(tmp_path: Path, sample_model_dir: Path) -> None:
    """Verify creating an uncompressed bundle directory and auditing its manifest."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    bundle_dir = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        version="v1.2.0",
        archive=False,
    )

    assert bundle_dir.is_dir()
    manifest_path = bundle_dir / "manifest.json"
    assert manifest_path.is_file()

    manifest = ModelBundleManifest.from_file(manifest_path)
    assert manifest.model_name == "m5_physics_base"
    assert manifest.version == "v1.2.0"
    assert manifest.model_type == "base"
    assert "weights.pt" in manifest.files
    assert "booster_01.txt" in manifest.files

    # Verify bundle audit passes
    verified = registry.verify_bundle(bundle_dir)
    assert verified.model_name == "m5_physics_base"
    assert verified.version == "v1.2.0"


def test_create_and_verify_bundle_tarball(tmp_path: Path, sample_model_dir: Path) -> None:
    """Verify creating a .tar.gz bundle archive and round-trip verification."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    archive_path = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        version="v1.0.0",
        archive=True,
    )

    assert archive_path.is_file()
    assert archive_path.name.endswith(".tar.gz")

    # Verify bundle directly on archive
    verified = registry.verify_bundle(archive_path)
    assert verified.model_name == "m5_physics_base"
    assert len(verified.files) >= 2


def test_bundle_tamper_detection(tmp_path: Path, sample_model_dir: Path) -> None:
    """T1-15: Modifying file content inside bundle raises BundleTamperedError."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    bundle_dir = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        archive=False,
    )

    # 1. Modifying a byte in weights.pt triggers tamper detection
    weights_path = bundle_dir / "weights.pt"
    weights_path.write_bytes(b"tampered_corrupted_neural_weights")

    with pytest.raises(BundleTamperedError, match=r"Bundle tampering detected.*SHA-256 mismatch"):
        registry.verify_bundle(bundle_dir)

    # 2. Deleting a declared file triggers tamper detection
    weights_path.unlink()
    with pytest.raises(BundleTamperedError, match=r"Bundle tampering detected.*missing file"):
        registry.verify_bundle(bundle_dir)


def test_calibrator_base_sha256_binding(tmp_path: Path, sample_model_dir: Path) -> None:
    """T1-15: Calibrator bundle must match active physics base_sha256 (Condition C3)."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    correct_base_sha = hashlib.sha256(b"active_base_model_weights_sha").hexdigest()
    mismatched_base_sha = hashlib.sha256(b"different_base_weights_sha").hexdigest()

    # 1. Calibrator creation without base_sha256 raises ValueError
    with pytest.raises(ValueError, match="requires explicit base_sha256 binding"):
        registry.create_bundle(
            model_name="m7_meta_calibrator",
            source_dir=sample_model_dir,
            output_dir=tmp_path / "bundles",
            model_type="residual_calibrator",
            base_sha256=None,
            archive=False,
        )

    # 2. Package calibrator bundle with correct_base_sha
    cal_bundle_dir = registry.create_bundle(
        model_name="m7_meta_calibrator",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        model_type="residual_calibrator",
        base_sha256=correct_base_sha,
        archive=False,
    )

    # 3. Auditing against matching active_base_sha succeeds
    manifest = registry.verify_bundle(
        cal_bundle_dir,
        active_base_sha256=correct_base_sha,
        verify_binding=True,
    )
    assert manifest.base_sha256 == correct_base_sha
    assert manifest.parameter_count == 47577
    assert manifest.step_sizes_count == 14

    # 4. Auditing against mismatched base_sha raises BaseModelBindingError
    with pytest.raises(BaseModelBindingError, match="Base model binding verification failed"):
        registry.verify_bundle(
            cal_bundle_dir,
            active_base_sha256=mismatched_base_sha,
            verify_binding=True,
        )

    # 5. verify_binding=False bypasses check
    bypassed = registry.verify_bundle(
        cal_bundle_dir,
        active_base_sha256=mismatched_base_sha,
        verify_binding=False,
    )
    assert bypassed is not None


def test_bundle_incompatible_feature_set(tmp_path: Path, sample_model_dir: Path) -> None:
    """Incompatible feature_set_version raises BundleIncompatibleError."""
    # Write a dummy features.yaml with feature_set_version fs1
    feats_cfg = tmp_path / "features.yaml"
    feats_cfg.write_text("feature_set_version: fs1\n", encoding="utf-8")

    registry = ModelRegistry(
        cache_dir=tmp_path / "cache",
        features_config_path=feats_cfg,
    )
    bundle_dir = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        archive=False,
    )

    # Manually tamper manifest feature_set_version to incompatible 'fs2'
    manifest_path = bundle_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["feature_set_version"] = "fs2_incompatible"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(BundleIncompatibleError, match=r"feature_set_version.*is incompatible"):
        registry.verify_bundle(bundle_dir, verify_compatibility=True)


def test_resolve_bundle_not_found(tmp_path: Path) -> None:
    """Resolving a non-existent bundle raises BundleNotFoundError."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    with pytest.raises(BundleNotFoundError):
        registry.resolve_bundle("non_existent_bundle_12345", auto_download=False)


def test_load_bundle_tarball_extraction(tmp_path: Path, sample_model_dir: Path) -> None:
    """load_bundle resolves archive, unpacks into persistent cache, and audits manifest."""
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    archive_path = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=sample_model_dir,
        output_dir=tmp_path / "bundles",
        version="v1.0.0",
        archive=True,
    )

    unpacked_dir, manifest = registry.load_bundle(str(archive_path))
    assert unpacked_dir.is_dir()
    assert (unpacked_dir / "manifest.json").is_file()
    assert (unpacked_dir / "weights.pt").is_file()
    assert manifest.model_name == "m5_physics_base"


def test_cli_registry_create_and_verify(tmp_path: Path, sample_model_dir: Path) -> None:
    """Test CLI commands: 'smogsense registry create-bundle' and 'verify'."""
    runner = CliRunner()
    bundle_out = tmp_path / "cli_bundles"

    # 1. create-bundle
    res_create = runner.invoke(
        app,
        [
            "registry",
            "create-bundle",
            "--model-name",
            "m5_physics_base",
            "--source-dir",
            str(sample_model_dir),
            "--output-dir",
            str(bundle_out),
            "--version",
            "v1.0.0",
        ],
    )
    assert res_create.exit_code == 0
    assert "SUCCESS: Created model bundle at" in res_create.stdout

    archive_file = bundle_out / "smogsense-model-m5_physics_base-v1.0.0.tar.gz"
    assert archive_file.is_file()

    # 2. verify
    res_verify = runner.invoke(
        app,
        [
            "registry",
            "verify",
            "--bundle-path",
            str(archive_file),
        ],
    )
    assert res_verify.exit_code == 0
    assert "BUNDLE VERIFICATION AUDIT: PASSED (T1-15)" in res_verify.stdout
    assert "m5_physics_base" in res_verify.stdout
