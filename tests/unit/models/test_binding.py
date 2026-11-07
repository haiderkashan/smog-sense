"""tests/unit/models/test_binding.py — Unit tests for Base / Calibrator cryptographic binding.

Specifications:
- docs/ml-architecture.md §6 ("Binding C3"), §10 ("Training protocol and determinism", "Fallbacks")
- docs/test-specifications.md §T1-15 ("Bundle integrity and binding")
- docs/PRD.md (FR-21, NFR-16)
- decisions.md (ADR-024 Revision 3)

Verifies:
1. test_resolve_base_digest: Resolves active base digest from manifests, archives, standalone files, and directories.
2. test_shared_base_binding_m6_and_m7_match: Models M6 and M7 bind to the single shared M5 physics base.
3. test_binding_mismatch_raises_base_model_binding_error: Tampered or mismatched base_sha256 raises BaseModelBindingError.
4. test_binding_missing_base_sha_raises: Calibrator with missing base_sha256 raises BaseModelBindingError.
5. test_non_calibrator_skips_binding_check: Non-calibrator model bundles pass binding check unconditionally.
6. test_audit_calibrator_binding: Produces complete structured audit card for matching and mismatched pairs.
7. test_pipeline_degrades_to_level_2_on_binding_mismatch: Automated Level 2 degradation when promoted bundle fails binding.
8. test_cli_verify_binding_command: CLI commands (models and registry alias) enforce binding verification and exit codes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from smogsense.cli import app
from smogsense.models.binding import (
    audit_calibrator_binding,
    resolve_base_digest,
    verify_base_calibrator_binding,
)
from smogsense.models.registry import (
    BaseModelBindingError,
    ModelBundleManifest,
    ModelRegistry,
    compute_file_sha256,
)
from smogsense.pipeline.orchestrator import (
    run_challenger_forecast,
    run_daily_pipeline,
)


def _make_test_manifest(
    model_name: str,
    model_type: str,
    version: str = "v1.0.0",
    base_sha256: str | None = None,
    oof_partition_hash: str | None = None,
    feature_set_version: str = "fs1",
    files: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelBundleManifest:
    """Helper to construct valid ModelBundleManifest for testing."""
    return ModelBundleManifest(
        manifest_version="1.0.0",
        model_name=model_name,
        model_type=model_type,
        version=version,
        git_sha="0" * 40,
        config_hash="0" * 64,
        feature_set_version=feature_set_version,
        created_at_utc="2026-10-15T00:00:00Z",
        files=files or {},
        base_sha256=base_sha256,
        oof_partition_hash=oof_partition_hash,
        metadata=metadata or {},
    )


@pytest.fixture
def clean_pipeline_state() -> Generator[None, None, None]:
    """Clean pipeline output and state directories before and after tests."""
    dirs = [
        Path(".state/manifests"),
        Path(".state/forecasts"),
        Path(".state/inputs/lahore"),
        Path("site/forecast"),
    ]
    files = [
        Path("site/latest.json"),
        Path("site/index.html"),
    ]
    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()

    yield

    for d in dirs:
        if d.is_dir():
            for f in d.glob("*"):
                if f.is_file():
                    with contextlib.suppress(Exception):
                        f.unlink()
    for f in files:
        if f.is_file():
            with contextlib.suppress(Exception):
                f.unlink()


@pytest.fixture
def base_and_calibrator_bundles(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, str]:
    """Create synthetic base M5 bundle and matching M6 & M7 calibrator bundles."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(cache_dir=tmp_path / "cache")

    # 1. Shared Base Model M5
    base_src = tmp_path / "base_src"
    base_src.mkdir(parents=True, exist_ok=True)
    base_file = base_src / "model.joblib"
    base_file.write_bytes(b"dummy_m5_lgbm_physics_base_model_content_bytes")
    base_sha = hashlib.sha256(b"dummy_m5_lgbm_physics_base_model_content_bytes").hexdigest()

    base_bundle_archive = registry.create_bundle(
        model_name="m5_physics_base",
        source_dir=base_src,
        output_dir=models_dir,
        version="v1.0.0",
        archive=True,
    )

    # 2. Fine-tuned Calibrator M6
    m6_src = tmp_path / "m6_src"
    m6_src.mkdir(parents=True, exist_ok=True)
    (m6_src / "calibrator.pt").write_bytes(b"dummy_m6_calibrator_weights_content")
    (m6_src / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "residual_calibrator",
                "base_sha256": base_sha,
                "oof_partition_hash": "a" * 64,
                "feature_set_version": "fs1",
                "parameter_count": 47577,
                "step_sizes_count": 0,
            }
        ),
        encoding="utf-8",
    )
    m6_bundle_archive = registry.create_bundle(
        model_name="m6_residual_finetune",
        source_dir=m6_src,
        output_dir=models_dir,
        version="v1.0.0",
        base_sha256=base_sha,
        archive=True,
    )

    # 3. Meta-SGD Calibrator M7
    m7_src = tmp_path / "m7_src"
    m7_src.mkdir(parents=True, exist_ok=True)
    (m7_src / "calibrator.pt").write_bytes(b"dummy_m7_meta_calibrator_weights_content")
    (m7_src / "step_sizes.pt").write_bytes(b"dummy_meta_sgd_step_sizes_content")
    (m7_src / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "residual_calibrator",
                "base_sha256": base_sha,
                "oof_partition_hash": "a" * 64,
                "feature_set_version": "fs1",
                "parameter_count": 47577,
                "step_sizes_count": 14,
            }
        ),
        encoding="utf-8",
    )
    m7_bundle_archive = registry.create_bundle(
        model_name="m7_hybrid_maml",
        source_dir=m7_src,
        output_dir=models_dir,
        version="v1.0.0",
        base_sha256=base_sha,
        archive=True,
    )

    # 4. Corrupted / Mismatched Calibrator M7
    m7_corrupt_src = tmp_path / "m7_corrupt_src"
    m7_corrupt_src.mkdir(parents=True, exist_ok=True)
    (m7_corrupt_src / "calibrator.pt").write_bytes(b"dummy_m7_corrupted_weights")
    (m7_corrupt_src / "step_sizes.pt").write_bytes(b"dummy_step_sizes")
    (m7_corrupt_src / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "residual_calibrator",
                "base_sha256": "bad0" * 16,
                "oof_partition_hash": "a" * 64,
                "feature_set_version": "fs1",
                "parameter_count": 47577,
                "step_sizes_count": 14,
            }
        ),
        encoding="utf-8",
    )
    m7_corrupt_archive = registry.create_bundle(
        model_name="m7_corrupted",
        source_dir=m7_corrupt_src,
        output_dir=models_dir,
        version="v1.0.0",
        base_sha256="bad0" * 16,
        archive=True,
    )

    return (
        Path(base_bundle_archive),
        Path(m6_bundle_archive),
        Path(m7_bundle_archive),
        Path(m7_corrupt_archive),
        base_sha,
    )


def test_resolve_base_digest(tmp_path: Path) -> None:
    """Verify resolve_base_digest handles manifests, files, directories, and archives."""
    # 1. From standalone file
    dummy_file = tmp_path / "dummy_base.joblib"
    dummy_file.write_bytes(b"physics_base_bytes_abc")
    expected_hash = compute_file_sha256(dummy_file)
    assert resolve_base_digest(dummy_file) == expected_hash

    # 2. From manifest object with files dict
    manifest = _make_test_manifest(
        model_name="m5_base",
        model_type="base",
        files={"model.joblib": expected_hash},
    )
    assert resolve_base_digest(manifest) == expected_hash

    # 3. From directory with residual_bound.json
    sub_dir = tmp_path / "dir_with_rb"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rb_file = sub_dir / "residual_bound.json"
    rb_file.write_text(json.dumps({"base_sha256": "abcdef123456"}), encoding="utf-8")
    assert resolve_base_digest(sub_dir) == "abcdef123456"

    # 4. Failure when digest cannot be resolved
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(BaseModelBindingError, match="Unable to resolve base model SHA-256"):
        resolve_base_digest(empty_dir, models_dir=tmp_path / "nonexistent")


def test_shared_base_binding_m6_and_m7_match(
    base_and_calibrator_bundles: tuple[Path, Path, Path, Path, str],
) -> None:
    """Verify both M6 and M7 calibrators cryptographically bind to the shared M5 physics base (Condition C3)."""
    base_bundle, m6_bundle, m7_bundle, _, base_sha = base_and_calibrator_bundles

    # Resolve base digest directly from base archive
    active_base_sha = resolve_base_digest(base_bundle)
    assert active_base_sha == base_sha

    # Verify M6 bundle manifest binding
    registry = ModelRegistry()
    m6_manifest = registry.verify_bundle(m6_bundle, verify_binding=False)
    assert verify_base_calibrator_binding(m6_manifest, active_base_sha, strict=True) is True
    assert m6_manifest.base_sha256 == active_base_sha

    # Verify M7 bundle manifest binding
    m7_manifest = registry.verify_bundle(m7_bundle, verify_binding=False)
    assert verify_base_calibrator_binding(m7_manifest, active_base_sha, strict=True) is True
    assert m7_manifest.base_sha256 == active_base_sha


def test_binding_mismatch_raises_base_model_binding_error(
    base_and_calibrator_bundles: tuple[Path, Path, Path, Path, str],
) -> None:
    """Verify mismatched base_sha256 raises BaseModelBindingError or returns False."""
    base_bundle, _, _, m7_corrupt_bundle, base_sha = base_and_calibrator_bundles

    active_base_sha = resolve_base_digest(base_bundle)
    assert active_base_sha == base_sha

    registry = ModelRegistry()
    corrupt_manifest = registry.verify_bundle(m7_corrupt_bundle, verify_binding=False)

    # In strict mode: raises BaseModelBindingError
    with pytest.raises(BaseModelBindingError, match="Base model binding verification failed"):
        verify_base_calibrator_binding(corrupt_manifest, active_base_sha, strict=True)

    # In non-strict mode: returns False
    assert verify_base_calibrator_binding(corrupt_manifest, active_base_sha, strict=False) is False


def test_binding_missing_base_sha_raises() -> None:
    """Verify calibrator manifest missing base_sha256 fails binding verification."""
    cal_manifest = _make_test_manifest(
        model_name="m7_unbound",
        model_type="residual_calibrator",
        base_sha256=None,
    )

    with pytest.raises(BaseModelBindingError, match="missing required 'base_sha256'"):
        verify_base_calibrator_binding(cal_manifest, "active_sha_12345", strict=True)

    assert verify_base_calibrator_binding(cal_manifest, "active_sha_12345", strict=False) is False


def test_non_calibrator_skips_binding_check() -> None:
    """Verify non-calibrator bundles (e.g. base or baseline) skip base_sha256 verification."""
    base_manifest = _make_test_manifest(
        model_name="m5_base",
        model_type="base",
        base_sha256=None,
    )
    assert verify_base_calibrator_binding(base_manifest, "any_sha_value", strict=True) is True


def test_audit_calibrator_binding(
    base_and_calibrator_bundles: tuple[Path, Path, Path, Path, str],
    tmp_path: Path,
) -> None:
    """Verify audit_calibrator_binding returns complete audit dictionary."""
    base_bundle, _, m7_bundle, m7_corrupt_bundle, base_sha = base_and_calibrator_bundles

    # Audit valid pair
    card_valid = audit_calibrator_binding(
        calibrator_bundle_path_or_tag=m7_bundle,
        base_bundle_path_or_tag=base_bundle,
        cache_dir=tmp_path / "cache",
    )
    assert card_valid["status"] == "verified"
    assert card_valid["is_bound"] is True
    assert card_valid["calibrator_model"] == "m7_hybrid_maml"
    assert card_valid["calibrator_base_sha256"] == base_sha
    assert card_valid["active_base_sha256"] == base_sha
    assert card_valid["parameter_count"] == 47577
    assert card_valid["step_sizes_count"] == 14

    # Audit mismatched pair
    card_invalid = audit_calibrator_binding(
        calibrator_bundle_path_or_tag=m7_corrupt_bundle,
        base_bundle_path_or_tag=base_bundle,
        cache_dir=tmp_path / "cache",
    )
    assert card_invalid["status"] == "failed"
    assert card_invalid["is_bound"] is False
    assert card_invalid["calibrator_base_sha256"] == "bad0" * 16
    assert card_invalid["active_base_sha256"] == base_sha


def test_pipeline_degrades_to_level_2_on_binding_mismatch(
    clean_pipeline_state: None,
    base_and_calibrator_bundles: tuple[Path, Path, Path, Path, str],
    tmp_path: Path,
) -> None:
    """Verify pipeline degrades to Level 2 when promoted bundle fails cryptographic binding."""
    base_bundle, _, _, m7_corrupt_bundle, _ = base_and_calibrator_bundles

    # 1. Test challenger forecast failure isolation
    challenger_conf = {
        "enabled": True,
        "method": "m7_hybrid_maml",
        "bundle_tag": str(m7_corrupt_bundle),
        "base_bundle_tag": str(base_bundle),
    }

    import pandas as pd

    empty_obs = pd.DataFrame(columns=["location_id", "datetime", "pm25"])
    res = run_challenger_forecast(
        issuance_utc=datetime(2026, 10, 15, 0, 0, tzinfo=UTC),
        stations_df=pd.DataFrame(),
        obs_df=empty_obs,
        cams_df=pd.DataFrame(),
        challenger_conf=challenger_conf,
        cache_dir=tmp_path / "cache",
    )
    assert res is not None
    assert res["error"] == "base_binding_mismatch_fallback_level_2"
    assert res["adaptation_status"] == "rejected_base_binding"

    # 2. Test full pipeline execution with promoted bundle set to corrupted bundle
    from smogsense.config import Settings

    real_settings = Settings.load("configs")
    fake_settings = real_settings.model_copy(deep=True)
    fake_settings.model["promoted_bundle"] = str(m7_corrupt_bundle)
    fake_settings.model["challenger"]["bundle_tag"] = str(m7_corrupt_bundle)
    fake_settings.model["challenger"]["base_bundle_tag"] = str(base_bundle)

    issuance_dt = datetime(2026, 10, 15, 0, 0, tzinfo=UTC)
    with patch("smogsense.config.Settings.load", return_value=fake_settings):
        # Run must complete safely (0 or 10) and NEVER crash (50)
        exit_code = run_daily_pipeline(issuance_dt, mode="fixtures", force=True)
        assert exit_code in (0, 10)

        # Public bulletin must fall back to baseline (Level 2 degradation)
        bulletin_path = Path("site/forecast/latest.json")
        assert bulletin_path.is_file()
        with bulletin_path.open("r", encoding="utf-8") as f:
            bulletin = json.load(f)

        # Candidate model was rejected due to binding failure
        assert bulletin["model"]["id"] != "m7_corrupted"
        assert bulletin["model"]["id"] in ("m1_cams_raw", "m2_cams_bc")


def test_cli_verify_binding_command(
    base_and_calibrator_bundles: tuple[Path, Path, Path, Path, str],
    tmp_path: Path,
) -> None:
    """Verify smogsense models verify-binding and registry verify-binding alias."""
    runner = CliRunner()
    base_bundle, _, m7_bundle, m7_corrupt_bundle, _ = base_and_calibrator_bundles
    cache_arg = str(tmp_path / "cache")

    # 1. Matching bundle via models verify-binding -> exit 0
    res_valid = runner.invoke(
        app,
        [
            "models",
            "verify-binding",
            "--calibrator-bundle",
            str(m7_bundle),
            "--base-bundle",
            str(base_bundle),
            "--cache-dir",
            cache_arg,
        ],
    )
    assert res_valid.exit_code == 0
    assert "[OK] Cryptographic binding verified" in res_valid.output

    # 2. Mismatching bundle via models verify-binding -> exit 30
    res_invalid = runner.invoke(
        app,
        [
            "models",
            "verify-binding",
            "--calibrator-bundle",
            str(m7_corrupt_bundle),
            "--base-bundle",
            str(base_bundle),
            "--cache-dir",
            cache_arg,
        ],
    )
    assert res_invalid.exit_code == 30
    assert "[FAIL] Base model binding mismatch" in res_invalid.output

    # 3. Alias under registry command group
    res_alias = runner.invoke(
        app,
        [
            "registry",
            "verify-binding",
            "-c",
            str(m7_bundle),
            "-b",
            str(base_bundle),
            "--cache-dir",
            cache_arg,
        ],
    )
    assert res_alias.exit_code == 0
    assert "[OK] Cryptographic binding verified" in res_alias.output
