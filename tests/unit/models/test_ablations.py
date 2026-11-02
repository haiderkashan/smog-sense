"""tests/unit/models/test_ablations.py — Unit tests for MAML scope ablations and Experiment X-11.

Covers:
  1. Support label corruption math matching input sequence corruption (Experiment X-11).
  2. Query labels stay 100% clean regulatory truth regardless of support label policy.
  3. Scope restrictions during inner loop (head_only freezes GRU; gru_only freezes head).
  4. Ablation runner condition execution and metrics computation (residual R, CRPSS, pinball).
  5. JSON serialization and schema round-trip of ablation summary tables.
"""

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from smogsense.models.ablations import (
    CANONICAL_ABLATIONS,
    MAMLAblationRunner,
    apply_task_label_policy,
    corrupt_support_labels,
    evaluate_ablation_condition,
    run_maml_ablations,
)
from smogsense.models.heads import (
    ResidualCalibrator,
)
from smogsense.models.maml import (
    MAMLTask,
    MetaSGDStepSizes,
    inner_loop_adapt,
)
from smogsense.models.pretraining import apply_sensor_corruption


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthetic_maml_task() -> MAMLTask:
    """Create a deterministic synthetic MAMLTask for fast unit testing."""
    torch.manual_seed(42)
    b_sup = 8
    b_qry = 8

    # Support tensors
    sup_x = torch.randn(b_sup, 72, 6)
    sup_u = torch.randn(b_sup, 14)
    sup_u[:, 1] = 80.0  # Set RH = 80% (10% excess over 70%)
    sup_base = torch.ones(b_sup, 3, 19) * 4.0
    sup_y = torch.ones(b_sup, 3) * 4.5
    sup_m = torch.ones(b_sup, 3)

    # Query tensors
    qry_x = torch.randn(b_qry, 72, 6)
    qry_u = torch.randn(b_qry, 14)
    qry_base = torch.ones(b_qry, 3, 19) * 4.0
    qry_y = torch.ones(b_qry, 3) * 4.5
    qry_m = torch.ones(b_qry, 3)

    return MAMLTask(
        domain="delhi",
        season="2023_winter",
        available_stations=[101, 102],
        held_out_stations=[103],
        support_x=sup_x,
        support_context=sup_u,
        support_base_q=sup_base,
        support_y=sup_y,
        support_mask=sup_m,
        query_x=qry_x,
        query_context=qry_u,
        query_base_q=qry_base,
        query_y=qry_y,
        query_mask=qry_m,
        n_available=2,
        corruption_params={"a": 1.25, "b": 0.8, "c": 12.0},
    )


class MockTaskSampler:
    """Mock task sampler returning pre-configured synthetic tasks."""

    def __init__(self, task: MAMLTask) -> None:
        self.task = task
        self.stations_df = [101, 102, 103]

    def sample_task(self, *args: Sequence[object], **kwargs: Sequence[object]) -> MAMLTask:
        return self.task


# -----------------------------------------------------------------------------
# 1. Support Label Corruption Math (Experiment X-11)
# -----------------------------------------------------------------------------
def test_corrupt_support_labels_math_matches_input_sequence_corruption() -> None:
    """Verify that corrupt_support_labels applies the exact same Barkjohn transformation

    as apply_sensor_corruption on input sequence channels.
    """
    raw_val = 50.0  # ug/m3
    rh_val = 85.0  # % RH
    a, b, c = 1.3, 0.9, -5.0

    # 1. Test via apply_sensor_corruption on a sequence tensor
    x_seq = torch.zeros(1, 72, 6)
    x_seq[:, :, 0] = torch.log1p(torch.tensor(raw_val))
    x_seq[:, :, 1] = 0.0  # obs - cams
    x_seq[:, :, 2] = rh_val  # rh_pct
    x_seq[:, :, 4] = 1.0  # valid mask

    corrupted_seq = apply_sensor_corruption(
        x_seq,
        p=1.0,
        gain_range=(a, a),
        rh_slope_range=(b, b),
        offset_range=(c, c),
    )
    expected_corrupted_raw = float(torch.expm1(corrupted_seq[0, 0, 0]).item())

    # 2. Test via corrupt_support_labels on target tensor
    y_sup = torch.tensor(
        [[np.log1p(raw_val), np.log1p(raw_val), np.log1p(raw_val)]], dtype=torch.float32
    )
    corrupted_y = corrupt_support_labels(
        y_sup=y_sup,
        corruption_params={"a": a, "b": b, "c": c},
        rh=rh_val,
        is_log1p=True,
    )
    actual_corrupted_raw = float(torch.expm1(corrupted_y[0, 0]).item())

    # Manual analytical calculation:
    # y_tilde = max(0, 1.3 * 50.0 + 0.9 * max(0, 85.0 - 70.0) - 5.0)
    #         = max(0, 65.0 + 0.9 * 15.0 - 5.0) = max(0, 65.0 + 13.5 - 5.0) = 73.5
    assert pytest.approx(expected_corrupted_raw, rel=1e-5) == 73.5
    assert pytest.approx(actual_corrupted_raw, rel=1e-5) == 73.5
    assert pytest.approx(actual_corrupted_raw, rel=1e-5) == expected_corrupted_raw


def test_corrupt_support_labels_zero_corruption_identity() -> None:
    """Verify that identity corruption (a=1, b=0, c=0) leaves support labels unchanged."""
    y_sup = torch.tensor([[1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    corrupted_y = corrupt_support_labels(
        y_sup=y_sup,
        corruption_params={"a": 1.0, "b": 0.0, "c": 0.0},
        rh=80.0,
        is_log1p=True,
    )
    assert torch.allclose(y_sup, corrupted_y, atol=1e-6)

    # When corruption_params is None, returns clone
    unchanged = corrupt_support_labels(y_sup=y_sup, corruption_params=None)
    assert torch.all(unchanged == y_sup)


def test_corrupt_support_labels_respects_validity_mask() -> None:
    """Verify that entries where support_mask is 0.0 are NOT modified."""
    y_sup = torch.tensor([[2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])  # middle column is invalid

    corrupted_y = corrupt_support_labels(
        y_sup=y_sup,
        corruption_params={"a": 1.5, "b": 1.0, "c": 10.0},
        rh=80.0,
        support_mask=mask,
        is_log1p=True,
    )
    # Valid columns (0 and 2) must be modified
    assert not torch.isclose(corrupted_y[0, 0], y_sup[0, 0])
    assert not torch.isclose(corrupted_y[0, 2], y_sup[0, 2])
    # Invalid column (1) must remain exactly original
    assert torch.isclose(corrupted_y[0, 1], y_sup[0, 1])


# -----------------------------------------------------------------------------
# 2. Support vs Query Label Policy (Experiment X-11)
# -----------------------------------------------------------------------------
def test_apply_task_label_policy_clean_keeps_labels_unmodified(
    synthetic_maml_task: MAMLTask,
) -> None:
    """Under label_policy='clean', support and query labels remain clean regulatory truth."""
    task = apply_task_label_policy(synthetic_maml_task, label_policy="clean")
    assert torch.all(task.support_y == synthetic_maml_task.support_y)
    assert torch.all(task.query_y == synthetic_maml_task.query_y)


def test_apply_task_label_policy_same_as_inputs_corrupts_support_only(
    synthetic_maml_task: MAMLTask,
) -> None:
    """Under label_policy='same_as_inputs', support_y is corrupted while query_y strictly stays clean."""
    task = apply_task_label_policy(synthetic_maml_task, label_policy="same_as_inputs")

    # Support labels must be corrupted
    assert not torch.allclose(task.support_y, synthetic_maml_task.support_y)
    assert torch.all(task.support_y > synthetic_maml_task.support_y)

    # Query labels MUST remain 100% clean regulatory truth!
    assert torch.all(task.query_y == synthetic_maml_task.query_y)


# -----------------------------------------------------------------------------
# 3. Adaptation Scope Restrictions (Condition C4)
# -----------------------------------------------------------------------------
def test_scope_head_only_freezes_gru_parameters(synthetic_maml_task: MAMLTask) -> None:
    """Verify that under adapt_scope='head_only', encoder parameters remain completely frozen."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    init_params = dict(calibrator.named_parameters())
    adapted_params = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=synthetic_maml_task,
        inner_steps=3,
        adapt_scope="head_only",
    )

    # All GRU encoder parameters must be identical to init_params
    for name, p_tensor in adapted_params.items():
        if name.startswith("encoder."):
            assert torch.all(p_tensor == init_params[name]), (
                f"Encoder parameter {name} changed under head_only!"
            )

    # Head parameters must have updated
    head_updated = any(
        not torch.all(adapted_params[name] == init_params[name])
        for name in adapted_params
        if name.startswith("head.")
    )
    assert head_updated, "Head parameters did not update under head_only scope!"


def test_scope_gru_only_freezes_head_parameters(synthetic_maml_task: MAMLTask) -> None:
    """Verify that under adapt_scope='gru_only', head parameters remain completely frozen."""
    calibrator = ResidualCalibrator()
    # Perturb head.fc2.weight slightly so that non-zero gradients propagate to the encoder
    with torch.no_grad():
        calibrator.head.fc2.weight.fill_(0.1)

    step_sizes = MetaSGDStepSizes(init_alpha=0.01)
    init_params = dict(calibrator.named_parameters())

    adapted_params = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=synthetic_maml_task,
        inner_steps=3,
        adapt_scope="gru_only",
    )

    # All Head parameters must be identical to init_params
    for name, p_tensor in adapted_params.items():
        if name.startswith("head."):
            assert torch.all(p_tensor == init_params[name]), (
                f"Head parameter {name} changed under gru_only!"
            )

    # Encoder parameters must have updated
    encoder_updated = any(
        not torch.all(adapted_params[name] == init_params[name])
        for name in adapted_params
        if name.startswith("encoder.")
    )
    assert encoder_updated, "Encoder parameters did not update under gru_only scope!"


# -----------------------------------------------------------------------------
# 4. Evaluation Function Metrics & Residual Bound R (Condition C6)
# -----------------------------------------------------------------------------
def test_evaluate_ablation_condition_metrics(synthetic_maml_task: MAMLTask) -> None:
    """Verify that evaluate_ablation_condition computes pinball, CRPS, CRPSS, and residual R."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    val_tasks = [synthetic_maml_task, synthetic_maml_task]
    metrics = evaluate_ablation_condition(
        calibrator=calibrator,
        step_sizes=step_sizes,
        val_tasks=val_tasks,
        inner_steps=3,
        adapt_scope="all",
        label_policy="clean",
    )

    required_keys = (
        "val_m7_pinball",
        "val_m5_pinball",
        "val_m7_crps",
        "val_m5_crps",
        "crpss_vs_base",
        "residual_r",
        "delta_z_p99",
    )
    for k in required_keys:
        assert k in metrics, f"Missing metric key: {k}"
        assert np.isfinite(metrics[k]), f"Metric {k} was not finite: {metrics[k]}"

    # With J=3 steps on support data, head adapts so residual_r > 0
    assert metrics["residual_r"] > 0.0
    assert metrics["delta_z_p99"] >= metrics["residual_r"]

    # At J=0 inner steps (no adaptation), delta_z == 0 so residual_r is identically 0.0
    metrics_j0 = evaluate_ablation_condition(
        calibrator=calibrator,
        step_sizes=step_sizes,
        val_tasks=val_tasks,
        inner_steps=0,
        adapt_scope="all",
        label_policy="clean",
    )
    assert pytest.approx(metrics_j0["residual_r"], abs=1e-5) == 0.0


# -----------------------------------------------------------------------------
# 5. Ablation Runner Execution Across All 5 Conditions
# -----------------------------------------------------------------------------
def test_maml_ablation_runner_all_5_conditions(synthetic_maml_task: MAMLTask) -> None:
    """Verify MAMLAblationRunner executes all 5 canonical conditions and serializes JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir) / "models"
        fs_dir = Path(tmp_dir) / "fs"

        runner = MAMLAblationRunner(
            feature_store_dir=fs_dir,
            models_dir=models_dir,
            domain="delhi",
            iterations=1,
            meta_batch_size=1,
            inner_steps=2,
            device="cpu",
            random_state=42,
        )

        # Mock initialized state for rapid testing
        runner.train_sampler = MockTaskSampler(synthetic_maml_task)  # type: ignore[assignment]
        runner.val_tasks = [synthetic_maml_task]
        runner.base_sha256 = "mock_base_hash_" + "0" * 50
        runner.oof_partition_hash = "mock_oof_hash_" + "1" * 50
        runner._initialized = True

        out_json = models_dir / "maml_ablations.json"
        results = runner.run_all(conditions=CANONICAL_ABLATIONS, output_path=out_json)

        # Verify all 5 conditions were executed
        assert len(results["conditions"]) == 5
        assert len(results["summary_table"]) == 5

        expected_names = {
            "m7_all_clean",
            "m7_head_clean",
            "m7_gru_clean",
            "m7_all_same_as_inputs",
            "m7_head_same_as_inputs",
        }
        assert set(results["conditions"].keys()) == expected_names

        # Verify JSON file exists and roundtrips cleanly
        assert out_json.exists()
        with out_json.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["domain"] == "delhi"
        assert loaded["base_sha256"] == runner.base_sha256
        assert loaded["oof_partition_hash"] == runner.oof_partition_hash
        assert len(loaded["summary_table"]) == 5

        # Check summary table fields
        first_row = loaded["summary_table"][0]
        assert "Condition" in first_row
        assert "Scope" in first_row
        assert "Label Policy" in first_row
        assert "Val CRPS (ug/m3)" in first_row
        assert "Residual R" in first_row


def test_run_maml_ablations_high_level_entrypoint(
    synthetic_maml_task: MAMLTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify run_maml_ablations wrapper function delegates to MAMLAblationRunner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir) / "models"
        fs_dir = Path(tmp_dir) / "fs"
        out_json = models_dir / "custom_ablations.json"

        # Monkeypatch setup so it bypasses file loading in test environment
        def mock_setup(self: MAMLAblationRunner) -> None:
            self.train_sampler = MockTaskSampler(synthetic_maml_task)  # type: ignore[assignment]
            self.val_tasks = [synthetic_maml_task]
            self.base_sha256 = "b" * 64
            self.oof_partition_hash = "o" * 64
            self._initialized = True

        monkeypatch.setattr(MAMLAblationRunner, "setup", mock_setup)

        custom_conditions = [
            {
                "name": "custom_cond_1",
                "adapt_scope": "all",
                "label_policy": "clean",
                "description": "Test condition",
            }
        ]

        res = run_maml_ablations(
            feature_store_dir=fs_dir,
            models_dir=models_dir,
            domain="delhi",
            iterations=1,
            meta_batch_size=1,
            conditions=custom_conditions,
            output_path=out_json,
        )

        assert "custom_cond_1" in res["conditions"]
        assert out_json.exists()
