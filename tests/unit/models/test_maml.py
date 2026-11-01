"""tests.unit.models.test_maml — Unit test suite for Phase 2 Task P2-08.

Tests:
1. Meta-SGD step sizes initialization (alpha_k = 0.01), softplus positivity, and parameter names.
2. Zero-init unadapted calibrator reproduces base model bit-for-bit.
3. Functional inner loop computes valid adapted parameter dictionary without mutating base model weights.
4. Inner loop decreases support loss over J=3 steps.
5. Outer loop backpropagation successfully updates both initial weights theta and Meta-SGD step sizes rho.
6. Synthetic sensor corruption applied to support inputs leaves query targets 100% uncorrupted.
7. N=0 task skips adaptation and returns base predictions identically (Condition C7).
8. Twin-aware station sampling never splits colocated twins across A and B.
9. Adaptation scope ('all', 'head_only', 'gru_only') restricts parameter updates appropriately.
10. Inner gradient clipping constrains gradient norm to <= 1.0.
11. Checkpoint serialization and round-trip restore preserves binding hashes (Condition C3).
12. TaskSampler builds valid embargoed tasks with proper tensor shapes.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
)
from smogsense.models.losses import MaskedResidualPinballLoss
from smogsense.models.maml import (
    PARAM_TENSOR_NAMES,
    MAMLTask,
    MetaSGDStepSizes,
    TaskSampler,
    evaluate_maml,
    inner_loop_adapt,
    maml_outer_step,
)


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------
@pytest.fixture
def dummy_task() -> MAMLTask:
    """Create a minimal synthetic MAMLTask for fast unit testing."""
    torch.manual_seed(42)
    b_sup = 8
    b_qry = 8

    # Support tensors
    sup_x = torch.randn(b_sup, 72, 6)
    sup_u = torch.randn(b_sup, 14)
    sup_base = torch.ones(b_sup, 3, 19) * 4.0
    sup_y = torch.ones(b_sup, 3) * 4.5  # Base is under-predicting
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
    )


@pytest.fixture
def dummy_n0_task(dummy_task: MAMLTask) -> MAMLTask:
    """Create a synthetic N=0 task where support set is empty."""
    return MAMLTask(
        domain="delhi",
        season="2023_winter",
        available_stations=[],
        held_out_stations=[101, 102, 103],
        support_x=torch.zeros(0, 72, 6),
        support_context=torch.zeros(0, 14),
        support_base_q=torch.zeros(0, 3, 19),
        support_y=torch.zeros(0, 3),
        support_mask=torch.zeros(0, 3),
        query_x=dummy_task.query_x,
        query_context=dummy_task.query_context,
        query_base_q=dummy_task.query_base_q,
        query_y=dummy_task.query_y,
        query_mask=dummy_task.query_mask,
        n_available=0,
    )


# -----------------------------------------------------------------------------
# 1. Meta-SGD Step Sizes Invariants
# -----------------------------------------------------------------------------
def test_meta_sgd_step_sizes_initialization() -> None:
    """Verify that Meta-SGD step sizes initialize at alpha = 0.01 via softplus."""
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    # Must contain exactly 14 learnable scalars
    assert len(step_sizes.param_names) == 14
    assert step_sizes.rho.shape == (14,)

    # Expected initial rho: ln(exp(0.01) - 1) approx -4.60015
    expected_rho = math.log(math.expm1(0.01))
    assert torch.allclose(step_sizes.rho, torch.tensor(expected_rho), atol=1e-4)

    # Initial step sizes must be approx 0.01
    alphas = step_sizes.forward()
    assert alphas.shape == (14,)
    assert torch.all(alphas > 0.0)
    assert torch.allclose(alphas, torch.tensor(0.01), atol=1e-4)

    # Dictionary lookup by parameter tensor name
    alpha_dict = step_sizes.get_step_sizes()
    for name in PARAM_TENSOR_NAMES:
        assert name in alpha_dict
        assert pytest.approx(float(alpha_dict[name].item()), rel=1e-4) == 0.01


def test_meta_sgd_step_sizes_softplus_positivity() -> None:
    """Verify that even with large negative rho, alpha is strictly positive."""
    step_sizes = MetaSGDStepSizes()
    with torch.no_grad():
        step_sizes.rho.fill_(-100.0)

    alphas = step_sizes.forward()
    assert torch.all(alphas >= 0.0)
    assert torch.all(torch.isfinite(alphas))


# -----------------------------------------------------------------------------
# 2. Zero-Init Invariant
# -----------------------------------------------------------------------------
def test_zero_init_unadapted_calibrator_reproduces_base(dummy_task: MAMLTask) -> None:
    """Verify zero-init invariant: untrained calibrator reproduces base predictions bit-for-bit."""
    calibrator = ResidualCalibrator()
    calibrator.eval()

    with torch.no_grad():
        pred = calibrator(
            dummy_task.query_x,
            dummy_task.query_context,
            base_quantiles=dummy_task.query_base_q,
            rearrange=False,
        )

    # Residual delta_z must be exactly zero, output equals base_quantiles
    assert torch.equal(pred, dummy_task.query_base_q)


# -----------------------------------------------------------------------------
# 3. Functional Inner-Loop Immutability
# -----------------------------------------------------------------------------
def test_inner_loop_does_not_mutate_model_weights(dummy_task: MAMLTask) -> None:
    """Verify functional inner loop returns updated dictionary without mutating model parameters."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()

    # Record parameter clones before adaptation
    orig_params = {name: p.clone() for name, p in calibrator.named_parameters()}

    init_params = dict(calibrator.named_parameters())
    adapted_params = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=dummy_task,
        inner_steps=3,
    )

    # Model parameters in calibrator must be 100% untouched
    for name, p in calibrator.named_parameters():
        assert torch.equal(p, orig_params[name]), f"Model weight {name} was mutated in place!"

    # Adapted parameter dictionary must have been updated
    diffs = [
        torch.max(torch.abs(adapted_params[name] - orig_params[name])).item()
        for name in adapted_params
    ]
    assert any(d > 0.0 for d in diffs), "Adapted parameters were not updated!"


# -----------------------------------------------------------------------------
# 4. Inner-Loop Loss Decrease
# -----------------------------------------------------------------------------
def test_inner_loop_decreases_support_loss(dummy_task: MAMLTask) -> None:
    """Verify that J=3 inner adaptation steps strictly decrease support pinball loss."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    init_params = dict(calibrator.named_parameters())

    # Initial loss
    with torch.no_grad():
        init_preds = calibrator(
            dummy_task.support_x,
            dummy_task.support_context,
            base_quantiles=dummy_task.support_base_q,
        )
        init_loss = criterion(
            dummy_task.support_y, init_preds, z_base=None, mask=dummy_task.support_mask
        )

    # Adapt J=3 steps
    adapted_params = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=dummy_task,
        inner_steps=3,
    )

    # Adapted loss
    with torch.no_grad():
        from torch.func import functional_call

        adapted_preds = functional_call(
            calibrator,
            adapted_params,
            (
                dummy_task.support_x,
                dummy_task.support_context,
                dummy_task.support_base_q,
                False,
            ),
        )
        adapted_loss = criterion(
            dummy_task.support_y, adapted_preds, z_base=None, mask=dummy_task.support_mask
        )

    assert adapted_loss.item() < init_loss.item()


# -----------------------------------------------------------------------------
# 5. Outer-Loop Backpropagation & Parameter Updates
# -----------------------------------------------------------------------------
def test_maml_outer_step_updates_weights_and_step_sizes(dummy_task: MAMLTask) -> None:
    """Verify FOMAML outer loop updates both initial weights theta and Meta-SGD step sizes rho."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()

    optimizer = torch.optim.Adam(
        list(calibrator.parameters()) + list(step_sizes.parameters()), lr=1e-2
    )

    orig_fc2 = calibrator.head.fc2.bias.clone()
    orig_rho = step_sizes.rho.clone()

    loss_val = maml_outer_step(
        calibrator=calibrator,
        step_sizes=step_sizes,
        outer_optimizer=optimizer,
        batch_tasks=[dummy_task],
        inner_steps=3,
        clip_norm=1.0,
    )

    assert loss_val > 0.0

    # Calibrator parameters must have updated
    assert not torch.equal(calibrator.head.fc2.bias, orig_fc2)

    # Meta-SGD step sizes rho must have updated
    assert not torch.equal(step_sizes.rho, orig_rho)


# -----------------------------------------------------------------------------
# 6. Sensor Corruption on Support Leaves Query Labels Clean
# -----------------------------------------------------------------------------
def test_synthetic_sensor_corruption_leaves_query_clean(dummy_task: MAMLTask) -> None:
    """Verify sensor corruption alters support inputs while query targets stay clean regulatory truth."""
    from smogsense.models.pretraining import apply_sensor_corruption

    orig_support_pm = dummy_task.support_x[:, 10, 0].clone()
    orig_query_y = dummy_task.query_y.clone()

    corrupted_x = apply_sensor_corruption(
        dummy_task.support_x,
        p=1.0,
        gain_range=(1.2, 1.2),
        rh_slope_range=(0.5, 0.5),
        offset_range=(10.0, 10.0),
    )

    # Support sequence channel 0 was corrupted
    assert not torch.equal(corrupted_x[:, 10, 0], orig_support_pm)

    # Query targets y must remain 100% untouched
    assert torch.equal(dummy_task.query_y, orig_query_y)


# -----------------------------------------------------------------------------
# 7. N = 0 Sensor-Free Invariant (Condition C7)
# -----------------------------------------------------------------------------
def test_n0_task_skips_adaptation_and_returns_base(dummy_n0_task: MAMLTask) -> None:
    """Verify that at N=0, inner adaptation is skipped and predictions equal M5 base identically."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()

    init_params = dict(calibrator.named_parameters())
    adapted_params = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=dummy_n0_task,
        inner_steps=3,
    )

    # Params must be completely identical
    for name in init_params:
        assert torch.equal(adapted_params[name], init_params[name])


# -----------------------------------------------------------------------------
# 8. Twin-Aware Station Sampling Invariant
# -----------------------------------------------------------------------------
def test_twin_aware_sampling_never_splits_colocated_pairs() -> None:
    """Verify that co-located stations (d <= 50m) always stay on the same side of A/B split."""
    from smogsense.features.sparse_network import sample_available_stations

    # Create dummy stations table where station 101 and 102 are colocated (< 50m)
    stations_df = pd.DataFrame(
        {
            "location_id": [101, 102, 201, 202, 301, 302],
            "latitude": [28.60000, 28.60010, 28.70000, 28.70000, 28.80000, 28.80000],
            "longitude": [77.20000, 77.20010, 77.30000, 77.40000, 77.50000, 77.60000],
        }
    )

    rng = np.random.default_rng(42)

    for _ in range(25):
        avail, _held, _n = sample_available_stations(
            stations_df, available_n_sampler=(1, 2, 3), rng=rng
        )
        in_avail_101 = 101 in avail
        in_avail_102 = 102 in avail

        # Both must be in available set or both in held-out set
        assert in_avail_101 == in_avail_102, (
            "Co-located twin pair (101, 102) was split across A and B!"
        )


# -----------------------------------------------------------------------------
# 9. Adaptation Scope Ablations (Condition C4)
# -----------------------------------------------------------------------------
def test_adaptation_scope_restricts_updated_parameters(dummy_task: MAMLTask) -> None:
    """Verify that 'head_only' (ANIL) and 'gru_only' adapt only their respective submodules."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()
    init_params = dict(calibrator.named_parameters())

    # 1. head_only: encoder must remain unchanged
    head_adapted = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=dummy_task,
        adapt_scope="head_only",
    )
    for name in init_params:
        if name.startswith("encoder."):
            assert torch.equal(head_adapted[name], init_params[name])
        elif name.startswith("head.fc2."):
            assert not torch.equal(head_adapted[name], init_params[name])

    # 2. gru_only: head must remain unchanged
    # Note: Zero-init head has fc2.weight == 0, which zeroes out gradients to the encoder.
    # We perturb fc2.weight so non-zero gradients flow back through the encoder.
    with torch.no_grad():
        calibrator.head.fc2.weight.normal_(0.0, 0.1)
    init_params_gru = dict(calibrator.named_parameters())

    gru_adapted = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params_gru,
        step_sizes=step_sizes,
        task=dummy_task,
        adapt_scope="gru_only",
    )
    for name in init_params_gru:
        if name.startswith("head."):
            assert torch.equal(gru_adapted[name], init_params_gru[name])
        elif name.startswith("encoder.gru."):
            assert not torch.equal(gru_adapted[name], init_params_gru[name])


# -----------------------------------------------------------------------------
# 10. Inner Gradient Clipping
# -----------------------------------------------------------------------------
def test_inner_gradient_clipping_bounds_gradient_norm(dummy_task: MAMLTask) -> None:
    """Verify inner gradient clipping constrains extreme gradient spikes."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()

    # Artificially amplify targets to induce massive gradients
    large_loss_task = MAMLTask(
        domain="delhi",
        season=dummy_task.season,
        available_stations=dummy_task.available_stations,
        held_out_stations=dummy_task.held_out_stations,
        support_x=dummy_task.support_x,
        support_context=dummy_task.support_context,
        support_base_q=dummy_task.support_base_q,
        support_y=dummy_task.support_y * 100.0,  # Extreme target spike
        support_mask=dummy_task.support_mask,
        query_x=dummy_task.query_x,
        query_context=dummy_task.query_context,
        query_base_q=dummy_task.query_base_q,
        query_y=dummy_task.query_y,
        query_mask=dummy_task.query_mask,
        n_available=2,
    )

    init_params = dict(calibrator.named_parameters())
    adapted = inner_loop_adapt(
        calibrator=calibrator,
        params=init_params,
        step_sizes=step_sizes,
        task=large_loss_task,
        clip_norm=1.0,
    )

    # Parameters should have changed, but boundedly (no NaNs or Infs)
    for name, p in adapted.items():
        assert torch.all(torch.isfinite(p))
        diff = torch.max(torch.abs(p - init_params[name])).item()
        # With clip=1.0, alpha=0.01, J=3 steps, max update is bounded by J * alpha * clip = 0.03
        assert diff <= 0.035, f"Parameter {name} update exceeded clip bounds: {diff}"


# -----------------------------------------------------------------------------
# 11. Checkpoint Serialization & Binding (Condition C3)
# -----------------------------------------------------------------------------
def test_maml_checkpoint_roundtrip_preserves_binding() -> None:
    """Verify checkpoint serialization preserves base_sha256, oof_hash, and learned step sizes."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes(init_alpha=0.02)

    dummy_base_sha256 = "c" * 64
    dummy_oof_hash = "d" * 64

    checkpoint_data = {
        "model_type": "M7",
        "model_state_dict": calibrator.state_dict(),
        "meta_sgd_step_sizes": {
            name: float(step_sizes.step_size_for(name).item()) for name in step_sizes.param_names
        },
        "meta_sgd_rho": step_sizes.rho.detach().cpu().numpy(),
        "base_sha256": dummy_base_sha256,
        "oof_partition_hash": dummy_oof_hash,
        "inner_steps": 3,
        "clip_norm": 1.0,
        "quantiles": CANONICAL_QUANTILES,
        "horizons": CANONICAL_HORIZONS,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        pt_path = Path(tmp_dir) / "m7_maml.pt"
        torch.save(checkpoint_data, pt_path)

        loaded = torch.load(pt_path, weights_only=False)
        assert loaded["model_type"] == "M7"
        assert loaded["base_sha256"] == dummy_base_sha256
        assert loaded["oof_partition_hash"] == dummy_oof_hash
        assert len(loaded["meta_sgd_step_sizes"]) == 14
        assert pytest.approx(loaded["meta_sgd_step_sizes"]["head.fc2.bias"], rel=1e-4) == 0.02


# -----------------------------------------------------------------------------
# 12. TaskSampler Window & Tensor Sanity
# -----------------------------------------------------------------------------
def test_task_sampler_synthetic_data() -> None:
    """Verify TaskSampler generates valid MAMLTask instances on tabular features."""
    # Synthetic DataFrame
    rng_syn = np.random.default_rng(42)
    n_rows = 100
    dates = pd.date_range("2023-11-01", periods=n_rows, freq="6h", tz="UTC")
    df = pd.DataFrame(
        {
            "location_id": rng_syn.choice([101, 102, 103, 104], size=n_rows),
            "issuance_utc": dates,
            "season": "2023_winter",
            "f_stat_latitude": 28.6,
            "f_stat_longitude": 77.2,
            "f_cams_bias_trailing_7d": 5.0,
            "f_met_rh_mean_h24": 60.0,
            "f_met_ln_vc_mean_h24": 7.0,
            "f_met_ln_vc_mean_h48": 7.0,
            "f_met_ln_vc_mean_h72": 7.0,
            "y_h24": 120.0,
            "valid_h24": True,
            "y_h48": 110.0,
            "valid_h48": True,
            "y_h72": 100.0,
            "valid_h72": True,
        }
    )

    # Add 9 base quantiles
    for h in [24, 48, 72]:
        for q in [10, 50, 90]:
            df[f"m5_oof_h{h}_q{q}"] = 100.0

    # Add 57 base quantiles
    for h in CANONICAL_HORIZONS:
        for q in CANONICAL_QUANTILES:
            df[f"m5_oof_h{h}_q{round(q * 100)}"] = 100.0

    sampler = TaskSampler(
        df_features=df,
        available_n_sampler=(1, 2),
        support_window_days=(7,),
        query_window_days=7,
        embargo_hours=72,
        corruption_enabled=False,
    )

    task = sampler.sample_task()
    assert isinstance(task, MAMLTask)
    assert task.query_x.ndim == 3
    assert task.query_x.shape[1] == 72
    assert task.query_x.shape[2] == 6
    assert task.query_context.shape[1] == 14
    assert task.query_base_q.shape[1] == 3
    assert task.query_base_q.shape[2] == 19
    assert task.query_y.shape[1] == 3
    assert task.query_mask.shape[1] == 3


def test_evaluate_maml_metrics(dummy_task: MAMLTask) -> None:
    """Verify evaluate_maml computes pinball, CRPS, and CRPSS over tasks."""
    calibrator = ResidualCalibrator()
    step_sizes = MetaSGDStepSizes()

    metrics = evaluate_maml(
        calibrator=calibrator,
        step_sizes=step_sizes,
        val_tasks=[dummy_task],
        inner_steps=1,
    )

    assert "m7_pinball" in metrics
    assert "m5_pinball" in metrics
    assert "m7_crps" in metrics
    assert "m5_crps" in metrics
    assert "crpss_vs_base" in metrics
    assert metrics["m7_pinball"] >= 0.0
    assert metrics["m7_crps"] >= 0.0
