"""tests.unit.models.test_neural_stack — Unit tests for Phase 2 Task P2-06 neural components.

Tests:
1. Exact parameter counts: LocalGRUEncoder (40,864), ResidualQuantileHead (6,713), Combined (47,577).
2. Zero-initialization invariant: Delta z == 0 everywhere before adaptation (reproduces M5 base exactly).
3. Forward shapes: batched (B, 72, 6) -> (B, 32) -> (B, 46) -> (B, 3, 19) / (B, 57) and unbatched.
4. Monotone rearrangement: guarantees non-crossing quantiles and satisfies the rearrangement inequality.
5. Asymmetric pinball loss: verifies exact penalty ratios (e.g. 9:1 at tau=0.90) and median MAE relation.
6. Masked residual pinball loss: target validity masking, NaN-free reduction, base-residual composition.
7. Gradient propagation: end-to-end backpropagation through loss, head, and encoder.
8. Functional call compatibility: verifies torch.func compatibility for Meta-SGD inner loops.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.func import functional_call

from smogsense.models.encoders.gru import LocalGRUEncoder
from smogsense.models.heads import (
    CANONICAL_QUANTILES,
    ResidualCalibrator,
    ResidualQuantileHead,
    rearrange_quantiles,
)
from smogsense.models.losses import MaskedResidualPinballLoss, pinball_loss


# -----------------------------------------------------------------------------
# 1. Exact Parameter Count Verification
# -----------------------------------------------------------------------------
def test_encoder_exact_parameter_count() -> None:
    """Verify that LocalGRUEncoder has exactly 40,864 parameters (docs/ml-architecture.md §4)."""
    encoder = LocalGRUEncoder(
        input_size=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        latent_dim=32,
    )

    # Layer 1 GRU: 3 * (64*6 + 64*64 + 2*64) = 13,824
    l1_params = sum(
        p.numel() for name, p in encoder.named_parameters() if "gru" in name and ("_l0" in name)
    )
    assert l1_params == 13_824, f"Layer 1 parameters expected 13,824, got {l1_params}"

    # Layer 2 GRU: 3 * (64*64 + 64*64 + 2*64) = 24,960
    l2_params = sum(
        p.numel() for name, p in encoder.named_parameters() if "gru" in name and ("_l1" in name)
    )
    assert l2_params == 24_960, f"Layer 2 parameters expected 24,960, got {l2_params}"

    # Linear projection (64 -> 32): 32 * 64 + 32 = 2,080
    proj_params = sum(p.numel() for p in encoder.proj.parameters())
    assert proj_params == 2_080, f"Projection parameters expected 2,080, got {proj_params}"

    # Total: 13,824 + 24,960 + 2,080 = 40,864
    assert encoder.num_parameters == 40_864
    assert sum(p.numel() for p in encoder.parameters()) == 40_864


def test_head_exact_parameter_count() -> None:
    """Verify that ResidualQuantileHead has exactly 6,713 parameters (docs/ml-architecture.md §4-5)."""
    head = ResidualQuantileHead(
        input_dim=46,
        hidden_dim=64,
        output_dim=57,
        n_horizons=3,
        n_quantiles=19,
    )

    # Layer 1 (46 -> 64): 64 * 46 + 64 = 3,008
    fc1_params = sum(p.numel() for p in head.fc1.parameters())
    assert fc1_params == 3_008, f"Layer 1 parameters expected 3,008, got {fc1_params}"

    # Layer 2 (64 -> 57): 57 * 64 + 57 = 3,705
    fc2_params = sum(p.numel() for p in head.fc2.parameters())
    assert fc2_params == 3_705, f"Layer 2 parameters expected 3,705, got {fc2_params}"

    # Total: 3,008 + 3,705 = 6,713
    assert head.num_parameters == 6_713
    assert sum(p.numel() for p in head.parameters()) == 6_713


def test_combined_stack_parameter_count() -> None:
    """Verify total combined neural stack parameters: 40,864 + 6,713 = 47,577."""
    calibrator = ResidualCalibrator()
    assert calibrator.encoder.num_parameters == 40_864
    assert calibrator.head.num_parameters == 6_713
    assert calibrator.num_parameters == 47_577
    assert sum(p.numel() for p in calibrator.parameters()) == 47_577


# -----------------------------------------------------------------------------
# 2. Zero-Initialization Invariant
# -----------------------------------------------------------------------------
def test_zero_initialization_invariant() -> None:
    """Verify that final layer zero-init guarantees Delta z == 0 everywhere before adaptation."""
    head = ResidualQuantileHead()

    # Pass diverse random inputs: normal, large positive, large negative
    torch.manual_seed(42)
    inputs = [
        torch.randn(16, 46),
        torch.ones(8, 46) * 100.0,
        torch.ones(8, 46) * -50.0,
        torch.zeros(4, 46),
    ]

    for u in inputs:
        delta_z = head(u, reshape=True)
        assert torch.all(delta_z == 0.0), "Head output must be identically zero on initialization"
        assert delta_z.shape == (u.shape[0], 3, 19)

        delta_z_flat = head(u, reshape=False)
        assert torch.all(delta_z_flat == 0.0)
        assert delta_z_flat.shape == (u.shape[0], 57)


def test_calibrator_reproduces_m5_at_initialization() -> None:
    """Verify that unadapted calibrator reproduces base predictions bit-for-bit (ADR-024)."""
    calibrator = ResidualCalibrator()
    batch_size = 5

    x_seq = torch.randn(batch_size, 72, 6)
    context_tab = torch.randn(batch_size, 14)
    base_quantiles = torch.rand(batch_size, 3, 19) + 2.0  # arbitrary log1p values

    q_pred = calibrator(x_seq, context_tab, base_quantiles=base_quantiles)
    # Since Delta z == 0, q_pred must equal base_quantiles exactly
    assert torch.all(q_pred == base_quantiles)


# -----------------------------------------------------------------------------
# 3. Forward Shapes and Dimension Consistency
# -----------------------------------------------------------------------------
def test_encoder_forward_shapes() -> None:
    """Verify forward shapes for batched and unbatched inputs to LocalGRUEncoder."""
    encoder = LocalGRUEncoder()

    # Batched input: (B, 72, 6) -> (B, 32)
    x_batched = torch.randn(8, 72, 6)
    z_batched = encoder(x_batched)
    assert z_batched.shape == (8, 32)

    # Unbatched input: (72, 6) -> (32,)
    x_unbatched = torch.randn(72, 6)
    z_unbatched = encoder(x_unbatched)
    assert z_unbatched.shape == (32,)


def test_head_forward_shapes_and_options() -> None:
    """Verify forward shapes and reshaping options for ResidualQuantileHead."""
    head = ResidualQuantileHead()

    # Batched default: (B, 46) -> (B, 3, 19)
    u_batched = torch.randn(10, 46)
    out_grid = head(u_batched, reshape=True)
    assert out_grid.shape == (10, 3, 19)

    # Batched flat: (B, 46) -> (B, 57)
    out_flat = head(u_batched, reshape=False)
    assert out_flat.shape == (10, 57)

    # Unbatched default: (46,) -> (3, 19)
    u_unbatched = torch.randn(46)
    out_single_grid = head(u_unbatched, reshape=True)
    assert out_single_grid.shape == (3, 19)

    # Unbatched flat: (46,) -> (57,)
    out_single_flat = head(u_unbatched, reshape=False)
    assert out_single_flat.shape == (57,)


def test_head_invalid_dimensions() -> None:
    """Verify that mismatch between horizons, quantiles, and output_dim raises ValueError."""
    with pytest.raises(ValueError, match="must equal output_dim"):
        ResidualQuantileHead(output_dim=50, n_horizons=3, n_quantiles=19)


# -----------------------------------------------------------------------------
# 4. Monotone Rearrangement
# -----------------------------------------------------------------------------
def test_rearrange_quantiles_torch() -> None:
    """Verify that rearrange_quantiles restores monotonicity in PyTorch tensors."""
    # Synthetic crossing quantiles
    torch.manual_seed(42)
    crossed = torch.tensor(
        [
            [10.0, 8.0, 15.0, 12.0, 20.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    rearranged = rearrange_quantiles(crossed, dim=-1)

    # Each row must be non-decreasing
    assert torch.all(rearranged[:, 1:] >= rearranged[:, :-1])
    # Exact sorted values
    expected = torch.tensor(
        [
            [8.0, 10.0, 12.0, 15.0, 20.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ]
    )
    assert torch.all(rearranged == expected)


def test_rearrange_quantiles_numpy() -> None:
    """Verify that rearrange_quantiles restores monotonicity in NumPy arrays."""
    crossed = np.array(
        [
            [10.0, 8.0, 15.0, 12.0, 20.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    rearranged = rearrange_quantiles(crossed, dim=-1)
    assert isinstance(rearranged, np.ndarray)
    assert np.all(rearranged[:, 1:] >= rearranged[:, :-1])


def test_rearrangement_inequality_on_pinball() -> None:
    """Verify Chernozhukov et al. (2010): sorting never increases total pinball loss."""
    torch.manual_seed(123)
    y_true = torch.tensor([12.0])
    taus = torch.tensor([0.10, 0.50, 0.90])

    # Inverted / crossing predictions: q_0.10 > q_0.50 > q_0.90
    q_crossed = torch.tensor([[18.0, 14.0, 8.0]])
    loss_crossed = pinball_loss(y_true, q_crossed, taus).sum()

    # Rearranged: q_0.10 <= q_0.50 <= q_0.90
    q_sorted = rearrange_quantiles(q_crossed, dim=-1)
    loss_sorted = pinball_loss(y_true, q_sorted, taus).sum()

    assert loss_sorted <= loss_crossed, (
        f"Rearranged loss {loss_sorted.item()} should be <= crossed loss {loss_crossed.item()}"
    )


# -----------------------------------------------------------------------------
# 5. Asymmetric Pinball Loss Properties
# -----------------------------------------------------------------------------
def test_pinball_loss_exact_values() -> None:
    """Verify pinball loss against exact analytical formula: max(tau*u, (tau-1)*u)."""
    y_true = torch.tensor([100.0])
    y_pred = torch.tensor([80.0])  # under-prediction by 20: u = +20

    # For tau = 0.90: under-prediction penalty is tau * u = 0.90 * 20 = 18.0
    loss_under = pinball_loss(y_true, y_pred, taus=0.90)
    assert torch.isclose(loss_under, torch.tensor([18.0]))

    # Over-prediction by 20: y_pred = 120 -> u = -20
    y_pred_over = torch.tensor([120.0])
    # For tau = 0.90: over-prediction penalty is (1 - tau) * |u| = 0.10 * 20 = 2.0
    loss_over = pinball_loss(y_true, y_pred_over, taus=0.90)
    assert torch.isclose(loss_over, torch.tensor([2.0]))

    # Asymmetry ratio under/over at tau=0.90 must be exactly 9:1
    ratio = (loss_under / loss_over).item()
    assert pytest.approx(ratio, rel=1e-5) == 9.0


def test_pinball_loss_median_equals_half_mae() -> None:
    """Verify that at tau=0.50, pinball loss equals exactly 0.5 * MAE."""
    torch.manual_seed(42)
    y_true = torch.randn(20)
    y_pred = torch.randn(20)

    loss_median = pinball_loss(y_true, y_pred, taus=0.50)
    mae = torch.abs(y_true - y_pred) * 0.5
    assert torch.allclose(loss_median, mae)


def test_pinball_loss_multi_quantile_broadcasting() -> None:
    """Verify multi-quantile evaluation with broadcasting over batch and horizons."""
    y_true = torch.tensor([[10.0, 20.0, 30.0]])  # (1, 3)
    # y_pred: (1, 3, 19)
    y_pred = torch.ones(1, 3, 19) * 15.0
    taus = torch.tensor(CANONICAL_QUANTILES)

    loss = pinball_loss(y_true.unsqueeze(-1), y_pred, taus)
    assert loss.shape == (1, 3, 19)
    assert torch.all(loss >= 0.0)


# -----------------------------------------------------------------------------
# 6. Masked Residual Pinball Loss
# -----------------------------------------------------------------------------
def test_masked_residual_pinball_loss_basic() -> None:
    """Verify MaskedResidualPinballLoss on valid inputs."""
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    batch_size = 4
    y_true = torch.rand(batch_size, 3) + 3.0
    delta_z = torch.randn(batch_size, 3, 19) * 0.1
    z_base = torch.rand(batch_size, 3, 19) + 3.0
    mask = torch.ones(batch_size, 3)

    loss = criterion(y_true, delta_z, z_base=z_base, mask=mask)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_masked_residual_pinball_loss_masking_effect() -> None:
    """Verify that masked-out targets (mask=0) contribute zero to loss and gradient."""
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    y_true = torch.tensor([[5.0, 5.0, 5.0]])  # B=1, 3 horizons
    z_base = torch.zeros(1, 3, 19)
    delta_z = torch.ones(1, 3, 19, requires_grad=True)

    # Mask only the first horizon valid: mask = [[1.0, 0.0, 0.0]]
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    loss = criterion(y_true, delta_z, z_base=z_base, mask=mask)
    loss.backward()

    # Gradients for horizon 0 must be non-zero; horizons 1 and 2 must have zero gradient
    assert delta_z.grad is not None
    assert torch.any(delta_z.grad[:, 0, :] != 0.0)
    assert torch.all(delta_z.grad[:, 1, :] == 0.0)
    assert torch.all(delta_z.grad[:, 2, :] == 0.0)


def test_masked_residual_pinball_loss_all_zero_mask_safety() -> None:
    """Verify that if all targets in a batch are invalid (mask sum = 0), loss is safe zero."""
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    y_true = torch.rand(2, 3)
    delta_z = torch.rand(2, 3, 19)
    mask = torch.zeros(2, 3)  # all invalid

    loss = criterion(y_true, delta_z, mask=mask)
    assert not torch.isnan(loss)
    assert loss.item() == 0.0


def test_masked_residual_pinball_loss_composition_equivalence() -> None:
    """Verify equivalence between (y_true, delta_z, z_base) and (y_true, delta_z + z_base)."""
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    y_true = torch.rand(3, 3)
    delta_z = torch.randn(3, 3, 19)
    z_base = torch.rand(3, 3, 19)

    loss_composite = criterion(y_true, delta_z, z_base=z_base)
    loss_direct = criterion(y_true, delta_z + z_base)

    assert torch.isclose(loss_composite, loss_direct)


# -----------------------------------------------------------------------------
# 7. End-to-End Gradient Propagation
# -----------------------------------------------------------------------------
def test_end_to_end_gradient_flow() -> None:
    """Verify that gradients propagate cleanly from loss through head and encoder."""
    encoder = LocalGRUEncoder()
    # Use zero_init=False to test full chain-rule backpropagation to inputs
    head = ResidualQuantileHead(zero_init=False)
    criterion = MaskedResidualPinballLoss()

    batch_size = 4
    x_seq = torch.randn(batch_size, 72, 6, requires_grad=True)
    context_tab = torch.randn(batch_size, 14, requires_grad=True)
    y_true = torch.rand(batch_size, 3) + 3.0
    z_base = torch.rand(batch_size, 3, 19) + 3.0
    mask = torch.ones(batch_size, 3)

    # Forward through stack
    z = encoder(x_seq)
    u = torch.cat([z, context_tab], dim=-1)
    delta_z = head(u, reshape=True)

    loss = criterion(y_true, delta_z, z_base=z_base, mask=mask)
    loss.backward()

    # Check input gradients
    assert x_seq.grad is not None
    assert not torch.isnan(x_seq.grad).any()
    assert torch.any(x_seq.grad != 0.0)

    # Check encoder parameter gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Encoder {name} missing gradient"
        assert not torch.isnan(param.grad).any(), f"Encoder {name} has NaN gradient"
        assert torch.any(param.grad != 0.0), f"Encoder {name} has zero gradient"

    # Check head parameter gradients
    for name, param in head.named_parameters():
        assert param.grad is not None, f"Head {name} missing gradient"
        assert not torch.isnan(param.grad).any(), f"Head {name} has NaN gradient"
        assert torch.any(param.grad != 0.0), f"Head {name} has zero gradient"


def test_zero_init_head_gradient_and_update() -> None:
    """Verify that a zero-initialized head receives non-zero gradients on fc2 to begin learning."""
    head = ResidualQuantileHead(zero_init=True)
    criterion = MaskedResidualPinballLoss()

    u = torch.randn(4, 46)
    y_true = torch.ones(4, 3) * 5.0
    z_base = torch.ones(4, 3, 19) * 2.0  # Under-predicting base

    delta_z = head(u, reshape=True)
    loss = criterion(y_true, delta_z, z_base=z_base)
    loss.backward()

    # fc2 weight and bias must receive non-zero gradients even though initialized to 0
    assert head.fc2.weight.grad is not None
    assert torch.any(head.fc2.weight.grad != 0.0)
    assert head.fc2.bias.grad is not None
    assert torch.any(head.fc2.bias.grad != 0.0)


# -----------------------------------------------------------------------------
# 8. Functional Call Compatibility (torch.func for Meta-SGD)
# -----------------------------------------------------------------------------
def test_functional_call_compatibility() -> None:
    """Verify that LocalGRUEncoder and ResidualQuantileHead work with torch.func.functional_call."""
    encoder = LocalGRUEncoder()
    head = ResidualQuantileHead()

    encoder_params = dict(encoder.named_parameters())
    head_params = dict(head.named_parameters())

    x_seq = torch.randn(2, 72, 6)
    z_out = functional_call(encoder, encoder_params, (x_seq,))
    assert z_out.shape == (2, 32)

    u = torch.cat([z_out, torch.randn(2, 14)], dim=-1)
    delta_out = functional_call(head, head_params, (u,))
    assert delta_out.shape == (2, 3, 19)
    assert torch.all(delta_out == 0.0)
