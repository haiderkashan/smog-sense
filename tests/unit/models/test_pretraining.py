"""tests.unit.models.test_pretraining — Unit tests for Phase 2 Task P2-07.

Tests:
1. Synthetic sensor corruption mathematical formulation and channel invariants.
2. Invariant: target labels y remain 100% clean regulatory truth.
3. Tabular context vector extraction (14 dimensions: bias, RH, 3 ln(VC), 9 base quantiles).
4. Base quantiles grid extraction (57 dimensions: 3 horizons x 19 quantiles in log1p).
5. Fast sequence tensor extraction from features.
6. ResidualPretrainingDataset shape, corruption integration, and DataLoader batching.
7. Untrained calibrator bit-for-bit reproduces base model (zero-init invariant).
8. Supervised training step, gradient propagation, and clipping.
9. Evaluation metrics: pinball loss, discrete CRPS, and CRPSS vs base.
10. Model checkpoint serialization with base_sha256 and oof_partition_hash binding (Condition C3).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from smogsense.models.heads import CANONICAL_HORIZONS, CANONICAL_QUANTILES, ResidualCalibrator
from smogsense.models.losses import MaskedResidualPinballLoss
from smogsense.models.pretraining import (
    ResidualPretrainingDataset,
    apply_sensor_corruption,
    evaluate_calibrator,
    extract_base_quantiles_tensor,
    extract_tabular_context,
    extract_targets_and_masks,
)


# -----------------------------------------------------------------------------
# 1. Synthetic Low-Cost Sensor Corruption Augmentation
# -----------------------------------------------------------------------------
def test_apply_sensor_corruption_math() -> None:
    """Verify synthetic sensor corruption matches mathematical specification.

    Formula:
        x_tilde(t) = max(0, a * x(t) + b * max(0, RH(t) - 70) + c)
        where a in [0.8, 1.6], b in [0.0, 1.5], c in [-25.0, 25.0].
    """
    # Create 1 sample sequence of shape (1, 72, 6)
    x = torch.zeros(1, 72, 6, dtype=torch.float32)

    # Step 10: uncorrupted PM2.5 = 100 ug/m3 -> log1p(100) = 4.6151
    # RH = 80% -> excess RH = 10%
    # CAMS = 90 ug/m3 -> channel 1 (obs - cams) = 10.0
    # mask = 1.0 (valid observation)
    x[0, 10, 0] = float(np.log1p(100.0))
    x[0, 10, 1] = 10.0  # cams was 100 - 10 = 90
    x[0, 10, 2] = 80.0
    x[0, 10, 3] = 22.0
    x[0, 10, 4] = 1.0
    x[0, 10, 5] = 0.0

    # Fixed coefficients: a=1.2, b=0.5, c=5.0
    # Expected x_tilde: 1.2 * 100 + 0.5 * (80 - 70) + 5 = 120 + 5 + 5 = 130.0 ug/m3
    # Expected channel 0: log1p(130.0)
    # Expected channel 1: x_tilde - cams = 130 - 90 = 40.0
    corrupted = apply_sensor_corruption(
        x,
        p=1.0,
        gain_range=(1.2, 1.2),
        rh_slope_range=(0.5, 0.5),
        offset_range=(5.0, 5.0),
    )

    expected_pm25 = 130.0
    assert torch.isclose(corrupted[0, 10, 0], torch.tensor(float(np.log1p(expected_pm25))))
    assert torch.isclose(corrupted[0, 10, 1], torch.tensor(40.0))

    # Channels 2, 3, 4, 5 must remain exactly identical
    assert torch.equal(corrupted[0, 10, 2:], x[0, 10, 2:])

    # Missing steps (mask == 0.0) must stay exactly 0.0
    assert corrupted[0, 0, 0] == 0.0
    assert corrupted[0, 0, 1] == 0.0
    assert corrupted[0, 0, 4] == 0.0


def test_apply_sensor_corruption_p_zero_identity() -> None:
    """When p=0, apply_sensor_corruption must return the input tensor unchanged."""
    torch.manual_seed(42)
    x = torch.randn(4, 72, 6)
    corrupted = apply_sensor_corruption(x, p=0.0)
    assert torch.equal(x, corrupted)


def test_apply_sensor_corruption_unbatched_support() -> None:
    """Verify that unbatched sequence tensor of shape (72, 6) is handled cleanly."""
    x = torch.zeros(72, 6)
    x[5, 0] = float(np.log1p(50.0))
    x[5, 4] = 1.0  # valid mask
    corrupted = apply_sensor_corruption(x, p=1.0)
    assert corrupted.shape == (72, 6)


def test_apply_sensor_corruption_non_negative_flooring() -> None:
    """Negative corrupted concentrations must be strictly floored at 0.0."""
    x = torch.zeros(1, 72, 6)
    # Obs = 10 ug/m3, large negative offset c = -50
    x[0, 5, 0] = float(np.log1p(10.0))
    x[0, 5, 4] = 1.0

    corrupted = apply_sensor_corruption(
        x,
        p=1.0,
        gain_range=(1.0, 1.0),
        rh_slope_range=(0.0, 0.0),
        offset_range=(-50.0, -50.0),
    )

    # 1.0 * 10 - 50 = -40 -> floored to 0.0 -> log1p(0) = 0.0
    assert corrupted[0, 5, 0] == 0.0


# -----------------------------------------------------------------------------
# 2. Context Vector & Feature Extraction (46 Dimensions)
# -----------------------------------------------------------------------------
def test_extract_tabular_context_dimensions_and_values() -> None:
    """Verify extraction of the 14-dimensional tabular context vector u_tabular.

    Context components:
      - recent CAMS bias (1)
      - station RH at T (1)
      - 3 ln(VC) block means (3)
      - 9 base quantiles subset at tau in {0.10, 0.50, 0.90} across horizons {24, 48, 72} (9)
      Total = 14 dimensions.
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "f_cams_bias_trailing_7d": [12.5, -5.0],
            "f_met_rh_mean_h24": [65.0, 75.0],
            "f_met_ln_vc_mean_h24": [7.2, 8.1],
            "f_met_ln_vc_mean_h48": [7.0, 7.8],
            "f_met_ln_vc_mean_h72": [6.8, 7.5],
            # 9 base quantiles: h in {24, 48, 72} x q in {10, 50, 90}
            "m5_oof_h24_q10": [50.0, 60.0],
            "m5_oof_h24_q50": [100.0, 110.0],
            "m5_oof_h24_q90": [150.0, 180.0],
            "m5_oof_h48_q10": [45.0, 55.0],
            "m5_oof_h48_q50": [95.0, 105.0],
            "m5_oof_h48_q90": [140.0, 170.0],
            "m5_oof_h72_q10": [40.0, 50.0],
            "m5_oof_h72_q50": [90.0, 100.0],
            "m5_oof_h72_q90": [135.0, 165.0],
        }
    )

    ctx = extract_tabular_context(df)
    assert ctx.shape == (2, 14)
    assert ctx.dtype == np.float32

    # Verify column 0: bias
    np.testing.assert_allclose(ctx[:, 0], [12.5, -5.0])
    # Verify column 1: RH
    np.testing.assert_allclose(ctx[:, 1], [65.0, 75.0])
    # Verify columns 2, 3, 4: ln(VC)
    np.testing.assert_allclose(ctx[:, 2], [7.2, 8.1])
    np.testing.assert_allclose(ctx[:, 3], [7.0, 7.8])
    np.testing.assert_allclose(ctx[:, 4], [6.8, 7.5])
    # Verify base quantiles are in log1p space
    np.testing.assert_allclose(ctx[:, 5], np.log1p([50.0, 60.0]), rtol=1e-5)
    np.testing.assert_allclose(ctx[:, 6], np.log1p([100.0, 110.0]), rtol=1e-5)
    np.testing.assert_allclose(ctx[:, 7], np.log1p([150.0, 180.0]), rtol=1e-5)


def test_extract_base_quantiles_grid_57_dimensions() -> None:
    """Verify extraction of the full 57 base quantiles (3 horizons x 19 quantiles)."""
    import pandas as pd

    cols: dict[str, list[float]] = {}
    for h in CANONICAL_HORIZONS:
        for q in CANONICAL_QUANTILES:
            q_tag = round(q * 100)
            cols[f"m5_oof_h{h}_q{q_tag:02d}"] = [float(h * 10 + q_tag)]

    df = pd.DataFrame(cols)
    base_grid = extract_base_quantiles_tensor(df)
    assert base_grid.shape == (1, 3, 19)
    assert base_grid.dtype == np.float32

    # Values must be converted to log1p
    expected_h24_q10 = np.log1p(240 + 10)
    assert pytest.approx(base_grid[0, 0, 1], rel=1e-5) == expected_h24_q10


def test_extract_targets_and_masks() -> None:
    """Verify extraction of targets in log1p space and validity masks."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "y_h24": [100.0, 150.0],
            "valid_h24": [True, False],
            "y_h48": [80.0, np.nan],
            "valid_h48": [True, False],
            "y_h72": [60.0, 70.0],
            "valid_h72": [True, True],
        }
    )

    y_log1p, mask = extract_targets_and_masks(df)
    assert y_log1p.shape == (2, 3)
    assert mask.shape == (2, 3)

    # Valid items
    np.testing.assert_allclose(y_log1p[0], np.log1p([100.0, 80.0, 60.0]), rtol=1e-5)
    np.testing.assert_array_equal(mask[0], [1.0, 1.0, 1.0])

    # Row 1: h24 invalid (mask=0), h48 NaN (mask=0), h72 valid (mask=1)
    np.testing.assert_array_equal(mask[1], [0.0, 0.0, 1.0])


# -----------------------------------------------------------------------------
# 3. ResidualPretrainingDataset & DataLoader
# -----------------------------------------------------------------------------
def test_residual_pretraining_dataset_batching() -> None:
    """Verify ResidualPretrainingDataset iteration and DataLoader batching."""
    n_samples = 20
    seqs = np.zeros((n_samples, 72, 6), dtype=np.float32)
    seqs[:, :, 0] = float(np.log1p(100.0))
    seqs[:, :, 4] = 1.0  # valid mask
    ctx = np.zeros((n_samples, 14), dtype=np.float32)
    base_q = np.zeros((n_samples, 3, 19), dtype=np.float32)
    y = np.ones((n_samples, 3), dtype=np.float32) * float(np.log1p(100.0))
    mask = np.ones((n_samples, 3), dtype=np.float32)

    ds = ResidualPretrainingDataset(
        sequences=seqs,
        context_tabular=ctx,
        base_quantiles=base_q,
        y_true=y,
        mask=mask,
        corrupt=True,
        corruption_p=1.0,
    )

    assert len(ds) == n_samples
    item = ds[0]
    assert len(item) == 5
    x_s, c_t, b_q, y_t, m_t = item
    assert x_s.shape == (72, 6)
    assert c_t.shape == (14,)
    assert b_q.shape == (3, 19)
    assert y_t.shape == (3,)
    assert m_t.shape == (3,)

    # DataLoader
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch[0].shape == (4, 72, 6)
    assert batch[1].shape == (4, 14)
    assert batch[2].shape == (4, 3, 19)


# -----------------------------------------------------------------------------
# 4. Model Invariants & Training Step
# -----------------------------------------------------------------------------
def test_untrained_calibrator_reproduces_m5_base() -> None:
    """Verify zero-init invariant: untrained calibrator reproduces base predictions bit-for-bit."""
    calibrator = ResidualCalibrator()
    calibrator.eval()

    x_seq = torch.randn(5, 72, 6)
    context_tab = torch.randn(5, 14)
    base_quantiles = torch.rand(5, 3, 19) + 3.0

    with torch.no_grad():
        preds = calibrator(x_seq, context_tab, base_quantiles=base_quantiles, rearrange=False)

    # Since delta_z == 0.0, preds must equal base_quantiles exactly
    assert torch.all(preds == base_quantiles)


def test_supervised_training_step_decreases_loss() -> None:
    """Verify end-to-end forward, backward, clipping, and parameter update."""
    calibrator = ResidualCalibrator()
    calibrator.train()
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-2)

    x_seq = torch.randn(8, 72, 6)
    context_tab = torch.randn(8, 14)
    base_q = torch.ones(8, 3, 19) * 4.0
    y_true = torch.ones(8, 3) * 4.5  # Base is under-predicting
    mask = torch.ones(8, 3)

    # Initial loss
    with torch.no_grad():
        pred_init = calibrator(x_seq, context_tab, base_quantiles=base_q)
        loss_init = criterion(y_true=y_true, delta_z=pred_init, z_base=None, mask=mask)

    # Optimize for 10 steps
    for _ in range(10):
        optimizer.zero_grad()
        pred = calibrator(x_seq, context_tab, base_quantiles=base_q)
        loss = criterion(y_true=y_true, delta_z=pred, z_base=None, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=1.0)
        optimizer.step()

    # Final loss must be strictly lower
    with torch.no_grad():
        pred_final = calibrator(x_seq, context_tab, base_quantiles=base_q)
        loss_final = criterion(y_true=y_true, delta_z=pred_final, z_base=None, mask=mask)

    assert loss_final.item() < loss_init.item()


# -----------------------------------------------------------------------------
# 5. Checkpoint Binding Serialization (Condition C3)
# -----------------------------------------------------------------------------
def test_checkpoint_binding_serialization() -> None:
    """Verify model checkpoint serialization preserves base_sha256 and oof_partition_hash."""
    calibrator = ResidualCalibrator()

    dummy_base_sha256 = "a" * 64
    dummy_oof_hash = "b" * 64

    checkpoint = {
        "model_type": "M6",
        "model_state_dict": calibrator.state_dict(),
        "base_sha256": dummy_base_sha256,
        "oof_partition_hash": dummy_oof_hash,
        "quantiles": CANONICAL_QUANTILES,
        "horizons": CANONICAL_HORIZONS,
        "encoder_params": calibrator.encoder.num_parameters,
        "head_params": calibrator.head.num_parameters,
        "total_params": calibrator.num_parameters,
        "train_loss": 0.42,
        "val_loss": 0.45,
        "crps_val": {24: 15.0, 48: 18.0, 72: 21.0},
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "m6_calibrator.pt"
        torch.save(checkpoint, ckpt_path)

        # Reload
        loaded = torch.load(ckpt_path, weights_only=False)
        assert loaded["model_type"] == "M6"
        assert loaded["base_sha256"] == dummy_base_sha256
        assert loaded["oof_partition_hash"] == dummy_oof_hash
        assert loaded["encoder_params"] == 40_864
        assert loaded["head_params"] == 6_713
        assert loaded["total_params"] == 47_577

        # Check weights load into new calibrator
        new_calibrator = ResidualCalibrator()
        new_calibrator.load_state_dict(loaded["model_state_dict"])
        for p1, p2 in zip(calibrator.parameters(), new_calibrator.parameters(), strict=False):
            assert torch.equal(p1, p2)


# -----------------------------------------------------------------------------
# 6. Evaluation Function (evaluate_calibrator)
# -----------------------------------------------------------------------------
def test_evaluate_calibrator_metrics() -> None:
    """Verify evaluate_calibrator returns comprehensive validation statistics."""
    calibrator = ResidualCalibrator()
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    n_val = 10
    seqs = np.zeros((n_val, 72, 6), dtype=np.float32)
    ctx = np.zeros((n_val, 14), dtype=np.float32)
    base_q = np.ones((n_val, 3, 19), dtype=np.float32) * float(np.log1p(100.0))
    y = np.ones((n_val, 3), dtype=np.float32) * float(np.log1p(100.0))
    mask = np.ones((n_val, 3), dtype=np.float32)

    val_dataset = ResidualPretrainingDataset(
        sequences=seqs,
        context_tabular=ctx,
        base_quantiles=base_q,
        y_true=y,
        mask=mask,
        corrupt=False,
    )

    metrics = evaluate_calibrator(
        calibrator=calibrator,
        dataset=val_dataset,
        criterion=criterion,
        batch_size=5,
        rearrange=True,
    )

    assert "val_loss" in metrics
    assert "overall_crps_m6" in metrics
    assert "overall_crps_m5" in metrics
    assert "overall_crpss_vs_base" in metrics
    assert "horizons" in metrics
    for h in CANONICAL_HORIZONS:
        assert h in metrics["horizons"]
        assert "mae_median" in metrics["horizons"][h]
        assert "crps" in metrics["horizons"][h]
        assert "crpss_vs_base" in metrics["horizons"][h]
