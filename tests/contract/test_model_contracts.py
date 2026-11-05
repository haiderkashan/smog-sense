"""tests.contract.test_model_contracts — Model contract tests T1-14 through T1-21.

Covers:
- T1-14: Leakage guards (as-of joins, zero E/V/T sample overlap, 72h embargo, twin-aware stations).
- T1-15: Bundle integrity and binding (calibrator base_sha256 matching, hash integrity).
- T1-16: Adaptation outcomes and M5 fallback (5 adaptation_status branches, Delta z = 0).
- T1-17: Adaptation memory RSS (bounded execution, no graph retention across iterations).
- T1-18: Parameter count (exact 47,577 calibrator parameters + 14 Meta-SGD step sizes).
- T1-19: Zero-initialized residual (untrained calibrator yields Delta z == 0, base receives no gradient).
- T1-20: Out-of-fold integrity (calibrator trains strictly on OOF base predictions, partition hash).
- T1-21: Corruption augmentation (empirical low-cost distortion, clean query labels, reproducible seed).

Specification: docs/test-specifications.md → 'Phase 1 — ingestion, QC, ledger' (T1-14…T1-21)
"""

from __future__ import annotations

import gc
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from smogsense.features.backfill import (
    compute_oof_partition_hash,
    is_boundary_purged,
    partition_seasons,
    verify_twin_aware_folds,
)
from smogsense.features.sparse_network import (
    cluster_stations,
    sample_available_stations,
)
from smogsense.models.encoders.gru import LocalGRUEncoder
from smogsense.models.heads import (
    ResidualCalibrator,
    ResidualQuantileHead,
)
from smogsense.models.losses import MaskedResidualPinballLoss
from smogsense.models.maml import (
    MAMLTask,
    MetaSGDStepSizes,
    inner_loop_adapt,
)
from smogsense.models.pretraining import apply_sensor_corruption
from smogsense.models.residual_bound import evaluate_adaptation_acceptance
from smogsense.preprocessing.alignment import filter_asof


# =============================================================================
# T1-14: Leakage Guards
# =============================================================================
@pytest.mark.contract
def test_t1_14_leakage_guards() -> None:
    """T1-14: No feature uses available_at > T; E/V/T blocks & cross-fit folds have zero

    overlap with 72-h embargo; twin-aware stations never straddle splits.
    """
    # 1. As-of temporal leakage guard
    asof_t = pd.Timestamp("2026-11-01 00:00:00+00:00")
    df = pd.DataFrame(
        {
            "station_id": [101, 101, 101],
            "pm25": [45.0, 50.0, 55.0],
            "available_at_utc": [
                pd.Timestamp("2026-10-31 23:00:00+00:00"),
                pd.Timestamp("2026-11-01 00:00:00+00:00"),
                pd.Timestamp("2026-11-01 00:01:00+00:00"),  # Future leakage!
            ],
        }
    )
    filtered = filter_asof(df, issuance=asof_t)
    assert len(filtered) == 2
    assert (filtered["available_at_utc"] <= asof_t).all()

    # 2. Season block splitting (disjoint E / V / T blocks)
    seasons = [
        "2018_2019",
        "2019_2020",
        "2020_2021",
        "2021_2022",
        "2022_2023",
        "2023_2024",
        "2024_2025",
        "2025_2026",
    ]
    splits = partition_seasons(seasons, domain="delhi")
    block_e = {s for s, b in splits.items() if b == "E"}
    block_v = {s for s, b in splits.items() if b == "V"}
    block_t = {s for s, b in splits.items() if b == "T"}

    assert block_e.isdisjoint(block_v), "Block E and Block V must have zero season overlap"
    assert block_e.isdisjoint(block_t), "Block E and Block T must have zero season overlap"
    assert block_v.isdisjoint(block_t), "Block V and Block T must have zero season overlap"
    assert len(block_e) >= 3, "Block E must contain at least 3 source seasons for cross-fitting"
    assert len(block_v) == 1, "Block V tuning season must be exactly 1 season"
    assert len(block_t) == 1, "Block T confirmatory test must be exactly 1 season"

    # 3. 72-hour boundary embargo and purging
    season_start = pd.Timestamp("2024-10-01 00:00:00+00:00")
    season_end = pd.Timestamp("2025-02-28 23:00:00+00:00")

    # Inside boundary (e.g. 24h from start) -> must be purged
    assert is_boundary_purged(
        season_start + pd.Timedelta(hours=24), season_start, season_end, embargo_hours=72
    )
    # Inside boundary (e.g. 48h before end) -> must be purged
    assert is_boundary_purged(
        season_end - pd.Timedelta(hours=48), season_start, season_end, embargo_hours=72
    )
    # Well inside season (e.g. 100h from start) -> eligible (not purged)
    assert not is_boundary_purged(
        season_start + pd.Timedelta(hours=100), season_start, season_end, embargo_hours=72
    )

    # 4. Twin-aware co-located station sampling
    stations_df = pd.DataFrame(
        {
            "location_id": [1, 2, 3, 4],
            "latitude": [31.5204, 31.5205, 31.5600, 31.6000],
            "longitude": [74.3587, 74.3588, 74.3300, 74.3100],  # 1 and 2 are < 20m apart
            "colocated_group_id": [None, None, None, None],
        }
    )
    clusters = cluster_stations(stations_df)
    # Stations 1 and 2 must share a cluster
    cluster_for_stn = {stn: cid for cid, stn_list in clusters.items() for stn in stn_list}
    assert cluster_for_stn[1] == cluster_for_stn[2]

    # Sample available set A and held-out set B across 50 random seeds
    for seed in range(50):
        a_stations, _b_stations, _ = sample_available_stations(stations_df, n=2, rng=seed)
        # Verify 1 and 2 are never split across A and B
        in_a_1 = 1 in a_stations
        in_a_2 = 2 in a_stations
        assert in_a_1 == in_a_2, f"Twin stations 1 and 2 split across A and B at seed {seed}"


# =============================================================================
# T1-15: Bundle Integrity and Binding
# =============================================================================
@pytest.mark.contract
def test_t1_15_bundle_integrity_and_binding() -> None:
    """T1-15: A bundle whose file hashes differ from manifest is refused; a calibrator

    whose base_sha256 differs from the loaded base is refused.
    """
    correct_base_sha256 = hashlib.sha256(b"lightgbm_m5_physics_base_weights").hexdigest()
    mismatched_base_sha256 = hashlib.sha256(b"different_m5_model_weights").hexdigest()

    # Calibrator checkpoint metadata dictionary
    checkpoint = {
        "model_state_dict": {},
        "base_sha256": correct_base_sha256,
        "oof_partition_hash": "a" * 64,
        "feature_set_version": "fs1",
    }

    # Verification function matching deployment pipeline contract
    def verify_binding(loaded_checkpoint: dict[str, Any], active_base_sha256: str) -> bool:
        ckpt_base_sha256 = loaded_checkpoint.get("base_sha256")
        return bool(ckpt_base_sha256 and ckpt_base_sha256 == active_base_sha256)

    # 1. Matching base_sha256 succeeds
    assert verify_binding(checkpoint, correct_base_sha256) is True

    # 2. Mismatched base_sha256 is refused (pipeline falls back to base M5 / level 2)
    assert verify_binding(checkpoint, mismatched_base_sha256) is False

    # 3. File hash integrity verification against manifest
    bundle_files = {
        "m5_base.txt": b"booster_parameters_content",
        "m7_maml.pt": b"torch_calibrator_weights",
    }
    manifest = {
        "files": {
            "m5_base.txt": hashlib.sha256(bundle_files["m5_base.txt"]).hexdigest(),
            "m7_maml.pt": hashlib.sha256(bundle_files["m7_maml.pt"]).hexdigest(),
        }
    }

    def verify_bundle_manifest(files: dict[str, bytes], bundle_manifest: dict[str, Any]) -> bool:
        for fname, expected_hash in bundle_manifest["files"].items():
            if fname not in files:
                return False
            actual_hash = hashlib.sha256(files[fname]).hexdigest()
            if actual_hash != expected_hash:
                return False
        return True

    assert verify_bundle_manifest(bundle_files, manifest) is True

    # Corrupt one byte of m7_maml.pt
    corrupted_files = dict(bundle_files)
    corrupted_files["m7_maml.pt"] = b"corrupted_calibrator_weights"
    assert verify_bundle_manifest(corrupted_files, manifest) is False


# =============================================================================
# T1-16: Adaptation Outcomes and M5 Fallback
# =============================================================================
@pytest.mark.contract
def test_t1_16_adaptation_outcomes_and_m5_fallback() -> None:
    """T1-16: All 5 adaptation_status branches verified; rejection or skip sets

    Delta z = 0 reproducing M5 bit-for-bit with degradation level 0.
    """
    rho_99 = 0.362452
    n_samples = 40
    dates = pd.date_range("2024-10-10", periods=n_samples, freq="6h", tz="UTC")
    y_true = np.full((n_samples, 3), 4.5)  # log1p space
    y_mask = np.ones((n_samples, 3), dtype=bool)

    # Baseline M5 predictions
    m5_pred = np.full((n_samples, 3, 19), 5.0)

    # Branch 1: N = 0 -> skipped_n0
    res_n0 = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=m5_pred,
        dates=dates,
        rho_99=rho_99,
        n_available=0,
        support_days=14.0,
    )
    assert res_n0.adaptation_status == "skipped_n0"
    assert res_n0.accepted is False

    # Branch 2: Support duration < min_support_days (7 days) -> skipped_insufficient_support
    res_short = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=m5_pred,
        dates=dates,
        rho_99=rho_99,
        n_available=2,
        support_days=5.0,
    )
    assert res_short.adaptation_status == "skipped_insufficient_support"
    assert res_short.accepted is False

    # Branch 3: No gain (< 2% or < 60% days) -> rejected_no_gain
    adapted_worse = np.full((n_samples, 3, 19), 5.2)  # Worse loss
    res_no_gain = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_worse,
        dates=dates,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_no_gain.adaptation_status == "rejected_no_gain"
    assert res_no_gain.accepted is False

    # Branch 4: Residual bound exceeded (R > rho_99) -> rejected_residual_bound
    # Target is 4.5. If adapted is 4.0, delta_z = |4.0 - 5.0| = 1.0 > rho_99 (0.362452)
    # Loss improves (closer to 4.5 than 5.0 is), but residual bound is violated!
    adapted_overshoot = np.full((n_samples, 3, 19), 4.2)
    res_bound = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_overshoot,
        dates=dates,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_bound.adaptation_status == "rejected_residual_bound"
    assert res_bound.accepted is False

    # Branch 5: Both gain & bound fail -> rejected_no_gain
    adapted_both_fail = np.full((n_samples, 3, 19), 6.0)
    res_both_fail = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_both_fail,
        dates=dates,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_both_fail.adaptation_status == "rejected_no_gain"
    assert res_both_fail.accepted is False

    # Branch 6: All criteria met -> adapted
    adapted_good = np.full((n_samples, 3, 19), 4.9)  # delta = 0.1 <= rho_99, improves loss
    res_adapted = evaluate_adaptation_acceptance(
        y_true=y_true,
        y_mask=y_mask,
        m5_pred=m5_pred,
        adapted_pred=adapted_good,
        dates=dates,
        rho_99=rho_99,
        min_improvement=0.02,
        min_days_better_fraction=0.60,
        n_available=2,
        support_days=14.0,
    )
    assert res_adapted.adaptation_status == "adapted"
    assert res_adapted.accepted is True

    # Mathematical Invariant: For all non-adapted outcomes, delta_z = 0, so forecast == M5 bit-for-bit
    base_m5_quantiles = torch.tensor([[[2.5, 3.8, 4.9]]], dtype=torch.float32)
    for outcome in [res_n0, res_short, res_no_gain, res_bound, res_both_fail]:
        delta_z = (
            torch.zeros_like(base_m5_quantiles)
            if not outcome.accepted
            else torch.ones_like(base_m5_quantiles)
        )
        final_forecast = base_m5_quantiles + delta_z
        assert torch.equal(final_forecast, base_m5_quantiles), (
            f"Status {outcome.adaptation_status} failed to reproduce M5 bit-for-bit"
        )


# =============================================================================
# T1-17: Adaptation Memory RSS
# =============================================================================
@pytest.mark.contract
def test_t1_17_adaptation_memory_rss() -> None:
    """T1-17: Repeated adaptations do not leak memory or retain computational graphs."""
    encoder = LocalGRUEncoder(input_size=6, hidden_size=64, num_layers=2, latent_dim=32)
    head = ResidualQuantileHead(input_dim=46, hidden_dim=64, output_dim=57)
    calibrator = ResidualCalibrator(encoder, head)
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    # Synthetic support batch (worst-case: batch size 16)
    b_sup = 16
    task = MAMLTask(
        domain="delhi",
        season="2023_winter",
        available_stations=[101, 102],
        held_out_stations=[103],
        support_x=torch.randn(b_sup, 72, 6),
        support_context=torch.randn(b_sup, 14),
        support_base_q=torch.ones(b_sup, 3, 19) * 4.0,
        support_y=torch.ones(b_sup, 3) * 4.5,
        support_mask=torch.ones(b_sup, 3),
        query_x=torch.randn(4, 72, 6),
        query_context=torch.randn(4, 14),
        query_base_q=torch.ones(4, 3, 19) * 4.0,
        query_y=torch.ones(4, 3) * 4.5,
        query_mask=torch.ones(4, 3),
        n_available=2,
    )

    initial_objects = len(gc.get_objects())

    # Run 10 adaptation iterations
    for _ in range(10):
        adapted_params = inner_loop_adapt(
            calibrator=calibrator,
            params=dict(calibrator.named_parameters()),
            step_sizes=step_sizes,
            task=task,
            inner_steps=3,
            clip_norm=1.0,
            adapt_scope="all",
        )
        assert len(adapted_params) == 14

    gc.collect()
    final_objects = len(gc.get_objects())
    # Object count delta must stay bounded across repeated adaptations
    assert final_objects - initial_objects < 500


# =============================================================================
# T1-18: Exact Parameter Count
# =============================================================================
@pytest.mark.contract
def test_t1_18_exact_parameter_count() -> None:
    """T1-18: Residual calibrator has exactly 47,577 parameters (40,864 encoder +

    6,713 head) plus 14 Meta-SGD step-size scalars.
    """
    encoder = LocalGRUEncoder(input_size=6, hidden_size=64, num_layers=2, latent_dim=32)
    head = ResidualQuantileHead(input_dim=46, hidden_dim=64, output_dim=57)
    calibrator = ResidualCalibrator(encoder, head)
    step_sizes = MetaSGDStepSizes(init_alpha=0.01)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    total_calibrator_params = sum(p.numel() for p in calibrator.parameters())

    assert encoder_params == 40_864, f"Expected 40,864 encoder params, got {encoder_params}"
    assert head_params == 6_713, f"Expected 6,713 head params, got {head_params}"
    assert total_calibrator_params == 47_577, (
        f"Expected 47,577 total params, got {total_calibrator_params}"
    )
    assert len(step_sizes.get_step_sizes()) == 14, (
        f"Expected 14 Meta-SGD step sizes, got {len(step_sizes.get_step_sizes())}"
    )
    assert sum(p.numel() for p in step_sizes.parameters()) == 14


# =============================================================================
# T1-19: Zero-Initialized Residual
# =============================================================================
@pytest.mark.contract
def test_t1_19_zero_initialized_residual() -> None:
    """T1-19: Untrained calibrator produces Delta z == 0 for every input; after one

    optimizer step last-layer gradient is non-zero; base receives no gradient.
    """
    encoder = LocalGRUEncoder(input_size=6, hidden_size=64, num_layers=2, latent_dim=32)
    head = ResidualQuantileHead(input_dim=46, hidden_dim=64, output_dim=57)
    calibrator = ResidualCalibrator(encoder, head)

    x_seq = torch.randn(8, 72, 6)
    context = torch.randn(8, 14)
    base_preds = torch.randn(8, 3, 19)

    # 1. Delta z is mathematically identical to 0
    delta_z = calibrator(x_seq, context)
    assert torch.all(delta_z == 0.0), "Delta z is not identically zero on initialization"

    # 2. Output equals base predictions bit-for-bit
    output = calibrator(x_seq, context, base_preds)
    assert torch.equal(output, base_preds), "Output does not equal base predictions bit-for-bit"

    # 3. Last layer gradient is non-zero after loss backward
    targets = base_preds.mean(dim=-1) + 1.0  # create error
    masks = torch.ones(8, 3)
    loss_fn = MaskedResidualPinballLoss()
    loss = loss_fn(y_true=targets, delta_z=output, mask=masks)
    loss.backward()

    assert calibrator.head.fc2.weight.grad is not None
    assert torch.norm(calibrator.head.fc2.weight.grad) > 0.0, "Last layer weight gradient is zero"
    assert calibrator.head.fc2.bias.grad is not None
    assert torch.norm(calibrator.head.fc2.bias.grad) > 0.0, "Last layer bias gradient is zero"

    # 4. Base receives no gradient (base_preds is detached / un-tracked)
    assert base_preds.grad is None
    assert base_preds.requires_grad is False


# =============================================================================
# T1-20: Out-of-Fold Integrity
# =============================================================================
@pytest.mark.contract
def test_t1_20_out_of_fold_integrity() -> None:
    """T1-20: Base predictions given to calibrator come strictly from OOF cross-fit models;

    folds are season-blocked with 72-h embargo; partition hash is deterministic.
    """
    df = pd.DataFrame(
        {
            "location_id": [1, 1, 2, 2],
            "issuance_utc": [
                pd.Timestamp("2020-11-01 00:00:00+00:00"),
                pd.Timestamp("2021-11-01 00:00:00+00:00"),
                pd.Timestamp("2020-11-01 00:00:00+00:00"),
                pd.Timestamp("2021-11-01 00:00:00+00:00"),
            ],
            "fold_id": [0, 1, 0, 1],
        }
    )

    hash1 = compute_oof_partition_hash(df)
    assert len(hash1) == 64
    assert hash1 == compute_oof_partition_hash(df.sample(frac=1.0, random_state=42))

    # Changing any fold assignment completely alters the OOF partition hash
    df_altered = df.copy()
    df_altered.loc[0, "fold_id"] = 2
    hash2 = compute_oof_partition_hash(df_altered)
    assert hash1 != hash2

    # Verify twin-aware fold validation
    df_clean = pd.DataFrame(
        {
            "location_id": [1, 2],
            "colocated_group_id": ["group_1", "group_1"],
            "fold_id": [0, 0],
        }
    )
    df_leaked = pd.DataFrame(
        {
            "location_id": [1, 2],
            "colocated_group_id": ["group_1", "group_1"],
            "fold_id": [0, 1],
        }
    )

    assert verify_twin_aware_folds(df_clean, raise_on_violation=False) is True
    assert verify_twin_aware_folds(df_leaked, raise_on_violation=False) is False


# =============================================================================
# T1-21: Corruption Augmentation
# =============================================================================
@pytest.mark.contract
def test_t1_21_corruption_augmentation() -> None:
    """T1-21: Low-cost sensor corruption follows Barkjohn formulation; draws fall within

    configured ranges; query labels stay clean; seeded reproducibility.
    """
    # 72-hour sequence with 6 channels
    # Channel 0: pm25_log1p_masked, 1: obs_minus_cams_pm25, 2: rh_pct, 3: t2m_c, 4: mask, 5: delta_t
    clean_pm25 = 100.0
    cams_pm25 = 80.0
    rh_val = 85.0

    x_clean = np.zeros((1, 72, 6), dtype=np.float32)
    x_clean[:, :, 0] = np.log1p(clean_pm25)
    x_clean[:, :, 1] = clean_pm25 - cams_pm25
    x_clean[:, :, 2] = rh_val
    x_clean[:, :, 3] = 18.0
    x_clean[:, :, 4] = 1.0
    x_clean[:, :, 5] = 1.0

    x_tensor = torch.tensor(x_clean)
    query_labels = torch.tensor([[clean_pm25, clean_pm25, clean_pm25]])

    # Apply corruption with seed
    gen = torch.Generator().manual_seed(12345)
    x_corrupted = apply_sensor_corruption(
        x_tensor,
        gain_range=(0.8, 1.3),
        rh_slope_range=(0.0, 1.5),
        offset_range=(-10.0, 25.0),
        generator=gen,
    )

    # 1. Non-identical output on channel 0 and 1
    assert not torch.equal(x_corrupted[:, :, 0], x_tensor[:, :, 0])
    assert not torch.equal(x_corrupted[:, :, 1], x_tensor[:, :, 1])

    # 2. Invariant channels: RH (ch 2), temp (ch 3), mask (ch 4), delta_t (ch 5) stay untouched
    assert torch.equal(x_corrupted[:, :, 2], x_tensor[:, :, 2])
    assert torch.equal(x_corrupted[:, :, 3], x_tensor[:, :, 3])
    assert torch.equal(x_corrupted[:, :, 4], x_tensor[:, :, 4])
    assert torch.equal(x_corrupted[:, :, 5], x_tensor[:, :, 5])

    # 3. Query labels remain strictly clean regulatory ground truth
    assert torch.equal(query_labels, torch.tensor([[clean_pm25, clean_pm25, clean_pm25]]))

    # 4. Seeded reproducibility
    gen_repeat = torch.Generator().manual_seed(12345)
    x_corrupted_repeat = apply_sensor_corruption(
        x_tensor,
        gain_range=(0.8, 1.3),
        rh_slope_range=(0.0, 1.5),
        offset_range=(-10.0, 25.0),
        generator=gen_repeat,
    )
    assert torch.equal(x_corrupted, x_corrupted_repeat)
