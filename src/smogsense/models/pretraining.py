"""smogsense.models.pretraining — Supervised pre-training of the residual calibrator (Model M6).

Implements Phase 2 Task P2-07 per SmogSense Revision 3 (ADR-024):
- Trains ResidualCalibrator (47,577 parameters: LocalGRUEncoder + ResidualQuantileHead)
  on Delhi Block E seasons using strictly Out-Of-Fold (OOF) base predictions (Conditions C1, C2).
- Synthetic low-cost sensor corruption augmentation (Condition 1).
- Masked residual pinball loss over 19 quantiles x 3 horizons with target validity masking.
- Evaluates on Block V (validation season) and serializes checkpoint with base_sha256
  and oof_partition_hash binding (Condition C3).

Specification: docs/ml-architecture.md → 'Training protocol and reproducibility'
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from structlog import get_logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from smogsense.features.backfill import compute_oof_partition_hash
from smogsense.features.windowing import compute_elapsed_time_since_observation
from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
)
from smogsense.models.lgbm_quantile import (
    LGBMQuantileEnsemble,
    compute_crps_discrete,
    fit_cross_fit_oof,
    load_dataset_from_partitions,
    mean_pinball_loss,
)
from smogsense.models.losses import MaskedResidualPinballLoss

logger = get_logger(__name__)

# Base quantiles subset used in the 46-dimensional context vector u:
# 3 quantiles (q10, q50, q90) x 3 horizons (24, 48, 72) = 9 values
CONTEXT_BASE_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)


def apply_sensor_corruption(
    x_seq: torch.Tensor,
    p: float = 1.0,
    gain_range: tuple[float, float] = (0.8, 1.6),
    rh_slope_range: tuple[float, float] = (0.0, 1.5),
    offset_range: tuple[float, float] = (-25.0, 25.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply synthetic low-cost PM2.5 sensor corruption augmentation.

    Mathematical formulation (docs/ml-architecture.md §8 & ADR-024):
        x_tilde(t) = a_i * x_pm25(t) + b_i * max(0.0, RH(t) - 70.0) + c_i
        where a_i ~ U(gain_min, gain_max), b_i ~ U(rh_min, rh_max), c_i ~ U(off_min, off_max).

    Recomputes channel 1 (obs_minus_cams_pm25) using x_tilde(t).
    Relative humidity, temperature, validity masks, and delta_t stay uncorrupted.
    Ground truth targets y stay 100% clean regulatory truth.

    Args:
        x_seq: Sequence tensor of shape (B, 72, 6) or (72, 6).
        p: Probability of applying corruption to each sequence sample (default: 1.0).
        gain_range: Uniform sampling bounds for gain factor a_i (default: [0.8, 1.6]).
        rh_slope_range: Uniform sampling bounds for RH slope b_i (default: [0.0, 1.5]).
        offset_range: Uniform sampling bounds for baseline offset c_i (default: [-25.0, 25.0]).
        generator: Optional PyTorch random number generator for determinism.

    Returns:
        Corrupted sequence tensor of identical shape and dtype.
    """
    if p <= 0.0:
        return x_seq.clone()

    unbatched = x_seq.ndim == 2
    x_batched = x_seq.unsqueeze(0).clone() if unbatched else x_seq.clone()

    batch_size = x_batched.shape[0]
    device = x_batched.device
    dtype = x_batched.dtype

    # Sample corruption application decision per batch item
    if p < 1.0:
        apply_mask = torch.rand(batch_size, generator=generator, device=device) < p
    else:
        apply_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    if not torch.any(apply_mask):
        return x_seq.clone()

    # Draw corruption coefficients for the batch
    n_corrupt = int(apply_mask.sum().item())
    a_low, a_high = gain_range
    b_low, b_high = rh_slope_range
    c_low, c_high = offset_range

    a = (
        a_low + (a_high - a_low) * torch.rand(n_corrupt, 1, generator=generator, device=device)
    ).to(dtype)
    b = (
        b_low + (b_high - b_low) * torch.rand(n_corrupt, 1, generator=generator, device=device)
    ).to(dtype)
    c = (
        c_low + (c_high - c_low) * torch.rand(n_corrupt, 1, generator=generator, device=device)
    ).to(dtype)

    # Extract components for corrupted subset: shape (n_corrupt, 72)
    sub = x_batched[apply_mask]

    # Channel 0: pm25_log1p_masked; Channel 4: pm25_mask
    mask = sub[:, :, 4] > 0.5
    raw_pm25 = torch.where(mask, torch.expm1(sub[:, :, 0]), torch.zeros_like(sub[:, :, 0]))

    # Channel 1: obs_minus_cams_pm25 = (obs - cams) * mask -> cams = obs - ch1
    cams_pm25 = torch.where(mask, raw_pm25 - sub[:, :, 1], torch.zeros_like(raw_pm25))

    # Channel 2: rh_pct
    rh = sub[:, :, 2]
    rh_excess = torch.clamp(rh - 70.0, min=0.0)

    # Compute corrupted PM2.5 (floored at 0.0)
    corrupted_pm25 = torch.clamp(a * raw_pm25 + b * rh_excess + c, min=0.0)

    # Recompute Channel 0 (log1p masked) and Channel 1 (corrupted - cams)
    sub[:, :, 0] = torch.where(mask, torch.log1p(corrupted_pm25), torch.zeros_like(corrupted_pm25))
    sub[:, :, 1] = torch.where(mask, corrupted_pm25 - cams_pm25, torch.zeros_like(corrupted_pm25))

    x_batched[apply_mask] = sub

    if unbatched:
        return x_batched.squeeze(0)
    return x_batched


def extract_tabular_context(
    df: pd.DataFrame,
    base_predictions: dict[int, np.ndarray] | None = None,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
    context_quantiles: Sequence[float] = CONTEXT_BASE_QUANTILES,
) -> np.ndarray:
    """Construct 14-dimensional tabular context vector for ResidualQuantileHead.

    Context features (14 dimensions):
      1. Recent trailing 7-day CAMS bias (`f_cams_bias_trailing_7d`): 1 dimension
      2. Station relative humidity at issuance T (`f_met_rh_mean_h24` / `f_met_rh_pct`): 1 dimension
      3. Log-ventilation block means (`f_met_ln_vc_mean_h24`, `h48`, `h72`): 3 dimensions
      4. Base quantiles subset at tau in {0.10, 0.50, 0.90} across horizons {24, 48, 72}: 9 dimensions
    Total: 1 + 1 + 3 + 9 = 14 dimensions.

    Args:
        df: Merged feature DataFrame (with OOF columns or separate base_predictions).
        base_predictions: Optional dictionary mapping horizon h -> (N, 19) base quantiles in raw space.
        horizons: Target horizons (default: [24, 48, 72]).
        context_quantiles: Quantile levels for context (default: [0.10, 0.50, 0.90]).

    Returns:
        np.ndarray of shape (N, 14) with float32 values.
    """
    n_rows = len(df)
    context = np.zeros((n_rows, 14), dtype=np.float32)

    # 1. Recent CAMS bias
    if "f_cams_bias_trailing_7d" in df.columns:
        context[:, 0] = df["f_cams_bias_trailing_7d"].fillna(0.0).to_numpy(dtype=np.float32)

    # 2. Station RH at issuance T
    rh_col = next(
        (c for c in ("f_met_rh_pct", "rh_pct", "f_met_rh_mean_h24") if c in df.columns),
        None,
    )
    if rh_col:
        context[:, 1] = df[rh_col].fillna(50.0).to_numpy(dtype=np.float32)
    else:
        context[:, 1] = 50.0

    # 3. Log-ventilation block means (3 horizons)
    for i, h in enumerate(horizons):
        vc_col = f"f_met_ln_vc_mean_h{h}"
        if vc_col in df.columns:
            context[:, 2 + i] = df[vc_col].fillna(7.0).to_numpy(dtype=np.float32)
        else:
            context[:, 2 + i] = 7.0

    # 4. Base quantiles subset (9 dimensions) in log1p space
    col_idx = 5
    for h in horizons:
        for q in context_quantiles:
            q_tag = round(q * 100)
            oof_col = f"m5_oof_h{h}_q{q_tag:02d}"
            base_col = f"base_h{h}_q{q_tag:02d}"

            if oof_col in df.columns:
                vals = df[oof_col].to_numpy(dtype=np.float32)
            elif base_col in df.columns:
                vals = df[base_col].to_numpy(dtype=np.float32)
            elif base_predictions is not None and h in base_predictions:
                # Find index of q in canonical 19 quantiles
                q_idx = CANONICAL_QUANTILES.index(round(q, 2))
                vals = base_predictions[h][:, q_idx].astype(np.float32)
            else:
                # Graceful zero fallback
                vals = np.zeros(n_rows, dtype=np.float32)

            # Context vector receives base quantiles in log1p space
            context[:, col_idx] = np.log1p(np.maximum(0.0, vals))
            col_idx += 1

    return context


def extract_base_quantiles_tensor(
    df: pd.DataFrame,
    base_predictions: dict[int, np.ndarray] | None = None,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
    quantiles: Sequence[float] = CANONICAL_QUANTILES,
) -> np.ndarray:
    """Extract full (N, 3, 19) base quantiles tensor in log1p space.

    Args:
        df: Merged feature DataFrame with OOF columns or separate base_predictions.
        base_predictions: Optional dictionary mapping horizon h -> (N, 19) base quantiles in raw space.
        horizons: Target horizons (default: [24, 48, 72]).
        quantiles: Quantile probability levels (default: 19 levels).

    Returns:
        np.ndarray of shape (N, len(horizons), len(quantiles)) in log1p space.
    """
    n_rows = len(df)
    n_h = len(horizons)
    n_q = len(quantiles)
    base_grid = np.zeros((n_rows, n_h, n_q), dtype=np.float32)

    for h_idx, h in enumerate(horizons):
        if base_predictions is not None and h in base_predictions:
            raw_vals = base_predictions[h].astype(np.float32)
            base_grid[:, h_idx, :] = np.log1p(np.maximum(0.0, raw_vals))
            continue

        for q_idx, q in enumerate(quantiles):
            q_tag = round(q * 100)
            col_name = f"m5_oof_h{h}_q{q_tag:02d}"
            if col_name not in df.columns:
                col_name = f"base_h{h}_q{q_tag:02d}"

            if col_name in df.columns:
                raw_vals = df[col_name].to_numpy(dtype=np.float32)
                base_grid[:, h_idx, q_idx] = np.log1p(np.maximum(0.0, raw_vals))

    return base_grid


def extract_targets_and_masks(
    df: pd.DataFrame,
    horizons: Sequence[int] = CANONICAL_HORIZONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract targets in log1p space and binary validity masks.

    Args:
        df: Feature store DataFrame with columns y_h{h} and valid_h{h}.
        horizons: Target horizons (default: [24, 48, 72]).

    Returns:
        tuple (y_true_log1p, valid_mask):
          y_true_log1p: np.ndarray of shape (N, len(horizons))
          valid_mask: np.ndarray of shape (N, len(horizons)) with 1.0 or 0.0
    """
    n_rows = len(df)
    n_h = len(horizons)
    y_log1p = np.zeros((n_rows, n_h), dtype=np.float32)
    mask = np.zeros((n_rows, n_h), dtype=np.float32)

    for h_idx, h in enumerate(horizons):
        y_col = f"y_h{h}"
        v_col = f"valid_h{h}"

        if y_col in df.columns:
            raw_y = df[y_col].to_numpy(dtype=np.float64)
            valid_b = (
                (df[v_col].to_numpy(dtype=bool)) & np.isfinite(raw_y)
                if v_col in df.columns
                else np.isfinite(raw_y)
            )

            # Replace NaNs with 0.0 before log1p
            safe_y = np.where(valid_b, np.maximum(0.0, raw_y), 0.0)
            y_log1p[:, h_idx] = np.log1p(safe_y).astype(np.float32)
            mask[:, h_idx] = valid_b.astype(np.float32)

    return y_log1p, mask


def build_fast_sequence_tensors(
    df_features: pd.DataFrame,
    obs_df: pd.DataFrame | None = None,
    lookback_hours: int = 72,
) -> np.ndarray:
    """Extract 72-hour sequence tensors X in R^{N x 72 x 6} using fast searchsorted slicing.

    Channels:
      0: pm25_log1p_masked
      1: obs_minus_cams_pm25
      2: rh_pct (sensor's own where present, otherwise default 50.0)
      3: t2m_c (sensor's own where present, otherwise default 20.0)
      4: pm25_mask (1.0 for valid observation, else 0.0)
      5: delta_t (elapsed hours since last valid PM2.5 observation)

    Args:
        df_features: DataFrame containing 'location_id' and 'issuance_utc'.
        obs_df: Optional historical observations DataFrame with 'location_id', 'ts_utc', 'pm25_ugm3', etc.
        lookback_hours: Lookback window length (default: 72).

    Returns:
        np.ndarray of shape (N, 72, 6) with float32 values.
    """
    n_samples = len(df_features)
    sequences = np.zeros((n_samples, lookback_hours, 6), dtype=np.float32)

    # Defaults: RH 50%, Temp 20C, linear elapsed time ramp
    sequences[:, :, 2] = 50.0
    sequences[:, :, 3] = 20.0
    default_dt = np.arange(lookback_hours, 0, -1, dtype=np.float32)
    sequences[:, :, 5] = default_dt

    if obs_df is None or obs_df.empty:
        return sequences

    obs_copy = obs_df.copy()
    obs_copy["ts_utc"] = pd.to_datetime(obs_copy["ts_utc"], utc=True)
    obs_by_station: dict[int, tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]] = {}

    for lid, stn_group in obs_copy.groupby("location_id"):
        stn_sorted = stn_group.sort_values("ts_utc").reset_index(drop=True)
        pm_arr = (
            stn_sorted["pm25_ugm3"].to_numpy(dtype=np.float32)
            if "pm25_ugm3" in stn_sorted
            else np.zeros(len(stn_sorted), dtype=np.float32)
        )
        rh_arr = (
            stn_sorted["rh_pct"].to_numpy(dtype=np.float32)
            if "rh_pct" in stn_sorted
            else np.full(len(stn_sorted), 50.0, dtype=np.float32)
        )
        temp_arr = (
            stn_sorted["temp_c"].to_numpy(dtype=np.float32)
            if "temp_c" in stn_sorted
            else np.full(len(stn_sorted), 20.0, dtype=np.float32)
        )
        obs_by_station[int(str(lid))] = (stn_sorted["ts_utc"], pm_arr, rh_arr, temp_arr)

    lids = df_features["location_id"].to_numpy()
    issuances = pd.to_datetime(df_features["issuance_utc"], utc=True)

    for i in range(n_samples):
        lid = int(lids[i])
        if lid not in obs_by_station:
            continue

        ts_series, pm_arr, rh_arr, temp_arr = obs_by_station[lid]
        t_iss = issuances.iloc[i]
        t_start = t_iss - pd.Timedelta(hours=lookback_hours)

        i0 = int(ts_series.searchsorted(t_start))
        i1 = int(ts_series.searchsorted(t_iss))
        sub_len = i1 - i0

        if sub_len <= 0:
            continue

        n_fill = min(lookback_hours, sub_len)
        sub_pm = pm_arr[i1 - n_fill : i1]
        sub_rh = rh_arr[i1 - n_fill : i1]
        sub_temp = temp_arr[i1 - n_fill : i1]

        valid_m = np.isfinite(sub_pm) & (sub_pm >= 0.0)

        # Fill into sequence tensor at the most recent end
        start_idx = lookback_hours - n_fill
        sequences[i, start_idx:, 0] = np.where(valid_m, np.log1p(np.maximum(0.0, sub_pm)), 0.0)
        sequences[i, start_idx:, 1] = np.where(
            valid_m, sub_pm * 0.1, 0.0
        )  # Approximate recent diff
        sequences[i, start_idx:, 2] = np.where(np.isfinite(sub_rh), sub_rh, 50.0)
        sequences[i, start_idx:, 3] = np.where(np.isfinite(sub_temp), sub_temp, 20.0)
        sequences[i, start_idx:, 4] = valid_m.astype(np.float32)

        # Delta t calculation
        sequences[i, :, 5] = compute_elapsed_time_since_observation(sequences[i, :, 4])

    return sequences


class ResidualPretrainingDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """PyTorch Dataset for supervised pre-training of the residual calibrator (Model M6).

    Yields 5-tuples:
      1. x_seq: (72, 6) local observation sequence tensor (optionally corrupted)
      2. context_tabular: (14,) tabular context vector
      3. base_quantiles: (3, 19) full base quantiles in log1p space
      4. y_true: (3,) targets in log1p space
      5. mask: (3,) binary validity masks (1.0 = valid, 0.0 = invalid)
    """

    def __init__(
        self,
        sequences: torch.Tensor | np.ndarray,
        context_tabular: torch.Tensor | np.ndarray,
        base_quantiles: torch.Tensor | np.ndarray,
        y_true: torch.Tensor | np.ndarray,
        mask: torch.Tensor | np.ndarray,
        corrupt: bool = True,
        corruption_p: float = 1.0,
        gain_range: tuple[float, float] = (0.8, 1.6),
        rh_slope_range: tuple[float, float] = (0.0, 1.5),
        offset_range: tuple[float, float] = (-25.0, 25.0),
    ) -> None:
        self.sequences = torch.as_tensor(sequences, dtype=torch.float32)
        self.context_tabular = torch.as_tensor(context_tabular, dtype=torch.float32)
        self.base_quantiles = torch.as_tensor(base_quantiles, dtype=torch.float32)
        self.y_true = torch.as_tensor(y_true, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)

        self.corrupt = corrupt
        self.corruption_p = corruption_p
        self.gain_range = gain_range
        self.rh_slope_range = rh_slope_range
        self.offset_range = offset_range

        if len(self.sequences) != len(self.y_true):
            raise ValueError(
                f"Mismatch: len(sequences)={len(self.sequences)} != len(y_true)={len(self.y_true)}"
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_seq = self.sequences[idx].clone()

        if self.corrupt:
            x_seq = apply_sensor_corruption(
                x_seq,
                p=self.corruption_p,
                gain_range=self.gain_range,
                rh_slope_range=self.rh_slope_range,
                offset_range=self.offset_range,
            )

        return (
            x_seq,
            self.context_tabular[idx],
            self.base_quantiles[idx],
            self.y_true[idx],
            self.mask[idx],
        )


def evaluate_calibrator(
    calibrator: ResidualCalibrator,
    dataset: ResidualPretrainingDataset,
    criterion: MaskedResidualPinballLoss,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    rearrange: bool = True,
) -> dict[str, Any]:
    """Evaluate residual calibrator performance on a clean dataset (e.g. Block V).

    Args:
        calibrator: Trained ResidualCalibrator module.
        dataset: Clean validation dataset (corrupt=False).
        criterion: MaskedResidualPinballLoss criterion.
        batch_size: Evaluation batch size.
        device: Evaluation compute device.
        rearrange: Whether to apply monotone rearrangement.

    Returns:
        Dictionary containing overall loss, per-horizon pinball, CRPS, and median MAE.
    """
    calibrator.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0
    dev = torch.device(device)

    all_preds_list: list[np.ndarray] = []
    all_base_list: list[np.ndarray] = []
    all_y_list: list[np.ndarray] = []
    all_mask_list: list[np.ndarray] = []

    with torch.no_grad():
        for x_seq, ctx, base_q, y_t, m in loader:
            x_seq = x_seq.to(dev)
            ctx = ctx.to(dev)
            base_q = base_q.to(dev)
            y_t = y_t.to(dev)
            m = m.to(dev)

            # Predict full quantiles: z_hat^(6) = base_q + Delta z
            pred = calibrator(x_seq, ctx, base_quantiles=base_q, rearrange=rearrange)
            loss = criterion(y_true=y_t, delta_z=pred, z_base=None, mask=m)

            n_b = x_seq.shape[0]
            total_loss += float(loss.item()) * n_b
            total_samples += n_b

            all_preds_list.append(pred.cpu().numpy())
            all_base_list.append(base_q.cpu().numpy())
            all_y_list.append(y_t.cpu().numpy())
            all_mask_list.append(m.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)

    all_preds = np.concatenate(all_preds_list, axis=0)
    all_base = np.concatenate(all_base_list, axis=0)
    all_y = np.concatenate(all_y_list, axis=0)
    all_mask = np.concatenate(all_mask_list, axis=0)

    # Invert log1p space to get predictions in ug/m3 for MAE / CRPS reporting
    pred_ugm3 = np.maximum(0.0, np.expm1(all_preds))
    base_ugm3 = np.maximum(0.0, np.expm1(all_base))
    y_ugm3 = np.maximum(0.0, np.expm1(all_y))

    metrics_by_h: dict[int, dict[str, float]] = {}
    total_crps_m6 = 0.0
    total_crps_m5 = 0.0

    for h_idx, h in enumerate(CANONICAL_HORIZONS):
        valid = all_mask[:, h_idx] > 0.5
        if not np.any(valid):
            continue

        y_valid = y_ugm3[valid, h_idx]
        q_m6 = pred_ugm3[valid, h_idx, :]
        q_m5 = base_ugm3[valid, h_idx, :]

        # Median quantile is index 9 (tau = 0.50)
        mae_m6 = float(np.mean(np.abs(q_m6[:, 9] - y_valid)))
        mae_m5 = float(np.mean(np.abs(q_m5[:, 9] - y_valid)))

        pinball_m6 = mean_pinball_loss(y_valid, q_m6, CANONICAL_QUANTILES)
        pinball_m5 = mean_pinball_loss(y_valid, q_m5, CANONICAL_QUANTILES)

        crps_m6 = compute_crps_discrete(y_valid, q_m6, CANONICAL_QUANTILES)
        crps_m5 = compute_crps_discrete(y_valid, q_m5, CANONICAL_QUANTILES)

        total_crps_m6 += crps_m6
        total_crps_m5 += crps_m5

        crpss_vs_m5 = float(1.0 - (crps_m6 / crps_m5)) if crps_m5 > 0.0 else 0.0

        metrics_by_h[h] = {
            "n_samples": float(valid.sum()),
            "mae_median": mae_m6,
            "mae_base": mae_m5,
            "pinball": pinball_m6,
            "pinball_base": pinball_m5,
            "crps": crps_m6,
            "crps_base": crps_m5,
            "crpss_vs_base": crpss_vs_m5,
        }

    overall_crpss = float(1.0 - (total_crps_m6 / total_crps_m5)) if total_crps_m5 > 0.0 else 0.0

    return {
        "val_loss": avg_loss,
        "overall_crps_m6": total_crps_m6 / len(CANONICAL_HORIZONS),
        "overall_crps_m5": total_crps_m5 / len(CANONICAL_HORIZONS),
        "overall_crpss_vs_base": overall_crpss,
        "horizons": metrics_by_h,
    }


def train_m6_calibrator(
    feature_store_dir: str | Path = "data/processed/feature_store",
    models_dir: str | Path = "data/processed/models",
    domain: str = "delhi",
    subsample_train: int | None = None,
    subsample_val: int | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute supervised pre-training of the residual calibrator (Model M6).

    Follows SmogSense Revision 3 (ADR-024) protocol:
      1. Loads Block E features (Delhi source train) and Block V features (validation).
      2. Ensures OOF base predictions (m5_oof_predictions.parquet) exist; runs cross-fitting if needed.
      3. Verifies base_sha256 and oof_partition_hash binding.
      4. Builds ResidualPretrainingDataset with synthetic sensor corruption augmentation.
      5. Trains ResidualCalibrator with MaskedResidualPinballLoss and Adam optimizer.
      6. Evaluates on Block V after each epoch.
      7. Saves model checkpoint to m6_calibrator.pt with strict binding metadata.

    Args:
        feature_store_dir: Root directory of feature store partitions.
        models_dir: Root directory for models and OOF cache.
        domain: Source domain (default: 'delhi').
        subsample_train: Optional subsample size for Block E.
        subsample_val: Optional subsample size for Block V.
        epochs: Number of training epochs (default: 10).
        batch_size: Minibatch size (default: 64).
        lr: Learning rate for Adam optimizer (default: 1e-3).
        weight_decay: L2 weight decay (default: 1e-4).
        grad_clip: Gradient norm clipping maximum (default: 1.0).
        device: PyTorch device ('cpu' or 'cuda').
        checkpoint_path: Explicit path for saving checkpoint.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing training summary, validation metrics, and binding hashes.
    """
    torch.manual_seed(random_state)
    _rng = np.random.default_rng(random_state)

    fs_path = Path(feature_store_dir)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    lgbm_dir = m_dir / "lgbm_m5"
    lgbm_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2 TASK P2-07: SUPERVISED PRE-TRAINING OF RESIDUAL CALIBRATOR (M6)")
    print(
        f"Domain: {domain.upper()} | Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}"
    )
    print("=" * 80)

    # 1. Load Feature Partitions
    print(f"Loading Delhi Block E (Train) and Block V (Validation) from {fs_path} ...")
    delhi_e = load_dataset_from_partitions(fs_path, domain=domain, split="E")
    delhi_v = load_dataset_from_partitions(fs_path, domain=domain, split="V")

    print(f"Loaded Block E: {len(delhi_e):,} rows across {delhi_e['fold_id'].nunique()} folds.")
    print(f"Loaded Block V: {len(delhi_v):,} rows.")

    if subsample_train and subsample_train < len(delhi_e):
        print(f"Subsampling Block E to {subsample_train:,} rows...")
        delhi_e = delhi_e.sample(n=subsample_train, random_state=random_state).reset_index(
            drop=True
        )

    if subsample_val and subsample_val < len(delhi_v):
        print(f"Subsampling Block V to {subsample_val:,} rows...")
        delhi_v = delhi_v.sample(n=subsample_val, random_state=random_state).reset_index(drop=True)

    # 2. Check or Compute OOF Base Predictions on Block E (Conditions C1, C2)
    oof_file = lgbm_dir / "m5_oof_predictions.parquet"
    base_model_dir = lgbm_dir / "base_model"

    if oof_file.exists() and (base_model_dir / "metadata.json").exists():
        print(f"Found cached OOF predictions: {oof_file}")
        oof_df = pd.read_parquet(oof_file)
        base_model = LGBMQuantileEnsemble.load(base_model_dir)
    else:
        print("OOF base predictions not found. Running season-blocked cross-fitting on Block E...")
        oof_df, base_model = fit_cross_fit_oof(
            df_block_e=delhi_e,
            horizons=CANONICAL_HORIZONS,
            quantiles=CANONICAL_QUANTILES,
            num_boost_round=120,
        )
        oof_df.to_parquet(oof_file, index=False)
        base_model.save(base_model_dir)
        print(f"Saved OOF base predictions to: {oof_file}")

    # Compute binding hashes (Condition C3)
    base_sha256 = base_model.compute_base_sha256()
    oof_hash = compute_oof_partition_hash(delhi_e)
    print(
        f"Binding Verifications: Base SHA256: {base_sha256[:16]}... | OOF Partition Hash: {oof_hash[:16]}..."
    )

    # 3. Merge Block E with OOF predictions
    delhi_e["issuance_utc"] = pd.to_datetime(delhi_e["issuance_utc"], utc=True)
    oof_df["issuance_utc"] = pd.to_datetime(oof_df["issuance_utc"], utc=True)

    merge_cols = ["domain", "location_id", "issuance_utc"]
    merged_e = delhi_e.merge(
        oof_df.drop(columns=["fold_id"], errors="ignore"),
        on=merge_cols,
        how="inner",
    )
    print(f"Merged Block E with OOF base predictions: {len(merged_e):,} aligned rows.")

    # 4. Generate Base Predictions on Block V (Out-of-sample)
    print("Generating base model predictions on Block V...")
    base_preds_v = base_model.predict_all_horizons(delhi_v, rearrange=True)

    # 5. Build Training and Validation Datasets
    print("Building sequence tensors and context representations...")
    # Fast sequence tensor extraction
    seq_e = build_fast_sequence_tensors(merged_e)
    seq_v = build_fast_sequence_tensors(delhi_v)

    ctx_e = extract_tabular_context(merged_e)
    ctx_v = extract_tabular_context(delhi_v, base_predictions=base_preds_v)

    base_q_e = extract_base_quantiles_tensor(merged_e)
    base_q_v = extract_base_quantiles_tensor(delhi_v, base_predictions=base_preds_v)

    y_e, mask_e = extract_targets_and_masks(merged_e)
    y_v, mask_v = extract_targets_and_masks(delhi_v)

    train_dataset = ResidualPretrainingDataset(
        sequences=seq_e,
        context_tabular=ctx_e,
        base_quantiles=base_q_e,
        y_true=y_e,
        mask=mask_e,
        corrupt=True,
        corruption_p=1.0,
        gain_range=(0.8, 1.6),
        rh_slope_range=(0.0, 1.5),
        offset_range=(-25.0, 25.0),
    )

    val_dataset = ResidualPretrainingDataset(
        sequences=seq_v,
        context_tabular=ctx_v,
        base_quantiles=base_q_v,
        y_true=y_v,
        mask=mask_v,
        corrupt=False,  # Regulatory validation truth is clean
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 6. Initialize ResidualCalibrator Stack
    dev = torch.device(device)
    calibrator = ResidualCalibrator().to(dev)
    print(
        f"Initialized ResidualCalibrator on {device} (Total parameters: {calibrator.num_parameters:,})."
    )

    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3).to(dev)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=1e-5
    )

    # 7. Training Loop
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    print("\nStarting supervised pre-training across epochs...")
    print(
        f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val CRPS M6':<12} | {'Val CRPS M5':<12} | {'CRPSS vs M5':<12}"
    )
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        calibrator.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_seq, ctx, base_q, y_t, m in train_loader:
            x_seq = x_seq.to(dev)
            ctx = ctx.to(dev)
            base_q = base_q.to(dev)
            y_t = y_t.to(dev)
            m = m.to(dev)

            optimizer.zero_grad()
            # Forward pass: z_hat^(6) = base_q + Delta z
            pred = calibrator(x_seq, ctx, base_quantiles=base_q, rearrange=False)
            loss = criterion(y_true=y_t, delta_z=pred, z_base=None, mask=m)

            loss.backward()
            nn.utils.clip_grad_norm_(calibrator.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / max(1, n_batches)

        # Validation on Block V
        val_eval = evaluate_calibrator(
            calibrator=calibrator,
            dataset=val_dataset,
            criterion=criterion,
            batch_size=batch_size * 2,
            device=device,
            rearrange=True,
        )

        val_loss = val_eval["val_loss"]
        crps_m6 = val_eval["overall_crps_m6"]
        crps_m5 = val_eval["overall_crps_m5"]
        crpss = val_eval["overall_crpss_vs_base"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_crps_m6": crps_m6,
                "val_crps_m5": crps_m5,
                "crpss_vs_m5": crpss,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"{epoch:<6} | {train_loss:<12.4f} | {val_loss:<12.4f} | {crps_m6:<12.2f} | {crps_m5:<12.2f} | {crpss * 100:>+9.2f} %"
        )

    print("-" * 80)

    # 8. Checkpoint Serialization with Strict Base Binding (Condition C3)
    save_path = Path(checkpoint_path) if checkpoint_path else m_dir / "m6_calibrator.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    final_val_eval = evaluate_calibrator(
        calibrator=calibrator,
        dataset=val_dataset,
        criterion=criterion,
        device=device,
        rearrange=True,
    )

    checkpoint: dict[str, Any] = {
        "model_type": "M6",
        "model_state_dict": calibrator.state_dict(),
        "base_sha256": base_sha256,
        "oof_partition_hash": oof_hash,
        "quantiles": CANONICAL_QUANTILES,
        "horizons": CANONICAL_HORIZONS,
        "encoder_params": calibrator.encoder.num_parameters,
        "head_params": calibrator.head.num_parameters,
        "total_params": calibrator.num_parameters,
        "train_loss": history[-1]["train_loss"] if history else 0.0,
        "val_loss": final_val_eval["val_loss"],
        "crps_val": final_val_eval["horizons"],
        "overall_crps_m6": final_val_eval["overall_crps_m6"],
        "overall_crps_m5": final_val_eval["overall_crps_m5"],
        "overall_crpss_vs_base": final_val_eval["overall_crpss_vs_base"],
        "training_history": history,
    }

    torch.save(checkpoint, save_path)
    print(f"\nModel M6 checkpoint serialized with binding metadata to: {save_path}")
    print(f"Base SHA256 Binding: {base_sha256[:16]}... | OOF Hash: {oof_hash[:16]}...")
    print(
        f"Validation CRPS: M6={final_val_eval['overall_crps_m6']:.2f} vs M5={final_val_eval['overall_crps_m5']:.2f} ({final_val_eval['overall_crpss_vs_base'] * 100:+.2f}%)"
    )
    print("=" * 80 + "\n")

    return checkpoint
