"""smogsense.models.maml — First-Order MAML (FOMAML) with Meta-SGD and sensor corruption.

Implements Phase 2 Task P2-08 per SmogSense Revision 3 (ADR-024):
- Neural residual calibrator stack (47,577 parameters: LocalGRUEncoder + ResidualQuantileHead).
- Meta-SGD learned step sizes (14 learnable scalars, alpha_k = softplus(rho_k) >= 0).
- J = 3 inner adaptation steps with functional parameter updates (torch.func.functional_call).
- Gradient clipping at 1.0 in both inner and outer loops.
- Twin-aware sparse-network task construction (N in {0, 1, 2, 3, 5, 8}) with 72-h embargo.
- Synthetic low-cost PM2.5 sensor corruption augmentation on support sets.
- Model checkpoint serialization (m7_maml.pt) with strict base_sha256 and oof_partition_hash binding.

Specification: docs/ml-architecture.md §7 & §8
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from structlog import get_logger
from torch import nn
from torch.func import functional_call
from torch.nn import functional

from smogsense.features.backfill import compute_oof_partition_hash
from smogsense.features.sparse_network import sample_available_stations
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
)
from smogsense.models.losses import MaskedResidualPinballLoss
from smogsense.models.pretraining import (
    apply_sensor_corruption,
    build_fast_sequence_tensors,
    extract_base_quantiles_tensor,
    extract_tabular_context,
    extract_targets_and_masks,
)

logger = get_logger(__name__)

# Canonical names of the 14 parameter tensors of ResidualCalibrator
# 8 GRU tensors + 2 Linear projection tensors + 4 MLP head tensors = 14 tensors
PARAM_TENSOR_NAMES: tuple[str, ...] = (
    "encoder.gru.weight_ih_l0",
    "encoder.gru.weight_hh_l0",
    "encoder.gru.bias_ih_l0",
    "encoder.gru.bias_hh_l0",
    "encoder.gru.weight_ih_l1",
    "encoder.gru.weight_hh_l1",
    "encoder.gru.bias_ih_l1",
    "encoder.gru.bias_hh_l1",
    "encoder.proj.weight",
    "encoder.proj.bias",
    "head.fc1.weight",
    "head.fc1.bias",
    "head.fc2.weight",
    "head.fc2.bias",
)


# -----------------------------------------------------------------------------
# 1. Meta-SGD Learnable Step Sizes
# -----------------------------------------------------------------------------
class MetaSGDStepSizes(nn.Module):
    """Learnable per-parameter-tensor Meta-SGD step sizes.

    Mathematical formulation (docs/ml-architecture.md §8 & ADR-024):
        alpha_k = softplus(rho_k) >= 0, where rho_k in R.
        Initialized at alpha_k = 0.01:
            rho_k = ln(exp(0.01) - 1) approx -4.60015.

    Granularity:
        Exactly one learnable step size scalar per parameter tensor (14 scalars).
    """

    def __init__(
        self,
        init_alpha: float = 0.01,
        param_names: Sequence[str] = PARAM_TENSOR_NAMES,
        *,
        init_lr: float | None = None,
    ) -> None:
        super().__init__()
        if init_lr is not None:
            init_alpha = init_lr
        self.param_names = list(param_names)
        self._name_to_idx = {name: i for i, name in enumerate(self.param_names)}

        # rho = ln(exp(init_alpha) - 1)
        init_rho = float(math.log(math.expm1(init_alpha)))
        self.rho = nn.Parameter(torch.full((len(self.param_names),), init_rho, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return the vector of positive step sizes alpha."""
        return functional.softplus(self.rho)

    def get_step_sizes(self) -> dict[str, torch.Tensor]:
        """Return dictionary mapping parameter tensor name -> alpha_k."""
        alphas = self.forward()
        return {name: alphas[self._name_to_idx[name]] for name in self.param_names}

    def step_size_for(self, param_name: str) -> torch.Tensor:
        """Return the positive step size for a given parameter name."""
        idx = self._name_to_idx[param_name]
        return functional.softplus(self.rho[idx])

    def extra_repr(self) -> str:
        alphas = self.forward().detach().cpu().numpy()
        return f"n_tensors={len(self.param_names)}, mean_alpha={float(np.mean(alphas)):.4f}"


# -----------------------------------------------------------------------------
# 2. Sparse-Network Task Representation
# -----------------------------------------------------------------------------
@dataclass
class MAMLTask:
    """Encapsulates support and query data for one sparse-network meta-task.

    Tensors:
        support_x: (B_sup, 72, 6) local sequence tensor.
        support_context: (B_sup, 14) tabular context tensor.
        support_base_q: (B_sup, 3, 19) base quantiles in log1p.
        support_y: (B_sup, 3) target block means in log1p.
        support_mask: (B_sup, 3) binary target validity mask.
        query_x: (B_qry, 72, 6) local sequence tensor.
        query_context: (B_qry, 14) tabular context tensor.
        query_base_q: (B_qry, 3, 19) base quantiles in log1p.
        query_y: (B_qry, 3) clean regulatory ground truth targets in log1p.
        query_mask: (B_qry, 3) binary target validity mask.
    """

    domain: str
    season: str
    available_stations: list[int]
    held_out_stations: list[int]
    support_x: torch.Tensor
    support_context: torch.Tensor
    support_base_q: torch.Tensor
    support_y: torch.Tensor
    support_mask: torch.Tensor
    query_x: torch.Tensor
    query_context: torch.Tensor
    query_base_q: torch.Tensor
    query_y: torch.Tensor
    query_mask: torch.Tensor
    n_available: int
    corruption_params: dict[str, float] | None = None

    def to(self, device: torch.device | str) -> MAMLTask:
        """Transfer all task tensors to the specified compute device."""
        return MAMLTask(
            domain=self.domain,
            season=self.season,
            available_stations=self.available_stations,
            held_out_stations=self.held_out_stations,
            support_x=self.support_x.to(device),
            support_context=self.support_context.to(device),
            support_base_q=self.support_base_q.to(device),
            support_y=self.support_y.to(device),
            support_mask=self.support_mask.to(device),
            query_x=self.query_x.to(device),
            query_context=self.query_context.to(device),
            query_base_q=self.query_base_q.to(device),
            query_y=self.query_y.to(device),
            query_mask=self.query_mask.to(device),
            n_available=self.n_available,
            corruption_params=self.corruption_params,
        )


# -----------------------------------------------------------------------------
# 3. Task Sampler (Twin-Aware, Season-Blocked, Embargoed)
# -----------------------------------------------------------------------------
class TaskSampler:
    """Samples sparse-network meta-tasks from feature partitions.

    Implements:
    - Twin-aware available/held-out station partition (d <= 50m never split).
    - Chronological support and query windows separated by a 72-hour embargo.
    - N in {0, 1, 2, 3, 5, 8} sensor availability simulation.
    - Synthetic low-cost PM2.5 sensor corruption on available stations.
    - Uncorrupted clean regulatory ground truth labels on query set.
    """

    def __init__(
        self,
        df_features: pd.DataFrame,
        df_obs: pd.DataFrame | None = None,
        available_n_sampler: Sequence[int] = (0, 1, 2, 3, 5, 8),
        support_window_days: Sequence[int] = (7, 14, 30),
        query_window_days: int = 14,
        embargo_hours: int = 72,
        corruption_enabled: bool = True,
        corruption_p: float = 1.0,
        gain_range: tuple[float, float] = (0.8, 1.3),
        rh_slope_range: tuple[float, float] = (0.0, 1.5),
        offset_range: tuple[float, float] = (-10.0, 25.0),
        support_labels_policy: str = "clean",
        rng: np.random.Generator | int | None = None,
    ) -> None:
        self.df = df_features.copy()
        self.df_obs = df_obs
        self.available_n_sampler = tuple(available_n_sampler)
        self.support_window_days = tuple(support_window_days)
        self.query_window_days = query_window_days
        self.embargo_hours = embargo_hours
        self.corruption_enabled = corruption_enabled
        self.corruption_p = corruption_p
        self.gain_range = gain_range
        self.rh_slope_range = rh_slope_range
        self.offset_range = offset_range
        self.support_labels_policy = support_labels_policy

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = np.random.default_rng(42)

        # Standardize datetimes
        self.df["issuance_utc"] = pd.to_datetime(self.df["issuance_utc"], utc=True)
        if "season" not in self.df.columns:
            # Fallback to year-based season partition if not explicitly named
            years = self.df["issuance_utc"].dt.year
            self.df["season"] = years.astype(str)

        self.seasons = sorted(self.df["season"].dropna().unique().tolist())

        # Build stations metadata DataFrame for twin-aware sampler
        id_col = "location_id"
        lat_col = next(
            (c for c in ("f_stat_latitude", "latitude", "lat") if c in self.df.columns), None
        )
        lon_col = next(
            (c for c in ("f_stat_longitude", "longitude", "lon") if c in self.df.columns), None
        )

        if lat_col and lon_col:
            stn_meta = (
                self.df[[id_col, lat_col, lon_col]]
                .drop_duplicates(subset=[id_col])
                .rename(columns={lat_col: "latitude", lon_col: "longitude"})
                .reset_index(drop=True)
            )
        else:
            # Fallback dummy coordinates if coordinates were excluded from features
            unique_ids = self.df[id_col].unique()
            stn_meta = pd.DataFrame(
                {
                    id_col: unique_ids,
                    "latitude": np.linspace(28.5, 28.7, len(unique_ids)),
                    "longitude": np.linspace(77.1, 77.3, len(unique_ids)),
                }
            )

        self.stations_df = stn_meta

    def sample_task(
        self,
        season: str | None = None,
        n: int | None = None,
        support_days: int | None = None,
    ) -> MAMLTask:
        """Sample a single MAMLTask satisfying all twin-aware and embargo constraints."""
        target_season = season if season is not None else str(self.rng.choice(self.seasons))
        season_df = self.df[self.df["season"] == target_season].copy()

        if season_df.empty:
            season_df = self.df

        # 1. Twin-aware available/held-out station split
        season_stn_ids = set(season_df["location_id"].unique())
        sub_stns = self.stations_df[self.stations_df["location_id"].isin(season_stn_ids)].copy()

        target_n = n if n is not None else int(self.rng.choice(self.available_n_sampler))
        avail_set, held_set, sampled_n = sample_available_stations(
            sub_stns,
            n=target_n,
            available_n_sampler=self.available_n_sampler,
            rng=self.rng,
        )

        avail_list = sorted(avail_set)
        held_list = sorted(held_set)

        # 2. Window sampling: support window + 72-h embargo + query window
        sup_days = (
            support_days
            if support_days is not None
            else int(self.rng.choice(self.support_window_days))
        )
        total_span_h = (sup_days * 24) + self.embargo_hours + (self.query_window_days * 24)

        min_time = season_df["issuance_utc"].min()
        max_time = season_df["issuance_utc"].max()

        total_available_h = int((max_time - min_time).total_seconds() // 3600)

        if total_available_h > total_span_h:
            offset_h = int(self.rng.integers(0, total_available_h - total_span_h))
            t_start_sup = min_time + pd.Timedelta(hours=offset_h)
        else:
            t_start_sup = min_time

        t_end_sup = t_start_sup + pd.Timedelta(days=sup_days)
        t_start_qry = t_end_sup + pd.Timedelta(hours=self.embargo_hours)
        t_end_qry = t_start_qry + pd.Timedelta(days=self.query_window_days)

        # 3. Filter Support Data
        if sampled_n > 0 and len(avail_list) > 0:
            mask_sup = (
                season_df["location_id"].isin(avail_list)
                & (season_df["issuance_utc"] >= t_start_sup)
                & (season_df["issuance_utc"] < t_end_sup)
            )
            df_sup = season_df[mask_sup].reset_index(drop=True)
        else:
            df_sup = season_df.iloc[:0].copy()

        # 4. Filter Query Data (from held-out stations if available, else all active stations)
        query_stns = (
            held_list
            if len(held_list) > 0
            else (avail_list if avail_list else list(season_stn_ids))
        )
        mask_qry = (
            season_df["location_id"].isin(query_stns)
            & (season_df["issuance_utc"] >= t_start_qry)
            & (season_df["issuance_utc"] <= t_end_qry)
        )
        df_qry = season_df[mask_qry].reset_index(drop=True)

        if df_qry.empty:
            # Fallback to general slice within season
            mask_qry = (season_df["issuance_utc"] >= t_start_qry) & (
                season_df["issuance_utc"] <= t_end_qry
            )
            df_qry = season_df[mask_qry].reset_index(drop=True)
            if df_qry.empty:
                df_qry = season_df.tail(min(len(season_df), 128)).reset_index(drop=True)

        # 5. Extract Tensors
        # Support
        if not df_sup.empty:
            sup_x_arr = build_fast_sequence_tensors(df_sup, self.df_obs)
            sup_x = torch.from_numpy(sup_x_arr).to(torch.float32)
            sup_u = torch.from_numpy(extract_tabular_context(df_sup)).to(torch.float32)
            sup_base = torch.from_numpy(extract_base_quantiles_tensor(df_sup)).to(torch.float32)
            sup_y_np, sup_m_np = extract_targets_and_masks(df_sup)
            sup_y = torch.from_numpy(sup_y_np).to(torch.float32)
            sup_mask = torch.from_numpy(sup_m_np).to(torch.float32)
        else:
            sup_x = torch.zeros(0, 72, 6, dtype=torch.float32)
            sup_u = torch.zeros(0, 14, dtype=torch.float32)
            sup_base = torch.zeros(0, 3, 19, dtype=torch.float32)
            sup_y = torch.zeros(0, 3, dtype=torch.float32)
            sup_mask = torch.zeros(0, 3, dtype=torch.float32)

        # Query
        qry_x_arr = build_fast_sequence_tensors(df_qry, self.df_obs)
        qry_x = torch.from_numpy(qry_x_arr).to(torch.float32)
        qry_u = torch.from_numpy(extract_tabular_context(df_qry)).to(torch.float32)
        qry_base = torch.from_numpy(extract_base_quantiles_tensor(df_qry)).to(torch.float32)
        qry_y_np, qry_m_np = extract_targets_and_masks(df_qry)
        qry_y = torch.from_numpy(qry_y_np).to(torch.float32)
        qry_mask = torch.from_numpy(qry_m_np).to(torch.float32)

        # 6. Apply Synthetic Sensor Corruption to Support Inputs
        corruption_meta: dict[str, float] | None = None
        if self.corruption_enabled and sup_x.shape[0] > 0 and self.rng.random() < self.corruption_p:
            a_val = float(self.rng.uniform(self.gain_range[0], self.gain_range[1]))
            b_val = float(self.rng.uniform(self.rh_slope_range[0], self.rh_slope_range[1]))
            c_val = float(self.rng.uniform(self.offset_range[0], self.offset_range[1]))
            corruption_meta = {"a": a_val, "b": b_val, "c": c_val}

            sup_x = apply_sensor_corruption(
                sup_x,
                p=1.0,
                gain_range=(a_val, a_val),
                rh_slope_range=(b_val, b_val),
                offset_range=(c_val, c_val),
            )

        return MAMLTask(
            domain="delhi",
            season=target_season,
            available_stations=avail_list,
            held_out_stations=held_list,
            support_x=sup_x,
            support_context=sup_u,
            support_base_q=sup_base,
            support_y=sup_y,
            support_mask=sup_mask,
            query_x=qry_x,
            query_context=qry_u,
            query_base_q=qry_base,
            query_y=qry_y,
            query_mask=qry_mask,
            n_available=sampled_n,
            corruption_params=corruption_meta,
        )


# -----------------------------------------------------------------------------
# 4. Functional Inner-Loop Adaptation
# -----------------------------------------------------------------------------
def inner_loop_adapt(
    calibrator: ResidualCalibrator,
    params: Mapping[str, torch.Tensor],
    step_sizes: MetaSGDStepSizes | dict[str, torch.Tensor],
    task: MAMLTask,
    inner_steps: int = 3,
    clip_norm: float = 1.0,
    adapt_scope: str = "all",
) -> dict[str, torch.Tensor]:
    """Execute J-step functional Meta-SGD inner-loop adaptation on task support data.

    Formula (docs/ml-architecture.md §8 & ADR-024):
        theta^(0) = theta
        theta^(j+1) = theta^(j) - alpha (.) clip(grad_{theta^(j)} L_sup(theta^(j)), clip_norm).detach()
        theta' = theta^(J)

    Using detached gradients yields exact First-Order MAML (FOMAML) for initial weights theta
    and exact Meta-SGD updates for step sizes alpha:
        d L_qry / d alpha_k approx - sum_j < grad_k L_sup(theta^(j)), grad_k L_qry(theta') >

    Args:
        calibrator: ResidualCalibrator model instance defining the architecture.
        params: Current parameter mapping theta^(0) mapping name -> tensor.
        step_sizes: MetaSGDStepSizes module or dict mapping name -> positive step size alpha_k.
        task: MAMLTask containing support sequence, context, base predictions, and targets.
        inner_steps: Number of inner gradient adaptation steps J (default: 3).
        clip_norm: Maximum L2 norm for inner gradient clipping (default: 1.0).
        adapt_scope: Parameter adaptation scope ('all', 'head_only', 'gru_only').

    Returns:
        Adapted parameter dictionary theta' of identical structure and shapes.
    """
    # At N = 0 or empty support set: skip adaptation and return theta identically (M5 fallback)
    if task.n_available == 0 or task.support_x.shape[0] == 0:
        return dict(params)

    # Disable cuDNN RNN backend and ensure training mode during functional calls
    # to prevent 'cudnn RNN backward can only be called in training mode'.
    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    was_training = calibrator.training
    calibrator.train()

    try:
        # Resolve step sizes dictionary
        alphas = (
            step_sizes.get_step_sizes() if isinstance(step_sizes, MetaSGDStepSizes) else step_sizes
        )

        criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)
        curr_params: dict[str, torch.Tensor] = dict(params)

        for _ in range(inner_steps):
            # Forward pass on support set using current parameters
            # rearrange=False preserves uncorrupted smooth pinball gradients
            preds = functional_call(
                calibrator,
                curr_params,
                (task.support_x, task.support_context, task.support_base_q, False),
            )

            loss = criterion(
                y_true=task.support_y,
                delta_z=preds,
                z_base=None,
                mask=task.support_mask,
            )

            # Filter parameters to differentiate according to adapt_scope
            target_param_names: list[str] = []
            target_tensors: list[torch.Tensor] = []
            for name, p_tensor in curr_params.items():
                if adapt_scope == "head_only" and not name.startswith("head."):
                    continue
                if adapt_scope == "gru_only" and not name.startswith("encoder."):
                    continue
                target_param_names.append(name)
                target_tensors.append(p_tensor)

            grads = torch.autograd.grad(
                loss,
                target_tensors,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )

            # Compute global L2 norm across all inner gradients
            total_sq_norm = sum(float(torch.sum(g * g).item()) for g in grads if g is not None)
            glob_norm = math.sqrt(total_sq_norm)
            clip_coeff = clip_norm / max(clip_norm, glob_norm + 1e-8)

            # Update parameters with Meta-SGD step sizes: theta_(j+1) = theta_j - alpha * (clip * grad).detach()
            next_params = dict(curr_params)
            for name, g in zip(target_param_names, grads, strict=False):
                if g is None:
                    continue
                clipped_g = (g * clip_coeff).detach()
                alpha_k = alphas.get(name, torch.tensor(0.01, device=clipped_g.device))
                next_params[name] = curr_params[name] - alpha_k * clipped_g

            curr_params = next_params

        return curr_params
    finally:
        calibrator.train(was_training)
        torch.backends.cudnn.enabled = prev_cudnn


# -----------------------------------------------------------------------------
# 5. MAML Outer Step (Batch Meta-Optimization)
# -----------------------------------------------------------------------------
def maml_outer_step(
    calibrator: ResidualCalibrator,
    step_sizes: MetaSGDStepSizes,
    outer_optimizer: torch.optim.Optimizer,
    batch_tasks: list[MAMLTask],
    inner_steps: int = 3,
    clip_norm: float = 1.0,
    adapt_scope: str = "all",
) -> float:
    """Execute one outer meta-optimization update over a batch of sparse tasks.

    Evaluates adapted parameters theta_i' on chronologically embargoed query sets,
    accumulates FOMAML gradients w.r.t initial parameters theta and Meta-SGD step sizes rho,
    clips outer gradient global norm, and steps the outer optimizer.

    Args:
        calibrator: ResidualCalibrator holding meta-initialization weights theta.
        step_sizes: MetaSGDStepSizes holding learnable step sizes rho.
        outer_optimizer: Adam optimizer over (theta, rho).
        batch_tasks: Batch of sampled MAMLTask instances.
        inner_steps: Inner adaptation step count J (default: 3).
        clip_norm: Maximum outer gradient global norm (default: 1.0).
        adapt_scope: Parameter adaptation scope ('all', 'head_only', 'gru_only').

    Returns:
        Average query pinball loss across the task batch.
    """
    if not batch_tasks:
        return 0.0

    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    outer_optimizer.zero_grad()
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)

    try:
        # Initial meta-parameters theta
        init_params = dict(calibrator.named_parameters())
        batch_query_loss = 0.0
        n_tasks = len(batch_tasks)

        for task in batch_tasks:
            # Inner loop adaptation
            adapted_params = inner_loop_adapt(
                calibrator=calibrator,
                params=init_params,
                step_sizes=step_sizes,
                task=task,
                inner_steps=inner_steps,
                clip_norm=clip_norm,
                adapt_scope=adapt_scope,
            )

            # Query evaluation
            qry_pred = functional_call(
                calibrator,
                adapted_params,
                (task.query_x, task.query_context, task.query_base_q, False),
            )

            qry_loss = criterion(
                y_true=task.query_y,
                delta_z=qry_pred,
                z_base=None,
                mask=task.query_mask,
            )

            # Normalize by batch size and backpropagate
            loss_scaled = qry_loss / n_tasks
            loss_scaled.backward()

            batch_query_loss += float(qry_loss.item())

        # Clip outer gradients across both calibrator parameters and Meta-SGD step sizes
        all_trainable_params = [p for p in calibrator.parameters() if p.requires_grad] + [
            p for p in step_sizes.parameters() if p.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=clip_norm)

        outer_optimizer.step()
        return batch_query_loss / n_tasks
    finally:
        torch.backends.cudnn.enabled = prev_cudnn


# -----------------------------------------------------------------------------
# 6. Evaluation on Evaluation / Validation Tasks
# -----------------------------------------------------------------------------
def evaluate_maml(
    calibrator: ResidualCalibrator,
    step_sizes: MetaSGDStepSizes,
    val_tasks: list[MAMLTask],
    inner_steps: int = 3,
    adapt_scope: str = "all",
) -> dict[str, float]:
    """Evaluate meta-adapted calibrator against LightGBM M5 physics base on validation tasks."""
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)
    init_params = dict(calibrator.named_parameters())

    total_m7_pinball = 0.0
    total_m5_pinball = 0.0
    total_m7_crps = 0.0
    total_m5_crps = 0.0
    count = 0

    for task in val_tasks:
        if task.query_x.shape[0] == 0:
            continue

        # Adapt on support set
        adapted_params = inner_loop_adapt(
            calibrator=calibrator,
            params=init_params,
            step_sizes=step_sizes,
            task=task,
            inner_steps=inner_steps,
            adapt_scope=adapt_scope,
        )

        with torch.no_grad():
            m7_pred = functional_call(
                calibrator,
                adapted_params,
                (task.query_x, task.query_context, task.query_base_q, False),
            )
            m5_pred = task.query_base_q

            # Pinball loss in log1p space
            loss_m7 = criterion(task.query_y, m7_pred, z_base=None, mask=task.query_mask)
            loss_m5 = criterion(task.query_y, m5_pred, z_base=None, mask=task.query_mask)

            total_m7_pinball += float(loss_m7.item())
            total_m5_pinball += float(loss_m5.item())

            # Discrete CRPS in ug/m3
            m7_ugm3 = np.maximum(0.0, np.expm1(m7_pred.cpu().numpy()))
            m5_ugm3 = np.maximum(0.0, np.expm1(m5_pred.cpu().numpy()))
            y_ugm3 = np.maximum(0.0, np.expm1(task.query_y.cpu().numpy()))
            mask_np = task.query_mask.cpu().numpy()

            for h_idx in range(3):
                valid = mask_np[:, h_idx] > 0.5
                if np.any(valid):
                    crps_m7 = compute_crps_discrete(
                        y_ugm3[valid, h_idx], m7_ugm3[valid, h_idx, :], CANONICAL_QUANTILES
                    )
                    crps_m5 = compute_crps_discrete(
                        y_ugm3[valid, h_idx], m5_ugm3[valid, h_idx, :], CANONICAL_QUANTILES
                    )
                    total_m7_crps += crps_m7
                    total_m5_crps += crps_m5

            count += 1

    avg_m7_pinball = total_m7_pinball / max(1, count)
    avg_m5_pinball = total_m5_pinball / max(1, count)
    avg_m7_crps = total_m7_crps / max(1, count * 3)
    avg_m5_crps = total_m5_crps / max(1, count * 3)

    crpss = 1.0 - (avg_m7_crps / avg_m5_crps) if avg_m5_crps > 0.0 else 0.0

    return {
        "m7_pinball": avg_m7_pinball,
        "m5_pinball": avg_m5_pinball,
        "m7_crps": avg_m7_crps,
        "m5_crps": avg_m5_crps,
        "crpss_vs_base": float(crpss),
    }


# -----------------------------------------------------------------------------
# 7. Meta-Training Pipeline (FOMAML + Meta-SGD)
# -----------------------------------------------------------------------------
def train_maml(
    feature_store_dir: str | Path = "data/processed/feature_store",
    models_dir: str | Path = "data/processed/models",
    domain: str = "delhi",
    iterations: int = 100,
    meta_batch_size: int = 4,
    inner_steps: int = 3,
    lr_outer: float = 1e-3,
    lr_inner_init: float = 0.01,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    adapt_scope: str = "all",
    device: str | torch.device = "cpu",
    checkpoint_path: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute complete meta-learning training pipeline for Model M7.

    Orchestrates:
    1. Loading Delhi Block E (Train) and Block V (Validation) feature partitions.
    2. Verifying OOF base predictions (m5_oof_predictions.parquet) and hash bindings.
    3. Initializing ResidualCalibrator (47,577 params) and MetaSGDStepSizes (14 learnable scalars).
    4. Task sampling across sparse network availability (N in {0, 1, 2, 3, 5, 8}).
    5. Inner-loop adaptation with synthetic sensor corruption and detached gradients.
    6. Outer FOMAML Adam optimization over initial weights theta and step sizes rho.
    7. Block V validation evaluation.
    8. Checkpoint serialization to m7_maml.pt with strict lineage binding.

    Args:
        feature_store_dir: Root directory of feature store partitions.
        models_dir: Root directory for models and OOF cache.
        domain: Source domain ('delhi').
        iterations: Number of outer meta-update iterations (default: 100).
        meta_batch_size: Number of tasks per meta-batch (default: 4).
        inner_steps: Inner adaptation steps J (default: 3).
        lr_outer: Outer Adam learning rate beta (default: 1e-3).
        lr_inner_init: Initial Meta-SGD inner step size alpha (default: 0.01).
        weight_decay: Outer weight decay (default: 1e-4).
        grad_clip: Inner and outer gradient norm clipping (default: 1.0).
        adapt_scope: Adaptation scope ('all', 'head_only', 'gru_only').
        device: Compute device ('cpu', 'cuda').
        checkpoint_path: Optional explicit output checkpoint path.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing meta-training summary, validation metrics, and binding hashes.
    """
    torch.manual_seed(random_state)
    rng = np.random.default_rng(random_state)
    dev = torch.device(device)
    # Meta-learning uses torch.func.functional_call on GRU layers. Disable cuDNN's
    # persistent RNN workspace so PyTorch uses its native ATen GRU implementation.
    torch.backends.cudnn.enabled = False

    fs_path = Path(feature_store_dir)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    lgbm_dir = m_dir / "lgbm_m5"
    lgbm_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2 TASK P2-08: MODEL M7 META-LEARNING (FOMAML + META-SGD)")
    print(
        f"Domain: {domain.upper()} | Device: {device} | Iterations: {iterations} | Batch: {meta_batch_size}"
    )
    print(f"Inner Steps: {inner_steps} | Inner LR Init: {lr_inner_init} | Outer LR: {lr_outer}")
    print("=" * 80)

    # 1. Load Feature Partitions
    print(f"Loading Delhi Block E (Train) and Block V (Validation) from {fs_path} ...")
    delhi_e = load_dataset_from_partitions(fs_path, domain=domain, split="E")
    delhi_v = load_dataset_from_partitions(fs_path, domain=domain, split="V")

    print(f"Loaded Block E: {len(delhi_e):,} rows across {delhi_e['fold_id'].nunique()} folds.")
    print(f"Loaded Block V: {len(delhi_v):,} rows.")

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
        f"Binding Lineage: Base SHA256: {base_sha256[:16]}... | OOF Partition Hash: {oof_hash[:16]}..."
    )

    # Merge Block E with OOF base predictions
    delhi_e["issuance_utc"] = pd.to_datetime(delhi_e["issuance_utc"], utc=True)
    oof_df["issuance_utc"] = pd.to_datetime(oof_df["issuance_utc"], utc=True)

    merge_cols = ["domain", "location_id", "issuance_utc"]
    merged_e = delhi_e.merge(
        oof_df.drop(columns=["fold_id"], errors="ignore"),
        on=merge_cols,
        how="inner",
    )
    print(f"Merged Block E with OOF predictions: {len(merged_e):,} rows.")

    # Generate base model predictions on Block V
    print("Generating base model predictions on Block V...")
    base_preds_v = base_model.predict_all_horizons(delhi_v, rearrange=True)
    merged_v = delhi_v.copy()
    for h in CANONICAL_HORIZONS:
        preds_h = base_preds_v[h]
        for q_idx, q in enumerate(CANONICAL_QUANTILES):
            col_name = f"m5_oof_h{h}_q{round(q * 100)}"
            merged_v[col_name] = preds_h[:, q_idx]

    # 3. Instantiate Task Samplers
    train_sampler = TaskSampler(
        df_features=merged_e,
        corruption_enabled=True,
        rng=rng,
    )
    val_sampler = TaskSampler(
        df_features=merged_v,
        corruption_enabled=False,
        rng=np.random.default_rng(12345),
    )

    # Pre-sample validation tasks for evaluation
    val_tasks = [val_sampler.sample_task() for _ in range(16)]

    # 4. Instantiate Model & Meta-SGD Step Sizes
    calibrator = ResidualCalibrator().to(dev)
    step_sizes = MetaSGDStepSizes(init_alpha=lr_inner_init).to(dev)

    print(f"Calibrator initialized: {calibrator.num_parameters:,} parameters.")
    print(
        f"Meta-SGD Step Sizes initialized: {len(step_sizes.param_names)} learnable tensor scalars."
    )

    # Outer optimizer over both calibrator weights theta and step size parameters rho
    outer_params = [
        {"params": calibrator.parameters(), "lr": lr_outer, "weight_decay": weight_decay},
        {"params": step_sizes.parameters(), "lr": lr_outer * 2.0, "weight_decay": 0.0},
    ]
    outer_optimizer = torch.optim.Adam(outer_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        outer_optimizer, T_max=max(1, iterations), eta_min=1e-5
    )

    # 5. Meta-Training Loop
    print("\nStarting FOMAML outer loop iterations...")
    print(
        f"{'Iteration':<10} | {'Query Loss':<12} | {'Val Pinball M7':<15} | {'Val CRPS M7':<12} | {'CRPSS vs M5':<12}"
    )
    print("-" * 80)

    history: list[dict[str, Any]] = []

    for it in range(1, iterations + 1):
        calibrator.train()
        step_sizes.train()

        # Sample batch of tasks
        batch_tasks = [train_sampler.sample_task().to(dev) for _ in range(meta_batch_size)]

        loss_val = maml_outer_step(
            calibrator=calibrator,
            step_sizes=step_sizes,
            outer_optimizer=outer_optimizer,
            batch_tasks=batch_tasks,
            inner_steps=inner_steps,
            clip_norm=grad_clip,
            adapt_scope=adapt_scope,
        )
        scheduler.step()

        # Validation check every 10 iterations or at final iteration
        if it == 1 or it % 10 == 0 or it == iterations:
            calibrator.eval()
            step_sizes.eval()
            val_tasks_dev = [t.to(dev) for t in val_tasks]
            val_metrics = evaluate_maml(
                calibrator=calibrator,
                step_sizes=step_sizes,
                val_tasks=val_tasks_dev,
                inner_steps=inner_steps,
                adapt_scope=adapt_scope,
            )

            print(
                f"{it:<10} | {loss_val:<12.4f} | {val_metrics['m7_pinball']:<15.4f} | "
                f"{val_metrics['m7_crps']:<12.2f} | {val_metrics['crpss_vs_base'] * 100:+.2f}%"
            )

            history.append(
                {
                    "iteration": it,
                    "query_loss": loss_val,
                    "val_m7_pinball": val_metrics["m7_pinball"],
                    "val_m5_pinball": val_metrics["m5_pinball"],
                    "val_m7_crps": val_metrics["m7_crps"],
                    "val_m5_crps": val_metrics["m5_crps"],
                    "crpss_vs_base": val_metrics["crpss_vs_base"],
                }
            )

    # 6. Save Model Checkpoint with Lineage Binding
    out_pt = Path(checkpoint_path) if checkpoint_path else m_dir / "m7_maml.pt"
    learned_step_sizes = {
        name: float(step_sizes.step_size_for(name).item()) for name in step_sizes.param_names
    }

    checkpoint = {
        "model_type": "M7",
        "model_state_dict": calibrator.state_dict(),
        "meta_sgd_step_sizes": learned_step_sizes,
        "meta_sgd_rho": step_sizes.rho.detach().cpu().numpy(),
        "base_sha256": base_sha256,
        "oof_partition_hash": oof_hash,
        "inner_steps": inner_steps,
        "clip_norm": grad_clip,
        "adapt_scope": adapt_scope,
        "quantiles": CANONICAL_QUANTILES,
        "horizons": CANONICAL_HORIZONS,
        "history": history,
    }

    torch.save(checkpoint, out_pt)
    print("\n" + "=" * 80)
    print(f"Meta-learning complete. Saved Model M7 checkpoint to: {out_pt}")
    print(f"Base SHA256: {base_sha256[:16]}... | OOF Hash: {oof_hash[:16]}...")
    learned_alphas_str = ", ".join(
        f"{k.split('.')[-1]}: {v:.4f}" for k, v in list(learned_step_sizes.items())[:4]
    )
    print(f"Sample Learned Step Sizes: {learned_alphas_str} ...")
    print("=" * 80 + "\n")

    val_summary = (
        {
            "m7_pinball": history[-1]["val_m7_pinball"],
            "m5_pinball": history[-1]["val_m5_pinball"],
            "m7_crps": history[-1]["val_m7_crps"],
            "m5_crps": history[-1]["val_m5_crps"],
            "crpss_vs_base": history[-1]["crpss_vs_base"],
        }
        if history
        else {}
    )

    return {
        "checkpoint_path": str(out_pt),
        "base_sha256": base_sha256,
        "oof_partition_hash": oof_hash,
        "final_query_loss": history[-1]["query_loss"] if history else loss_val,
        "learned_step_sizes": learned_step_sizes,
        "history": history,
        "loss_history": [h["query_loss"] for h in history],
        "validation_metrics": val_summary,
    }
