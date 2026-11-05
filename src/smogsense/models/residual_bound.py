"""smogsense.models.residual_bound — Residual Bound Calibration and Acceptance Check.

Calibrates the deployment residual bound threshold rho_99 exclusively on Delhi Block V
(tuning season) and implements the deployment acceptance gate under Revision 3 (ADR-024).

Mathematical & Architectural Specification:
1. Support Window Split (Condition C6):
   Chronological split of available stations' support window:
   - fit part (earlier)
   - 72-hour embargo
   - check part (latest)
2. Inner Loop Adaptation (Condition C5):
   Adapt theta -> theta' on fit part using the shared recipe:
   J = 3 steps, Meta-SGD alpha step sizes (or fixed alpha = 0.01), pinball loss on 19 levels,
   gradient clip at 1.0.
3. Check Part Acceptance Gate (Condition C6):
   - Check pinball relative improvement >= min_improvement (default: 2% / 0.02)
   - Daily win fraction >= min_days_better_fraction (default: 60% / 0.60)
   - Residual statistic R = mean(|Delta z_hat|) <= rho_99 (over check rows, horizons, levels in log1p units)
   - Acceptance status in {"adapted", "rejected_no_gain", "rejected_residual_bound", "skipped_n0", "skipped_insufficient_support"}
   - Rejection sets Delta z = 0, publishing M5 bit-for-bit without degrading pipeline level.
4. Calibration of rho_99 on Block V:
   Simulates deployments on Block V tasks (drawing N in {1, 2, 3, 5, 8} and support windows from {7, 14, 30} days),
   executes the fit/embargo/check adaptation recipe, records R on each check window, and sets
   rho_99 = percentile(R, 99.0).

Specification: docs/ml-architecture.md §8 & ADR-024 §6, §7.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from structlog import get_logger
from torch.func import functional_call

from smogsense.features.sparse_network import sample_available_stations
from smogsense.models.heads import (
    CANONICAL_QUANTILES,
    ResidualCalibrator,
)
from smogsense.models.losses import pinball_loss
from smogsense.models.maml import (
    MAMLTask,
    MetaSGDStepSizes,
    inner_loop_adapt,
)
from smogsense.models.pretraining import (
    apply_sensor_corruption,
    build_fast_sequence_tensors,
    extract_base_quantiles_tensor,
    extract_tabular_context,
    extract_targets_and_masks,
)

logger = get_logger(__name__)

AdaptationStatus = Literal[
    "adapted",
    "rejected_no_gain",
    "rejected_residual_bound",
    "skipped_n0",
    "skipped_insufficient_support",
]


@dataclass(frozen=True)
class AdaptationAcceptanceResult:
    """Result of deployment adaptation acceptance gate (Condition C6)."""

    accepted: bool
    adaptation_status: AdaptationStatus
    relative_improvement: float
    days_better_fraction: float
    residual_r: float
    rho_99: float
    check_pinball_m5: float
    check_pinball_adapted: float
    check_pinball_unadapted: float | None = None
    n_check_samples: int = 0
    n_check_days: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return asdict(self)


def split_support_window(
    df: pd.DataFrame,
    embargo_hours: int = 72,
    check_fraction: float = 0.35,
    max_check_days: int = 7,
    min_check_hours: int = 24,
    min_fit_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split support window chronologically into fit part, embargo, and check part.

    Args:
        df: Support observations dataframe sorted by issuance_utc.
        embargo_hours: Length of boundary embargo in hours (default: 72).
        check_fraction: Fraction of usable window allocated to check part (default: 0.35).
        max_check_days: Cap on check part duration in days (default: 7).
        min_check_hours: Minimum check duration in hours (default: 24).
        min_fit_hours: Minimum fit duration in hours (default: 24).

    Returns:
        tuple (df_fit, df_check). Returns (empty, empty) if window is too short.
    """
    if df.empty or "issuance_utc" not in df.columns:
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    df_sorted = df.sort_values("issuance_utc").reset_index(drop=True)
    t_min = pd.to_datetime(df_sorted["issuance_utc"].iloc[0], utc=True)
    t_max = pd.to_datetime(df_sorted["issuance_utc"].iloc[-1], utc=True)

    total_h = (t_max - t_min).total_seconds() / 3600.0
    usable_h = total_h - embargo_hours

    if usable_h < (min_fit_hours + min_check_hours):
        return df.iloc[:0].copy(), df.iloc[:0].copy()

    check_h = min(max_check_days * 24.0, max(min_check_hours * 1.0, usable_h * check_fraction))
    t_check_start = t_max - pd.Timedelta(hours=check_h)
    t_fit_end = t_check_start - pd.Timedelta(hours=embargo_hours)

    ts_series = pd.to_datetime(df_sorted["issuance_utc"], utc=True)
    df_fit = df_sorted[ts_series < t_fit_end].reset_index(drop=True)
    df_check = df_sorted[ts_series >= t_check_start].reset_index(drop=True)

    return df_fit, df_check


def compute_residual_statistic(delta_z: np.ndarray | torch.Tensor) -> float:
    """Compute deployment residual statistic R = mean(|Delta z_hat|).

    Formula (docs/ml-architecture.md §8 & ADR-024):
        R = 1 / (|D_check| x 3 x 19) * sum_{i, h, tau} |Delta z_hat_{tau, h}(x_i)|
    Evaluated in log1p units across all check window rows, horizons, and quantile levels.

    Args:
        delta_z: Tensor or array of shape (N, 3, 19) or (N, 57).

    Returns:
        Scalar mean absolute residual R >= 0.0.
    """
    if isinstance(delta_z, torch.Tensor):
        if delta_z.numel() == 0:
            return 0.0
        return float(torch.mean(torch.abs(delta_z)).item())

    arr = np.asarray(delta_z)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def evaluate_adaptation_acceptance(
    y_true: np.ndarray | torch.Tensor,
    y_mask: np.ndarray | torch.Tensor,
    m5_pred: np.ndarray | torch.Tensor,
    adapted_pred: np.ndarray | torch.Tensor,
    dates: Sequence[Any] | np.ndarray | pd.Series,
    rho_99: float,
    unadapted_pred: np.ndarray | torch.Tensor | None = None,
    min_improvement: float = 0.02,
    min_days_better_fraction: float = 0.60,
    n_available: int = 1,
    support_days: float = 14.0,
    in_log1p: bool = True,
    quantiles: Sequence[float] = CANONICAL_QUANTILES,
) -> AdaptationAcceptanceResult:
    """Evaluate deployment acceptance gate on check window (Condition C6).

    Args:
        y_true: Ground truth target array/tensor of shape (N, 3).
        y_mask: Boolean target validity mask of shape (N, 3).
        m5_pred: M5 baseline quantile predictions of shape (N, 3, 19).
        adapted_pred: Adapted model quantile predictions of shape (N, 3, 19).
        dates: Sequence of dates or datetimes corresponding to each check sample (length N).
        rho_99: Calibrated residual bound threshold from Block V.
        unadapted_pred: Optional J=0 unadapted predictions of shape (N, 3, 19) for diagnostics.
        min_improvement: Minimum relative pinball loss improvement (default: 0.02 / 2%).
        min_days_better_fraction: Minimum fraction of days adapted beats M5 (default: 0.60 / 60%).
        n_available: Number of available stations N.
        support_days: Duration of support window in days.
        in_log1p: Whether inputs are already in log1p space (default: True).
        quantiles: Canonical 19 quantile levels.

    Returns:
        AdaptationAcceptanceResult containing acceptance decision and diagnostics.
    """
    # 1. Edge Case: N = 0 (Sensor-free mode)
    if n_available == 0:
        return AdaptationAcceptanceResult(
            accepted=False,
            adaptation_status="skipped_n0",
            relative_improvement=0.0,
            days_better_fraction=0.0,
            residual_r=0.0,
            rho_99=rho_99,
            check_pinball_m5=0.0,
            check_pinball_adapted=0.0,
            n_check_samples=0,
            n_check_days=0,
            details={"reason": "N=0 available stations; sensor-free mode uses frozen base M5"},
        )

    # 2. Edge Case: Insufficient Support
    n_samples = len(dates)
    if support_days < 7.0 or n_samples == 0:
        return AdaptationAcceptanceResult(
            accepted=False,
            adaptation_status="skipped_insufficient_support",
            relative_improvement=0.0,
            days_better_fraction=0.0,
            residual_r=0.0,
            rho_99=rho_99,
            check_pinball_m5=0.0,
            check_pinball_adapted=0.0,
            n_check_samples=n_samples,
            n_check_days=0,
            details={"reason": f"Insufficient support: {support_days} days, {n_samples} samples"},
        )

    # Convert inputs to float64 numpy arrays
    y_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    mask_np = (
        y_mask.detach().cpu().numpy().astype(bool)
        if isinstance(y_mask, torch.Tensor)
        else np.asarray(y_mask, dtype=bool)
    )
    m5_np = (
        m5_pred.detach().cpu().numpy() if isinstance(m5_pred, torch.Tensor) else np.asarray(m5_pred)
    )
    adp_np = (
        adapted_pred.detach().cpu().numpy()
        if isinstance(adapted_pred, torch.Tensor)
        else np.asarray(adapted_pred)
    )

    if not in_log1p:
        y_np = np.log1p(np.maximum(0.0, y_np))
        m5_np = np.log1p(np.maximum(0.0, m5_np))
        adp_np = np.log1p(np.maximum(0.0, adp_np))

    # Compute residual statistic R = mean(|Delta z_hat|)
    delta_z = adp_np - m5_np
    residual_r = compute_residual_statistic(delta_z)

    # Compute unadapted diagnostic pinball loss if provided
    unadapted_loss: float | None = None
    if unadapted_pred is not None:
        unadp_np = (
            unadapted_pred.detach().cpu().numpy()
            if isinstance(unadapted_pred, torch.Tensor)
            else np.asarray(unadapted_pred)
        )
        if not in_log1p:
            unadp_np = np.log1p(np.maximum(0.0, unadp_np))
        u_losses = [
            pinball_loss(y_np[mask_np[:, h], h], unadp_np[mask_np[:, h], h, i], tau)
            for h in range(3)
            if mask_np[:, h].any()
            for i, tau in enumerate(quantiles)
        ]
        unadapted_loss = float(np.mean(np.concatenate(u_losses))) if u_losses else None

    # Compute overall check pinball losses
    m5_losses = [
        pinball_loss(y_np[mask_np[:, h], h], m5_np[mask_np[:, h], h, i], tau)
        for h in range(3)
        if mask_np[:, h].any()
        for i, tau in enumerate(quantiles)
    ]
    adp_losses = [
        pinball_loss(y_np[mask_np[:, h], h], adp_np[mask_np[:, h], h, i], tau)
        for h in range(3)
        if mask_np[:, h].any()
        for i, tau in enumerate(quantiles)
    ]

    if not m5_losses or not adp_losses:
        return AdaptationAcceptanceResult(
            accepted=False,
            adaptation_status="skipped_insufficient_support",
            relative_improvement=0.0,
            days_better_fraction=0.0,
            residual_r=residual_r,
            rho_99=rho_99,
            check_pinball_m5=0.0,
            check_pinball_adapted=0.0,
            check_pinball_unadapted=unadapted_loss,
            n_check_samples=n_samples,
            n_check_days=0,
            details={"reason": "No valid masked targets found in check window"},
        )

    check_pb_m5 = float(np.mean(np.concatenate(m5_losses)))
    check_pb_adp = float(np.mean(np.concatenate(adp_losses)))

    rel_improvement = 0.0 if check_pb_m5 <= 1e-8 else (check_pb_m5 - check_pb_adp) / check_pb_m5

    # Compute daily win fraction
    # Standardize dates to strings YYYY-MM-DD
    dt_series = pd.to_datetime(pd.Series(list(dates)), utc=True).dt.strftime("%Y-%m-%d")
    unique_days = sorted(dt_series.unique())

    daily_m5_wins = 0
    daily_adp_wins = 0
    valid_days = 0

    for day_str in unique_days:
        day_indices = np.where(dt_series == day_str)[0]
        day_m5_losses = [
            pinball_loss(y_np[day_idx, h], m5_np[day_idx, h, i], tau)
            for day_idx in day_indices
            for h in range(3)
            if mask_np[day_idx, h]
            for i, tau in enumerate(quantiles)
        ]
        day_adp_losses = [
            pinball_loss(y_np[day_idx, h], adp_np[day_idx, h, i], tau)
            for day_idx in day_indices
            for h in range(3)
            if mask_np[day_idx, h]
            for i, tau in enumerate(quantiles)
        ]
        if day_m5_losses and day_adp_losses:
            valid_days += 1
            if np.mean(day_adp_losses) < np.mean(day_m5_losses):
                daily_adp_wins += 1
            else:
                daily_m5_wins += 1

    days_better_frac = (daily_adp_wins / valid_days) if valid_days > 0 else 0.0

    # Acceptance Decision (ADR-024 & docs/ml-architecture.md §8)
    gain_passed = (rel_improvement >= min_improvement) and (
        days_better_frac >= min_days_better_fraction
    )
    bound_passed = residual_r <= rho_99

    if gain_passed and bound_passed:
        status: AdaptationStatus = "adapted"
        accepted = True
    elif not gain_passed:
        status = "rejected_no_gain"
        accepted = False
    else:
        status = "rejected_residual_bound"
        accepted = False

    return AdaptationAcceptanceResult(
        accepted=accepted,
        adaptation_status=status,
        relative_improvement=rel_improvement,
        days_better_fraction=days_better_frac,
        residual_r=residual_r,
        rho_99=rho_99,
        check_pinball_m5=check_pb_m5,
        check_pinball_adapted=check_pb_adp,
        check_pinball_unadapted=unadapted_loss,
        n_check_samples=n_samples,
        n_check_days=valid_days,
        details={
            "gain_passed": gain_passed,
            "bound_passed": bound_passed,
            "min_improvement": min_improvement,
            "min_days_better_fraction": min_days_better_fraction,
            "daily_adp_wins": daily_adp_wins,
            "daily_m5_wins": daily_m5_wins,
        },
    )


def simulate_deployments_on_block_v(
    feature_store_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    df_block_v: pd.DataFrame | None = None,
    calibrator: ResidualCalibrator | None = None,
    step_sizes: MetaSGDStepSizes | dict[str, torch.Tensor] | None = None,
    n_simulations: int = 200,
    available_n_sampler: Sequence[int] = (1, 2, 3, 5, 8),
    support_window_days: Sequence[int] = (7, 14, 30),
    embargo_hours: int = 72,
    inner_steps: int = 3,
    clip_norm: float = 1.0,
    corruption_enabled: bool = True,
    seed: int = 42,
    device: torch.device | str = "cpu",
    output_json: Path | str | None = None,
) -> dict[str, Any]:
    """Calibrate residual bound threshold rho_99 exclusively on Delhi Block V.

    Simulates M deployments across available station counts N and support durations D,
    executes the shared adaptation recipe on the fit part, and records the residual
    statistic R = mean(|Delta z_hat|) on each simulated check window.

    Args:
        feature_store_dir: Path to processed feature store root.
        models_dir: Path to directory containing trained models (m7_maml.pt / m6_calibrator.pt).
        df_block_v: Optional pre-loaded Block V feature dataframe.
        calibrator: Optional pre-instantiated ResidualCalibrator.
        step_sizes: Optional pre-instantiated MetaSGDStepSizes or step-size dict.
        n_simulations: Total number of deployment simulations M (default: 200).
        available_n_sampler: Distribution of available station counts N (default: (1, 2, 3, 5, 8)).
        support_window_days: Possible support durations in days (default: (7, 14, 30)).
        embargo_hours: Embargo separating fit and check parts in hours (default: 72).
        inner_steps: Inner loop gradient steps J (default: 3).
        clip_norm: Maximum L2 gradient norm (default: 1.0).
        corruption_enabled: Whether to apply synthetic low-cost corruption to fit inputs (default: True).
        seed: Random seed for deterministic reproducibility.
        device: Torch compute device (default: 'cpu').
        output_json: Optional destination path to write residual_bound.json.

    Returns:
        Summary dictionary containing rho_99, percentiles, simulation count, and metadata.
    """
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    # 1. Load Block V Data if not directly provided
    if df_block_v is None:
        if feature_store_dir is None:
            raise ValueError("Either df_block_v or feature_store_dir must be provided.")
        fs_path = Path(feature_store_dir)
        v_files = sorted(fs_path.glob("delhi/season=2024-2025/**/features.parquet"))
        if not v_files:
            v_files = sorted(fs_path.glob("delhi/**/features.parquet"))
        dfs: list[pd.DataFrame] = []
        for f in v_files:
            df_part = pd.read_parquet(f)
            if "source_split" in df_part.columns:
                df_part = df_part[df_part["source_split"] == "V"]
            if not df_part.empty:
                dfs.append(df_part)
        if not dfs:
            raise FileNotFoundError(f"No Delhi Block V partitions found under {fs_path}")
        df_v = pd.concat(dfs, ignore_index=True)
    else:
        df_v = df_block_v.copy()

    df_v["issuance_utc"] = pd.to_datetime(df_v["issuance_utc"], utc=True)
    df_v = df_v.sort_values("issuance_utc").reset_index(drop=True)

    # 2. Extract Station Metadata for Twin-Aware Sampler
    id_col = "location_id"
    lat_col = next((c for c in ("f_stat_latitude", "latitude", "lat") if c in df_v.columns), None)
    lon_col = next((c for c in ("f_stat_longitude", "longitude", "lon") if c in df_v.columns), None)

    if lat_col and lon_col:
        stations_df = (
            df_v[[id_col, lat_col, lon_col]]
            .drop_duplicates(subset=[id_col])
            .rename(columns={lat_col: "latitude", lon_col: "longitude"})
            .reset_index(drop=True)
        )
    else:
        u_ids = df_v[id_col].unique()
        stations_df = pd.DataFrame(
            {
                id_col: u_ids,
                "latitude": np.linspace(28.5, 28.7, len(u_ids)),
                "longitude": np.linspace(77.1, 77.3, len(u_ids)),
            }
        )

    # 3. Load Calibrator and Step Sizes if not directly provided
    base_sha256 = "none"
    ckpt_name = "in_memory"

    if calibrator is None:
        if models_dir is None:
            raise ValueError("Either calibrator or models_dir must be provided.")
        m_dir = Path(models_dir)
        m7_path = m_dir / "m7_maml.pt"
        m6_path = m_dir / "m6_calibrator.pt"

        model = ResidualCalibrator().to(dev)
        if m7_path.exists():
            ckpt = torch.load(m7_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            base_sha256 = ckpt.get("base_sha256", "none")
            ckpt_name = "m7_maml.pt"
            if step_sizes is None:
                raw_sizes = ckpt.get("meta_sgd_step_sizes")
                if isinstance(raw_sizes, dict):
                    step_sizes = {
                        k: (
                            v.to(dev)
                            if isinstance(v, torch.Tensor)
                            else torch.tensor(float(v), device=dev)
                        )
                        for k, v in raw_sizes.items()
                    }
                else:
                    step_sizes = MetaSGDStepSizes(init_alpha=0.01).to(dev)
        elif m6_path.exists():
            ckpt = torch.load(m6_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            base_sha256 = ckpt.get("base_sha256", "none")
            ckpt_name = "m6_calibrator.pt"
            if step_sizes is None:
                step_sizes = MetaSGDStepSizes(init_alpha=0.01).to(dev)
        else:
            logger.warning(
                "No trained checkpoint found; using zero-initialized calibrator", dir=str(m_dir)
            )
            ckpt_name = "zero_init"
            if step_sizes is None:
                step_sizes = MetaSGDStepSizes(init_alpha=0.01).to(dev)
        calibrator = model

    if step_sizes is None:
        step_sizes = MetaSGDStepSizes(init_alpha=0.01).to(dev)

    calibrator.eval()
    initial_params: dict[str, torch.Tensor] = {
        k: v.clone().detach() for k, v in calibrator.named_parameters()
    }

    # 4. Run M Deployment Simulations
    r_values: list[float] = []
    min_time = df_v["issuance_utc"].min()
    max_time = df_v["issuance_utc"].max()
    total_hours = int((max_time - min_time).total_seconds() // 3600)

    logger.info(
        "starting_block_v_deployment_simulations",
        n_simulations=n_simulations,
        n_rows=len(df_v),
        span_hours=total_hours,
        calibrator=ckpt_name,
    )

    attempts = 0
    max_attempts = n_simulations * 5

    while len(r_values) < n_simulations and attempts < max_attempts:
        attempts += 1
        n = int(rng.choice(available_n_sampler))
        d_days = int(rng.choice(support_window_days))

        # Sample available stations
        avail_set, _, _ = sample_available_stations(
            stations_df,
            n=n,
            available_n_sampler=available_n_sampler,
            rng=rng,
        )
        if not avail_set:
            continue

        avail_list = sorted(avail_set)
        df_avail = df_v[df_v["location_id"].isin(avail_list)].copy()

        # Sample window start
        window_span_h = d_days * 24
        if total_hours > window_span_h:
            offset_h = int(rng.integers(0, total_hours - window_span_h))
            t_start = min_time + pd.Timedelta(hours=offset_h)
        else:
            t_start = min_time

        t_end = t_start + pd.Timedelta(days=d_days)
        df_window = df_avail[
            (df_avail["issuance_utc"] >= t_start) & (df_avail["issuance_utc"] <= t_end)
        ].copy()

        if df_window.empty:
            continue

        # Split into fit part, embargo, check part
        df_fit, df_check = split_support_window(df_window, embargo_hours=embargo_hours)
        if df_fit.empty or df_check.empty:
            continue

        # Extract tensors
        fit_x = torch.from_numpy(build_fast_sequence_tensors(df_fit)).float().to(dev)
        fit_u = torch.from_numpy(extract_tabular_context(df_fit)).float().to(dev)
        fit_base = torch.from_numpy(extract_base_quantiles_tensor(df_fit)).float().to(dev)
        fit_y_np, fit_m_np = extract_targets_and_masks(df_fit)
        fit_y = torch.from_numpy(fit_y_np).float().to(dev)
        fit_mask = torch.from_numpy(fit_m_np).float().to(dev)

        if fit_mask.sum() == 0:
            continue

        # Apply synthetic low-cost corruption to fit inputs if enabled
        if corruption_enabled:
            a_val = float(rng.uniform(0.8, 1.3))
            b_val = float(rng.uniform(0.0, 1.5))
            c_val = float(rng.uniform(-10.0, 25.0))
            fit_x = apply_sensor_corruption(
                fit_x,
                p=1.0,
                gain_range=(a_val, a_val),
                rh_slope_range=(b_val, b_val),
                offset_range=(c_val, c_val),
            )

        # Check tensors
        chk_x = torch.from_numpy(build_fast_sequence_tensors(df_check)).float().to(dev)
        chk_u = torch.from_numpy(extract_tabular_context(df_check)).float().to(dev)

        fit_task = MAMLTask(
            domain="delhi",
            season="2024-2025",
            available_stations=avail_list,
            held_out_stations=[],
            support_x=fit_x,
            support_context=fit_u,
            support_base_q=fit_base,
            support_y=fit_y,
            support_mask=fit_mask,
            query_x=chk_x,
            query_context=chk_u,
            query_base_q=torch.zeros(0, 3, 19, device=dev),
            query_y=torch.zeros(0, 3, device=dev),
            query_mask=torch.zeros(0, 3, device=dev),
            n_available=n,
        )

        # Adapt theta -> theta' on fit part
        adapted_params = inner_loop_adapt(
            calibrator=calibrator,
            params=initial_params,
            step_sizes=step_sizes,
            task=fit_task,
            inner_steps=inner_steps,
            clip_norm=clip_norm,
        )

        # Evaluate Delta z on check part
        with torch.no_grad():
            delta_z_chk = functional_call(
                calibrator,
                adapted_params,
                (chk_x, chk_u, None, False),
            )
            r_sim = compute_residual_statistic(delta_z_chk)
            r_values.append(r_sim)

    if not r_values:
        # Fallback safeguard if all simulation windows were too sparse
        r_values = [0.0]

    r_arr = np.array(r_values, dtype=np.float64)
    rho_50 = float(np.percentile(r_arr, 50.0))
    rho_90 = float(np.percentile(r_arr, 90.0))
    rho_95 = float(np.percentile(r_arr, 95.0))
    rho_99 = float(np.percentile(r_arr, 99.0))
    r_max = float(np.max(r_arr))

    summary = {
        "statistic": "mean_abs_residual",
        "threshold_quantile": 0.99,
        "rho_99": rho_99,
        "percentiles": {
            "p50": rho_50,
            "p90": rho_90,
            "p95": rho_95,
            "p99": rho_99,
            "max": r_max,
        },
        "n_simulations": len(r_values),
        "source_domain": "delhi",
        "season": "2024-2025",
        "block": "V",
        "base_sha256": base_sha256,
        "calibrator_checkpoint": ckpt_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    logger.info(
        "calibrated_residual_bound_rho_99",
        rho_99=rho_99,
        p95=rho_95,
        p50=rho_50,
        n_sims=len(r_values),
    )

    if output_json is not None:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", newline="\n") as json_file:
            json.dump(summary, json_file, indent=2)
            json_file.write("\n")
        logger.info("saved_residual_bound_json", path=str(out_p))

    return summary


__all__ = [
    "AdaptationAcceptanceResult",
    "AdaptationStatus",
    "compute_residual_statistic",
    "evaluate_adaptation_acceptance",
    "simulate_deployments_on_block_v",
    "split_support_window",
]
