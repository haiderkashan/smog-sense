"""smogsense.models.adaptation — Few-shot deployment adaptation and acceptance gate engine.

Runs J=3 inner-loop steps on the neural residual calibrator (ResidualCalibrator: GRU + head)
using only available stations' labels and strictly earlier data. LightGBM physics base M5 is frozen.
Includes chronological fit/embargo/check split, residual-bound check (R <= rho_99),
acceptance gate (Condition C6), canonical adaptation_status assignment (Condition C7),
and exact fallback Delta z = 0 (= M5) on skip or rejection with degradation level 0.

Specification:
- docs/ml-architecture.md §6, §8 ('Meta-learning (MAML, Meta-SGD) and adaptation')
- docs/test-specifications.md §T1-16 ('Adaptation outcomes') & §T1-17 ('Peak RSS Memory Guard')
- decisions.md (ADR-002, ADR-003, ADR-015, ADR-024)
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from structlog import get_logger
from torch.func import functional_call

from smogsense.models.heads import (
    CANONICAL_HORIZONS,
    CANONICAL_QUANTILES,
    ResidualCalibrator,
)
from smogsense.models.lgbm_quantile import (
    LGBMQuantileEnsemble,
    rearrange_quantiles,
)
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
from smogsense.models.residual_bound import (
    AdaptationAcceptanceResult,
    AdaptationStatus,
    evaluate_adaptation_acceptance,
    split_support_window,
)

logger = get_logger(__name__)

# Calibrated residual bound threshold from Block V (Task P2-12)
DEFAULT_RHO_99 = 0.362452

# Configured Barkjohn sensor corruption parameter ranges
DEFAULT_CORRUPTION_PRIORS: dict[str, tuple[float, float]] = {
    "gain": (0.8, 1.3),
    "rh_slope": (0.0, 1.5),
    "offset": (-10.0, 25.0),
}


def get_process_rss_mb() -> float:
    """Return current process resident set size in megabytes using OS standard facilities."""
    try:
        import resource

        getrusage = getattr(resource, "getrusage", None)
        rusage_self = getattr(resource, "RUSAGE_SELF", 0)
        if callable(getrusage):
            ru = getrusage(rusage_self)
            return float(getattr(ru, "ru_maxrss", 0.0)) / 1024.0
    except (ImportError, AttributeError):
        pass

    try:
        with Path("/proc/self/status").open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0
    except (FileNotFoundError, PermissionError, IndexError, ValueError):
        pass

    return 0.0


class DeploymentAdaptationEngine:
    """Production few-shot deployment adaptation and inference engine.

    Coordinates the adaptation of the neural residual calibrator (Model M6 or M7)
    on top of the frozen LightGBM physics base (Model M5) following ADR-024
    (Physics-Residual Meta-Quantile Learning) Conditions C5, C6, C7, and C8.
    """

    def __init__(
        self,
        base_model: LGBMQuantileEnsemble,
        calibrator: ResidualCalibrator,
        step_sizes: MetaSGDStepSizes | Mapping[str, torch.Tensor] | float = 0.01,
        rho_99: float = DEFAULT_RHO_99,
        embargo_hours: int = 72,
        inner_steps: int = 3,
        clip_norm: float = 1.0,
        min_improvement: float = 0.02,
        min_days_better_fraction: float = 0.60,
        min_support_days: float = 7.0,
        adapt_scope: str = "all",
        corruption_priors: dict[str, tuple[float, float]] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.rho_99 = float(rho_99)
        self.embargo_hours = int(embargo_hours)
        self.inner_steps = int(inner_steps)
        self.clip_norm = float(clip_norm)
        self.min_improvement = float(min_improvement)
        self.min_days_better_fraction = float(min_days_better_fraction)
        self.min_support_days = float(min_support_days)
        self.adapt_scope = adapt_scope
        self.corruption_priors = corruption_priors or DEFAULT_CORRUPTION_PRIORS
        self.device = torch.device(device)

        self.calibrator.to(self.device)
        self.calibrator.eval()

        if not isinstance(step_sizes, (int, float, MetaSGDStepSizes, Mapping)):
            raise TypeError(f"Unsupported step_sizes type: {type(step_sizes).__name__}")
        self.step_sizes = step_sizes
        if isinstance(self.step_sizes, MetaSGDStepSizes):
            self.step_sizes.to(self.device)

        # Store adapted parameters; None indicates fallback to M5 (Delta z = 0)
        self.adapted_params: dict[str, torch.Tensor] | None = None
        self.last_result: AdaptationAcceptanceResult | None = None

    @property
    def is_adapted(self) -> bool:
        """Whether the engine currently holds an accepted adapted residual model."""
        return self.adapted_params is not None

    @property
    def adaptation_status(self) -> AdaptationStatus:
        """Current adaptation status (Condition C7)."""
        if self.last_result is not None:
            return self.last_result.adaptation_status
        return "skipped_n0"

    def _resolve_step_sizes(self) -> dict[str, torch.Tensor]:
        """Resolve step sizes to parameter-name mapping."""
        if isinstance(self.step_sizes, MetaSGDStepSizes):
            return self.step_sizes.get_step_sizes()
        if isinstance(self.step_sizes, Mapping):
            return {k: v.to(self.device) for k, v in self.step_sizes.items()}
        if isinstance(self.step_sizes, (int, float)):
            val = float(self.step_sizes)
            return {
                name: torch.tensor(val, dtype=torch.float32, device=self.device)
                for name, _ in self.calibrator.named_parameters()
            }
        raise TypeError(f"Unsupported step_sizes type: {type(self.step_sizes).__name__}")

    def _ensure_base_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """Extract or generate (N, 3, 19) base quantiles in log1p space."""
        has_oof = any(c.startswith("m5_oof_") or c.startswith("base_") for c in df.columns)
        if has_oof:
            return extract_base_quantiles_tensor(df)
        base_preds = self.base_model.predict_all_horizons(df, rearrange=True)
        return extract_base_quantiles_tensor(df, base_predictions=base_preds)

    def adapt(
        self,
        support_df: pd.DataFrame,
        n_available: int | None = None,
        is_reference_sensor: bool = False,
        seed: int = 42,
    ) -> AdaptationAcceptanceResult:
        """Execute full deployment adaptation lifecycle and acceptance gate.

        Lifecycle (ADR-024 Condition C6 & C7):
        1. Boundary check: N=0 => skipped_n0; duration < 7 days => skipped_insufficient_support.
        2. Split support window chronologically: fit part, 72h embargo, check part.
        3. Run J=3 inner-loop adaptation steps on fit part using shared recipe (Condition C5).
        4. Evaluate check window acceptance criteria (>=2% pinball gain, >=60% daily win, R <= rho_99).
        5. If accepted: re-adapt on full support window and store parameters.
           If rejected or skipped: enforce Delta z = 0 (Model M5 fallback).
        6. Release memory graphs and return AdaptationAcceptanceResult.

        Args:
            support_df: DataFrame containing support window observations up to issuance T.
            n_available: Number of available stations N (inferred if None).
            is_reference_sensor: True if adapting on reference monitor (no sensor corruption).
            seed: Random seed for deterministic reproducibility.

        Returns:
            AdaptationAcceptanceResult containing acceptance decision and diagnostics.
        """
        # Determine N available stations
        if n_available is None:
            if not support_df.empty and "location_id" in support_df.columns:
                n_avail = int(support_df["location_id"].nunique())
            else:
                n_avail = 0 if support_df.empty else 1
        else:
            n_avail = int(n_available)

        # 1. Edge Case: N = 0 (Sensor-Free Mode, Condition C7)
        if n_avail == 0 or support_df.empty:
            self.adapted_params = None
            res = evaluate_adaptation_acceptance(
                y_true=np.zeros((0, 3)),
                y_mask=np.zeros((0, 3), dtype=bool),
                m5_pred=np.zeros((0, 3, 19)),
                adapted_pred=np.zeros((0, 3, 19)),
                dates=[],
                rho_99=self.rho_99,
                n_available=0,
                support_days=0.0,
            )
            self.last_result = res
            return res

        # 2. Edge Case: Insufficient Support Duration (< 7 days)
        support_sorted = support_df.sort_values("issuance_utc").reset_index(drop=True)
        t_start = pd.to_datetime(support_sorted["issuance_utc"].iloc[0], utc=True)
        t_end = pd.to_datetime(support_sorted["issuance_utc"].iloc[-1], utc=True)
        total_days = max(0.0, (t_end - t_start).total_seconds() / 86400.0)

        if total_days < self.min_support_days or len(support_sorted) < 4:
            self.adapted_params = None
            res = evaluate_adaptation_acceptance(
                y_true=np.zeros((len(support_sorted), 3)),
                y_mask=np.zeros((len(support_sorted), 3), dtype=bool),
                m5_pred=np.zeros((len(support_sorted), 3, 19)),
                adapted_pred=np.zeros((len(support_sorted), 3, 19)),
                dates=support_sorted["issuance_utc"],
                rho_99=self.rho_99,
                min_support_days=self.min_support_days,
                n_available=n_avail,
                support_days=total_days,
            )
            self.last_result = res
            return res

        # 3. Chronological Support Split: Fit, 72h Embargo, Check (Condition C6)
        df_fit, df_check = split_support_window(support_sorted, embargo_hours=self.embargo_hours)
        if df_fit.empty or df_check.empty:
            self.adapted_params = None
            res = evaluate_adaptation_acceptance(
                y_true=np.zeros((0, 3)),
                y_mask=np.zeros((0, 3), dtype=bool),
                m5_pred=np.zeros((0, 3, 19)),
                adapted_pred=np.zeros((0, 3, 19)),
                dates=[],
                rho_99=self.rho_99,
                n_available=n_avail,
                support_days=total_days,
            )
            self.last_result = res
            return res

        try:
            # 4. Prepare Fit Tensors
            fit_x = torch.from_numpy(build_fast_sequence_tensors(df_fit)).float().to(self.device)
            fit_u = torch.from_numpy(extract_tabular_context(df_fit)).float().to(self.device)
            fit_base_np = self._ensure_base_quantiles(df_fit)
            fit_base = torch.from_numpy(fit_base_np).float().to(self.device)
            fit_y_np, fit_m_np = extract_targets_and_masks(df_fit)
            fit_y = torch.from_numpy(fit_y_np).float().to(self.device)
            fit_mask = torch.from_numpy(fit_m_np).float().to(self.device)

            if fit_mask.sum() == 0:
                self.adapted_params = None
                res = evaluate_adaptation_acceptance(
                    y_true=np.zeros((0, 3)),
                    y_mask=np.zeros((0, 3), dtype=bool),
                    m5_pred=np.zeros((0, 3, 19)),
                    adapted_pred=np.zeros((0, 3, 19)),
                    dates=[],
                    rho_99=self.rho_99,
                    n_available=n_avail,
                    support_days=total_days,
                )
                self.last_result = res
                return res

            # Apply synthetic corruption if adapting on low-cost sensor
            if not is_reference_sensor:
                rng = np.random.default_rng(seed)
                g_min, g_max = self.corruption_priors["gain"]
                s_min, s_max = self.corruption_priors["rh_slope"]
                o_min, o_max = self.corruption_priors["offset"]
                a_val = float(rng.uniform(g_min, g_max))
                b_val = float(rng.uniform(s_min, s_max))
                c_val = float(rng.uniform(o_min, o_max))
                fit_x = apply_sensor_corruption(
                    fit_x,
                    p=1.0,
                    gain_range=(a_val, a_val),
                    rh_slope_range=(b_val, b_val),
                    offset_range=(c_val, c_val),
                )

            # 5. Execute Inner-Loop Adaptation on Fit Part
            fit_task = MAMLTask(
                domain=str(df_fit["domain"].iloc[0]) if "domain" in df_fit.columns else "target",
                season="deployment",
                available_stations=(
                    list(df_fit["location_id"].unique()) if "location_id" in df_fit.columns else [1]
                ),
                held_out_stations=[],
                support_x=fit_x,
                support_context=fit_u,
                support_base_q=fit_base,
                support_y=fit_y,
                support_mask=fit_mask,
                query_x=torch.zeros(0, 72, 6, device=self.device),
                query_context=torch.zeros(0, 14, device=self.device),
                query_base_q=torch.zeros(0, 3, 19, device=self.device),
                query_y=torch.zeros(0, 3, device=self.device),
                query_mask=torch.zeros(0, 3, device=self.device),
                n_available=n_avail,
            )

            initial_params = {
                k: v.clone().detach().requires_grad_(True)
                for k, v in self.calibrator.named_parameters()
            }
            step_sizes_dict = self._resolve_step_sizes()

            adapted_fit_params = inner_loop_adapt(
                calibrator=self.calibrator,
                params=initial_params,
                step_sizes=step_sizes_dict,
                task=fit_task,
                inner_steps=self.inner_steps,
                clip_norm=self.clip_norm,
                adapt_scope=self.adapt_scope,
            )

            # 6. Evaluate Acceptance Gate on Check Part (Condition C6)
            chk_x = torch.from_numpy(build_fast_sequence_tensors(df_check)).float().to(self.device)
            chk_u = torch.from_numpy(extract_tabular_context(df_check)).float().to(self.device)
            chk_base_np = self._ensure_base_quantiles(df_check)
            chk_base = torch.from_numpy(chk_base_np).float().to(self.device)
            chk_y_np, chk_m_np = extract_targets_and_masks(df_check)

            with torch.no_grad():
                chk_adapted_q = functional_call(
                    self.calibrator,
                    adapted_fit_params,
                    (chk_x, chk_u, chk_base, False),
                )
                chk_unadapted_q = functional_call(
                    self.calibrator,
                    initial_params,
                    (chk_x, chk_u, chk_base, False),
                )

            acceptance_res = evaluate_adaptation_acceptance(
                y_true=chk_y_np,
                y_mask=chk_m_np,
                m5_pred=chk_base_np,
                adapted_pred=chk_adapted_q.cpu().numpy(),
                dates=df_check["issuance_utc"],
                rho_99=self.rho_99,
                unadapted_pred=chk_unadapted_q.cpu().numpy(),
                min_improvement=self.min_improvement,
                min_days_better_fraction=self.min_days_better_fraction,
                min_support_days=self.min_support_days,
                n_available=n_avail,
                support_days=total_days,
                in_log1p=True,
                details={"n_fit_samples": len(df_fit)},
            )

            # 7. Final Parameter Decision & Full Support Window Re-adaptation
            if acceptance_res.accepted:
                logger.info(
                    "deployment_adaptation_accepted",
                    improvement=f"{acceptance_res.relative_improvement * 100:.2f}%",
                    win_fraction=f"{acceptance_res.days_better_fraction * 100:.1f}%",
                    residual_r=f"{acceptance_res.residual_r:.4f}",
                    rho_99=f"{self.rho_99:.4f}",
                )

                # Re-adapt on full support window (Condition C6 step 5)
                full_x = (
                    torch.from_numpy(build_fast_sequence_tensors(support_sorted))
                    .float()
                    .to(self.device)
                )
                full_u = (
                    torch.from_numpy(extract_tabular_context(support_sorted))
                    .float()
                    .to(self.device)
                )
                full_base = (
                    torch.from_numpy(self._ensure_base_quantiles(support_sorted))
                    .float()
                    .to(self.device)
                )
                full_y_np, full_m_np = extract_targets_and_masks(support_sorted)
                full_y = torch.from_numpy(full_y_np).float().to(self.device)
                full_mask = torch.from_numpy(full_m_np).float().to(self.device)

                if not is_reference_sensor:
                    full_x = apply_sensor_corruption(
                        full_x,
                        p=1.0,
                        gain_range=(a_val, a_val),
                        rh_slope_range=(b_val, b_val),
                        offset_range=(c_val, c_val),
                    )

                full_task = MAMLTask(
                    domain=(
                        str(support_sorted["domain"].iloc[0])
                        if "domain" in support_sorted.columns
                        else "target"
                    ),
                    season="deployment",
                    available_stations=(
                        list(support_sorted["location_id"].unique())
                        if "location_id" in support_sorted.columns
                        else [1]
                    ),
                    held_out_stations=[],
                    support_x=full_x,
                    support_context=full_u,
                    support_base_q=full_base,
                    support_y=full_y,
                    support_mask=full_mask,
                    query_x=torch.zeros(0, 72, 6, device=self.device),
                    query_context=torch.zeros(0, 14, device=self.device),
                    query_base_q=torch.zeros(0, 3, 19, device=self.device),
                    query_y=torch.zeros(0, 3, device=self.device),
                    query_mask=torch.zeros(0, 3, device=self.device),
                    n_available=n_avail,
                )

                final_adapted = inner_loop_adapt(
                    calibrator=self.calibrator,
                    params={
                        k: v.clone().detach().requires_grad_(True)
                        for k, v in self.calibrator.named_parameters()
                    },
                    step_sizes=step_sizes_dict,
                    task=full_task,
                    inner_steps=self.inner_steps,
                    clip_norm=self.clip_norm,
                    adapt_scope=self.adapt_scope,
                )
                self.adapted_params = {k: v.detach().clone() for k, v in final_adapted.items()}
            else:
                logger.info(
                    "deployment_adaptation_rejected",
                    status=acceptance_res.adaptation_status,
                    improvement=f"{acceptance_res.relative_improvement * 100:.2f}%",
                    win_fraction=f"{acceptance_res.days_better_fraction * 100:.1f}%",
                    residual_r=f"{acceptance_res.residual_r:.4f}",
                    rho_99=f"{self.rho_99:.4f}",
                )
                self.adapted_params = None

            self.last_result = acceptance_res
            return acceptance_res

        finally:
            # Memory Hygiene (Condition C6 / T1-17): release local references and clean up
            gc.collect()

    def predict(
        self,
        inference_df: pd.DataFrame,
        rearrange: bool = True,
    ) -> dict[int, np.ndarray]:
        """Predict multi-quantile forecasts across all 3 horizons.

        If adaptation succeeded (status='adapted'), applies adapted local residual corrections Delta z_hat.
        If adaptation was skipped or rejected, Delta z_hat is identically zero, producing Model M5
        predictions bit-for-bit with degradation level 0 (Condition C6 & C7).

        Args:
            inference_df: Feature DataFrame at issuance T for target locations.
            rearrange: Whether to apply monotone quantile rearrangement (default: True).

        Returns:
            Dictionary mapping horizon h -> (N, 19) physical PM2.5 quantile forecasts in ug/m3.
        """
        # 1. Base Model Physical Forecasts (Model M5)
        m5_preds = self.base_model.predict_all_horizons(inference_df, rearrange=False)

        # Fallback branch: Delta z = 0 (exact M5 bit-for-bit match)
        if self.adapted_params is None or len(inference_df) == 0:
            if rearrange:
                return {
                    h: rearrange_quantiles(m5_preds[h], clip_min=0.0, clip_max=2000.0)
                    for h in CANONICAL_HORIZONS
                }
            return m5_preds

        # 2. Adapted Residual Calibration Branch
        x_seq = torch.from_numpy(build_fast_sequence_tensors(inference_df)).float().to(self.device)
        context = torch.from_numpy(extract_tabular_context(inference_df)).float().to(self.device)

        # Base quantiles in log1p
        base_grid = np.zeros(
            (len(inference_df), len(CANONICAL_HORIZONS), len(CANONICAL_QUANTILES)),
            dtype=np.float32,
        )
        for h_idx, h in enumerate(CANONICAL_HORIZONS):
            base_grid[:, h_idx, :] = np.log1p(np.maximum(0.0, m5_preds[h].astype(np.float32)))

        base_tensor = torch.from_numpy(base_grid).float().to(self.device)

        with torch.no_grad():
            calibrator_out = functional_call(
                self.calibrator,
                self.adapted_params,
                (x_seq, context, base_tensor, False),
            )
            cal_np = calibrator_out.cpu().numpy()

        results: dict[int, np.ndarray] = {}
        for h_idx, h in enumerate(CANONICAL_HORIZONS):
            # Invert log1p back to physical ug/m3
            pred_ugm3 = np.expm1(np.maximum(0.0, cal_np[:, h_idx, :]))
            if rearrange:
                pred_ugm3 = rearrange_quantiles(pred_ugm3, clip_min=0.0, clip_max=2000.0)
            results[h] = pred_ugm3

        return results

    def predict_residual(
        self,
        inference_df: pd.DataFrame,
    ) -> np.ndarray:
        """Predict local residual corrections Delta z_hat in log1p space.

        Args:
            inference_df: Feature DataFrame at issuance T.

        Returns:
            np.ndarray of shape (N, 3, 19) containing residual corrections Delta z_hat.
            If skipped or rejected, returns all zeros identically.
        """
        n_samples = len(inference_df)
        if self.adapted_params is None or n_samples == 0:
            return np.zeros(
                (n_samples, len(CANONICAL_HORIZONS), len(CANONICAL_QUANTILES)), dtype=np.float32
            )

        x_seq = torch.from_numpy(build_fast_sequence_tensors(inference_df)).float().to(self.device)
        context = torch.from_numpy(extract_tabular_context(inference_df)).float().to(self.device)

        with torch.no_grad():
            delta_z = functional_call(
                self.calibrator,
                self.adapted_params,
                (x_seq, context, None, False),
            )
            return cast(np.ndarray, delta_z.cpu().numpy().astype(np.float32))

    @classmethod
    def from_paths(
        cls,
        base_model_dir: Path | str,
        calibrator_path: Path | str,
        step_sizes_path: Path | str | None = None,
        rho_99: float = DEFAULT_RHO_99,
        verify_binding: bool = True,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> DeploymentAdaptationEngine:
        """Instantiate deployment engine from model files on disk.

        Args:
            base_model_dir: Path to LightGBM base model directory containing metadata.json.
            calibrator_path: Path to PyTorch calibrator weights (.pt).
            step_sizes_path: Optional path to MetaSGDStepSizes checkpoint (.pt).
            rho_99: Calibrated residual bound threshold.
            verify_binding: Whether to verify cryptographic base_sha256 binding.
            device: Compute device.
            **kwargs: Extra parameters passed to DeploymentAdaptationEngine constructor.

        Returns:
            Configured DeploymentAdaptationEngine instance.
        """
        from smogsense.models.encoders.gru import LocalGRUEncoder
        from smogsense.models.heads import ResidualQuantileHead

        # 1. Load LightGBM Base Model
        base_model = LGBMQuantileEnsemble.load(base_model_dir, verify_sha256=verify_binding)
        base_sha = base_model.compute_base_sha256()

        # 2. Load Residual Calibrator
        encoder = LocalGRUEncoder(input_size=6, hidden_size=64, num_layers=2, latent_dim=32)
        head = ResidualQuantileHead(input_dim=46, hidden_dim=64, output_dim=57)
        calibrator = ResidualCalibrator(encoder, head)

        ckpt_path = Path(calibrator_path)
        ckpt_data = torch.load(ckpt_path, map_location=device)

        if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
            calibrator.load_state_dict(ckpt_data["model_state_dict"])
            # Verify cryptographic base_sha256 binding (Condition C3 / T1-15)
            if verify_binding and "base_sha256" in ckpt_data:
                ckpt_base_sha = ckpt_data["base_sha256"]
                if ckpt_base_sha != base_sha:
                    raise ValueError(
                        f"Base model binding verification failed: calibrator checkpoint references "
                        f"base_sha256='{ckpt_base_sha}' but loaded base is '{base_sha}'"
                    )
        elif isinstance(ckpt_data, dict):
            calibrator.load_state_dict(ckpt_data)

        # 3. Resolve Step Sizes
        step_sizes: MetaSGDStepSizes | float = 0.01
        if isinstance(ckpt_data, dict) and "step_sizes_state_dict" in ckpt_data:
            meta_sgd = MetaSGDStepSizes(init_alpha=0.01)
            meta_sgd.load_state_dict(ckpt_data["step_sizes_state_dict"])
            step_sizes = meta_sgd
        elif step_sizes_path is not None:
            meta_sgd = MetaSGDStepSizes(init_alpha=0.01)
            step_ckpt = torch.load(step_sizes_path, map_location=device)
            meta_sgd.load_state_dict(step_ckpt.get("step_sizes_state_dict", step_ckpt))
            step_sizes = meta_sgd

        return cls(
            base_model=base_model,
            calibrator=calibrator,
            step_sizes=step_sizes,
            rho_99=rho_99,
            device=device,
            **kwargs,
        )
