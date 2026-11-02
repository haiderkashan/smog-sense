"""smogsense.models.ablations — MAML adaptation scope ablations and Experiment X-11.

Orchestrates the formal meta-learning ablation matrix for Model M7 (Revision 3, ADR-024):
  1. Adaptation Scope Ablation (Condition C4):
     - all (default): adapts all 14 parameter tensors (8 GRU, 2 projection, 4 MLP head)
     - head_only (ANIL): adapts only 4 MLP head tensors (fc1/fc2 weights and biases)
     - gru_only: adapts only 10 GRU / projection tensors (encoder.*)
  2. Synthetic Corruption Support Label Policy (Experiment X-11):
     - clean (default): support labels y_sup are clean regulatory truth
     - same_as_inputs: support labels y_sup are corrupted with identical Barkjohn
       (a_i, b_i, c_i) drawing as support inputs (mirrors low-cost deployments)
     - Query labels y_qry ALWAYS stay 100% clean regulatory truth.

Specification: docs/ml-architecture.md §8, ADR-024, Condition C4 & C6.
"""

import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from structlog import get_logger
from torch.func import functional_call

from smogsense.features.backfill import compute_oof_partition_hash
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
from smogsense.models.maml import (
    MAMLTask,
    MetaSGDStepSizes,
    TaskSampler,
    inner_loop_adapt,
    maml_outer_step,
)

logger = get_logger(__name__)

# Canonical 5-condition ablation matrix per ADR-024 and docs/ml-architecture.md §8
CANONICAL_ABLATIONS: tuple[dict[str, str], ...] = (
    {
        "name": "m7_all_clean",
        "adapt_scope": "all",
        "label_policy": "clean",
        "description": "Baseline M7: Full parameter adaptation with clean support labels",
    },
    {
        "name": "m7_head_clean",
        "adapt_scope": "head_only",
        "label_policy": "clean",
        "description": "ANIL: Head-only adaptation with clean support labels",
    },
    {
        "name": "m7_gru_clean",
        "adapt_scope": "gru_only",
        "label_policy": "clean",
        "description": "GRU-only: GRU parameter adaptation with clean support labels",
    },
    {
        "name": "m7_all_same_as_inputs",
        "adapt_scope": "all",
        "label_policy": "same_as_inputs",
        "description": "X-11 Full: Full parameter adaptation with corrupted support labels",
    },
    {
        "name": "m7_head_same_as_inputs",
        "adapt_scope": "head_only",
        "label_policy": "same_as_inputs",
        "description": "X-11 Head: Head-only adaptation with corrupted support labels",
    },
)


# -----------------------------------------------------------------------------
# 1. Support Label Corruption Engine (Experiment X-11)
# -----------------------------------------------------------------------------
def corrupt_support_labels(
    y_sup: torch.Tensor,
    corruption_params: dict[str, float] | None,
    rh: torch.Tensor | float | None = None,
    support_mask: torch.Tensor | None = None,
    is_log1p: bool = True,
) -> torch.Tensor:
    """Apply Barkjohn synthetic low-cost sensor corruption to support labels.

    Mathematical formulation (docs/ml-architecture.md §8 & Experiment X-11):
        y_tilde = max(0.0, a_i * y_raw + b_i * max(0.0, RH - 70.0) + c_i)

    Used when label_policy == 'same_as_inputs' to simulate real-world deployments
    where low-cost stations only have their own corrupted sensor history for
    both inputs and local inner-loop adaptation.

    Args:
        y_sup: Support targets tensor of shape (N, n_horizons) or (n_horizons,).
        corruption_params: Dict with 'a' (gain), 'b' (RH slope), 'c' (offset).
        rh: Relative humidity values (percentage 0-100), as float or tensor.
        support_mask: Optional binary mask (N, n_horizons) indicating valid targets.
        is_log1p: Whether y_sup is in log1p space (default: True).

    Returns:
        Corrupted support targets tensor of identical shape, device, and dtype.
    """
    if corruption_params is None or y_sup.numel() == 0:
        return y_sup.clone()

    a = float(corruption_params.get("a", 1.0))
    b = float(corruption_params.get("b", 0.0))
    c = float(corruption_params.get("c", 0.0))

    device = y_sup.device
    dtype = y_sup.dtype

    # Convert to linear PM2.5 space
    y_raw = torch.expm1(y_sup) if is_log1p else y_sup.clone()

    # Determine RH excess above 70%
    if rh is None:
        rh_excess = torch.tensor(0.0, device=device, dtype=dtype)
    elif isinstance(rh, (int, float)):
        rh_excess = torch.tensor(max(0.0, float(rh) - 70.0), device=device, dtype=dtype)
    elif isinstance(rh, torch.Tensor):
        rh_tensor = rh.to(device=device, dtype=dtype)
        if rh_tensor.ndim == 1 and y_sup.ndim == 2:
            rh_tensor = rh_tensor.unsqueeze(1)
        rh_excess = torch.clamp(rh_tensor - 70.0, min=0.0)
    else:
        rh_excess = torch.tensor(0.0, device=device, dtype=dtype)

    # Barkjohn empirical sensor corruption formula (floored at 0.0)
    y_corrupted_raw = torch.clamp(a * y_raw + b * rh_excess + c, min=0.0)

    # Convert back to log1p space if input was in log1p space
    y_out = torch.log1p(y_corrupted_raw) if is_log1p else y_corrupted_raw

    # Preserve unmasked/invalid values if mask provided
    if support_mask is not None:
        mask_bool = support_mask.to(device=device) > 0.5
        y_out = torch.where(mask_bool, y_out, y_sup)

    return y_out


def apply_task_label_policy(
    task: MAMLTask,
    label_policy: str = "clean",
) -> MAMLTask:
    """Apply support label policy to a MAMLTask instance.

    Under Revision 3 (ADR-024 and Experiment X-11):
      - If label_policy == 'clean': support labels stay clean regulatory truth.
      - If label_policy == 'same_as_inputs': support labels are corrupted with
        the identical Barkjohn (a_i, b_i, c_i) drawing as the support inputs.
      - Query labels task.query_y ALWAYS stay 100% clean regulatory truth.

    Args:
        task: Sampled MAMLTask instance.
        label_policy: Either 'clean' (default) or 'same_as_inputs'.

    Returns:
        MAMLTask with potentially corrupted support_y and strictly clean query_y.
    """
    if (
        label_policy != "same_as_inputs"
        or task.corruption_params is None
        or task.support_y.shape[0] == 0
    ):
        return task

    # Extract station RH at reference time T from context vector column 1
    rh_val = task.support_context[:, 1] if task.support_context.shape[1] > 1 else None

    corrupted_sup_y = corrupt_support_labels(
        y_sup=task.support_y,
        corruption_params=task.corruption_params,
        rh=rh_val,
        support_mask=task.support_mask,
        is_log1p=True,
    )

    return MAMLTask(
        domain=task.domain,
        season=task.season,
        available_stations=task.available_stations,
        held_out_stations=task.held_out_stations,
        support_x=task.support_x,
        support_context=task.support_context,
        support_base_q=task.support_base_q,
        support_y=corrupted_sup_y,
        support_mask=task.support_mask,
        query_x=task.query_x,
        query_context=task.query_context,
        query_base_q=task.query_base_q,
        query_y=task.query_y,  # Query labels ALWAYS stay clean regulatory truth!
        query_mask=task.query_mask,
        n_available=task.n_available,
        corruption_params=task.corruption_params,
    )


# -----------------------------------------------------------------------------
# 2. Evaluation Engine for Ablation Conditions
# -----------------------------------------------------------------------------
def evaluate_ablation_condition(
    calibrator: ResidualCalibrator,
    step_sizes: MetaSGDStepSizes,
    val_tasks: Sequence[MAMLTask],
    inner_steps: int = 3,
    adapt_scope: str = "all",
    label_policy: str = "clean",
) -> dict[str, float]:
    """Evaluate adapted model on validation tasks, computing pinball, CRPS, and residual R.

    Calculates:
      - val_m7_pinball: Masked pinball loss across 19 quantiles x 3 horizons in log1p space.
      - val_m5_pinball: Unadapted M5 physics base pinball loss.
      - val_m7_crps: Continuous Ranked Probability Score in ug/m3.
      - val_m5_crps: Unadapted M5 physics base CRPS in ug/m3.
      - crpss_vs_base: 1.0 - (val_m7_crps / val_m5_crps).
      - residual_r: Mean absolute residual R = mean|Delta z| in log1p units (Condition C6).
      - delta_z_p99: 99th percentile of absolute residual |Delta z| in log1p units.

    Args:
        calibrator: ResidualCalibrator model instance.
        step_sizes: MetaSGDStepSizes module.
        val_tasks: Sequence of validation MAMLTask instances.
        inner_steps: Inner adaptation step count J.
        adapt_scope: Parameter adaptation scope ('all', 'head_only', 'gru_only').
        label_policy: Support label policy ('clean', 'same_as_inputs').

    Returns:
        Dictionary of computed validation metrics.
    """
    criterion = MaskedResidualPinballLoss(quantiles=CANONICAL_QUANTILES, n_horizons=3)
    init_params = dict(calibrator.named_parameters())

    total_m7_pinball = 0.0
    total_m5_pinball = 0.0
    total_m7_crps = 0.0
    total_m5_crps = 0.0
    count = 0

    all_abs_residuals: list[np.ndarray] = []

    for raw_task in val_tasks:
        if raw_task.query_x.shape[0] == 0:
            continue

        # Apply label policy for support adaptation
        task = apply_task_label_policy(raw_task, label_policy=label_policy)

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

            # Compute residual Delta z in log1p units
            delta_z = m7_pred - m5_pred  # shape (N, 3, 19)
            delta_z_np = delta_z.cpu().numpy()
            for h_idx in range(3):
                valid = mask_np[:, h_idx] > 0.5
                if np.any(valid):
                    abs_dz = np.abs(delta_z_np[valid, h_idx, :]).ravel()
                    all_abs_residuals.append(abs_dz)

            count += 1

    avg_m7_pinball = total_m7_pinball / max(1, count)
    avg_m5_pinball = total_m5_pinball / max(1, count)
    avg_m7_crps = total_m7_crps / max(1, count * 3)
    avg_m5_crps = total_m5_crps / max(1, count * 3)
    crpss = 1.0 - (avg_m7_crps / avg_m5_crps) if avg_m5_crps > 0.0 else 0.0

    if all_abs_residuals:
        concat_residuals = np.concatenate(all_abs_residuals)
        residual_r = float(np.mean(concat_residuals))
        delta_z_p99 = float(np.percentile(concat_residuals, 99))
    else:
        residual_r = 0.0
        delta_z_p99 = 0.0

    return {
        "val_m7_pinball": avg_m7_pinball,
        "val_m5_pinball": avg_m5_pinball,
        "val_m7_crps": avg_m7_crps,
        "val_m5_crps": avg_m5_crps,
        "crpss_vs_base": float(crpss),
        "residual_r": residual_r,
        "delta_z_p99": delta_z_p99,
    }


# -----------------------------------------------------------------------------
# 3. MAML Ablation Runner Class
# -----------------------------------------------------------------------------
class MAMLAblationRunner:
    """Orchestrates Phase 2 Task P2-09 ablation experiments for Model M7.

    Evaluates the 5 canonical conditions:
      1. Baseline M7: scope=all, label_policy=clean
      2. ANIL: scope=head_only, label_policy=clean
      3. GRU-only: scope=gru_only, label_policy=clean
      4. X-11 Full: scope=all, label_policy=same_as_inputs
      5. X-11 Head: scope=head_only, label_policy=same_as_inputs
    """

    def __init__(
        self,
        feature_store_dir: str | Path = "data/processed/feature_store",
        models_dir: str | Path = "data/processed/models",
        domain: str = "delhi",
        iterations: int = 50,
        meta_batch_size: int = 4,
        inner_steps: int = 3,
        lr_outer: float = 1e-3,
        lr_inner_init: float = 0.01,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        device: str | torch.device = "cpu",
        random_state: int = 42,
    ) -> None:
        self.feature_store_dir = Path(feature_store_dir)
        self.models_dir = Path(models_dir)
        self.domain = domain
        self.iterations = iterations
        self.meta_batch_size = meta_batch_size
        self.inner_steps = inner_steps
        self.lr_outer = lr_outer
        self.lr_inner_init = lr_inner_init
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.random_state = random_state

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.lgbm_dir = self.models_dir / "lgbm_m5"
        self.lgbm_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = False
        self.base_sha256 = ""
        self.oof_partition_hash = ""
        self.train_sampler: TaskSampler | None = None
        self.val_tasks: list[MAMLTask] = []

    def setup(self) -> None:
        """Load datasets, OOF base predictions, and initialize task samplers."""
        if self._initialized:
            return

        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)

        logger.info(
            "Loading datasets for MAML ablations",
            fs=str(self.feature_store_dir),
            domain=self.domain,
        )
        delhi_e = load_dataset_from_partitions(
            self.feature_store_dir, domain=self.domain, split="E"
        )
        delhi_v = load_dataset_from_partitions(
            self.feature_store_dir, domain=self.domain, split="V"
        )

        oof_file = self.lgbm_dir / "m5_oof_predictions.parquet"
        base_model_dir = self.lgbm_dir / "base_model"

        if oof_file.exists() and (base_model_dir / "metadata.json").exists():
            oof_df = pd.read_parquet(oof_file)
            base_model = LGBMQuantileEnsemble.load(base_model_dir)
        else:
            oof_df, base_model = fit_cross_fit_oof(
                df_block_e=delhi_e,
                horizons=CANONICAL_HORIZONS,
                quantiles=CANONICAL_QUANTILES,
                num_boost_round=120,
            )
            oof_df.to_parquet(oof_file, index=False)
            base_model.save(base_model_dir)

        self.base_sha256 = base_model.compute_base_sha256()
        self.oof_partition_hash = compute_oof_partition_hash(delhi_e)

        # Merge OOF base predictions into Block E
        delhi_e["issuance_utc"] = pd.to_datetime(delhi_e["issuance_utc"], utc=True)
        oof_df["issuance_utc"] = pd.to_datetime(oof_df["issuance_utc"], utc=True)

        merge_cols = [
            c
            for c in ["domain", "location_id", "issuance_utc"]
            if c in delhi_e.columns and c in oof_df.columns
        ]
        if merge_cols:
            merged_e = delhi_e.merge(
                oof_df.drop(columns=["fold_id"], errors="ignore"),
                on=merge_cols,
                how="inner",
            )
        elif len(delhi_e) == len(oof_df):
            merged_e = delhi_e.copy()
            for c in oof_df.columns:
                if c.startswith("m5_oof_"):
                    merged_e[c] = oof_df[c].to_numpy()
        else:
            merged_e = oof_df.copy()

        # Predict M5 base on Block V
        base_preds_v = base_model.predict_all_horizons(delhi_v)
        merged_v = delhi_v.copy()
        for h in CANONICAL_HORIZONS:
            preds_h = base_preds_v[h]
            for q_idx, q in enumerate(CANONICAL_QUANTILES):
                col_name = f"m5_oof_h{h}_q{round(q * 100)}"
                merged_v[col_name] = preds_h[:, q_idx]

        # Build task samplers
        self.train_sampler = TaskSampler(
            df_features=merged_e,
            corruption_enabled=True,
            rng=rng,
        )
        val_sampler = TaskSampler(
            df_features=merged_v,
            corruption_enabled=False,
            rng=np.random.default_rng(12345),
        )

        # Fixed validation tasks (pre-sampled for exact apples-to-apples evaluation across conditions)
        self.val_tasks = [val_sampler.sample_task().to(self.device) for _ in range(16)]
        self._initialized = True
        logger.info(
            "MAML ablation setup complete",
            train_stations=len(self.train_sampler.stations_df),
            val_tasks=len(self.val_tasks),
        )

    def run_condition(
        self,
        condition: dict[str, str],
        iterations: int | None = None,
    ) -> dict[str, Any]:
        """Execute training and evaluation for a single ablation condition.

        Args:
            condition: Dict with 'name', 'adapt_scope', 'label_policy', and optional 'description'.
            iterations: Optional override for outer iteration count.

        Returns:
            Dictionary containing condition performance metrics and lineage metadata.
        """
        self.setup()
        if self.train_sampler is None:
            msg = "Train sampler was not properly initialized during setup."
            raise RuntimeError(msg)

        c_name = condition["name"]
        scope = condition["adapt_scope"]
        policy = condition["label_policy"]
        desc = condition.get("description", "")
        n_iters = iterations if iterations is not None else self.iterations

        logger.info(
            "Starting ablation condition",
            condition=c_name,
            scope=scope,
            label_policy=policy,
            iterations=n_iters,
        )

        # Seed for reproducible initial weights across conditions
        torch.manual_seed(self.random_state)
        calibrator = ResidualCalibrator().to(self.device)
        step_sizes = MetaSGDStepSizes(init_alpha=self.lr_inner_init).to(self.device)

        outer_params = [
            {
                "params": calibrator.parameters(),
                "lr": self.lr_outer,
                "weight_decay": self.weight_decay,
            },
            {"params": step_sizes.parameters(), "lr": self.lr_outer * 2.0, "weight_decay": 0.0},
        ]
        outer_optimizer = torch.optim.Adam(outer_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            outer_optimizer, T_max=max(1, n_iters), eta_min=1e-5
        )

        loss_history: list[float] = []
        t0 = time.perf_counter()

        for _ in range(1, n_iters + 1):
            calibrator.train()
            step_sizes.train()

            # Sample batch tasks
            batch_raw = [
                self.train_sampler.sample_task().to(self.device)
                for _ in range(self.meta_batch_size)
            ]
            batch_tasks = [apply_task_label_policy(t, label_policy=policy) for t in batch_raw]

            outer_loss = maml_outer_step(
                calibrator=calibrator,
                step_sizes=step_sizes,
                outer_optimizer=outer_optimizer,
                batch_tasks=batch_tasks,
                inner_steps=self.inner_steps,
                clip_norm=self.grad_clip,
                adapt_scope=scope,
            )
            scheduler.step()
            loss_history.append(float(outer_loss))

        elapsed = time.perf_counter() - t0

        # Evaluate on fixed validation set
        calibrator.eval()
        step_sizes.eval()
        metrics = evaluate_ablation_condition(
            calibrator=calibrator,
            step_sizes=step_sizes,
            val_tasks=self.val_tasks,
            inner_steps=self.inner_steps,
            adapt_scope=scope,
            label_policy=policy,
        )

        step_sizes_dict = {
            name: round(float(step_sizes.step_size_for(name).item()), 5)
            for name in step_sizes.param_names
        }

        result = {
            "name": c_name,
            "adapt_scope": scope,
            "label_policy": policy,
            "description": desc,
            "iterations": n_iters,
            "wall_clock_seconds": round(elapsed, 2),
            "val_m7_pinball": float(metrics["val_m7_pinball"]),
            "val_m5_pinball": float(metrics["val_m5_pinball"]),
            "val_m7_crps": float(metrics["val_m7_crps"]),
            "val_m5_crps": float(metrics["val_m5_crps"]),
            "crpss_vs_base": float(metrics["crpss_vs_base"]),
            "residual_r": float(metrics["residual_r"]),
            "delta_z_p99": float(metrics["delta_z_p99"]),
            "loss_history": [round(loss_val, 4) for loss_val in loss_history],
            "learned_step_sizes": step_sizes_dict,
        }

        logger.info(
            "Condition completed",
            condition=c_name,
            crpss=f"{metrics['crpss_vs_base'] * 100:+.2f}%",
            val_crps=round(metrics["val_m7_crps"], 2),
            residual_r=round(metrics["residual_r"], 4),
            elapsed_s=round(elapsed, 2),
        )

        return result

    def run_all(
        self,
        conditions: Sequence[dict[str, str]] | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute full ablation suite and optionally persist to JSON.

        Args:
            conditions: Optional custom list of condition dictionaries.
            output_path: Optional output JSON path (default: data/processed/models/maml_ablations.json).

        Returns:
            Dictionary with condition summaries, summary table, and lineage metadata.
        """
        self.setup()
        active_conditions = (
            list(conditions) if conditions is not None else list(CANONICAL_ABLATIONS)
        )

        logger.info(
            "Launching MAML ablation suite",
            num_conditions=len(active_conditions),
            iterations_per_condition=self.iterations,
        )

        results: dict[str, dict[str, Any]] = {}
        summary_table: list[dict[str, Any]] = []

        for cond in active_conditions:
            res = self.run_condition(cond)
            results[res["name"]] = res
            summary_table.append(
                {
                    "Condition": res["name"],
                    "Scope": res["adapt_scope"],
                    "Label Policy": res["label_policy"],
                    "Iterations": res["iterations"],
                    "Val Pinball": round(res["val_m7_pinball"], 4),
                    "Val CRPS (ug/m3)": round(res["val_m7_crps"], 2),
                    "CRPSS vs M5 (%)": round(res["crpss_vs_base"] * 100, 2),
                    "Residual R": round(res["residual_r"], 4),
                    "Time (s)": round(res["wall_clock_seconds"], 2),
                }
            )

        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "domain": self.domain,
            "base_sha256": self.base_sha256,
            "oof_partition_hash": self.oof_partition_hash,
            "iterations": self.iterations,
            "meta_batch_size": self.meta_batch_size,
            "inner_steps": self.inner_steps,
            "conditions": results,
            "summary_table": summary_table,
        }

        out_p = (
            Path(output_path)
            if output_path is not None
            else self.models_dir / "maml_ablations.json"
        )
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

        logger.info("Ablation results successfully serialized", path=str(out_p))
        return payload


# -----------------------------------------------------------------------------
# 4. High-Level Entrypoint Function
# -----------------------------------------------------------------------------
def run_maml_ablations(
    feature_store_dir: str | Path = "data/processed/feature_store",
    models_dir: str | Path = "data/processed/models",
    domain: str = "delhi",
    iterations: int = 50,
    meta_batch_size: int = 4,
    inner_steps: int = 3,
    lr_outer: float = 1e-3,
    lr_inner_init: float = 0.01,
    device: str | torch.device = "cpu",
    output_path: str | Path | None = None,
    conditions: Sequence[dict[str, str]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Execute complete MAML ablation suite across scope and label policy conditions.

    Args:
        feature_store_dir: Directory containing feature store partitions.
        models_dir: Directory containing trained LightGBM M5 base and OOF cache.
        domain: Target source domain ('delhi').
        iterations: Number of outer meta-update iterations per condition.
        meta_batch_size: Number of tasks per meta-batch.
        inner_steps: Inner adaptation step count J (default: 3).
        lr_outer: Outer Adam learning rate beta.
        lr_inner_init: Initial Meta-SGD inner step size alpha.
        device: Compute device ('cpu', 'cuda').
        output_path: Optional explicit output JSON path.
        conditions: Optional explicit condition list.
        random_state: Random seed for reproducibility.

    Returns:
        Structured results dictionary with summary table and serialized JSON.
    """
    torch.backends.cudnn.enabled = False
    runner = MAMLAblationRunner(
        feature_store_dir=feature_store_dir,
        models_dir=models_dir,
        domain=domain,
        iterations=iterations,
        meta_batch_size=meta_batch_size,
        inner_steps=inner_steps,
        lr_outer=lr_outer,
        lr_inner_init=lr_inner_init,
        device=device,
        random_state=random_state,
    )
    return runner.run_all(conditions=conditions, output_path=output_path)
