"""smogsense.models.lgbm_quantile — LightGBM quantile ensemble (the frozen physics base).

Trains 57 gradient-boosted quantile regressors (3 horizons x 19 quantiles) with
monotone rearrangement (Chernozhukov et al., 2010). Supports out-of-fold (OOF)
season-blocked cross-fitting on Block E and deterministic SHA256 model binding.

Specification: docs/ml-architecture.md §6 & ADR-024.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from structlog import get_logger

logger = get_logger(__name__)

# Standard 19 quantile levels: tau in {0.05, 0.10, ..., 0.95}
DEFAULT_QUANTILES: tuple[float, ...] = tuple(round(0.05 + 0.05 * i, 2) for i in range(19))

# Standard forecast horizons in hours
DEFAULT_HORIZONS: tuple[int, ...] = (24, 48, 72)

# Features strictly excluded from M5 physics base (docs/ml-architecture.md §6, docs/test-specifications.md §55)
# - No recent CAMS bias: pure physical transfer with zero local calibration
# - No absolute coordinates: prevents arbitrary leaf shifts across domains
# - No Indian-only event flags: prevent silent domain mismatch
M5_EXCLUDED_COLUMNS: set[str] = {
    "f_cams_bias_trailing_7d",
    "m_cams_bias_trailing_7d",
    "f_cal_diwali_h24",
    "m_cal_diwali_h24",
    "f_cal_diwali_h48",
    "m_cal_diwali_h48",
    "f_cal_diwali_h72",
    "m_cal_diwali_h72",
    "f_stat_lat",
    "m_stat_lat",
    "f_stat_lon",
    "m_stat_lon",
}

# Features excluded from M4 within-domain model (may use coordinates, but not Diwali for transfer)
M4_EXCLUDED_COLUMNS: set[str] = {
    "f_cal_diwali_h24",
    "m_cal_diwali_h24",
    "f_cal_diwali_h48",
    "m_cal_diwali_h48",
    "f_cal_diwali_h72",
    "m_cal_diwali_h72",
}


def rearrange_quantiles(q_preds: np.ndarray) -> np.ndarray:
    """Apply monotone rearrangement to quantile predictions along the last axis.

    Sorting restores monotonicity without increasing total pinball loss
    (Chernozhukov et al., 2010; docs/ml-architecture.md §5).

    Args:
        q_preds: Array of shape (N, K) or (K,) containing quantile estimates.

    Returns:
        Monotonically sorted quantile predictions of the same shape.
    """
    return np.asarray(np.sort(q_preds, axis=-1))


def pinball_loss(
    y_true: float | np.ndarray,
    y_pred: float | np.ndarray,
    tau: float,
) -> np.ndarray:
    """Compute pinball loss for quantile tau.

    Loss = max((tau - 1) * (y - q), tau * (y - q)).
    """
    y = np.asarray(y_true)
    q = np.asarray(y_pred)
    error = y - q
    return np.asarray(np.maximum((tau - 1.0) * error, tau * error))


def mean_pinball_loss(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    taus: Sequence[float] = DEFAULT_QUANTILES,
) -> float:
    """Compute mean pinball loss across all K quantile levels.

    Args:
        y_true: Array of shape (N,) containing true target values.
        q_preds: Array of shape (N, K) containing quantile predictions.
        taus: Sequence of K quantile levels.

    Returns:
        Mean scalar pinball loss.
    """
    y = np.asarray(y_true)
    q = np.asarray(q_preds)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if len(taus) != q.shape[1]:
        raise ValueError(f"Mismatch between len(taus)={len(taus)} and q.shape[1]={q.shape[1]}")

    losses = [pinball_loss(y, q[:, i], tau) for i, tau in enumerate(taus)]
    return float(np.mean(losses))


def filter_features(
    columns: Sequence[str],
    model_type: str = "M5",
) -> list[str]:
    """Filter candidate feature store columns according to model architecture rules.

    Args:
        columns: Available column names from feature store.
        model_type: 'M5' (physical transfer base) or 'M4' (within-domain tabular).

    Returns:
        List of selected feature column names.
    """
    excluded = M5_EXCLUDED_COLUMNS if model_type.upper() == "M5" else M4_EXCLUDED_COLUMNS
    selected = [
        col
        for col in columns
        if (col.startswith("f_") or col.startswith("m_")) and col not in excluded
    ]
    return sorted(selected)


class LGBMQuantileEnsemble:
    """Ensemble of gradient boosted quantile regression trees for probabilistic forecasting.

    Trains K boosters per horizon H (total K x H boosters) with objective='quantile',
    applying monotone rearrangement to enforce non-crossing predictive distributions.
    """

    def __init__(
        self,
        model_type: str = "M5",
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        lgbm_params: dict[str, Any] | None = None,
    ) -> None:
        self.model_type = model_type.upper()
        self.quantiles = tuple(quantiles)
        self.horizons = tuple(horizons)

        default_params: dict[str, Any] = {
            "objective": "quantile",
            "metric": "quantile",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 5.0,
            "max_bin": 255,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 2,
            "verbosity": -1,
            "random_state": 42,
        }
        if lgbm_params:
            default_params.update(lgbm_params)
        self.params = default_params

        # Booster storage: boosters[horizon][quantile] = Booster
        self.boosters: dict[int, dict[float, lgb.Booster]] = {h: {} for h in self.horizons}
        self.feature_names: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        num_boost_round: int = 150,
    ) -> LGBMQuantileEnsemble:
        """Fit all K x H quantile boosters on valid observations.

        Args:
            train_df: Training feature table.
            feature_cols: Explicit feature columns, or auto-filtered if None.
            val_df: Optional validation set for early stopping.
            num_boost_round: Maximum boosting iterations.

        Returns:
            Fitted LGBMQuantileEnsemble instance.
        """
        if feature_cols is None:
            self.feature_names = filter_features(list(train_df.columns), model_type=self.model_type)
        else:
            self.feature_names = list(feature_cols)

        logger.info(
            "training_lgbm_quantile_ensemble",
            model_type=self.model_type,
            horizons=self.horizons,
            n_quantiles=len(self.quantiles),
            n_features=len(self.feature_names),
            n_rows=len(train_df),
        )

        for h in self.horizons:
            y_col = f"y_h{h}"
            v_col = f"valid_h{h}"

            # Filter for valid, complete target ground truth
            mask_train = (
                (train_df[v_col]) & (train_df[y_col].notna())
                if v_col in train_df
                else train_df[y_col].notna()
            )
            x_train = train_df.loc[mask_train, self.feature_names]
            y_train = train_df.loc[mask_train, y_col]

            if len(x_train) == 0:
                raise ValueError(f"No valid training samples found for horizon h={h}.")

            dtrain = lgb.Dataset(x_train, label=y_train, free_raw_data=False)

            dval: lgb.Dataset | None = None
            if val_df is not None:
                mask_val = (
                    (val_df[v_col]) & (val_df[y_col].notna())
                    if v_col in val_df
                    else val_df[y_col].notna()
                )
                if mask_val.sum() > 0:
                    dval = lgb.Dataset(
                        val_df.loc[mask_val, self.feature_names],
                        label=val_df.loc[mask_val, y_col],
                        reference=dtrain,
                        free_raw_data=False,
                    )

            for q in self.quantiles:
                p = dict(self.params)
                p["alpha"] = q

                callbacks: list[Any] = []
                valid_sets = [dtrain]
                if dval is not None:
                    valid_sets.append(dval)
                    callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

                bst = lgb.train(
                    p,
                    dtrain,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    callbacks=callbacks,
                )
                self.boosters[h][q] = bst

        return self

    def predict(
        self,
        df_x: pd.DataFrame,
        horizon: int,
        rearrange: bool = True,
    ) -> np.ndarray:
        """Predict quantile levels for a specific horizon with monotone rearrangement.

        Args:
            df_x: Feature table containing self.feature_names.
            horizon: Target horizon (e.g. 24, 48, 72).
            rearrange: Whether to apply monotone sorting (default: True).

        Returns:
            np.ndarray of shape (N, len(self.quantiles)) with quantile predictions.
        """
        if horizon not in self.boosters:
            raise KeyError(f"Horizon h={horizon} not found in trained boosters.")

        x_in = df_x[self.feature_names]
        n_samples = len(x_in)
        n_q = len(self.quantiles)
        preds = np.zeros((n_samples, n_q), dtype=np.float64)

        for i, q in enumerate(self.quantiles):
            bst = self.boosters[horizon][q]
            preds[:, i] = bst.predict(x_in)

        if rearrange:
            preds = rearrange_quantiles(preds)

        return preds

    def predict_all_horizons(
        self,
        df_x: pd.DataFrame,
        rearrange: bool = True,
    ) -> dict[int, np.ndarray]:
        """Predict quantiles across all configured horizons.

        Returns:
            Dictionary mapping horizon h -> (N, K) quantile prediction array.
        """
        return {h: self.predict(df_x, horizon=h, rearrange=rearrange) for h in self.horizons}

    def compute_base_sha256(self) -> str:
        """Compute deterministic SHA256 over all booster model strings.

        Guarantees exact binding verification between physics base and residual calibrator
        (ADR-024 Condition C3).
        """
        hasher = hashlib.sha256()
        for h in sorted(self.horizons):
            for q in sorted(self.quantiles):
                bst = self.boosters[h].get(q)
                if bst is not None:
                    hasher.update(f"h={h}:q={q}:".encode())
                    hasher.update(bst.model_to_string().encode())
        return hasher.hexdigest()

    def save(self, out_dir: Path | str) -> Path:
        """Save boosters and metadata to directory."""
        dir_path = Path(out_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        meta = {
            "model_type": self.model_type,
            "quantiles": list(self.quantiles),
            "horizons": list(self.horizons),
            "feature_names": self.feature_names,
            "base_sha256": self.compute_base_sha256(),
            "params": self.params,
        }
        with (dir_path / "metadata.json").open("w") as f:
            json.dump(meta, f, indent=2)

        for h in self.horizons:
            for q in self.quantiles:
                bst = self.boosters[h][q]
                q_tag = round(q * 100)
                bst_file = dir_path / f"booster_h{h}_q{q_tag:02d}.txt"
                bst.save_model(str(bst_file))

        logger.info("saved_lgbm_quantile_ensemble", dir=str(dir_path), sha256=meta["base_sha256"])
        return dir_path

    @classmethod
    def load(cls, dir_path: Path | str) -> LGBMQuantileEnsemble:
        """Load trained ensemble from directory."""
        p = Path(dir_path)
        with (p / "metadata.json").open("r") as f:
            meta = json.load(f)

        ensemble = cls(
            model_type=meta["model_type"],
            quantiles=meta["quantiles"],
            horizons=meta["horizons"],
            lgbm_params=meta.get("params"),
        )
        ensemble.feature_names = meta["feature_names"]

        for h in ensemble.horizons:
            for q in ensemble.quantiles:
                q_tag = round(q * 100)
                bst_file = p / f"booster_h{h}_q{q_tag:02d}.txt"
                bst = lgb.Booster(model_file=str(bst_file))
                ensemble.boosters[h][q] = bst

        return ensemble


def fit_cross_fit_oof(
    df_block_e: pd.DataFrame,
    feature_cols: list[str] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    lgbm_params: dict[str, Any] | None = None,
    num_boost_round: int = 120,
) -> tuple[pd.DataFrame, LGBMQuantileEnsemble]:
    """Train season-blocked cross-fitting folds on Block E and return un-leaked OOF base predictions.

    Conditions C1, C2 (ADR-024):
    Partitions Block E by fold_id (0..K-1). For each fold k:
      1. Fits boosters on all folds != k.
      2. Predicts out-of-fold quantiles for fold k.
    Finally, fits the full Base ensemble on all of Block E.

    Args:
        df_block_e: Block E DataFrame with 'fold_id' column.
        feature_cols: Feature columns (or filtered M5 features if None).
        horizons: Target horizons.
        quantiles: Quantile probability levels.
        lgbm_params: Optional LightGBM parameters.
        num_boost_round: Boosting rounds.

    Returns:
        tuple (oof_predictions_df, final_base_ensemble).
    """
    if "fold_id" not in df_block_e.columns:
        raise ValueError("Block E dataframe must contain 'fold_id' for cross-fitting.")

    if feature_cols is None:
        feature_cols = filter_features(list(df_block_e.columns), model_type="M5")

    fold_ids = sorted([int(f) for f in df_block_e["fold_id"].dropna().unique()])
    logger.info("running_season_blocked_cross_fitting", n_folds=len(fold_ids), folds=fold_ids)

    oof_dfs: list[pd.DataFrame] = []

    for k in fold_ids:
        train_folds = df_block_e[df_block_e["fold_id"] != k].copy()
        test_fold = df_block_e[df_block_e["fold_id"] == k].copy()

        logger.info(
            "cross_fitting_fold", test_fold=k, train_rows=len(train_folds), test_rows=len(test_fold)
        )

        fold_ensemble = LGBMQuantileEnsemble(
            model_type="M5",
            quantiles=quantiles,
            horizons=horizons,
            lgbm_params=lgbm_params,
        )
        fold_ensemble.fit(train_folds, feature_cols=feature_cols, num_boost_round=num_boost_round)

        fold_oof = test_fold[["domain", "location_id", "issuance_utc", "fold_id"]].copy()
        for h in horizons:
            preds = fold_ensemble.predict(test_fold, horizon=h, rearrange=True)
            for i, q in enumerate(quantiles):
                q_tag = round(q * 100)
                q_col = f"m5_oof_h{h}_q{q_tag:02d}"
                fold_oof[q_col] = preds[:, i]

        oof_dfs.append(fold_oof)

    oof_combined = (
        pd.concat(oof_dfs, ignore_index=True)
        .sort_values(by=["issuance_utc", "location_id"])
        .reset_index(drop=True)
    )

    # Train final base model on all of Block E
    logger.info("training_final_base_ensemble_on_all_block_e", rows=len(df_block_e))
    final_base = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=quantiles,
        horizons=horizons,
        lgbm_params=lgbm_params,
    )
    final_base.fit(df_block_e, feature_cols=feature_cols, num_boost_round=num_boost_round)

    return oof_combined, final_base


def load_dataset_from_partitions(
    base_dir: Path,
    domain: str,
    split: str | None = None,
    max_partitions: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate Parquet feature store partitions for a domain and optional split."""
    pattern = f"{domain}/**/features.parquet"
    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No feature store partitions found under {base_dir / domain}")

    target_files = files[:max_partitions] if max_partitions else files
    dfs: list[pd.DataFrame] = []
    for f in target_files:
        df = pd.read_parquet(f)
        if split is not None and "source_split" in df.columns:
            df = df[df["source_split"] == split]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No rows found for domain={domain}, split={split}")

    return pd.concat(dfs, ignore_index=True)


def compute_crps_discrete(
    y_true: np.ndarray,
    q_preds: np.ndarray,
    taus: tuple[float, ...] = DEFAULT_QUANTILES,
) -> float:
    """Compute discrete 19-quantile CRPS approximation."""
    mean_pb = mean_pinball_loss(y_true, q_preds, taus)
    return 2.0 * mean_pb


def run_early_transfer_probe(
    feature_store_dir: str | Path = "data/processed/feature_store",
    subsample_train: int | None = None,
    eval_location_id: int = 8664,
    num_boost_round: int = 120,
    out_json: Path | str | None = None,
) -> dict[str, Any]:
    """Execute Gate G2 early transfer probe: LightGBM M5 zero-shot on Lahore vs M2/M1."""
    fs_path = Path(feature_store_dir)

    print("=" * 80)
    print("PHASE 2 TASK P2-04: EARLY TRANSFER PROBE (GATE G2)")
    print("Delhi -> Lahore Zero-Shot Physical Transfer (LightGBM M5 vs Baselines)")
    print("=" * 80)

    # 1. Load Delhi Block E (Train) and Block V (Validation)
    print(f"Loading Delhi feature partitions from {fs_path} ...")
    delhi_e = load_dataset_from_partitions(fs_path, domain="delhi", split="E")
    delhi_v = load_dataset_from_partitions(fs_path, domain="delhi", split="V")

    print(
        f"Loaded Delhi Block E: {len(delhi_e):,} rows across {delhi_e['fold_id'].nunique()} folds."
    )
    print(f"Loaded Delhi Block V: {len(delhi_v):,} rows.")

    if subsample_train and subsample_train < len(delhi_e):
        print(
            f"Subsampling Delhi Block E to {subsample_train:,} rows for rapid probe evaluation ..."
        )
        delhi_e = delhi_e.sample(n=subsample_train, random_state=42).reset_index(drop=True)

    # 2. Load Lahore Evaluation Set
    print(f"Loading Lahore feature partitions from {fs_path} ...")
    lahore_df = load_dataset_from_partitions(fs_path, domain="lahore")
    print(
        f"Loaded Lahore total: {len(lahore_df):,} rows across stations: {sorted(lahore_df['location_id'].unique())}"
    )

    # Focus evaluation on the primary held-out reference monitor (US Consulate 8664)
    lahore_eval = (
        lahore_df[lahore_df["location_id"] == eval_location_id].copy().reset_index(drop=True)
    )
    if lahore_eval.empty:
        print(f"Warning: Location {eval_location_id} empty. Evaluating on entire Lahore dataset.")
        lahore_eval = lahore_df

    print(f"Evaluating on Lahore target station ({eval_location_id}): {len(lahore_eval):,} rows.")

    # 3. Train LightGBM M5 Physics Base
    print("\nTraining LightGBM M5 physics base (57 boosters across 3 horizons x 19 quantiles) ...")
    m5 = LGBMQuantileEnsemble(
        model_type="M5",
        quantiles=DEFAULT_QUANTILES,
        horizons=DEFAULT_HORIZONS,
    )
    m5.fit(train_df=delhi_e, val_df=delhi_v, num_boost_round=num_boost_round)
    base_sha256 = m5.compute_base_sha256()
    print(
        f"M5 Physics Base fitted. Features: {len(m5.feature_names)}. Base SHA256: {base_sha256[:16]}..."
    )

    # 4. Generate Predictions on Lahore Held-Out Monitor
    m5_preds = m5.predict_all_horizons(lahore_eval, rearrange=True)

    results: dict[str, Any] = {
        "eval_location_id": eval_location_id,
        "n_eval_rows": len(lahore_eval),
        "base_sha256": base_sha256,
        "horizons": {},
        "overall_crpss_m2": 0.0,
        "overall_crpss_m1": 0.0,
        "gate_g2_passed": False,
    }

    print("\n" + "=" * 80)
    print("PROBE RESULTS BY HORIZON (Held-out Monitor 8664)")
    print("=" * 80)
    print(
        f"{'Horizon':<8} | {'Model':<12} | {'MAE (ug/m3)':<12} | {'Pinball':<10} | {'CRPS':<10} | {'CRPSS vs M2':<12}"
    )
    print("-" * 80)

    total_crps_m5 = 0.0
    total_crps_m2 = 0.0
    total_crps_m1 = 0.0

    for h in DEFAULT_HORIZONS:
        y_col = f"y_h{h}"
        v_col = f"valid_h{h}"
        cams_col = f"f_cams_pm25_h{h}" if f"f_cams_pm25_h{h}" in lahore_eval.columns else None

        valid_mask = (
            (lahore_eval[v_col]) & (lahore_eval[y_col].notna())
            if v_col in lahore_eval.columns
            else lahore_eval[y_col].notna()
        )
        y_obs = lahore_eval.loc[valid_mask, y_col].to_numpy()
        n_obs = len(y_obs)

        # 1. M5 Predictions
        m5_q = m5_preds[h][valid_mask]
        m5_median = m5_q[:, 9]  # tau=0.50 is index 9
        m5_mae = float(np.mean(np.abs(m5_median - y_obs)))
        m5_pinball = mean_pinball_loss(y_obs, m5_q, DEFAULT_QUANTILES)
        m5_crps = compute_crps_discrete(y_obs, m5_q, DEFAULT_QUANTILES)

        # 2. M1 (Operational CAMS) Baseline
        if cams_col and cams_col in lahore_eval.columns:
            m1_pred = lahore_eval.loc[valid_mask, cams_col].to_numpy()
        else:
            m1_pred = np.full(n_obs, np.mean(y_obs))

        m1_mae = float(np.mean(np.abs(m1_pred - y_obs)))
        m1_crps = float(np.mean(np.abs(m1_pred - y_obs)))

        # 3. M2 (CAMS-BC with rolling residual) Baseline
        bias_col = "f_cams_bias_trailing_7d"
        if bias_col in lahore_eval.columns and cams_col:
            cams_bias = lahore_eval.loc[valid_mask, bias_col].fillna(0.0).to_numpy()
            m2_pred = np.maximum(0.0, m1_pred + cams_bias)
        else:
            m2_pred = m1_pred * 1.05

        m2_mae = float(np.mean(np.abs(m2_pred - y_obs)))
        m2_crps = float(np.mean(np.abs(m2_pred - y_obs)))

        # CRPSS
        crpss_m2 = float(1.0 - (m5_crps / m2_crps)) if m2_crps > 0 else 0.0
        crpss_m1 = float(1.0 - (m5_crps / m1_crps)) if m1_crps > 0 else 0.0

        total_crps_m5 += m5_crps
        total_crps_m2 += m2_crps
        total_crps_m1 += m1_crps

        results["horizons"][h] = {
            "n_samples": n_obs,
            "m5_mae": m5_mae,
            "m5_pinball": m5_pinball,
            "m5_crps": m5_crps,
            "m2_mae": m2_mae,
            "m2_crps": m2_crps,
            "m1_mae": m1_mae,
            "m1_crps": m1_crps,
            "crpss_vs_m2": crpss_m2,
            "crpss_vs_m1": crpss_m1,
        }

        print(
            f"h={h:<6} | {'M1 (Raw CAMS)':<12} | {m1_mae:<12.2f} | {'-':<10} | {m1_crps:<10.2f} | {'-':<12}"
        )
        print(
            f"h={h:<6} | {'M2 (CAMS-BC)':<12} | {m2_mae:<12.2f} | {'-':<10} | {m2_crps:<10.2f} | {'0.0% (ref)':<12}"
        )
        print(
            f"h={h:<6} | {'M5 (Physics)':<12} | {m5_mae:<12.2f} | {m5_pinball:<10.2f} | {m5_crps:<10.2f} | {crpss_m2 * 100:>+9.2f} %"
        )
        print("-" * 80)

    overall_crpss_m2 = float(1.0 - (total_crps_m5 / total_crps_m2)) if total_crps_m2 > 0 else 0.0
    overall_crpss_m1 = float(1.0 - (total_crps_m5 / total_crps_m1)) if total_crps_m1 > 0 else 0.0
    results["overall_crpss_m2"] = overall_crpss_m2
    results["overall_crpss_m1"] = overall_crpss_m1
    results["gate_g2_passed"] = bool(overall_crpss_m2 > 0.0)

    print(f"\nOVERALL CRPSS(M5 vs M2): {overall_crpss_m2 * 100:+.2f} %")
    print(f"OVERALL CRPSS(M5 vs M1): {overall_crpss_m1 * 100:+.2f} %")
    print(f"GATE G2 STATUS: {'PASSED [CRPSS > 0]' if results['gate_g2_passed'] else 'FAILED'}")
    print("=" * 80 + "\n")

    if out_json:
        p_json = Path(out_json)
        p_json.parent.mkdir(parents=True, exist_ok=True)
        with p_json.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Probe results written to: {p_json}")

    return results
