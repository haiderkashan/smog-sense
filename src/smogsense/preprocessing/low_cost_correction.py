"""smogsense.preprocessing.low_cost_correction — Empirical humidity-aware low-cost sensor correction.

Implements the linear empirical correction model (Barkjohn et al., 2021):
    pm25_corrected = max(0.0, a * pm25_raw + b * RH + c)

Fitted via robust Huber regression against regulatory reference monitors in Lahore
(e.g. US Consulate 8664 StateAir BAM-1020) for candidate sensors within <= 8 km.
Provides 5-fold cross-validation, decision rule evaluation, and corruption prior
auditing against configs/model.yaml augmentation ranges.

Specification: docs/data-engineering.md §5, docs/ml-architecture.md §8, and PRD FR-11.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points in kilometres."""
    r = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def find_candidate_stations(
    registry_df: pd.DataFrame,
    reference_location_id: int = 8664,
    max_distance_km: float = 8.0,
) -> pd.DataFrame:
    """Identify candidate low-cost monitoring stations within max_distance_km of reference monitor.

    Parameters
    ----------
    registry_df : pd.DataFrame
        Station registry dataframe with location_id, lat, lon, and is_reference.
    reference_location_id : int
        Location ID of the reference anchor station (default 8664).
    max_distance_km : float
        Maximum allowable distance in km from reference station (default 8.0).

    Returns
    -------
    pd.DataFrame
        Filtered candidate stations dataframe with 'distance_km' column, sorted by distance.
    """
    ref_rows = registry_df[registry_df["location_id"] == reference_location_id]
    if ref_rows.empty:
        raise ValueError(f"Reference location_id={reference_location_id} not found in registry.")

    ref_lat = float(ref_rows.iloc[0]["lat"])
    ref_lon = float(ref_rows.iloc[0]["lon"])

    candidates = registry_df[registry_df["location_id"] != reference_location_id].copy()
    if "is_reference" in candidates.columns:
        candidates = candidates[~candidates["is_reference"]].copy()

    distances: list[float] = []
    for _, row in candidates.iterrows():
        d = haversine_distance_km(ref_lat, ref_lon, float(row["lat"]), float(row["lon"]))
        distances.append(round(d, 3))

    candidates["distance_km"] = distances
    eligible = candidates[candidates["distance_km"] <= max_distance_km].copy()
    return eligible.sort_values(by="distance_km").reset_index(drop=True)


def build_paired_observations(
    obs_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    reference_location_id: int = 8664,
    max_distance_km: float = 8.0,
    window_days: int | None = 60,
    min_pairs: int = 500,
) -> pd.DataFrame:
    """Build paired reference vs low-cost hourly observations for eligible candidate stations.

    Parameters
    ----------
    obs_df : pd.DataFrame
        Historical or QC'd hourly observations containing location_id, ts_utc,
        pm25 (or pm25_ugm3, pm25_raw_ugm3), and rh_pct (or rh).
    registry_df : pd.DataFrame
        Station registry dataframe.
    reference_location_id : int
        Reference anchor location ID.
    max_distance_km : float
        Maximum distance threshold in km.
    window_days : int | None
        Optional temporal window in days (e.g. 60 days). If None, all paired data is used.
    min_pairs : int
        Minimum required hourly paired records.

    Returns
    -------
    pd.DataFrame
        Paired observations dataframe with columns:
        ['ts_utc', 'location_id', 'sensor_model', 'distance_km', 'pm25_raw', 'rh', 'pm25_ref']
    """
    candidates = find_candidate_stations(
        registry_df,
        reference_location_id=reference_location_id,
        max_distance_km=max_distance_km,
    )
    if candidates.empty:
        raise ValueError(
            f"No candidate low-cost stations found within {max_distance_km} km of reference {reference_location_id}."
        )

    # Resolve column names in obs_df
    ref_pm25_col = "pm25_ugm3" if "pm25_ugm3" in obs_df.columns else "pm25"
    raw_pm25_col = (
        "pm25_raw_ugm3"
        if "pm25_raw_ugm3" in obs_df.columns
        else ("pm25_raw" if "pm25_raw" in obs_df.columns else ref_pm25_col)
    )
    rh_col = "rh_pct" if "rh_pct" in obs_df.columns else "rh"

    # Filter reference observations
    ref_obs = obs_df[obs_df["location_id"] == reference_location_id].copy()
    if ref_obs.empty:
        raise ValueError(f"No observations found for reference monitor {reference_location_id}.")

    ref_subset = (
        ref_obs[["ts_utc", ref_pm25_col]].dropna().rename(columns={ref_pm25_col: "pm25_ref"})
    )

    # Restrict to window_days if specified
    if window_days is not None:
        max_ts = pd.to_datetime(ref_subset["ts_utc"]).max()
        min_ts = max_ts - timedelta(days=window_days)
        ref_subset = ref_subset[pd.to_datetime(ref_subset["ts_utc"]) >= min_ts].copy()

    paired_dfs: list[pd.DataFrame] = []

    for _, cand_meta in candidates.iterrows():
        cand_id = int(cand_meta["location_id"])
        cand_obs = obs_df[obs_df["location_id"] == cand_id].copy()
        if cand_obs.empty:
            continue

        cand_subset = (
            cand_obs[["ts_utc", raw_pm25_col, rh_col]]
            .dropna()
            .rename(columns={raw_pm25_col: "pm25_raw", rh_col: "rh"})
        )

        merged = ref_subset.merge(cand_subset, on="ts_utc", how="inner").dropna()
        if not merged.empty:
            merged["location_id"] = cand_id
            merged["sensor_model"] = cand_meta.get("sensor_model", "low_cost")
            merged["distance_km"] = float(cand_meta["distance_km"])
            paired_dfs.append(merged)

    if not paired_dfs:
        raise ValueError(
            "No overlapping paired observations found between reference and candidates."
        )

    all_paired = pd.concat(paired_dfs, ignore_index=True)
    all_paired = all_paired[
        (all_paired["pm25_raw"] >= 0.0)
        & (all_paired["pm25_ref"] >= 0.0)
        & (all_paired["rh"] >= 0.0)
        & (all_paired["rh"] <= 100.0)
    ].reset_index(drop=True)

    if len(all_paired) < min_pairs:
        raise ValueError(
            f"Insufficient paired observations: {len(all_paired)} pairs < {min_pairs} required threshold."
        )

    # Standardize column order
    cols = ["ts_utc", "location_id", "sensor_model", "distance_km", "pm25_raw", "rh", "pm25_ref"]
    return all_paired[cols]


@dataclass
class LowCostCorrectionModel:
    """Humidity-aware linear empirical correction model for low-cost PM2.5 sensors.

    Formulation:
        pm25_corrected = max(0.0, a * pm25_raw + b * RH + c)

    Fitted via scikit-learn HuberRegressor with epsilon=1.35.
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    epsilon: float = 1.35
    alpha: float = 0.0001
    is_fitted: bool = False
    metrics_: dict[str, float] = field(default_factory=dict)
    sensor_model: str | None = None

    def _resolve_cols(
        self,
        df: pd.DataFrame,
        pm25_col: str = "pm25_raw",
        rh_col: str = "rh",
        ref_col: str = "pm25_ref",
    ) -> tuple[str, str, str]:
        """Resolve available column names in dataframe."""
        if pm25_col not in df.columns:
            if "pm25_raw_ugm3" in df.columns:
                pm25_col = "pm25_raw_ugm3"
            elif "pm25" in df.columns:
                pm25_col = "pm25"
        if rh_col not in df.columns and "rh_pct" in df.columns:
            rh_col = "rh_pct"
        if ref_col not in df.columns:
            if "pm25_ugm3" in df.columns:
                ref_col = "pm25_ugm3"
            elif "pm25_ref_ugm3" in df.columns:
                ref_col = "pm25_ref_ugm3"
        return pm25_col, rh_col, ref_col

    def fit(
        self,
        paired_df: pd.DataFrame,
        pm25_col: str = "pm25_raw",
        rh_col: str = "rh",
        ref_col: str = "pm25_ref",
    ) -> LowCostCorrectionModel:
        """Fit empirical correction coefficients using Huber regression.

        Parameters
        ----------
        paired_df : pd.DataFrame
            Paired observations containing raw PM2.5, relative humidity, and reference PM2.5.
        pm25_col : str
            Column name for raw low-cost PM2.5 readings.
        rh_col : str
            Column name for relative humidity in percent (0-100).
        ref_col : str
            Column name for reference ground truth PM2.5.

        Returns
        -------
        LowCostCorrectionModel
            Self fitted instance.
        """
        p_col, r_col, y_col = self._resolve_cols(paired_df, pm25_col, rh_col, ref_col)
        valid = paired_df[[p_col, r_col, y_col]].dropna()
        if len(valid) < 5:
            raise ValueError(f"Insufficient valid data to fit Huber regression: {len(valid)} rows.")

        x_mat = valid[[p_col, r_col]].to_numpy(dtype=float)
        y_vec = valid[y_col].to_numpy(dtype=float)

        huber = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
        huber.fit(x_mat, y_vec)

        self.a = float(huber.coef_[0])
        self.b = float(huber.coef_[1])
        self.c = float(huber.intercept_)
        self.is_fitted = True

        if "sensor_model" in paired_df.columns:
            models = paired_df["sensor_model"].dropna().unique()
            if len(models) == 1:
                self.sensor_model = str(models[0])

        return self

    def predict(self, pm25_raw: np.ndarray, rh: np.ndarray) -> np.ndarray:
        """Apply fitted correction model to raw PM2.5 and humidity readings.

        Enforces non-negativity constraint: pm25_corrected >= 0.0.

        Parameters
        ----------
        pm25_raw : np.ndarray
            Array of raw optical PM2.5 measurements (ug/m3).
        rh : np.ndarray
            Array of relative humidity measurements in percent (0-100).

        Returns
        -------
        np.ndarray
            Array of corrected PM2.5 concentrations (ug/m3), floored at 0.0.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "LowCostCorrectionModel must be fitted before predict() can be called."
            )

        raw_arr = np.asarray(pm25_raw, dtype=float)
        rh_arr = np.asarray(rh, dtype=float)

        corrected = self.a * raw_arr + self.b * rh_arr + self.c
        return np.maximum(0.0, corrected)

    def cross_validate(
        self,
        paired_df: pd.DataFrame,
        n_splits: int = 5,
        pm25_col: str = "pm25_raw",
        rh_col: str = "rh",
        ref_col: str = "pm25_ref",
        random_state: int = 42,
    ) -> dict[str, float]:
        """Perform K-fold cross-validation of empirical correction vs raw sensor error.

        Computes out-of-fold RMSE, MAE, and R^2 for both raw and corrected readings,
        calculating relative improvement metrics.

        Parameters
        ----------
        paired_df : pd.DataFrame
            Paired observations dataframe.
        n_splits : int
            Number of folds (default 5).
        pm25_col : str
            Raw PM2.5 column name.
        rh_col : str
            Relative humidity column name.
        ref_col : str
            Reference PM2.5 ground truth column name.
        random_state : int
            Random seed for KFold split.

        Returns
        -------
        dict[str, float]
            Cross-validation metrics summary.
        """
        p_col, r_col, y_col = self._resolve_cols(paired_df, pm25_col, rh_col, ref_col)
        valid = paired_df[[p_col, r_col, y_col]].dropna().reset_index(drop=True)
        if len(valid) < n_splits * 2:
            raise ValueError(
                f"Insufficient samples ({len(valid)}) for {n_splits}-fold cross-validation."
            )

        x_mat = valid[[p_col, r_col]].to_numpy(dtype=float)
        y_vec = valid[y_col].to_numpy(dtype=float)
        raw_x0 = x_mat[:, 0]

        oof_corrected = np.zeros(len(valid), dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, test_idx in kf.split(x_mat):
            x_train, y_train = x_mat[train_idx], y_vec[train_idx]
            x_test = x_mat[test_idx]

            fold_huber = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
            fold_huber.fit(x_train, y_train)

            pred_fold = (
                fold_huber.coef_[0] * x_test[:, 0]
                + fold_huber.coef_[1] * x_test[:, 1]
                + fold_huber.intercept_
            )
            oof_corrected[test_idx] = np.maximum(0.0, pred_fold)

        # Compute Out-of-Fold overall metrics
        raw_rmse = float(np.sqrt(mean_squared_error(y_vec, raw_x0)))
        corr_rmse = float(np.sqrt(mean_squared_error(y_vec, oof_corrected)))

        raw_mae = float(mean_absolute_error(y_vec, raw_x0))
        corr_mae = float(mean_absolute_error(y_vec, oof_corrected))

        raw_r2 = float(r2_score(y_vec, raw_x0))
        corr_r2 = float(r2_score(y_vec, oof_corrected))

        rmse_imprv = (raw_rmse - corr_rmse) / raw_rmse if raw_rmse > 0 else 0.0
        mae_imprv = (raw_mae - corr_mae) / raw_mae if raw_mae > 0 else 0.0

        metrics: dict[str, float] = {
            "n_samples": float(len(valid)),
            "n_splits": float(n_splits),
            "raw_rmse": round(raw_rmse, 3),
            "corrected_rmse": round(corr_rmse, 3),
            "rmse_improvement": round(float(rmse_imprv), 4),
            "raw_mae": round(raw_mae, 3),
            "corrected_mae": round(corr_mae, 3),
            "mae_improvement": round(float(mae_imprv), 4),
            "raw_r2": round(raw_r2, 4),
            "corrected_r2": round(corr_r2, 4),
            "beneficial": 1.0 if rmse_imprv >= 0.05 else 0.0,
        }

        self.metrics_ = metrics
        return metrics

    def get_inverted_corruption_priors(self) -> dict[str, float]:
        """Invert fitted coefficients to obtain synthetic corruption augmentation priors.

        Formulation:
            pm25_raw ~ gain * pm25_ref + rh_slope * RH + offset
        Inverting pm25_ref = a * pm25_raw + b * RH + c:
            Gain prior:     1 / a
            RH slope prior: b / a
            Offset prior:   -c / a

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'gain_prior', 'rh_slope_prior', and 'offset_prior'.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before extracting inverted priors.")

        if abs(self.a) < 1e-9:
            gain_prior = float("inf")
            rh_slope_prior = float("inf")
            offset_prior = float("inf")
        else:
            gain_prior = 1.0 / self.a
            rh_slope_prior = self.b / self.a
            offset_prior = -self.c / self.a

        return {
            "gain_prior": round(float(gain_prior), 4),
            "rh_slope_prior": round(float(rh_slope_prior), 4),
            "offset_prior": round(float(offset_prior), 4),
        }

    def save(self, path: Path | str) -> None:
        """Serialize model coefficients, metrics, and inverted priors to JSON or Parquet file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.suffix == ".parquet":
            priors = self.get_inverted_corruption_priors() if self.is_fitted else {}
            record = {
                "a": [self.a],
                "b": [self.b],
                "c": [self.c],
                "epsilon": [self.epsilon],
                "alpha": [self.alpha],
                "is_fitted": [self.is_fitted],
                "sensor_model": [self.sensor_model or "generic"],
                "gain_prior": [priors.get("gain_prior", np.nan)],
                "rh_slope_prior": [priors.get("rh_slope_prior", np.nan)],
                "offset_prior": [priors.get("offset_prior", np.nan)],
            }
            pd.DataFrame(record).to_parquet(target_path, index=False)
        else:
            data = {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "is_fitted": self.is_fitted,
                "sensor_model": self.sensor_model,
                "metrics": self.metrics_,
                "priors": self.get_inverted_corruption_priors() if self.is_fitted else {},
            }
            with target_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, indent=2)
                f.write("\n")

    @classmethod
    def load(cls, path: Path | str) -> LowCostCorrectionModel:
        """Load serialized model from JSON or Parquet file."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Correction model file not found at {target_path}")

        if target_path.suffix == ".parquet":
            df = pd.read_parquet(target_path)
            row = df.iloc[0]
            model = cls(
                a=float(row["a"]),
                b=float(row["b"]),
                c=float(row["c"]),
                epsilon=float(row["epsilon"]),
                alpha=float(row["alpha"]),
                is_fitted=bool(row["is_fitted"]),
                sensor_model=str(row["sensor_model"]) if pd.notna(row["sensor_model"]) else None,
            )
            return model

        with target_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(
            a=float(data["a"]),
            b=float(data["b"]),
            c=float(data["c"]),
            epsilon=float(data.get("epsilon", 1.35)),
            alpha=float(data.get("alpha", 0.0001)),
            is_fitted=bool(data["is_fitted"]),
            sensor_model=data.get("sensor_model"),
            metrics_=data.get("metrics", {}),
        )
        return model


def audit_corruption_priors(
    priors: dict[str, float],
    model_config_path: Path | str = "configs/model.yaml",
) -> dict[str, Any]:
    """Compare empirical inverted priors against model corruption augmentation ranges.

    Parameters
    ----------
    priors : dict[str, float]
        Inverted corruption priors containing 'gain_prior', 'rh_slope_prior', 'offset_prior'.
    model_config_path : Path | str
        Path to configs/model.yaml.

    Returns
    -------
    dict[str, Any]
        Comparison audit report.
    """
    cfg_path = Path(model_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found at {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    corruption = config.get("corruption", {})

    gain_range = corruption.get("gain_range", [0.8, 1.3])
    rh_slope_range = corruption.get("rh_slope_range", [0.0, 1.5])
    offset_range = corruption.get("offset_range", [-10.0, 25.0])

    g_val = priors.get("gain_prior", float("nan"))
    rh_val = priors.get("rh_slope_prior", float("nan"))
    off_val = priors.get("offset_prior", float("nan"))

    g_inside = bool(gain_range[0] <= g_val <= gain_range[1])
    rh_inside = bool(rh_slope_range[0] <= rh_val <= rh_slope_range[1])
    off_inside = bool(offset_range[0] <= off_val <= offset_range[1])

    return {
        "gain": {
            "empirical": g_val,
            "configured_range": gain_range,
            "within_prior": g_inside,
        },
        "rh_slope": {
            "empirical": rh_val,
            "configured_range": rh_slope_range,
            "within_prior": rh_inside,
        },
        "offset": {
            "empirical": off_val,
            "configured_range": offset_range,
            "within_prior": off_inside,
        },
        "all_within_priors": bool(g_inside and rh_inside and off_inside),
    }


def update_preprocessing_config_decision(
    rmse_improvement: float,
    config_path: Path | str = "configs/preprocessing.yaml",
    threshold: float = 0.05,
) -> bool:
    """Update configs/preprocessing.yaml with the empirical decision rule (FR-11).

    Enables lowcost_correction ONLY IF cross-validated RMSE improves by >= threshold (5%).

    Parameters
    ----------
    rmse_improvement : float
        Relative cross-validated RMSE improvement (e.g. 0.08 for 8%).
    config_path : Path | str
        Path to configs/preprocessing.yaml.
    threshold : float
        Decision threshold (default 0.05 = 5%).

    Returns
    -------
    bool
        Whether lowcost_correction was enabled.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing config not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    enabled = bool(rmse_improvement >= threshold)

    if "lowcost_correction" not in config:
        config["lowcost_correction"] = {}

    config["lowcost_correction"]["enabled"] = enabled

    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.dump(config, f, sort_keys=False)

    return enabled


def load_backfilled_domain_observations(backfill_dir: Path | str, domain: str) -> pd.DataFrame:
    """Load all backfilled hourly observations for a target domain."""
    path = Path(backfill_dir)
    pattern = f"openaq-archive_{domain}_*.parquet"
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No backfill observation files matching {path / pattern}")

    cols = ["location_id", "ts_utc", "pm25_ugm3", "pm25_raw_ugm3", "rh_pct"]
    dfs = [pd.read_parquet(f, columns=cols) for f in files]
    return pd.concat(dfs, ignore_index=True)


def run_fitting_pipeline(
    domain: str = "lahore",
    backfill_dir: str = "data/processed/backfill",
    registry_path: str | None = None,
    window_days: int = 60,
    max_distance_km: float = 8.0,
    min_pairs: int = 500,
    output_json: str | None = None,
    output_model: str | None = None,
    update_config: bool = False,
    config_path: str = "configs/preprocessing.yaml",
    model_config_path: str = "configs/model.yaml",
) -> dict[str, Any]:
    """Execute complete low-cost correction fit, cross-validation, and audit pipeline."""
    print("=" * 80)
    print(f"LOW-COST SENSOR CORRECTION FIT & VALIDATION (Domain: {domain.upper()})")
    print("=" * 80)

    # 1. Load Registry & Identify Candidates
    reg_file = Path(registry_path or f"data/registry/registry_{domain}.parquet")
    if not reg_file.exists():
        raise FileNotFoundError(f"Registry not found at {reg_file}")

    registry_df = pd.read_parquet(reg_file)
    ref_rows = registry_df[registry_df["is_reference"]]
    if ref_rows.empty:
        raise ValueError(f"No reference monitors found in {reg_file}")

    ref_id = int(ref_rows.iloc[0]["location_id"])
    ref_name = str(ref_rows.iloc[0]["name"])
    print(f"Reference Monitor Anchor: {ref_id} ({ref_name})")

    candidates = find_candidate_stations(
        registry_df, reference_location_id=ref_id, max_distance_km=max_distance_km
    )
    print(f"Candidate Low-Cost Stations within <={max_distance_km} km:")
    for _, row in candidates.iterrows():
        print(
            f"  - Location {row['location_id']} ({row['name']}): "
            f"distance = {row['distance_km']} km, model = {row.get('sensor_model', 'unknown')}"
        )

    # 2. Load Observations & Pair Data
    print(f"\nLoading backfilled observations from {backfill_dir} ...")
    obs_df = load_backfilled_domain_observations(backfill_dir, domain)
    print(f"Total domain observations loaded: {len(obs_df):,} records.")

    print(f"Building paired dataset over the last {window_days} days (min_pairs={min_pairs}) ...")
    paired_df = build_paired_observations(
        obs_df=obs_df,
        registry_df=registry_df,
        reference_location_id=ref_id,
        max_distance_km=max_distance_km,
        window_days=window_days,
        min_pairs=min_pairs,
    )
    print(f"Total paired hourly records: {len(paired_df):,}")

    # 3. Fit Huber Regression Model
    model = LowCostCorrectionModel()
    model.fit(paired_df)
    print("\nFitted Huber Regression Equation:")
    print(
        f"  pm25_corrected = max(0.0, {model.a:.4f} * pm25_raw + {model.b:.4f} * RH + ({model.c:.4f}))"
    )

    # 4. 5-Fold Cross-Validation
    print("\nExecuting 5-Fold Cross-Validation ...")
    cv_metrics = model.cross_validate(paired_df, n_splits=5)
    print(f"  Samples:                 {int(cv_metrics['n_samples']):,}")
    print(f"  Raw PM2.5 RMSE:          {cv_metrics['raw_rmse']:.2f} ug/m3")
    print(f"  Corrected PM2.5 RMSE:    {cv_metrics['corrected_rmse']:.2f} ug/m3")
    print(f"  RMSE Improvement:        {cv_metrics['rmse_improvement'] * 100:.2f} %")
    print(f"  Raw PM2.5 MAE:           {cv_metrics['raw_mae']:.2f} ug/m3")
    print(f"  Corrected PM2.5 MAE:     {cv_metrics['corrected_mae']:.2f} ug/m3")
    print(f"  MAE Improvement:         {cv_metrics['mae_improvement'] * 100:.2f} %")
    print(f"  Raw R^2:                 {cv_metrics['raw_r2']:.4f}")
    print(f"  Corrected R^2:           {cv_metrics['corrected_r2']:.4f}")

    # 5. Inverted Corruption Priors & Audit
    priors = model.get_inverted_corruption_priors()
    print("\nEmpirical Inverted Corruption Priors (Barkjohn Inversion):")
    print(f"  Gain Prior (1/a):        {priors['gain_prior']:.4f}")
    print(f"  RH Slope Prior (b/a):    {priors['rh_slope_prior']:.4f}")
    print(f"  Offset Prior (-c/a):     {priors['offset_prior']:.4f}")

    audit = audit_corruption_priors(priors, model_config_path=model_config_path)
    print(f"\nAudit against Configured Augmentation Priors ({model_config_path}):")
    print(
        f"  Gain:      empirical={audit['gain']['empirical']:.4f} in {audit['gain']['configured_range']} -> {audit['gain']['within_prior']}"
    )
    print(
        f"  RH Slope:  empirical={audit['rh_slope']['empirical']:.4f} in {audit['rh_slope']['configured_range']} -> {audit['rh_slope']['within_prior']}"
    )
    print(
        f"  Offset:    empirical={audit['offset']['empirical']:.4f} in {audit['offset']['configured_range']} -> {audit['offset']['within_prior']}"
    )
    print(f"  All Priors Enveloped:    {audit['all_within_priors']}")

    # 6. Decision Rule & Config Update (PRD FR-11)
    beneficial = cv_metrics["rmse_improvement"] >= 0.05
    print("\nPRD FR-11 Decision Rule:")
    print(f"  RMSE Improvement: {cv_metrics['rmse_improvement'] * 100:.2f}% (Threshold >= 5.00%)")
    print(f"  Correction Beneficial: {beneficial}")

    if update_config:
        enabled = update_preprocessing_config_decision(
            rmse_improvement=cv_metrics["rmse_improvement"],
            config_path=config_path,
            threshold=0.05,
        )
        print(f"  Updated {config_path}: lowcost_correction.enabled = {enabled}")

    # 7. Model and JSON Output
    if output_model:
        model.save(output_model)
        print(f"\nSaved correction model to: {output_model}")

    report: dict[str, Any] = {
        "domain": domain,
        "reference_station": {"location_id": ref_id, "name": ref_name},
        "candidate_stations": candidates[
            ["location_id", "name", "distance_km", "sensor_model"]
        ].to_dict(orient="records"),
        "window_days": window_days,
        "paired_samples": int(cv_metrics["n_samples"]),
        "coefficients": {"a": model.a, "b": model.b, "c": model.c},
        "cross_validation": cv_metrics,
        "inverted_priors": priors,
        "corruption_prior_audit": audit,
        "decision": {
            "threshold": 0.05,
            "rmse_improvement": cv_metrics["rmse_improvement"],
            "beneficial": beneficial,
        },
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"Saved audit report to: {output_json}")

    print("=" * 80)
    return report
