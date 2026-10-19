"""smogsense.models.baselines - Reference forecasters M0-M3b.

Persistence, raw CAMS, CAMS with trailing-window bias correction, climatological quantile
baseline, and linear quantile regression (M3b); all emit the same QuantileFunction object
as learned models and operate strictly on the available set A at issuance time T.

Specification: docs/evaluation-strategy.md -> 'Baselines'
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from smogsense.models.distribution import QuantileFunction

CANONICAL_TAUS: np.ndarray = np.round(np.arange(0.05, 0.951, 0.05), 2)

DEFAULT_M3B_FEATURES: tuple[str, ...] = (
    "lag_24h_pm25",
    "cams_pm25",
    "blh_m",
    "wind_speed",
)


def _check_pit_dataframe(
    df: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp | None,
) -> pd.DataFrame:
    """Validate point-in-time correctness on a training/history DataFrame."""
    if issuance_utc is None or df.empty:
        return df

    t_ref = pd.to_datetime(issuance_utc, utc=True)
    if "available_at_utc" in df.columns:
        avail = pd.to_datetime(df["available_at_utc"], utc=True)
        if (avail > t_ref).any():
            raise ValueError(
                f"Point-in-time leakage: baseline training data contains available_at_utc > {t_ref}"
            )
    elif "ts_utc" in df.columns:
        ts = pd.to_datetime(df["ts_utc"], utc=True)
        if (ts > t_ref).any():
            raise ValueError(
                f"Point-in-time leakage: baseline training data contains ts_utc > {t_ref}"
            )
    return df


class M0Persistence:
    """Baseline M0: Persistence.

    Uses the latest observed 24-h block mean repeated at all horizons.
    Probabilistic via residual quantiles from the available set A.
    Undefined at N=0 (raises ValueError if residual_quantiles is None).
    """

    method_id: str = "m0_persistence"

    def __init__(self, residual_quantiles: Sequence[float] | np.ndarray | None) -> None:
        """Initialize M0.

        Args:
            residual_quantiles: 19 quantiles of historical residuals from set A. None if N=0.
        """
        self.residual_quantiles = None
        if residual_quantiles is not None:
            res_arr = np.array(residual_quantiles, dtype=float)
            if len(res_arr) != 19:
                raise ValueError("M0 requires exactly 19 residual quantiles.")
            self.residual_quantiles = res_arr

    def predict(self, recent_24h_obs: pd.Series | np.ndarray | list[float]) -> QuantileFunction:
        """Predict the distribution for any horizon based on the last 24h block.

        Args:
            recent_24h_obs: The latest available 24-h block of observations.

        Returns:
            QuantileFunction representing the probabilistic forecast.

        Raises:
            ValueError: if N=0 (no residual_quantiles provided) or if recent_24h_obs is invalid.
        """
        if self.residual_quantiles is None:
            raise ValueError("M0 Persistence is undefined at N=0 (no residual quantiles).")

        obs = np.asarray(recent_24h_obs, dtype=float)
        if np.isnan(obs).any() or len(obs) == 0:
            raise ValueError("M0 Persistence requires a complete valid 24h block.")

        mean_val = float(np.mean(obs))
        return QuantileFunction(np.maximum(0.0, mean_val + self.residual_quantiles))


class M1Cams:
    """Baseline M1: Raw CAMS.

    Uses the raw CAMS PM2.5 forecast at the correct lead time.
    Deterministic at N=0, probabilistic via residual quantiles from A at N>=1.
    """

    method_id: str = "m1_cams_raw"

    def __init__(self, residual_quantiles: Sequence[float] | np.ndarray | None) -> None:
        """Initialize M1.

        Args:
            residual_quantiles: 19 quantiles of historical residuals from set A. None if N=0.
        """
        self.residual_quantiles = None
        if residual_quantiles is not None:
            res_arr = np.array(residual_quantiles, dtype=float)
            if len(res_arr) != 19:
                raise ValueError("M1 requires exactly 19 residual quantiles.")
            self.residual_quantiles = res_arr

    def predict(self, cams_forecast: float) -> QuantileFunction | float:
        """Predict the distribution for a specific horizon given its CAMS forecast.

        Args:
            cams_forecast: The raw CAMS point forecast for the target horizon.

        Returns:
            QuantileFunction if N>=1, else float (deterministic) if N=0.
        """
        if self.residual_quantiles is None:
            return float(cams_forecast)

        return QuantileFunction(cams_forecast + self.residual_quantiles)


class M2CamsBiasCorrected:
    """Baseline M2: CAMS with trailing-window bias correction (m2_cams_bc).

    Fits a linear bias correction (in linear or log1p space) of observed station PM2.5
    against CAMS PM2.5 over a trailing calibration window from the available set A,
    plus 19 empirical residual quantiles mapped to a non-crossing, non-negative
    QuantileFunction.

    Gracefully falls back to M1Cams when historical pairs are fewer than `min_samples`
    (default 7 days) or at N=0.
    """

    method_id: str = "m2_cams_bc"

    def __init__(
        self,
        cams_history: Sequence[float] | np.ndarray | pd.Series | None = None,
        obs_history: Sequence[float] | np.ndarray | pd.Series | None = None,
        history_df: pd.DataFrame | None = None,
        min_samples: int = 7,
        window_days: int = 30,
        log_space: bool = False,
        shrinkage_prior_weight: float = 0.0,
        fallback_residual_quantiles: Sequence[float] | np.ndarray | None = None,
        issuance_utc: datetime | pd.Timestamp | None = None,
    ) -> None:
        self.min_samples = int(min_samples)
        self.window_days = int(window_days)
        self.log_space = bool(log_space)
        self.shrinkage_prior_weight = float(shrinkage_prior_weight)
        self.intercept_: float = 0.0
        self.slope_: float = 1.0
        self.residual_quantiles_: np.ndarray = np.zeros(19, dtype=float)
        self.n_samples_: int = 0
        self.is_fallback_: bool = True
        self._m1_fallback = M1Cams(fallback_residual_quantiles)

        if history_df is not None or (cams_history is not None and obs_history is not None):
            self.fit(
                cams_history=cams_history,
                obs_history=obs_history,
                history_df=history_df,
                issuance_utc=issuance_utc,
            )

    def fit(
        self,
        cams_history: Sequence[float] | np.ndarray | pd.Series | None = None,
        obs_history: Sequence[float] | np.ndarray | pd.Series | None = None,
        history_df: pd.DataFrame | None = None,
        issuance_utc: datetime | pd.Timestamp | None = None,
    ) -> M2CamsBiasCorrected:
        """Fit bias correction parameters and empirical residual quantiles on available set A."""
        if history_df is not None:
            df = _check_pit_dataframe(history_df.copy(), issuance_utc)
            if not df.empty and "ts_utc" in df.columns and self.window_days > 0:
                ts = pd.to_datetime(df["ts_utc"], utc=True)
                max_ts = pd.to_datetime(issuance_utc, utc=True) if issuance_utc else ts.max()
                cutoff = max_ts - timedelta(days=self.window_days)
                df = df[(ts >= cutoff) & (ts <= max_ts)]

            cams_col = next(
                (c for c in ("cams_pm25", "cams_pred", "cams_ugm3") if c in df.columns),
                None,
            )
            obs_col = next(
                (c for c in ("obs_pm25", "y_obs", "pm25_ugm3") if c in df.columns),
                None,
            )
            if cams_col is None or obs_col is None:
                raise ValueError(
                    "history_df must contain CAMS column ('cams_pm25'/'cams_pred') "
                    "and observation column ('obs_pm25'/'y_obs'/'pm25_ugm3')."
                )
            c_arr = pd.to_numeric(df[cams_col], errors="coerce").to_numpy(dtype=float)
            y_arr = pd.to_numeric(df[obs_col], errors="coerce").to_numpy(dtype=float)
        elif cams_history is not None and obs_history is not None:
            c_arr = np.asarray(cams_history, dtype=float).ravel()
            y_arr = np.asarray(obs_history, dtype=float).ravel()
            if len(c_arr) != len(y_arr):
                raise ValueError("cams_history and obs_history must have the same length.")
            if self.window_days > 0 and len(c_arr) > self.window_days:
                c_arr = c_arr[-self.window_days :]
                y_arr = y_arr[-self.window_days :]
        else:
            self.is_fallback_ = True
            self.n_samples_ = 0
            return self

        valid_mask = np.isfinite(c_arr) & np.isfinite(y_arr) & (c_arr >= 0.0) & (y_arr >= 0.0)
        c_valid = c_arr[valid_mask]
        y_valid = y_arr[valid_mask]
        self.n_samples_ = len(c_valid)

        if self.n_samples_ < self.min_samples:
            self.is_fallback_ = True
            self.intercept_ = 0.0
            self.slope_ = 1.0
            if self.n_samples_ > 0:
                residuals = y_valid - c_valid
                self.residual_quantiles_ = np.quantile(residuals, CANONICAL_TAUS)
            return self

        self.is_fallback_ = False
        if self.log_space:
            x = np.log1p(c_valid)
            z = np.log1p(y_valid)
        else:
            x = c_valid
            z = y_valid

        x_var = float(np.var(x))
        if x_var < 1e-8:
            # Constant CAMS predictor: purely additive shift (slope = 1.0)
            slope = 1.0
            intercept = float(np.mean(z - x))
        else:
            x_mean = float(np.mean(x))
            z_mean = float(np.mean(z))
            cov_xz = float(np.mean((x - x_mean) * (z - z_mean)))
            raw_slope = cov_xz / x_var
            raw_intercept = z_mean - raw_slope * x_mean

            # Optional shrinkage toward identity (intercept=0, slope=1)
            w = self.n_samples_ / (self.n_samples_ + max(0.0, self.shrinkage_prior_weight))
            slope = w * raw_slope + (1.0 - w) * 1.0
            intercept = w * raw_intercept

        self.intercept_ = float(intercept)
        self.slope_ = float(slope)

        if self.log_space:
            z_hat = self.intercept_ + self.slope_ * x
            y_hat = np.maximum(0.0, np.expm1(z_hat))
        else:
            y_hat = np.maximum(0.0, self.intercept_ + self.slope_ * x)

        residuals = y_valid - y_hat
        self.residual_quantiles_ = np.quantile(residuals, CANONICAL_TAUS)
        return self

    def predict_point(self, cams_forecast: float) -> float:
        """Return bias-corrected point forecast (in ug/m3, floored at 0)."""
        c_val = max(0.0, float(cams_forecast))
        if self.is_fallback_:
            return c_val
        if self.log_space:
            z_hat = self.intercept_ + self.slope_ * np.log1p(c_val)
            return float(max(0.0, np.expm1(z_hat)))
        return float(max(0.0, self.intercept_ + self.slope_ * c_val))

    def predict(
        self,
        cams_forecast: float,
        deterministic_at_n0: bool = False,
    ) -> QuantileFunction | float:
        """Predict 19-quantile distribution for a target horizon given raw CAMS forecast."""
        if self.is_fallback_:
            m1_pred = self._m1_fallback.predict(cams_forecast)
            if isinstance(m1_pred, QuantileFunction):
                return QuantileFunction(np.maximum(0.0, m1_pred.q))
            if deterministic_at_n0:
                return float(m1_pred)
            c_floor = max(0.0, float(m1_pred))
            return QuantileFunction(np.maximum(0.0, c_floor + self.residual_quantiles_))

        y_hat = self.predict_point(cams_forecast)
        q_vals = np.maximum(0.0, y_hat + self.residual_quantiles_)
        return QuantileFunction(q_vals)


class M3Climatology:
    """Baseline M3: Day-of-year seasonal climatology (m3_climatology).

    Computes the 19 empirical quantiles of historical PM2.5 observations in the
    available set A within a +/- `window_half_days` (default 15 days) circular
    day-of-year window around the target date. Falls back to full available
    history when the seasonal window has fewer than `min_window_samples` points.
    Undefined at N=0 (raises ValueError if no valid observations exist).
    """

    method_id: str = "m3_climatology"

    def __init__(
        self,
        history_df: pd.DataFrame | None = None,
        values: Sequence[float] | np.ndarray | pd.Series | None = None,
        window_half_days: int = 15,
        min_window_samples: int = 5,
        issuance_utc: datetime | pd.Timestamp | None = None,
    ) -> None:
        self.window_half_days = int(window_half_days)
        self.min_window_samples = int(min_window_samples)
        self._history_df: pd.DataFrame = pd.DataFrame()
        self._values: np.ndarray = np.array([], dtype=float)

        if history_df is not None or values is not None:
            self.fit(history_df=history_df, values=values, issuance_utc=issuance_utc)

    def fit(
        self,
        history_df: pd.DataFrame | None = None,
        values: Sequence[float] | np.ndarray | pd.Series | None = None,
        issuance_utc: datetime | pd.Timestamp | None = None,
    ) -> M3Climatology:
        """Fit climatology store from historical observations in available set A."""
        if history_df is not None:
            df = _check_pit_dataframe(history_df.copy(), issuance_utc)
            val_col = next(
                (c for c in ("pm25_ugm3", "y_obs", "obs_pm25", "value") if c in df.columns),
                None,
            )
            if val_col is None:
                raise ValueError(
                    "history_df must contain a PM2.5 column ('pm25_ugm3', 'y_obs', or 'obs_pm25')."
                )
            if "imputed" in df.columns:
                df = df[~df["imputed"].fillna(False).astype(bool)]

            vals = pd.to_numeric(df[val_col], errors="coerce")
            valid_mask = vals.notna() & (vals >= 0.0)
            df_clean = df.loc[valid_mask].copy()
            df_clean["_pm25_val"] = vals.loc[valid_mask].to_numpy(dtype=float)

            ts_col = next(
                (c for c in ("ts_utc", "target_date", "date") if c in df_clean.columns),
                None,
            )
            if ts_col is not None:
                ts = pd.to_datetime(df_clean[ts_col], utc=True)
                df_clean["_doy"] = ts.dt.dayofyear.to_numpy(dtype=int)
            else:
                df_clean["_doy"] = np.nan

            self._history_df = df_clean.reset_index(drop=True)
            self._values = df_clean["_pm25_val"].to_numpy(dtype=float)
        elif values is not None:
            arr = np.asarray(values, dtype=float).ravel()
            arr = arr[np.isfinite(arr) & (arr >= 0.0)]
            self._values = arr
            self._history_df = pd.DataFrame({"_pm25_val": arr, "_doy": np.nan})
        else:
            self._values = np.array([], dtype=float)
            self._history_df = pd.DataFrame()

        if len(self._values) == 0:
            raise ValueError(
                "M3 Climatology is undefined at N=0 (no valid historical observations)."
            )

        return self

    @staticmethod
    def _circular_doy_distance(doy_array: np.ndarray, target_doy: int) -> np.ndarray:
        """Compute circular day-of-year distance on a 365.25-day cycle."""
        diff = np.abs(doy_array - target_doy)
        return np.asarray(np.minimum(diff, 365 - diff), dtype=int)

    def predict(
        self,
        target_date: datetime | pd.Timestamp | str | None = None,
        horizon_h: int = 24,
    ) -> QuantileFunction:
        """Predict 19-quantile climatological distribution for target date / horizon."""
        if len(self._values) == 0:
            raise ValueError("M3 Climatology is undefined at N=0 (not fitted or empty history).")

        if (
            target_date is not None
            and not self._history_df.empty
            and "_doy" in self._history_df.columns
        ):
            has_doy = self._history_df["_doy"].notna()
            if has_doy.any():
                dt = pd.to_datetime(target_date, utc=True)
                # Shift by horizon if target_date is an issuance timestamp
                target_dt = dt + timedelta(hours=int(horizon_h) - 12) if horizon_h > 0 else dt
                target_doy = int(target_dt.dayofyear)
                doy_vals = self._history_df.loc[has_doy, "_doy"].to_numpy(dtype=int)
                dists = self._circular_doy_distance(doy_vals, target_doy)
                in_win = dists <= self.window_half_days
                win_samples = self._history_df.loc[has_doy, "_pm25_val"].to_numpy(dtype=float)[
                    in_win
                ]
                if len(win_samples) >= self.min_window_samples:
                    q_win = np.maximum(0.0, np.quantile(win_samples, CANONICAL_TAUS))
                    return QuantileFunction(q_win)

        q_all = np.maximum(0.0, np.quantile(self._values, CANONICAL_TAUS))
        return QuantileFunction(q_all)

    def predict_horizons(
        self,
        issuance_utc: datetime | pd.Timestamp,
        horizons: Sequence[int] = (24, 48, 72),
    ) -> dict[int, QuantileFunction]:
        """Predict climatological QuantileFunction for each horizon in {24, 48, 72}."""
        return {int(h): self.predict(target_date=issuance_utc, horizon_h=int(h)) for h in horizons}


class M3bLinearQuantileRegression:
    """Baseline M3b: Linear Quantile Regression (m3b_linear_qr).

    Fits 19 independent linear quantile regression models (pinball loss with L1
    regularisation `alpha` solved via HiGHS linear programming) across
    tau in {0.05, 0.10, ..., 0.95} on compact point-in-time predictors from
    the available set A, followed by monotone rearrangement (`QuantileFunction`)
    and non-negative flooring at 0.0.
    """

    method_id: str = "m3b_linear_qr"

    def __init__(
        self,
        feature_cols: Sequence[str] | None = None,
        alpha: float = 1e-3,
        min_samples: int = 5,
    ) -> None:
        self.feature_cols: list[str] | None = (
            list(feature_cols) if feature_cols is not None else None
        )
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.models_: dict[float, QuantileRegressor] = {}
        self.feature_medians_: np.ndarray | None = None
        self.feature_scales_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def _extract_matrix(
        self,
        X: pd.DataFrame | np.ndarray | Mapping[str, Any] | Sequence[Any],  # noqa: N803
        is_training: bool = False,
    ) -> np.ndarray:
        """Convert DataFrame, dict, or array into a 2D float matrix."""
        if isinstance(X, pd.DataFrame):
            if is_training and self.feature_cols is None:
                preferred = [c for c in DEFAULT_M3B_FEATURES if c in X.columns]
                if preferred:
                    self.feature_cols = preferred
                else:
                    exclude = {
                        "y",
                        "y_obs",
                        "obs_pm25",
                        "pm25_ugm3",
                        "ts_utc",
                        "available_at_utc",
                        "issuance_utc",
                        "location_id",
                        "domain",
                    }
                    self.feature_cols = [
                        c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude
                    ]
            if not self.feature_cols:
                raise ValueError(
                    "No numeric predictor columns found for M3bLinearQuantileRegression."
                )
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required predictor columns for M3b: {missing}")
            mat = X[self.feature_cols].to_numpy(dtype=float)
        elif isinstance(X, Mapping):
            if not self.feature_cols:
                self.feature_cols = list(X.keys())
            row = [float(X[c]) for c in self.feature_cols]
            mat = np.asarray([row], dtype=float)
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim == 1:
                mat = arr.reshape(1, -1)
            elif arr.ndim == 2:
                mat = arr
            else:
                raise ValueError(f"Expected 1D or 2D predictor array, got shape {arr.shape}")
        return mat

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,  # noqa: N803
        y: Sequence[float] | np.ndarray | pd.Series | None = None,
        target_col: str = "y_obs",
        issuance_utc: datetime | pd.Timestamp | None = None,
    ) -> M3bLinearQuantileRegression:
        """Fit 19 linear quantile regressors on point-in-time training set from A."""
        if isinstance(X, pd.DataFrame):
            df = _check_pit_dataframe(X.copy(), issuance_utc)
            if y is None:
                t_col = next(
                    (
                        c
                        for c in (target_col, "y", "y_obs", "obs_pm25", "pm25_ugm3")
                        if c in df.columns
                    ),
                    None,
                )
                if t_col is None:
                    raise ValueError("Target `y` must be provided or present in DataFrame.")
                y_arr = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
            else:
                y_arr = np.asarray(y, dtype=float).ravel()
            x_mat = self._extract_matrix(df, is_training=True)
        else:
            if y is None:
                raise ValueError("Target `y` must be provided when `X` is an array.")
            x_mat = self._extract_matrix(X, is_training=True)
            y_arr = np.asarray(y, dtype=float).ravel()

        if len(x_mat) != len(y_arr):
            raise ValueError("Predictor matrix X and target vector y must have the same length.")

        valid_target = np.isfinite(y_arr) & (y_arr >= 0.0)
        x_mat = x_mat[valid_target]
        y_arr = y_arr[valid_target]

        if len(y_arr) < self.min_samples:
            raise ValueError(
                f"M3bLinearQuantileRegression requires at least {self.min_samples} valid training "
                f"samples in available set A (got {len(y_arr)})."
            )

        # Impute any missing feature values with column medians from the training set
        col_medians = np.nanmedian(x_mat, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        self.feature_medians_ = col_medians

        x_clean = x_mat.copy()
        for j in range(x_clean.shape[1]):
            nan_mask = ~np.isfinite(x_clean[:, j])
            if np.any(nan_mask):
                x_clean[nan_mask, j] = col_medians[j]

        col_scales = np.nanstd(x_clean, axis=0)
        col_scales = np.where(col_scales > 1e-8, col_scales, 1.0)
        self.feature_scales_ = col_scales

        x_scaled = (x_clean - self.feature_medians_) / self.feature_scales_

        self.models_.clear()
        for tau in CANONICAL_TAUS:
            qr = QuantileRegressor(quantile=float(tau), alpha=self.alpha, solver="highs")
            qr.fit(x_scaled, y_arr)
            self.models_[float(tau)] = qr

        self.is_fitted_ = True
        return self

    def predict_quantiles(
        self,
        X: pd.DataFrame | np.ndarray | Mapping[str, Any] | Sequence[float],  # noqa: N803
    ) -> np.ndarray:
        """Predict rearranged, non-negative (n_samples, 19) quantile matrix."""
        if not self.is_fitted_ or self.feature_medians_ is None or self.feature_scales_ is None:
            raise ValueError("M3bLinearQuantileRegression must be fitted before calling predict.")

        x_mat = self._extract_matrix(X, is_training=False)
        if x_mat.shape[1] != len(self.feature_medians_):
            raise ValueError(
                f"Expected {len(self.feature_medians_)} features, got {x_mat.shape[1]}."
            )

        x_clean = x_mat.copy()
        for j in range(x_clean.shape[1]):
            nan_mask = ~np.isfinite(x_clean[:, j])
            if np.any(nan_mask):
                x_clean[nan_mask, j] = self.feature_medians_[j]

        x_scaled = (x_clean - self.feature_medians_) / self.feature_scales_

        n = x_scaled.shape[0]
        raw_preds = np.zeros((n, len(CANONICAL_TAUS)), dtype=float)
        for idx, tau in enumerate(CANONICAL_TAUS):
            raw_preds[:, idx] = self.models_[float(tau)].predict(x_scaled)

        # Floor at 0.0 and apply monotone rearrangement across the 19 quantiles
        floored = np.maximum(0.0, raw_preds)
        rearranged = np.sort(floored, axis=1)
        return rearranged

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | Mapping[str, Any] | Sequence[float],  # noqa: N803
    ) -> QuantileFunction | list[QuantileFunction]:
        """Predict QuantileFunction (for single row) or list[QuantileFunction] (for multiple rows)."""
        q_mat = self.predict_quantiles(X)
        if q_mat.shape[0] == 1 and not (isinstance(X, pd.DataFrame) and len(X) > 1):
            return QuantileFunction(q_mat[0])
        return [QuantileFunction(q_mat[i]) for i in range(q_mat.shape[0])]
