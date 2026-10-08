"""smogsense.preprocessing.imputation - Gap handling: shape-preserving short-gap fill, masked long gaps.

Gaps of at most 3 consecutive hours are filled with PCHIP in log1p space (cubic splines are
rejected because they overshoot); longer gaps stay missing with an explicit mask. Imputed values
are inputs only and are never used as training or scoring targets.

Public contract (implemented in Phase 1):
- impute_short_gaps(series, max_gap_h=3) -> (series, imputed_mask)
- pchip_impute(series, max_gap=3) -> series

Specification: docs/data-engineering.md -> 'Gap handling and imputation'
"""

import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate  # type: ignore


def pchip_impute(series: pd.Series, max_gap: int = 3) -> pd.Series:
    """
    Impute short internal gaps using PCHIP interpolation in log1p space.

    Args:
        series: The PM2.5 series to impute.
        max_gap: Maximum number of consecutive NaNs to fill.

    Returns:
        pd.Series with eligible gaps filled.
    """
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")

    out = series.copy()
    if len(out) == 0:
        return out

    arr = out.to_numpy()
    mask_na = pd.isna(arr)

    valid_idx = np.where(~mask_na)[0]
    if len(valid_idx) < 2:
        return out

    first_v, last_v = valid_idx[0], valid_idx[-1]

    eligible = mask_na.copy()
    eligible[:first_v] = False
    eligible[last_v + 1 :] = False

    run_id_arr = np.cumsum(mask_na != np.roll(mask_na, 1))
    for rid in np.unique(run_id_arr[eligible]):
        if np.sum(run_id_arr == rid) > max_gap:
            eligible[run_id_arr == rid] = False

    if not np.any(eligible):
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        x_all = out.index.astype(np.int64).to_numpy()
    else:
        x_all = np.arange(len(out))

    x_valid = x_all[~mask_na]
    y_valid = arr[~mask_na]
    x_interp = x_all[eligible]

    y_log = np.log1p(y_valid)
    y_interp_log = pchip_interpolate(x_valid, y_log, x_interp)
    y_interp = np.expm1(y_interp_log)

    out.iloc[np.where(eligible)[0]] = y_interp
    return out


def impute_short_gaps(series: pd.Series, max_gap_h: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Legacy/contractual wrapper.
    Returns (imputed_series, imputed_mask).
    """
    imputed = pchip_impute(series, max_gap=max_gap_h)
    mask = imputed.notna() & series.isna()
    return imputed, mask
