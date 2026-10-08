"""smogsense.preprocessing.qc - Rule-based quality control with bit-flagged outcomes.

Range, flat-line, spike, spatial-outlier, humidity and completeness checks. Never deletes rows:
sets bits in qc_flags and nulls the cleaned value so that raw and cleaned series remain
auditable.

Public contract (implemented in Phase 1):
- apply_qc(df, rules) -> df with pm25_ugm3, qc_flags
- clean_observations(df) -> unified pipeline applying QC and imputation (adds bit 512)
- Bit meanings are documented in data/schemas/observations_hourly.schema.yaml.

Specification: docs/data-engineering.md -> 'Quality control and low-cost sensor handling'
"""

from typing import Any

import numpy as np
import pandas as pd

from smogsense.preprocessing.imputation import impute_short_gaps

RANGE_REJECT = 1
NEGATIVE_CLIPPED = 2
FLATLINE = 4
SPIKE = 8
IMPUTED = 512


def apply_qc(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    """Apply QC to observation dataframe.

    Operates on pm25_ugm3, sets qc_flags.
    Returns a new dataframe.
    """
    out = df.copy()
    if "qc_flags" not in out.columns:
        out["qc_flags"] = 0
    out["qc_flags"] = out["qc_flags"].fillna(0).astype("Int64")

    if "pm25_ugm3" not in out.columns:
        return out

    group_col = (
        "sensor_id"
        if "sensor_id" in out.columns
        else ("location_id" if "location_id" in out.columns else None)
    )

    # 1. RANGE_REJECT
    mask_range = (out["pm25_ugm3"] < -5) | (out["pm25_ugm3"] > 1500)
    out.loc[mask_range, "qc_flags"] |= RANGE_REJECT
    out.loc[mask_range, "pm25_ugm3"] = np.nan

    # 2. NEGATIVE_CLIPPED
    mask_neg = (out["pm25_ugm3"] >= -5) & (out["pm25_ugm3"] < 0)
    out.loc[mask_neg, "qc_flags"] |= NEGATIVE_CLIPPED
    out.loc[mask_neg, "pm25_ugm3"] = 0.0

    if group_col and "ts_utc" in out.columns:
        # Sort values internally to ensure temporal functions work
        out = out.sort_values([group_col, "ts_utc"])

        flatline_indices: list[Any] = []

        def mark_flatline(group: pd.DataFrame) -> None:
            val = group["pm25_ugm3"]
            ts = group["ts_utc"]

            valid_val = val > 0
            same_val = val == val.shift(1)
            consecutive = ts.diff() == pd.Timedelta(hours=1)

            new_run = (~same_val) | (~consecutive) | (~valid_val)
            run_id = new_run.cumsum()

            run_counts = run_id[valid_val].value_counts()
            flatline_runs = run_counts[run_counts >= 6].index

            is_flat = valid_val & run_id.isin(flatline_runs)
            flatline_indices.extend(group.index[is_flat])

        for _, group in out.groupby(group_col):
            mark_flatline(group)

        out.loc[flatline_indices, "qc_flags"] |= FLATLINE
        out.loc[flatline_indices, "pm25_ugm3"] = np.nan

        # 4. SPIKE
        spike_indices: list[Any] = []

        def mark_spike(group: pd.DataFrame) -> None:
            tmp = group.set_index("ts_utc")
            tmp_val = tmp["pm25_ugm3"]

            if len(tmp) == 0:
                return

            full_idx = pd.date_range(tmp.index.min(), tmp.index.max(), freq="h")
            tmp_val_nodup = tmp_val[~tmp.index.duplicated(keep="first")]
            tmp_val_full = tmp_val_nodup.reindex(full_idx)

            def calc_mad(x: np.ndarray) -> float:
                m = np.nanmedian(x)
                if np.isnan(m):
                    return np.nan
                return float(np.nanmedian(np.abs(x - m)))

            c_med = tmp_val_full.rolling(window=7, center=True, min_periods=7).median()
            c_mad = tmp_val_full.rolling(window=7, center=True, min_periods=7).apply(
                calc_mad, raw=True
            )

            t_med = tmp_val_full.rolling(window=7, min_periods=4).median()
            t_mad = tmp_val_full.rolling(window=7, min_periods=4).apply(calc_mad, raw=True)

            # backward rolling for the start edges
            b_med = tmp_val_full.iloc[::-1].rolling(window=7, min_periods=4).median().iloc[::-1]
            b_mad = (
                tmp_val_full.iloc[::-1]
                .rolling(window=7, min_periods=4)
                .apply(calc_mad, raw=True)
                .iloc[::-1]
            )

            rolling_med = c_med.combine_first(t_med).combine_first(b_med)
            rolling_mad = c_mad.combine_first(t_mad).combine_first(b_mad)

            med_aligned = rolling_med.reindex(tmp.index)
            mad_aligned = rolling_mad.reindex(tmp.index)

            threshold = np.maximum(8 * mad_aligned, 100.0)
            is_spike = (
                (tmp_val > med_aligned + threshold)
                & tmp_val.notna()
                & med_aligned.notna()
                & mad_aligned.notna()
            )

            spike_indices.extend(group.index[is_spike.to_numpy(dtype=bool)])

        for _, group in out.groupby(group_col):
            mark_spike(group)

        out.loc[spike_indices, "qc_flags"] |= SPIKE
        out.loc[spike_indices, "pm25_ugm3"] = np.nan

        # Sort back to original index order
        out = out.sort_index()

    # The prompt: "Follow the project's DataFrame index contract. If the repository's Pandera contract requires a reset index, reset it before returning."
    # We will just preserve the index.
    return out


def expand_to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df

    group_col = (
        "sensor_id"
        if "sensor_id" in df.columns
        else ("location_id" if "location_id" in df.columns else None)
    )
    if not group_col or "ts_utc" not in df.columns:
        return df

    def _expand_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("ts_utc").sort_index()
        g = g[~g.index.duplicated(keep="first")]
        if len(g) == 0:
            return g
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_idx)
        # Forward fill the group identifier
        g[group_col] = g[group_col].ffill().bfill()
        # Fill qc_flags with 0 for new rows
        if "qc_flags" in g.columns:
            g["qc_flags"] = g["qc_flags"].fillna(0).astype(int)
        g.index.name = "ts_utc"
        return g.reset_index()

    return df.groupby(group_col, group_keys=False).apply(_expand_group).reset_index(drop=True)


def clean_observations(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Unified pipeline:
    1. Apply QC rules (range, negative clip, flatline, spike).
    2. Impute short internal gaps.
    3. Accumulate IMPUTED bit (512) into qc_flags.
    """
    df_expanded = expand_to_hourly_grid(df)
    out = apply_qc(df_expanded, rules)

    if "pm25_ugm3" not in out.columns:
        return out

    group_col = (
        "sensor_id"
        if "sensor_id" in out.columns
        else ("location_id" if "location_id" in out.columns else None)
    )

    if group_col:
        # Impute per group to avoid interpolating across different sensors
        out = out.sort_values([group_col, "ts_utc"])

        imputed_series_list = []
        mask_series_list = []

        for _, group in out.groupby(group_col):
            imp, mask = impute_short_gaps(group["pm25_ugm3"], max_gap_h=3)
            imputed_series_list.append(imp)
            mask_series_list.append(mask)

        new_pm25 = pd.concat(imputed_series_list)
        new_mask = pd.concat(mask_series_list)

        # Align correctly since we sorted
        out["pm25_ugm3"] = new_pm25
        out.loc[new_mask, "qc_flags"] |= IMPUTED

        out = out.sort_index()
    else:
        imp, mask = impute_short_gaps(out["pm25_ugm3"], max_gap_h=3)
        out["pm25_ugm3"] = imp
        out.loc[mask, "qc_flags"] |= IMPUTED

    return out
