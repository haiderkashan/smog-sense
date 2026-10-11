"""smogsense.preprocessing.qc - Rule-based quality control with bit-flagged outcomes.

Range, negative-clip, flat-line, spike, spatial-outlier, humidity, sub-hourly coverage,
high-range sensor limit, drift, and imputation checks. Never deletes rows: sets bits in
qc_flags and nulls the cleaned value where rejected so that raw and cleaned series remain
auditable.

Public contract (implemented in Phase 1):
- apply_qc(df, rules=None, stations_df=None) -> df with pm25_ugm3, qc_flags
- clean_observations(df, rules=None, stations_df=None) -> unified pipeline applying QC and imputation (adds bit 512)
- Bit meanings are documented in data/schemas/observations_hourly.schema.yaml:
    1: RANGE_REJECT
    2: NEGATIVE_CLIPPED
    4: FLATLINE
    8: SPIKE
   16: SPATIAL_OUTLIER
   32: HIGH_RH (>90%)
   64: LOW_COVERAGE (<75% sub-hourly)
  128: HIGH_RANGE (>500 ug/m3, retained but flagged)
  256: DRIFT_SUSPECT (slope outside [0.5, 2.0] vs neighbours)
  512: IMPUTED (short-gap PCHIP interpolation)

Specification: docs/data-engineering.md -> 'Quality control and low-cost sensor handling'
"""

from typing import Any

import numpy as np
import pandas as pd
import pyproj

from smogsense.preprocessing.imputation import impute_short_gaps

RANGE_REJECT = 1
NEGATIVE_CLIPPED = 2
FLATLINE = 4
SPIKE = 8
SPATIAL_OUTLIER = 16
HIGH_RH = 32
LOW_COVERAGE = 64
HIGH_RANGE = 128
DRIFT_SUSPECT = 256
IMPUTED = 512

DEFAULT_QC_RULES: dict[str, Any] = {
    "hard_range_ugm3": {"min": 0.0, "max": 1500.0},
    "negative_clip": {"floor_ugm3": -5.0},
    "high_range_flag_ugm3": 500.0,
    "flatline": {"min_consecutive_hours": 6, "ignore_value_leq_ugm3": 0.0},
    "spike": {"window_hours": 7, "k_mad": 8.0, "floor_ugm3": 100.0},
    "spatial_outlier": {
        "radius_km": 15.0,
        "min_neighbors": 3,
        "k_mad": 6.0,
        "floor_ugm3": 80.0,
    },
    "humidity": {"flag_above_pct": 90.0},
    "coverage": {"min_fraction_subhourly": 0.75},
    "drift": {
        "window_days": 7,
        "min_overlap_hours": 72,
        "slope_bounds": [0.5, 2.0],
        "reference": "neighbours_median",
    },
}


def _calc_mad(x: np.ndarray) -> float:
    """Calculate median absolute deviation."""
    m = np.nanmedian(x)
    if np.isnan(m):
        return float(np.nan)
    return float(np.nanmedian(np.abs(x - m)))


def apply_qc(
    df: pd.DataFrame,
    rules: dict[str, Any] | None = None,
    stations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply rule-based quality control to observation dataframe.

    Operates on pm25_ugm3, sets bits in qc_flags, and nulls rejected values.
    Returns a new dataframe preserving the input index.

    Parameters
    ----------
    df : pd.DataFrame
        Observation dataframe containing pm25_ugm3, ts_utc, and station/sensor identifiers.
    rules : dict[str, Any] | None, optional
        Quality control rule thresholds matching configs/preprocessing.yaml.
    stations_df : pd.DataFrame | None, optional
        Station metadata table containing lat and lon coordinates for spatial checks.
    """
    out = df.copy()
    if "qc_flags" not in out.columns:
        out["qc_flags"] = 0
    out["qc_flags"] = out["qc_flags"].fillna(0).astype("Int64")

    if "pm25_ugm3" not in out.columns or len(out) == 0:
        return out

    cfg = DEFAULT_QC_RULES.copy()
    if rules:
        cfg.update(rules)

    group_col = (
        "sensor_id"
        if "sensor_id" in out.columns
        else ("location_id" if "location_id" in out.columns else None)
    )

    # -------------------------------------------------------------------------
    # 1. RANGE_REJECT (Bit 1): value < -5 or > 1500 ug/m3
    # -------------------------------------------------------------------------
    hard_range = cfg.get("hard_range_ugm3", {"min": 0.0, "max": 1500.0})
    neg_clip_cfg = cfg.get("negative_clip", {"floor_ugm3": -5.0})
    min_bound = neg_clip_cfg.get("floor_ugm3", -5.0)
    max_bound = hard_range.get("max", 1500.0)

    mask_range = (out["pm25_ugm3"] < min_bound) | (out["pm25_ugm3"] > max_bound)
    out.loc[mask_range, "qc_flags"] |= RANGE_REJECT
    out.loc[mask_range, "pm25_ugm3"] = np.nan

    # -------------------------------------------------------------------------
    # 2. NEGATIVE_CLIPPED (Bit 2): value in [-5, 0)
    # -------------------------------------------------------------------------
    mask_neg = (out["pm25_ugm3"] >= min_bound) & (out["pm25_ugm3"] < 0)
    out.loc[mask_neg, "qc_flags"] |= NEGATIVE_CLIPPED
    out.loc[mask_neg, "pm25_ugm3"] = 0.0

    # -------------------------------------------------------------------------
    # 3. LOW_COVERAGE (Bit 64): < 75% sub-hourly readings present
    # -------------------------------------------------------------------------
    coverage_cfg = cfg.get("coverage", {"min_fraction_subhourly": 0.75})
    min_cov_frac = coverage_cfg.get("min_fraction_subhourly", 0.75)
    for cov_col in ["coverage_fraction", "subhourly_fraction", "reading_coverage"]:
        if cov_col in out.columns:
            mask_cov = (out[cov_col] < min_cov_frac) & out[cov_col].notna()
            out.loc[mask_cov, "qc_flags"] |= LOW_COVERAGE
            out.loc[mask_cov, "pm25_ugm3"] = np.nan
            break

    # -------------------------------------------------------------------------
    # Temporal Checks (Per Station): FLATLINE and SPIKE
    # -------------------------------------------------------------------------
    if group_col and "ts_utc" in out.columns:
        out = out.sort_values([group_col, "ts_utc"])

        # 4. FLATLINE (Bit 4): >= 6 identical consecutive hourly values > 0
        flatline_cfg = cfg.get(
            "flatline", {"min_consecutive_hours": 6, "ignore_value_leq_ugm3": 0.0}
        )
        min_flat_h = flatline_cfg.get("min_consecutive_hours", 6)
        ignore_leq = flatline_cfg.get("ignore_value_leq_ugm3", 0.0)

        flatline_indices: list[Any] = []

        def mark_flatline(group: pd.DataFrame) -> None:
            val = group["pm25_ugm3"]
            ts = group["ts_utc"]

            valid_val = val > ignore_leq
            same_val = val == val.shift(1)
            consecutive = ts.diff() == pd.Timedelta(hours=1)

            new_run = (~same_val) | (~consecutive) | (~valid_val)
            run_id = new_run.cumsum()

            run_counts = run_id[valid_val].value_counts()
            flatline_runs = run_counts[run_counts >= min_flat_h].index

            is_flat = valid_val & run_id.isin(flatline_runs)
            flatline_indices.extend(group.index[is_flat])

        for _, group in out.groupby(group_col):
            mark_flatline(group)

        out.loc[flatline_indices, "qc_flags"] |= FLATLINE
        out.loc[flatline_indices, "pm25_ugm3"] = np.nan

        # 5. SPIKE (Bit 8): exceeds centred 7-h rolling median by max(8*MAD, 100 ug/m3)
        spike_cfg = cfg.get("spike", {"window_hours": 7, "k_mad": 8.0, "floor_ugm3": 100.0})
        spike_w = spike_cfg.get("window_hours", 7)
        k_mad_spike = spike_cfg.get("k_mad", 8.0)
        floor_spike = spike_cfg.get("floor_ugm3", 100.0)

        spike_indices: list[Any] = []

        def mark_spike(group: pd.DataFrame) -> None:
            tmp = group.set_index("ts_utc")
            tmp_val = tmp["pm25_ugm3"]

            if len(tmp) == 0:
                return

            full_idx = pd.date_range(tmp.index.min(), tmp.index.max(), freq="h")
            tmp_val_nodup = tmp_val[~tmp.index.duplicated(keep="first")]
            tmp_val_full = tmp_val_nodup.reindex(full_idx)

            c_med = tmp_val_full.rolling(window=spike_w, center=True, min_periods=spike_w).median()
            c_mad = tmp_val_full.rolling(window=spike_w, center=True, min_periods=spike_w).apply(
                _calc_mad, raw=True
            )

            t_med = tmp_val_full.rolling(window=spike_w, min_periods=max(2, spike_w // 2)).median()
            t_mad = tmp_val_full.rolling(window=spike_w, min_periods=max(2, spike_w // 2)).apply(
                _calc_mad, raw=True
            )

            # backward rolling for the start edges
            b_med = (
                tmp_val_full.iloc[::-1]
                .rolling(window=spike_w, min_periods=max(2, spike_w // 2))
                .median()
                .iloc[::-1]
            )
            b_mad = (
                tmp_val_full.iloc[::-1]
                .rolling(window=spike_w, min_periods=max(2, spike_w // 2))
                .apply(_calc_mad, raw=True)
                .iloc[::-1]
            )

            rolling_med = c_med.combine_first(t_med).combine_first(b_med)
            rolling_mad = c_mad.combine_first(t_mad).combine_first(b_mad)

            med_aligned = rolling_med.reindex(tmp.index)
            mad_aligned = rolling_mad.reindex(tmp.index)

            threshold = np.maximum(k_mad_spike * mad_aligned, floor_spike)
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

    # -------------------------------------------------------------------------
    # Spatial Checks: SPATIAL_OUTLIER (Bit 16) and DRIFT_SUSPECT (Bit 256)
    # -------------------------------------------------------------------------
    out = _apply_spatial_and_drift_qc(out, cfg, group_col, stations_df)

    # -------------------------------------------------------------------------
    # 6. HIGH_RH (Bit 32): RH > 90% (flag only, retains PM2.5)
    # -------------------------------------------------------------------------
    humidity_cfg = cfg.get("humidity", {"flag_above_pct": 90.0})
    rh_thresh = humidity_cfg.get("flag_above_pct", 90.0)
    if "rh_pct" in out.columns:
        mask_rh = (out["rh_pct"] > rh_thresh) & out["rh_pct"].notna()
        out.loc[mask_rh, "qc_flags"] |= HIGH_RH

    # -------------------------------------------------------------------------
    # 7. HIGH_RANGE (Bit 128): > 500 ug/m3 (flag only, retains PM2.5)
    # -------------------------------------------------------------------------
    high_range_thresh = cfg.get("high_range_flag_ugm3", 500.0)
    mask_high = (out["pm25_ugm3"] > high_range_thresh) & out["pm25_ugm3"].notna()
    out.loc[mask_high, "qc_flags"] |= HIGH_RANGE

    return out


def _apply_spatial_and_drift_qc(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    group_col: str | None,
    stations_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Evaluate spatial outlier and drift suspect checks when coordinates are present."""
    if not group_col or "ts_utc" not in df.columns:
        return df

    # Resolve coordinates
    coords_df = df[[group_col]].copy()
    has_coords = False

    if "lat" in df.columns and "lon" in df.columns:
        coords_df["lat"] = df["lat"]
        coords_df["lon"] = df["lon"]
        has_coords = True
    elif stations_df is not None and "lat" in stations_df.columns and "lon" in stations_df.columns:
        stat_id_col = (
            group_col
            if group_col in stations_df.columns
            else (
                "location_id"
                if "location_id" in stations_df.columns
                else ("sensor_id" if "sensor_id" in stations_df.columns else None)
            )
        )
        if stat_id_col:
            meta = stations_df[[stat_id_col, "lat", "lon"]].drop_duplicates(subset=[stat_id_col])
            meta_map = meta.set_index(stat_id_col)
            coords_df["lat"] = coords_df[group_col].map(meta_map["lat"])
            coords_df["lon"] = coords_df[group_col].map(meta_map["lon"])
            if coords_df["lat"].notna().any() and coords_df["lon"].notna().any():
                has_coords = True

    if not has_coords:
        return df

    # Unique stations with valid coordinates
    unique_stats = coords_df[[group_col, "lat", "lon"]].dropna().drop_duplicates(subset=[group_col])
    if len(unique_stats) < 2:
        return df

    # Compute pairwise geodesic distances in km
    geod = pyproj.Geod(ellps="WGS84")
    stats_list = unique_stats[group_col].tolist()
    lats = unique_stats["lat"].to_numpy()
    lons = unique_stats["lon"].to_numpy()
    n_stats = len(stats_list)

    spatial_cfg = cfg.get(
        "spatial_outlier",
        {"radius_km": 15.0, "min_neighbors": 3, "k_mad": 6.0, "floor_ugm3": 80.0},
    )
    radius_km = spatial_cfg.get("radius_km", 15.0)
    min_neighbors = spatial_cfg.get("min_neighbors", 3)
    k_mad_spatial = spatial_cfg.get("k_mad", 6.0)
    floor_spatial = spatial_cfg.get("floor_ugm3", 80.0)

    # Build neighbor graph within radius_km
    neighbors_dict: dict[Any, list[Any]] = {s: [] for s in stats_list}
    for i in range(n_stats):
        for j in range(i + 1, n_stats):
            _, _, dist_m = geod.inv(lons[i], lats[i], lons[j], lats[j])
            dist_km = dist_m / 1000.0
            if dist_km <= radius_km:
                neighbors_dict[stats_list[i]].append(stats_list[j])
                neighbors_dict[stats_list[j]].append(stats_list[i])

    # -------------------------------------------------------------------------
    # Spatial Outlier Check (Bit 16)
    # -------------------------------------------------------------------------
    spatial_outlier_indices: list[Any] = []

    # Pivot valid observations by timestamp for fast neighbor lookup
    valid_obs = df[df["pm25_ugm3"].notna() & df[group_col].isin(stats_list)]
    if not valid_obs.empty:
        obs_pivot = valid_obs.pivot_table(
            index="ts_utc", columns=group_col, values="pm25_ugm3", aggfunc="first"
        )

        for ts, row in obs_pivot.iterrows():
            valid_stations_at_t = row.dropna()
            for station_id, val in valid_stations_at_t.items():
                nbrs = [nb for nb in neighbors_dict[station_id] if nb in valid_stations_at_t.index]
                if len(nbrs) >= min_neighbors:
                    nbr_vals = valid_stations_at_t.loc[nbrs].to_numpy(dtype=float)
                    nbr_median = float(np.median(nbr_vals))
                    nbr_mad = float(np.median(np.abs(nbr_vals - nbr_median)))
                    threshold = max(k_mad_spatial * nbr_mad, floor_spatial)
                    if val > nbr_median + threshold:
                        idx = df[(df["ts_utc"] == ts) & (df[group_col] == station_id)].index
                        spatial_outlier_indices.extend(idx)

    if spatial_outlier_indices:
        df.loc[spatial_outlier_indices, "qc_flags"] |= SPATIAL_OUTLIER
        df.loc[spatial_outlier_indices, "pm25_ugm3"] = np.nan

    # -------------------------------------------------------------------------
    # Drift Suspect Check (Bit 256): slope outside [0.5, 2.0] vs neighbours
    # -------------------------------------------------------------------------
    drift_cfg = cfg.get(
        "drift",
        {
            "window_days": 7,
            "min_overlap_hours": 72,
            "slope_bounds": [0.5, 2.0],
            "reference": "neighbours_median",
        },
    )
    min_overlap = drift_cfg.get("min_overlap_hours", 72)
    slope_bounds = drift_cfg.get("slope_bounds", [0.5, 2.0])

    # Re-pivot after spatial outlier nulling
    valid_obs_after = df[df["pm25_ugm3"].notna() & df[group_col].isin(stats_list)]
    if not valid_obs_after.empty:
        pivot_after = valid_obs_after.pivot_table(
            index="ts_utc", columns=group_col, values="pm25_ugm3", aggfunc="first"
        )

        drift_suspect_indices: list[Any] = []
        for station_id in stats_list:
            nbrs = neighbors_dict[station_id]
            if len(nbrs) < min_neighbors or station_id not in pivot_after.columns:
                continue

            station_series = pivot_after[station_id].dropna()
            # Calculate neighbours' median for each hour with >= min_neighbors available
            nbrs_in_pivot = [nb for nb in nbrs if nb in pivot_after.columns]
            if not nbrs_in_pivot:
                continue

            nbrs_data = pivot_after[nbrs_in_pivot]
            valid_nbr_counts = nbrs_data.notna().sum(axis=1)
            eligible_hours = valid_nbr_counts[valid_nbr_counts >= min_neighbors].index

            nbr_median = nbrs_data.loc[eligible_hours].median(axis=1)

            # Find overlapping valid hours
            common_hours = station_series.index.intersection(eligible_hours)
            if len(common_hours) >= min_overlap:
                x = nbr_median.loc[common_hours].to_numpy(dtype=float)
                y = station_series.loc[common_hours].to_numpy(dtype=float)

                var_x = float(np.var(x))
                if var_x > 1e-4:
                    cov_xy = float(np.cov(x, y)[0, 1])
                    slope = cov_xy / var_x
                    if slope < slope_bounds[0] or slope > slope_bounds[1]:
                        match_idx = df[
                            (df[group_col] == station_id) & (df["ts_utc"].isin(common_hours))
                        ].index
                        drift_suspect_indices.extend(match_idx)

        if drift_suspect_indices:
            df.loc[drift_suspect_indices, "qc_flags"] |= DRIFT_SUSPECT

    return df


def expand_to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Expand observations dataframe to a contiguous hourly temporal grid."""
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
        loc_val = g[group_col].iloc[0] if group_col in g.columns else g.name

        g = g.set_index("ts_utc").sort_index()
        g = g[~g.index.duplicated(keep="first")]
        if len(g) == 0:
            return g
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_idx)

        # Forward fill the group identifier
        g[group_col] = loc_val

        # Fill qc_flags with 0 for new rows
        if "qc_flags" in g.columns:
            g["qc_flags"] = g["qc_flags"].fillna(0).astype("Int64")
        g.index.name = "ts_utc"
        return g.reset_index()

    return df.groupby(group_col, group_keys=False).apply(_expand_group).reset_index(drop=True)


def clean_observations(
    df: pd.DataFrame,
    rules: dict[str, Any] | None = None,
    stations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Unified cleaning pipeline:

    1. Expands dataframe to regular hourly grid.
    2. Applies full rule-based QC checks.
    3. Imputes short internal gaps (<= 3 hours) using log1p PCHIP interpolation.
    4. Accumulates IMPUTED bit (512) into qc_flags.
    """
    df_expanded = expand_to_hourly_grid(df)
    out = apply_qc(df_expanded, rules=rules, stations_df=stations_df)

    if "pm25_ugm3" not in out.columns:
        return out

    group_col = (
        "sensor_id"
        if "sensor_id" in out.columns
        else ("location_id" if "location_id" in out.columns else None)
    )

    if group_col:
        out = out.sort_values([group_col, "ts_utc"])

        imputed_series_list = []
        mask_series_list = []

        for _, group in out.groupby(group_col):
            imp, mask = impute_short_gaps(group["pm25_ugm3"], max_gap_h=3)
            imputed_series_list.append(imp)
            mask_series_list.append(mask)

        new_pm25 = pd.concat(imputed_series_list)
        new_mask = pd.concat(mask_series_list)

        out["pm25_ugm3"] = new_pm25
        out.loc[new_mask, "qc_flags"] |= IMPUTED

        out = out.sort_index()
    else:
        imp, mask = impute_short_gaps(out["pm25_ugm3"], max_gap_h=3)
        out["pm25_ugm3"] = imp
        out.loc[mask, "qc_flags"] |= IMPUTED

    return out
