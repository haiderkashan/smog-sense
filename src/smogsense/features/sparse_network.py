"""smogsense.features.sparse_network — Sparse-network simulation and neighbour-history signal.

Samples available-station subsets of size N from a dense network, builds the inverse-distance-
weighted history signal from available stations, and applies modality dropout so the model
learns to forecast with N = 0, 1, 2, 5, ... observed stations.

Specification:
- docs/ml-architecture.md §7 ('Sparse-network task construction') & §8 ('Meta-learning and adaptation')
- docs/data-engineering.md §11 & §13 ('Temporal alignment' & 'Station audit criteria')
- configs/features.yaml ('sparse_network')
- decisions.md (ADR-015: Sparse-network tasks and sensor-free mode define N)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pyproj

DEFAULT_MAX_COLOCATION_DIST_M = 50.0
DEFAULT_IDW_POWER = 2.0
DEFAULT_IDW_MAX_RADIUS_KM = 25.0
DEFAULT_AVAILABLE_N_SAMPLER: tuple[int, ...] = (0, 1, 2, 3, 5, 8)


def _get_station_id_col(df: pd.DataFrame) -> str:
    for col in ("location_id", "station_id", "id", "sensor_id"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not find station ID column in {list(df.columns)}")


def _get_coord_cols(df: pd.DataFrame) -> tuple[str, str]:
    lat_col = next((c for c in ("latitude", "lat") if c in df.columns), None)
    lon_col = next((c for c in ("longitude", "lon") if c in df.columns), None)
    if not lat_col or not lon_col:
        raise ValueError(f"Could not find coordinate columns (lat, lon) in {list(df.columns)}")
    return lat_col, lon_col


def cluster_stations(
    stations_df: pd.DataFrame,
    max_colocation_dist_m: float = DEFAULT_MAX_COLOCATION_DIST_M,
) -> dict[int, list[int]]:
    """Group stations into atomic clusters based on colocated_group_id or 50m WGS84 geodesic distance.

    Stations within `max_colocation_dist_m` (default 50.0 m) form an indivisible cluster.
    Singletons form 1-station clusters.

    Returns:
    dict[int, list[int]]: Mapping of cluster_id -> list of station IDs.
    """
    if stations_df.empty:
        return {}

    id_col = _get_station_id_col(stations_df)
    lat_col, lon_col = _get_coord_cols(stations_df)

    df = stations_df.copy().reset_index(drop=True)
    n = len(df)
    station_ids = [int(x) for x in df[id_col]]

    # Disjoint Set Union (Union-Find)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # 1. Cluster by explicit colocated_group_id if present
    group_col = next((c for c in ("colocated_group_id", "colocated_group", "group_id") if c in df.columns), None)
    if group_col is not None:
        seen_groups: dict[Any, int] = {}
        for i, val in enumerate(df[group_col]):
            if pd.notna(val) and val != "":
                if val in seen_groups:
                    union(i, seen_groups[val])
                else:
                    seen_groups[val] = i

    # 2. Geodesic distance clustering for stations within max_colocation_dist_m
    lats = df[lat_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)
    geod = pyproj.Geod(ellps="WGS84")

    for i in range(n):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            continue
        for j in range(i + 1, n):
            if np.isnan(lats[j]) or np.isnan(lons[j]):
                continue
            _, _, dist_m = geod.inv(lons[i], lats[i], lons[j], lats[j])
            if float(abs(dist_m)) <= max_colocation_dist_m:
                union(i, j)

    # Collect grouped station IDs
    clusters_map: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters_map.setdefault(root, []).append(station_ids[i])

    # Re-key clusters by integer index 0, 1, 2, ...
    ordered_clusters = sorted(clusters_map.values(), key=lambda c: (min(c), len(c)))
    return {cluster_idx: sorted(members) for cluster_idx, members in enumerate(ordered_clusters)}


def compute_pairwise_geodesic_distances(
    stations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all-pairs WGS84 geodesic distances in kilometers between stations.

    Returns:
    pd.DataFrame: Symmetric distance matrix in kilometers with station IDs as rows and columns.
    """
    if stations_df.empty:
        return pd.DataFrame()

    id_col = _get_station_id_col(stations_df)
    lat_col, lon_col = _get_coord_cols(stations_df)

    df = stations_df.copy().reset_index(drop=True)
    station_ids = [int(x) for x in df[id_col]]
    n = len(df)

    lats = df[lat_col].to_numpy(dtype=np.float64)
    lons = df[lon_col].to_numpy(dtype=np.float64)

    geod = pyproj.Geod(ellps="WGS84")
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            dist_matrix[i, :] = np.nan
            dist_matrix[:, i] = np.nan
            continue
        for j in range(i + 1, n):
            if np.isnan(lats[j]) or np.isnan(lons[j]):
                dist_matrix[i, j] = np.nan
                dist_matrix[j, i] = np.nan
            else:
                _, _, dist_m = geod.inv(lons[i], lats[i], lons[j], lats[j])
                dist_km = float(abs(dist_m)) / 1000.0
                dist_matrix[i, j] = dist_km
                dist_matrix[j, i] = dist_km

    return pd.DataFrame(dist_matrix, index=station_ids, columns=station_ids)


def sample_available_stations(
    stations_df: pd.DataFrame,
    n: int | None = None,
    available_n_sampler: Sequence[int] = DEFAULT_AVAILABLE_N_SAMPLER,
    rng: np.random.Generator | int | None = None,
) -> tuple[set[int], set[int], int]:
    """Twin-aware available-set sampler returning (available_ids, held_out_ids, sampled_n).

    Clusters co-located stations (d <= 50m) and guarantees that co-located pairs/groups
    are NEVER split between the available set A and held-out set B.

    Parameters:
    - stations_df: DataFrame of stations with coordinates.
    - n: Optional target number of available stations. If None, sampled from available_n_sampler.
    - available_n_sampler: Sequence of target N values to draw from.
    - rng: Optional numpy Generator or integer seed for reproducibility.

    Returns:
    tuple[set[int], set[int], int]: (A, B, sampled_n) where A + B = all stations.
    """
    if stations_df.empty:
        return set(), set(), 0

    id_col = _get_station_id_col(stations_df)
    all_stations = {int(x) for x in stations_df[id_col].dropna().unique()}
    total_stations = len(all_stations)

    # Initialize random generator
    if isinstance(rng, np.random.Generator):
        gen = rng
    elif isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(int(rng))
    else:
        gen = np.random.default_rng()

    # Determine target N
    if n is not None:
        target_n = int(n)
    else:
        sampler_candidates = list(available_n_sampler)
        target_n = int(gen.choice(sampler_candidates))

    # Boundary conditions
    if target_n <= 0:
        return set(), set(all_stations), target_n
    if target_n >= total_stations:
        return set(all_stations), set(), target_n

    # Compute atomic clusters
    clusters_dict = cluster_stations(stations_df)
    clusters = list(clusters_dict.values())

    # Shuffle clusters randomly
    cluster_order = gen.permutation(len(clusters))
    available_set: set[int] = set()

    for idx in cluster_order:
        cluster = clusters[idx]
        if len(available_set) + len(cluster) <= target_n:
            available_set.update(cluster)

    # If greedy packing remained empty, select the single cluster closest in size
    if not available_set:
        best_cluster = min(clusters, key=lambda c: (abs(len(c) - target_n), len(c)))
        available_set.update(best_cluster)

    held_out_set = all_stations - available_set
    return available_set, held_out_set, target_n


def compute_idw_history(
    target_location_id: int,
    available_station_ids: set[int] | Sequence[int],
    obs_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    power: float = DEFAULT_IDW_POWER,
    max_radius_km: float = DEFAULT_IDW_MAX_RADIUS_KM,
) -> tuple[pd.Series, float, pd.Series]:
    """Compute IDW historical PM2.5 series, distance to nearest available station, and validity mask.

    Formula:
        H_A(s, t) = sum_{a in A} w_a(s) * x_{a, t} / sum_{a in A} w_a(s)
        w_a(s) = 1 / d(s, a)^power for d(s, a) <= max_radius_km, else 0.

    Singularity treatment:
        If d(s, a) <= 10 m (0.010 km) or s == a:
            w_a(s) = 1.0 and all other weights 0.

    Companion feature dist_nearest_available_km:
        - If A is empty: 25.0 km (truncation radius).
        - If A is non-empty: geodesic distance in km to the nearest station in A.

    Sensor-free mode (N = 0 or no station within max_radius_km):
        H_A(s, t) is 0.0, mask is False.

    Returns:
    tuple[pd.Series, float, pd.Series]: (idw_history_series, dist_nearest_km, valid_mask_series)
    """
    stn_id_col = _get_station_id_col(stations_df)
    lat_col, lon_col = _get_coord_cols(stations_df)

    # 1. Lookup target station coordinates
    target_match = stations_df[stations_df[stn_id_col] == target_location_id]
    if target_match.empty:
        raise ValueError(f"Target location_id {target_location_id} not found in stations_df")

    target_lat = float(target_match[lat_col].iloc[0])
    target_lon = float(target_match[lon_col].iloc[0])

    avail_set = {int(x) for x in available_station_ids}
    avail_stns = stations_df[stations_df[stn_id_col].isin(avail_set)].copy()

    # Determine unique timestamps from obs_df
    if obs_df.empty or "ts_utc" not in obs_df.columns:
        empty_series = pd.Series(dtype=np.float64)
        empty_mask = pd.Series(dtype=bool)
        dist_nearest = float(max_radius_km) if not avail_set else 0.0
        return empty_series, dist_nearest, empty_mask

    all_timestamps = pd.DatetimeIndex(sorted(pd.to_datetime(obs_df["ts_utc"].unique(), utc=True)))

    # Handle N = 0 (sensor-free mode)
    if avail_stns.empty or not avail_set:
        idw_zero = pd.Series(0.0, index=all_timestamps, dtype=np.float64, name="idw_history")
        mask_false = pd.Series(False, index=all_timestamps, dtype=bool, name="idw_mask")
        return idw_zero, float(max_radius_km), mask_false

    # Compute geodesic distances to available stations
    geod = pyproj.Geod(ellps="WGS84")
    distances_km: dict[int, float] = {}

    for _, row in avail_stns.iterrows():
        sid = int(row[stn_id_col])
        s_lat = float(row[lat_col])
        s_lon = float(row[lon_col])
        if np.isnan(s_lat) or np.isnan(s_lon):
            continue
        _, _, dist_m = geod.inv(target_lon, target_lat, s_lon, s_lat)
        distances_km[sid] = float(abs(dist_m)) / 1000.0

    if not distances_km:
        idw_zero = pd.Series(0.0, index=all_timestamps, dtype=np.float64, name="idw_history")
        mask_false = pd.Series(False, index=all_timestamps, dtype=bool, name="idw_mask")
        return idw_zero, float(max_radius_km), mask_false

    dist_nearest_km = min(distances_km.values())

    # Check for singularity: exact match or d <= 10 m (0.010 km)
    singular_stations = [sid for sid, d in distances_km.items() if d <= 0.010 or sid == target_location_id]
    weights: dict[int, float] = {}

    if singular_stations:
        # Co-located target: only the singular station(s) receive weight
        k = len(singular_stations)
        for sid in singular_stations:
            weights[sid] = 1.0 / k
    else:
        # IDW inverse power weighting up to max_radius_km
        for sid, d in distances_km.items():
            if d <= max_radius_km and d > 0.0:
                weights[sid] = 1.0 / (d ** float(power))

    # If no stations within max_radius_km
    if not weights:
        idw_zero = pd.Series(0.0, index=all_timestamps, dtype=np.float64, name="idw_history")
        mask_false = pd.Series(False, index=all_timestamps, dtype=bool, name="idw_mask")
        return idw_zero, float(dist_nearest_km), mask_false

    # Prepare observation slice
    obs_id_col = next((c for c in ("location_id", "station_id", "id", "sensor_id") if c in obs_df.columns), None)
    if not obs_id_col:
        raise ValueError(f"Could not find station ID column in obs_df: {list(obs_df.columns)}")

    pm25_col = next((c for c in ("pm25_ugm3", "pm25", "value") if c in obs_df.columns), None)
    if not pm25_col:
        raise ValueError(f"Could not find PM2.5 column in obs_df: {list(obs_df.columns)}")

    obs_sub = obs_df[obs_df[obs_id_col].isin(weights.keys())].copy()
    obs_sub["ts_utc"] = pd.to_datetime(obs_sub["ts_utc"], utc=True)

    # Invalidate negative and imputed readings
    pm_series = pd.to_numeric(obs_sub[pm25_col], errors="coerce")
    pm_series = pm_series.where(pm_series >= 0.0)
    if "imputed" in obs_sub.columns:
        pm_series = pm_series.where(~obs_sub["imputed"].astype(bool))
    obs_sub["clean_pm25"] = pm_series

    obs_dedup = obs_sub.dropna(subset=["clean_pm25"]).drop_duplicates(subset=["ts_utc", obs_id_col], keep="last")

    # Pivot to timestamp x station table
    if obs_dedup.empty:
        idw_zero = pd.Series(0.0, index=all_timestamps, dtype=np.float64, name="idw_history")
        mask_false = pd.Series(False, index=all_timestamps, dtype=bool, name="idw_mask")
        return idw_zero, float(dist_nearest_km), mask_false

    pivot = obs_dedup.pivot_table(
        index="ts_utc", columns=obs_id_col, values="clean_pm25", aggfunc="last"
    ).reindex(index=all_timestamps, columns=list(weights.keys()))

    w_vec = np.array([weights[int(sid)] for sid in pivot.columns], dtype=np.float64)

    # Valid mask for each station
    valid_mat = pivot.notna().to_numpy()
    vals_mat = np.nan_to_num(pivot.to_numpy(), nan=0.0)

    # Weighted numerator and denominator
    numer = np.sum(vals_mat * w_vec, axis=1)
    denom = np.sum(valid_mat * w_vec, axis=1)

    has_data = denom > 0.0
    idw_values = np.zeros(len(all_timestamps), dtype=np.float64)
    idw_values[has_data] = numer[has_data] / denom[has_data]

    idw_series = pd.Series(idw_values, index=all_timestamps, dtype=np.float64, name="idw_history")
    mask_series = pd.Series(has_data, index=all_timestamps, dtype=bool, name="idw_mask")

    return idw_series, float(dist_nearest_km), mask_series

