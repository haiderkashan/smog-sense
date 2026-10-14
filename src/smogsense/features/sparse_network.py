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
