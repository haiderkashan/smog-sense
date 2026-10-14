"""smogsense.features.targets — 24-hour block-mean targets with completeness rule.

y_h is the mean PM2.5 over the 24 hours [T0+h-24h, T0+h) (hour-start labels) for h in {24,48,72}; a block is valid only with at
least 18 of 24 valid, non-imputed hourly values (75% completeness).

Public contract (implemented in Phase 1b / Phase 2):
- block_targets(obs, issuance) -> DataFrame with y_h and valid_h
- compute_city_panel_target(obs_df, registry_df, ...) -> CityPanelTargetResult

Specification: docs/ml-architecture.md → 'Problem formulation'
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pyproj


@dataclass
class CityPanelTargetResult:
    """Result of city-panel target aggregation over a 24-hour block.

    Supports both attribute access (`result.y_city`, `result.valid`) and
    dictionary-style subscripting (`result["y_city"]`, `result["valid"]`).
    """

    y_city: float | None
    valid: bool
    n_valid_stations: int
    n_raw_valid_stations: int = 0
    n_reference: int = 0
    n_low_cost: int = 0
    truth_basis: str = "insufficient"
    group_means: dict[Any, float] = field(default_factory=dict)
    station_means: dict[Any, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "y_city": self.y_city,
            "valid": self.valid,
            "n_valid_stations": self.n_valid_stations,
            "n_raw_valid_stations": self.n_raw_valid_stations,
            "n_reference": self.n_reference,
            "n_low_cost": self.n_low_cost,
            "truth_basis": self.truth_basis,
            "group_means": dict(self.group_means),
            "station_means": dict(self.station_means),
        }


def _assign_colocation_groups(
    registry_df: pd.DataFrame,
    radius_m: float = 50.0,
) -> dict[Any, Any]:
    """Map each location_id in registry_df to a deterministic co-location group ID.

    Stations within `radius_m` meters of each other are merged via union-find;
    the group ID is the minimum location_id in the connected component.
    """
    if registry_df.empty or "location_id" not in registry_df.columns:
        return {}

    df = registry_df.drop_duplicates(subset=["location_id"]).reset_index(drop=True)
    loc_ids = df["location_id"].tolist()
    n = len(loc_ids)

    # If explicit group column is already populated, respect it where non-null
    for col in ("colocated_group_id", "group_id"):
        if col in df.columns and df[col].notna().any():
            mapping: dict[Any, Any] = {}
            for _, row in df.iterrows():
                loc = row["location_id"]
                grp = row[col]
                mapping[loc] = grp if pd.notna(grp) else loc
            return mapping

    lat_col = next((c for c in ("lat", "latitude") if c in df.columns), None)
    lon_col = next((c for c in ("lon", "longitude") if c in df.columns), None)
    if lat_col is None or lon_col is None:
        return {loc: loc for loc in loc_ids}

    geod = pyproj.Geod(ellps="WGS84")
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    lats = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)

    for i in range(n):
        if np.isnan(lats[i]) or np.isnan(lons[i]):
            continue
        for j in range(i + 1, n):
            if np.isnan(lats[j]) or np.isnan(lons[j]):
                continue
            _, _, dist_m = geod.inv(lons[i], lats[i], lons[j], lats[j])
            if abs(float(dist_m)) <= radius_m:
                union(i, j)

    root_to_min_id: dict[int, Any] = {}
    for i in range(n):
        r = find(i)
        lid = loc_ids[i]
        if r not in root_to_min_id or str(lid) < str(root_to_min_id[r]):
            try:
                if (
                    r in root_to_min_id and float(lid) < float(root_to_min_id[r])
                ) or r not in root_to_min_id:
                    root_to_min_id[r] = lid
            except (TypeError, ValueError):
                root_to_min_id[r] = lid

    return {loc_ids[i]: root_to_min_id[find(i)] for i in range(n)}


def compute_city_panel_target(
    obs_df: pd.DataFrame,
    registry_df: pd.DataFrame | None = None,
    panel_station_ids: Iterable[Any] | None = None,
    min_valid_stations: int = 3,
    min_valid_hours: int = 18,
    colocation_radius_m: float = 50.0,
    window_start_utc: datetime | pd.Timestamp | None = None,
    window_end_utc: datetime | pd.Timestamp | None = None,
) -> CityPanelTargetResult:
    """Compute the city pseudo-location 24-hour block mean target over a station panel.

    Rules (docs/ml-architecture.md §1, docs/evaluation-strategy.md §2):
    1. Filter to frozen panel stations (if `panel_station_ids` is given, or eligible
       stations in `registry_df`).
    2. For each station, require >= `min_valid_hours` (default 18 of 24) valid,
       non-imputed hourly PM2.5 observations. Imputed hours never count toward targets.
    3. Collapse co-located station groups (<= `colocation_radius_m`, default 50m) using
       the co-location hierarchy:
       - If the group has >= 1 valid reference monitor (`is_reference=True` or `is_monitor=True`),
         use the mean of valid reference monitors in that group.
       - Otherwise, use the median of valid low-cost sensors in that group.
    4. Require >= `min_valid_stations` (default 3) valid collapsed groups to emit a valid
       city target (`y_city = mean(group_means)`). Otherwise `valid=False` and `y_city=None`.
    """
    if obs_df is None or obs_df.empty or "pm25_ugm3" not in obs_df.columns:
        return CityPanelTargetResult(
            y_city=None,
            valid=False,
            n_valid_stations=0,
        )

    df = obs_df.copy()
    if "location_id" not in df.columns:
        if "sensor_id" in df.columns:
            df["location_id"] = df["sensor_id"]
        else:
            return CityPanelTargetResult(
                y_city=None,
                valid=False,
                n_valid_stations=0,
            )

    # Time window filter if specified
    if (window_start_utc is not None or window_end_utc is not None) and "ts_utc" in df.columns:
        ts = pd.to_datetime(df["ts_utc"], utc=True)
        if window_start_utc is not None:
            w_start = pd.to_datetime(window_start_utc, utc=True)
            df = df[ts >= w_start]
            ts = pd.to_datetime(df["ts_utc"], utc=True)
        if window_end_utc is not None:
            w_end = pd.to_datetime(window_end_utc, utc=True)
            df = df[ts < w_end]

    # Determine panel stations
    allowed_ids: set[Any] | None = None
    if panel_station_ids is not None:
        allowed_ids = set(panel_station_ids)
    elif registry_df is not None and not registry_df.empty and "location_id" in registry_df.columns:
        if "eligible" in registry_df.columns:
            elig_ids = set(registry_df.loc[registry_df["eligible"].astype(bool), "location_id"])
            allowed_ids = elig_ids
        else:
            allowed_ids = set(registry_df["location_id"])

    if allowed_ids is not None:
        df = df[df["location_id"].isin(allowed_ids)]

    if df.empty:
        return CityPanelTargetResult(
            y_city=None,
            valid=False,
            n_valid_stations=0,
        )

    # Exclude imputed and invalid observations
    valid_mask = df["pm25_ugm3"].notna() & (pd.to_numeric(df["pm25_ugm3"], errors="coerce") >= 0.0)
    if "imputed" in df.columns:
        valid_mask = valid_mask & (~df["imputed"].fillna(False).astype(bool))

    df_valid = df[valid_mask].copy()
    if df_valid.empty:
        return CityPanelTargetResult(
            y_city=None,
            valid=False,
            n_valid_stations=0,
        )

    # Station-level completeness and 24h block mean
    station_means: dict[Any, float] = {}
    for loc_id, grp in df_valid.groupby("location_id"):
        if "ts_utc" in grp.columns:
            n_hours = int(pd.to_datetime(grp["ts_utc"], utc=True).dt.floor("h").nunique())
        else:
            n_hours = len(grp)
        if n_hours >= min_valid_hours:
            station_means[loc_id] = float(pd.to_numeric(grp["pm25_ugm3"]).mean())

    if not station_means:
        return CityPanelTargetResult(
            y_city=None,
            valid=False,
            n_valid_stations=0,
        )

    # Determine reference status per station
    ref_map: dict[Any, bool] = {}
    if registry_df is not None and not registry_df.empty and "location_id" in registry_df.columns:
        for _, row in registry_df.iterrows():
            lid = row["location_id"]
            is_ref = bool(row.get("is_reference", row.get("is_monitor", False)))
            ref_map[lid] = is_ref

    for loc_id, grp in df_valid.groupby("location_id"):
        if loc_id not in ref_map:
            if "is_reference" in grp.columns:
                ref_map[loc_id] = bool(grp["is_reference"].fillna(False).any())
            elif "is_monitor" in grp.columns:
                ref_map[loc_id] = bool(grp["is_monitor"].fillna(False).any())
            else:
                ref_map[loc_id] = False

    # Assign co-location groups
    if registry_df is not None and not registry_df.empty:
        group_map = _assign_colocation_groups(registry_df, radius_m=colocation_radius_m)
    elif ("lat" in df_valid.columns or "latitude" in df_valid.columns) and (
        "lon" in df_valid.columns or "longitude" in df_valid.columns
    ):
        station_coords = df_valid.drop_duplicates(subset=["location_id"])
        group_map = _assign_colocation_groups(station_coords, radius_m=colocation_radius_m)
    else:
        group_map = {loc_id: loc_id for loc_id in station_means}

    # Collapse each co-located group using the hierarchy: reference first, else median of low-cost
    grouped_stations: dict[Any, list[Any]] = {}
    for loc_id in station_means:
        gid = group_map.get(loc_id, loc_id)
        grouped_stations.setdefault(gid, []).append(loc_id)

    group_means: dict[Any, float] = {}
    n_ref_groups = 0
    n_lowcost_groups = 0

    for gid, member_ids in grouped_stations.items():
        ref_members = [m for m in member_ids if ref_map.get(m, False)]
        if ref_members:
            group_means[gid] = float(np.mean([station_means[m] for m in ref_members]))
            n_ref_groups += 1
        else:
            group_means[gid] = float(np.median([station_means[m] for m in member_ids]))
            n_lowcost_groups += 1

    n_valid_groups = len(group_means)
    is_valid = n_valid_groups >= min_valid_stations
    y_city = float(np.mean(list(group_means.values()))) if is_valid else None

    if not is_valid:
        basis = "insufficient"
    elif n_ref_groups > 0 and n_lowcost_groups == 0:
        basis = "reference"
    elif n_lowcost_groups > 0 and n_ref_groups == 0:
        basis = "low_cost"
    else:
        basis = "mixed"

    return CityPanelTargetResult(
        y_city=y_city,
        valid=is_valid,
        n_valid_stations=n_valid_groups,
        n_raw_valid_stations=len(station_means),
        n_reference=n_ref_groups,
        n_low_cost=n_lowcost_groups,
        truth_basis=basis,
        group_means=group_means,
        station_means=station_means,
    )


def block_targets(
    obs_df: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp,
    horizons: Sequence[int] = (24, 48, 72),
    min_valid_hours: int = 18,
) -> pd.DataFrame:
    """Compute 24-hour block-mean targets y_h and validity flags valid_h per station.

    For each horizon h in {24, 48, 72}, the target block spans [T0 + h - 24h, T0 + h).
    A block is valid iff >= `min_valid_hours` valid, non-imputed hours are present.
    """
    t0 = pd.to_datetime(issuance_utc, utc=True)
    if obs_df is None or obs_df.empty or "location_id" not in obs_df.columns:
        cols = ["location_id", "issuance_utc"]
        for h in horizons:
            cols.extend([f"y_{h}", f"y_h{h}", f"valid_{h}", f"valid_h{h}"])
        return pd.DataFrame(columns=cols)

    df = obs_df.copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    valid_obs_mask = df["pm25_ugm3"].notna() & (
        pd.to_numeric(df["pm25_ugm3"], errors="coerce") >= 0.0
    )
    if "imputed" in df.columns:
        valid_obs_mask = valid_obs_mask & (~df["imputed"].fillna(False).astype(bool))
    df_valid = df[valid_obs_mask]

    stations = sorted(df["location_id"].dropna().unique(), key=str)
    records: list[dict[str, Any]] = []

    for loc_id in stations:
        st_df = df_valid[df_valid["location_id"] == loc_id]
        row: dict[str, Any] = {
            "location_id": loc_id,
            "issuance_utc": t0,
        }
        for h in horizons:
            w_start = t0 + timedelta(hours=int(h) - 24)
            w_end = t0 + timedelta(hours=int(h))
            blk = st_df[(st_df["ts_utc"] >= w_start) & (st_df["ts_utc"] < w_end)]
            n_hrs = int(blk["ts_utc"].dt.floor("h").nunique())
            if n_hrs >= min_valid_hours:
                mean_val = float(pd.to_numeric(blk["pm25_ugm3"]).mean())
                row[f"y_{h}"] = mean_val
                row[f"y_h{h}"] = mean_val
                row[f"valid_{h}"] = True
                row[f"valid_h{h}"] = True
            else:
                row[f"y_{h}"] = np.nan
                row[f"y_h{h}"] = np.nan
                row[f"valid_{h}"] = False
                row[f"valid_h{h}"] = False
        records.append(row)

    return pd.DataFrame(records)


def assert_no_target_leakage(
    feature_matrix: pd.DataFrame,
    issuance_utc: datetime | pd.Timestamp,
) -> None:
    """Assert that no target columns or future observation data leak into the feature matrix.

    Raises ValueError if target variables or records beyond issuance_utc are present.
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)

    # 1. Check for target column names in feature columns
    forbidden_prefixes = ("y_", "y_h", "target_", "label_")
    forbidden_found = [
        c for c in feature_matrix.columns if any(c.startswith(p) for p in forbidden_prefixes)
    ]
    if forbidden_found:
        raise ValueError(
            f"Target leakage detected! Found target columns in feature matrix: {forbidden_found}"
        )

    # 2. Check for timestamp leakage if timestamp column is present
    for ts_col in ("available_at_utc", "available_at", "ts_utc"):
        if ts_col in feature_matrix.columns:
            ts_series = pd.to_datetime(feature_matrix[ts_col], utc=True)
            future_mask = ts_series > t_iss
            if future_mask.any():
                n_future = int(future_mask.sum())
                max_ts = ts_series.max()
                raise ValueError(
                    f"Lookahead leakage detected! {n_future} records have {ts_col} > issuance {t_iss} (max={max_ts})"
                )
