"""smogsense.data_ingestion.station_registry -- Station registry and co-location logic.

Loads and saves registry Parquet files, applying the 50m distance co-location rule
and computing uptime/history eligibility.

Specification: docs/data-engineering.md -> 'Station registry and eligibility'
"""

import hashlib
from pathlib import Path

import pandas as pd
import pyproj
from structlog import get_logger

logger = get_logger(__name__)


class StationRegistry:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _compute_eligibility(self, df: pd.DataFrame, min_uptime: float) -> pd.Series:
        # history_days measures calendar span (first observation to last observation)
        history_days = (df["last_datetime"] - df["first_datetime"]).dt.total_seconds() / 86400

        # P1-02 semantic decision: lifecycle_uptime cannot satisfy the 90-day uptime requirement.
        # Until pm25_uptime_90d is computed by the historical backfill layer (P1-07/P1-13),
        # no station passes the uptime eligibility gate.
        if "pm25_uptime_90d" in df.columns:
            uptime_check = df["pm25_uptime_90d"] >= min_uptime
        else:
            uptime_check = pd.Series(False, index=df.index)

        return (history_days >= 30) & uptime_check

    def _compute_colocation(self, df: pd.DataFrame) -> pd.Series:
        """Assign co-located group ID.

        Stations <= 50m with different providers get the same ID.
        Deterministic: the group ID is the minimum location_id in the group.
        Isolated stations receive pd.NA.
        """
        geod = pyproj.Geod(ellps="WGS84")
        n = len(df)

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

        # Grouping
        for i in range(n):
            row_i = df.iloc[i]
            for j in range(i + 1, n):
                row_j = df.iloc[j]
                if row_i["provider"] != row_j["provider"]:
                    _, _, dist = geod.inv(row_i["lon"], row_i["lat"], row_j["lon"], row_j["lat"])
                    if dist <= 50.0:
                        union(i, j)

        # Map root to minimum location_id in that root's group for determinism
        root_to_min_loc = {}
        for i in range(n):
            root = find(i)
            loc_id = df.iloc[i]["location_id"]
            if root not in root_to_min_loc:
                root_to_min_loc[root] = loc_id
            else:
                root_to_min_loc[root] = min(root_to_min_loc[root], loc_id)

        # Compute sizes to isolate singletons
        root_sizes: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            root_sizes[root] = root_sizes.get(root, 0) + 1

        group_ids = pd.Series(pd.NA, index=df.index, dtype="Int64")

        for i in range(n):
            root = find(i)
            if root_sizes[root] > 1:
                group_ids.iloc[i] = root_to_min_loc[root]

        return group_ids

    def _compute_version(self, df: pd.DataFrame) -> str:
        """Create a deterministic registry version based on sorted location IDs and their datetimes."""
        if df.empty:
            return "empty"
        sorted_df = df.sort_values("location_id")
        hash_input = "".join(f"{row['location_id']}:{row['last_datetime']}" for _, row in sorted_df.iterrows())
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]

    def build_and_save(self, locations_df: pd.DataFrame, domain: str, min_uptime: float) -> pd.DataFrame:
        if locations_df.empty:
            df = locations_df.copy()
            df["eligible"] = pd.Series(dtype=bool)
            df["colocated_group_id"] = pd.Series(dtype="Int64")
            df["registry_version"] = pd.Series(dtype=str)
        else:
            df = locations_df.copy()
            df["eligible"] = self._compute_eligibility(df, min_uptime)
            df["colocated_group_id"] = self._compute_colocation(df)
            version = self._compute_version(df)
            df["registry_version"] = version

        path = self.data_dir / f"registry_{domain}.parquet"
        df.to_parquet(path, index=False)
        logger.info("registry_saved", domain=domain, path=str(path), count=len(df))
        return df

    def load(self, domain: str) -> pd.DataFrame:
        path = self.data_dir / f"registry_{domain}.parquet"
        return pd.read_parquet(path)
