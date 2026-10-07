"""smogsense.data_ingestion.station_registry -- Station registry and co-location logic.

Loads and saves registry Parquet files, applying the 50m distance co-location rule
and computing uptime/history eligibility.

Specification: docs/data-engineering.md -> 'Station registry and eligibility'
"""

from datetime import datetime, UTC
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
        history_days = (df["last_datetime"] - df["first_datetime"]).dt.total_seconds() / 86400
        return (history_days >= 30) & (df["pm25_uptime_90d"] >= min_uptime)
        
    def _compute_colocation(self, df: pd.DataFrame) -> pd.Series:
        """Assign co-located group ID. Stations <= 50m with different providers get the same ID."""
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
                
        for i in range(n):
            row_i = df.iloc[i]
            for j in range(i + 1, n):
                row_j = df.iloc[j]
                if row_i["provider"] != row_j["provider"]:
                    _, _, dist = geod.inv(row_i["lon"], row_i["lat"], row_j["lon"], row_j["lat"])
                    if dist <= 50.0:
                        union(i, j)
                        
        group_ids = pd.Series(-1, index=df.index, dtype=int)
        next_group_id = 1
        
        # Map root to group ID
        root_to_id: dict[int, int] = {}
        for i in range(n):
            root = find(i)
            if root not in root_to_id:
                root_to_id[root] = next_group_id
                next_group_id += 1
            group_ids.iloc[i] = root_to_id[root]
            
        return group_ids

    def build_and_save(self, locations_df: pd.DataFrame, domain: str, min_uptime: float) -> pd.DataFrame:
        if locations_df.empty:
            df = locations_df.copy()
            df["eligible"] = pd.Series(dtype=bool)
            df["colocated_group_id"] = pd.Series(dtype=int)
            df["registry_version"] = pd.Series(dtype=str)
        else:
            df = locations_df.copy()
            df["eligible"] = self._compute_eligibility(df, min_uptime)
            df["colocated_group_id"] = self._compute_colocation(df)
            df["registry_version"] = datetime.now(UTC).isoformat()
        
        path = self.data_dir / f"registry_{domain}.parquet"
        df.to_parquet(path, index=False)
        logger.info("registry_saved", domain=domain, path=str(path), count=len(df))
        return df

    def load(self, domain: str) -> pd.DataFrame:
        path = self.data_dir / f"registry_{domain}.parquet"
        return pd.read_parquet(path)
