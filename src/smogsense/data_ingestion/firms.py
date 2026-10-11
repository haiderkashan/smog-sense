"""smogsense.data_ingestion.firms — NASA FIRMS area-API client (active fire detections).

Queries /api/area/csv/<MAP_KEY>/<SOURCE>/<west,south,east,north>/<DAY_RANGE>/<DATE> in windows
of at most 10 days for VIIRS NOAA-21 and NOAA-20 (operational) plus legacy MODIS/Suomi-NPP
(historical only). CSV only; 5,000 transactions per 10 minutes per MAP_KEY.

Public contract (implemented in Phase 1):
- detections(source, bbox, start, end) -> DataFrame
- Platform availability table guards against requesting Suomi-NPP after 2026-11-01T13:00Z.

Specification: docs/data-engineering.md → 'NASA FIRMS fire ingestion'
"""

import asyncio
import io
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pyproj
from structlog import get_logger

from smogsense.data_ingestion.base import (
    CircuitBreaker,
    RateBudget,
    ResilientClient,
    create_client,
)
from smogsense.utils.io import validate_frame

logger = get_logger(__name__)

# Constants
DEFAULT_FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DEFAULT_AVAILABILITY_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv"

# FR-03: Suomi-NPP product delivery ends 2026-11-01 13:00 UTC
SUOMI_NPP_CUTOFF = datetime(2026, 11, 1, 13, 0, 0, tzinfo=UTC)

# Centroids from configs/domains.yaml: lat, lon
DOMAIN_CENTROIDS: dict[str, tuple[float, float]] = {
    "lahore": (31.5204, 74.3587),
    "delhi": (28.6139, 77.2090),
    "amritsar": (31.6340, 74.8723),
    "ludhiana": (30.9010, 75.8573),
}

# FIRMS query box: west, south, east, north
DEFAULT_FIRE_SEARCH_BOX: dict[str, float] = {
    "west": 71.0,
    "south": 27.0,
    "east": 79.5,
    "north": 34.0,
}

# Standard 8 compass sectors: 45° each, centered on cardinal/intercardinal bearings
SECTOR_NAMES: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _load_fire_exposure_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/fire_exposure.schema.yaml"),
        Path("/app/data/schemas/fire_exposure.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/fire_exposure.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(str(p))
    raise FileNotFoundError(f"fire_exposure.schema.yaml not found in candidates: {candidates}")


def _to_date(val: datetime | date | str) -> date:
    """Normalize datetime, date, or ISO string to a date object."""
    if isinstance(val, str):
        return pd.to_datetime(val, utc=True).date()
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    raise TypeError(f"Expected datetime, date, or str, got {type(val)}")


def split_date_range(
    start_date: datetime | date | str,
    end_date: datetime | date | str,
    max_days: int = 10,
    inclusive: bool = True,
) -> list[tuple[date, int]]:
    """Partition a multi-week interval into consecutive chunks of <= 10 days for FIRMS area API.

    Returns a list of (chunk_start_date, day_count) tuples where 1 <= day_count <= max_days.
    """
    if max_days < 1:
        raise ValueError(f"max_days must be >= 1, got {max_days}")
    clamped_max_days = min(max_days, 10)

    start_d = _to_date(start_date)
    end_d = _to_date(end_date)

    if start_d > end_d:
        raise ValueError(f"start_date ({start_d}) must be <= end_date ({end_d})")

    total_days = (end_d - start_d).days + (1 if inclusive else 0)
    if total_days <= 0:
        total_days = 1

    chunks: list[tuple[date, int]] = []
    current_d = start_d
    remaining = total_days

    while remaining > 0:
        chunk_len = min(clamped_max_days, remaining)
        chunks.append((current_d, chunk_len))
        current_d += timedelta(days=chunk_len)
        remaining -= chunk_len

    return chunks


def check_suomi_npp_cutoff(source: str, query_time_or_date: datetime | date | str) -> None:
    """Enforce platform continuity policy (FR-03).

    Suomi-NPP product delivery ends 2026-11-01 13:00 UTC. Refuse queries for SNPP after this timestamp.
    """
    src_upper = source.upper()
    if "SNPP" in src_upper:
        if isinstance(query_time_or_date, str):
            dt = pd.to_datetime(query_time_or_date, utc=True).to_pydatetime()
        elif isinstance(query_time_or_date, datetime):
            dt = (
                query_time_or_date
                if query_time_or_date.tzinfo
                else query_time_or_date.replace(tzinfo=UTC)
            )
        elif isinstance(query_time_or_date, date):
            if query_time_or_date > date(2026, 11, 1):
                dt = SUOMI_NPP_CUTOFF
            else:
                dt = datetime.combine(query_time_or_date, datetime.min.time(), tzinfo=UTC)
        else:
            raise TypeError(f"Unexpected date type: {type(query_time_or_date)}")

        if dt >= SUOMI_NPP_CUTOFF:
            raise ValueError("Suomi-NPP product delivery ended 2026-11-01T13:00:00Z")


def filter_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Filter detections to retain nominal and high confidence, dropping low confidence.

    VIIRS confidence: {'n', 'h'} kept, 'l' dropped.
    MODIS confidence: numeric >= 30 kept, < 30 dropped.
    """
    if df.empty or "confidence" not in df.columns:
        return df

    def is_confident(val: Any) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip().lower()
        if s in {"n", "h", "nominal", "high"}:
            return True
        if s in {"l", "low"}:
            return False
        try:
            v = float(s)
            return v >= 30.0
        except ValueError:
            return False

    mask = df["confidence"].apply(is_confident)
    return df[mask].copy()


def _empty_detections_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": pd.Series(dtype=float),
            "longitude": pd.Series(dtype=float),
            "acq_date": pd.Series(dtype="string"),
            "acq_time": pd.Series(dtype="string"),
            "confidence": pd.Series(dtype="string"),
            "frp": pd.Series(dtype=float),
            "acq_datetime_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def parse_detections_csv(csv_content: str | bytes | io.StringIO | io.BytesIO) -> pd.DataFrame:
    """Parse raw NASA FIRMS CSV detections into a structured DataFrame.

    Filters low-confidence rows, converts acquisition timestamps to UTC datetime,
    computes hour bin `ts_utc`, and applies the 3-hour availability rule `available_at_utc`.
    """
    if isinstance(csv_content, (bytes, bytearray)):
        csv_text = csv_content.decode("utf-8", errors="replace")
    elif isinstance(csv_content, io.BytesIO):
        csv_text = csv_content.getvalue().decode("utf-8", errors="replace")
    elif isinstance(csv_content, io.StringIO):
        csv_text = csv_content.getvalue()
    else:
        csv_text = str(csv_content)

    csv_text = csv_text.strip()
    if not csv_text or csv_text.startswith("No data") or csv_text.startswith("Invalid"):
        return _empty_detections_df()

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return _empty_detections_df()

    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return _empty_detections_df()

    # Drop low confidence
    df = filter_confidence(df)
    if df.empty:
        return _empty_detections_df()

    # Normalize acquisition time (e.g. 830 -> 0830)
    acq_times = df["acq_time"].astype(str).str.split(".").str[0].str.zfill(4)
    dt_str = df["acq_date"].astype(str) + " " + acq_times
    df["acq_datetime_utc"] = pd.to_datetime(dt_str, format="%Y-%m-%d %H%M", utc=True)
    df["ts_utc"] = df["acq_datetime_utc"].dt.floor("h")
    # Universal availability rule: acq_datetime + 3 hours
    df["available_at_utc"] = df["acq_datetime_utc"] + pd.Timedelta(hours=3)

    # Ensure numeric columns
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["frp"] = 0.0

    df = df.dropna(subset=["latitude", "longitude", "acq_datetime_utc"]).reset_index(drop=True)
    return df


def compute_geodesic_geometry(
    df: pd.DataFrame, target_lat: float, target_lon: float
) -> pd.DataFrame:
    """Compute target-centric geodesic distance, azimuth, bearing, ring, and sector.

    - Geodesic distance d in km via pyproj.Geod(ellps='WGS84').
    - Target->fire azimuth theta in degrees clockwise from north [0, 360).
    - Fire->target bearing beta = (theta + 180) % 360.
    - Radial rings: [0, 50), [50, 150), [150, 300), [300, 600] km (drops d > 600 km).
    - Compass sectors: 8 sectors of 45° centered on cardinal/intercardinal bearings.
    """
    if df.empty:
        res = df.copy()
        res["dist_km"] = pd.Series(dtype=float)
        res["azimuth_deg"] = pd.Series(dtype=float)
        res["bearing_deg"] = pd.Series(dtype=float)
        res["ring"] = pd.Series(dtype="string")
        res["sector"] = pd.Series(dtype="string")
        return res

    geod = pyproj.Geod(ellps="WGS84")
    fwd_az, _back_az, dist_m = geod.inv(
        np.full(len(df), target_lon),
        np.full(len(df), target_lat),
        df["longitude"].to_numpy(dtype=float),
        df["latitude"].to_numpy(dtype=float),
    )
    fwd_az_arr = np.asarray(fwd_az)
    dist_m_arr = np.asarray(dist_m)

    dist_km = dist_m_arr / 1000.0
    theta = (fwd_az_arr % 360.0 + 360.0) % 360.0
    beta = (theta + 180.0) % 360.0

    res = df.copy()
    res["dist_km"] = dist_km
    res["azimuth_deg"] = theta
    res["bearing_deg"] = beta

    # Filter to radius <= 600 km
    res = res[res["dist_km"] <= 600.0].copy()
    if res.empty:
        res["ring"] = pd.Series(dtype="string")
        res["sector"] = pd.Series(dtype="string")
        return res

    # Assign radial rings
    rings = pd.Series(index=res.index, dtype="string")
    dists = res["dist_km"]
    rings.loc[(dists >= 0.0) & (dists < 50.0)] = "r0_50"
    rings.loc[(dists >= 50.0) & (dists < 150.0)] = "r50_150"
    rings.loc[(dists >= 150.0) & (dists < 300.0)] = "r150_300"
    rings.loc[(dists >= 300.0) & (dists <= 600.0)] = "r300_600"
    res["ring"] = rings

    # Assign compass sectors
    sector_indices = np.floor((res["azimuth_deg"].to_numpy() + 22.5) / 45.0).astype(int) % 8
    res["sector"] = [SECTOR_NAMES[i] for i in sector_indices]
    return res


def aggregate_fire_exposure(
    detections_df: pd.DataFrame,
    domain: str,
    start_utc: datetime | str,
    end_utc: datetime | str,
    centroid: tuple[float, float] | None = None,
    platforms_active: int = 2,
    coverage_complete: bool = True,
) -> pd.DataFrame:
    """Aggregate active fire detections into hourly, target-centric radial rings.

    Computes platform-normalised fire counts and FRP sums conforming strictly to
    data/schemas/fire_exposure.schema.yaml.
    """
    domain_clean = domain.lower().strip()
    if domain_clean not in DOMAIN_CENTROIDS:
        raise ValueError(
            f"Unknown domain: '{domain}'. Expected one of {list(DOMAIN_CENTROIDS.keys())}"
        )

    if centroid is None:
        target_lat, target_lon = DOMAIN_CENTROIDS[domain_clean]
    else:
        target_lat, target_lon = centroid

    # Compute target-centric geometry if not already done
    if "dist_km" not in detections_df.columns or "ring" not in detections_df.columns:
        dets = compute_geodesic_geometry(
            detections_df, target_lat=target_lat, target_lon=target_lon
        )
    else:
        dets = detections_df[detections_df["dist_km"] <= 600.0].copy()

    start_dt = pd.to_datetime(start_utc, utc=True).floor("h")
    end_dt = pd.to_datetime(end_utc, utc=True).floor("h")
    if end_dt < start_dt:
        raise ValueError(f"end_utc ({end_utc}) must be >= start_utc ({start_utc})")

    if end_dt == start_dt:
        hours = pd.DatetimeIndex([start_dt], tz=UTC)
    else:
        hours = pd.date_range(start_dt, end_dt, freq="1h", inclusive="left", tz=UTC)

    norm_plat = float(platforms_active) if platforms_active > 0 else 1.0

    rows: list[dict[str, Any]] = []
    for h in hours:
        if not dets.empty and "ts_utc" in dets.columns:
            h_dets = dets[dets["ts_utc"] == h]
        else:
            h_dets = pd.DataFrame()

        n_total = len(h_dets)
        frp_total = float(h_dets["frp"].sum()) if n_total > 0 else 0.0

        ring_counts: dict[str, float] = {}
        ring_frp: dict[str, float] = {}
        for r in ("r0_50", "r50_150", "r150_300", "r300_600"):
            if n_total > 0 and "ring" in h_dets.columns:
                r_slice = h_dets[h_dets["ring"] == r]
                ring_counts[r] = float(len(r_slice)) / norm_plat
                ring_frp[r] = (
                    float(r_slice["frp"].sum()) / norm_plat if not r_slice.empty else 0.0
                )
            else:
                ring_counts[r] = 0.0
                ring_frp[r] = 0.0

        rows.append(
            {
                "domain": domain_clean,
                "ts_utc": h,
                "platforms_active": np.int32(platforms_active),
                "n_detections_total": np.int32(n_total),
                "frp_sum_mw_total": float(frp_total),
                "fire_count_r0_50": ring_counts["r0_50"],
                "frp_mw_r0_50": ring_frp["r0_50"],
                "fire_count_r50_150": ring_counts["r50_150"],
                "frp_mw_r50_150": ring_frp["r50_150"],
                "fire_count_r150_300": ring_counts["r150_300"],
                "frp_mw_r150_300": ring_frp["r150_300"],
                "fire_count_r300_600": ring_counts["r300_600"],
                "frp_mw_r300_600": ring_frp["r300_600"],
                "exposure_e24": 0.0,
                "exposure_e48": 0.0,
                "exposure_e72": 0.0,
                "coverage_complete": bool(coverage_complete),
            }
        )

    df_out = pd.DataFrame(rows)
    schema = _load_fire_exposure_schema()
    return validate_frame(df_out, schema)


class FirmsClient:
    """NASA FIRMS area-API client.

    Enforces 50% safety budget on 5,000 tx / 10 min, max 10 days per query,
    and Suomi-NPP cutoff policy (FR-03).
    """

    def __init__(
        self,
        map_key: str | None = None,
        base_url: str = DEFAULT_FIRMS_BASE_URL,
        rate_budget: RateBudget | None = None,
        resilient_client: ResilientClient | None = None,
    ) -> None:
        self.map_key = map_key or os.getenv("FIRMS_MAP_KEY", "")
        self.base_url = base_url.rstrip("/")
        if rate_budget is None:
            # FIRMS limit: 5000 tx / 10 min = 500/min.
            # 50% safety margin: 250 tx/min, 15000 tx/hr.
            self.rate_budget = RateBudget(per_minute=500, per_hour=30000, safety_margin=0.5)
        else:
            self.rate_budget = rate_budget

        self.circuit_breaker = CircuitBreaker(consecutive_429_to_open=3, open_seconds=900)
        self._custom_resilient = resilient_client

    def build_area_url(
        self,
        source: str,
        bbox: tuple[float, float, float, float] | dict[str, float] | str,
        day_range: int,
        date_str: str,
    ) -> str:
        """Construct the URL for /api/area/csv/<MAP_KEY>/<SOURCE>/<west,south,east,north>/<DAY_RANGE>/<DATE>."""
        if isinstance(bbox, dict):
            bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"
        elif isinstance(bbox, (list, tuple)):
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        else:
            bbox_str = str(bbox)

        return f"{self.base_url}/{self.map_key}/{source}/{bbox_str}/{day_range}/{date_str}"

    async def query_area(
        self,
        source: str,
        bbox: tuple[float, float, float, float] | dict[str, float] | str | None = None,
        start_date: datetime | date | str = "2026-10-01",
        end_date: datetime | date | str = "2026-10-01",
    ) -> pd.DataFrame:
        """Query FIRMS area API over consecutive chunks of <= 10 days, returning all detections."""
        check_suomi_npp_cutoff(source, start_date)
        check_suomi_npp_cutoff(source, end_date)

        if not self.map_key:
            raise ValueError("FIRMS_MAP_KEY is required to query NASA FIRMS API")

        if bbox is None:
            bbox = DEFAULT_FIRE_SEARCH_BOX

        chunks = split_date_range(start_date, end_date, max_days=10, inclusive=True)
        dfs: list[pd.DataFrame] = []

        if self._custom_resilient is not None:
            for chunk_date, day_count in chunks:
                check_suomi_npp_cutoff(source, chunk_date)
                url = self.build_area_url(
                    source, bbox, day_count, chunk_date.strftime("%Y-%m-%d")
                )
                resp = await self._custom_resilient.get(url)
                dfs.append(parse_detections_csv(resp.text))
        else:
            async with create_client(connect_timeout=10.0, read_timeout=60.0) as client:
                resilient = ResilientClient(client, self.rate_budget, self.circuit_breaker)
                for chunk_date, day_count in chunks:
                    check_suomi_npp_cutoff(source, chunk_date)
                    url = self.build_area_url(
                        source, bbox, day_count, chunk_date.strftime("%Y-%m-%d")
                    )
                    resp = await resilient.get(url)
                    dfs.append(parse_detections_csv(resp.text))

        if not dfs:
            return _empty_detections_df()

        combined = pd.concat(dfs, ignore_index=True)
        if combined.empty:
            return combined

        dedup_cols = [
            c
            for c in ["latitude", "longitude", "acq_date", "acq_time"]
            if c in combined.columns
        ]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

        if "acq_datetime_utc" in combined.columns:
            combined = combined.sort_values("acq_datetime_utc").reset_index(drop=True)

        return combined

    def query_area_sync(
        self,
        source: str,
        bbox: tuple[float, float, float, float] | dict[str, float] | str | None = None,
        start_date: datetime | date | str = "2026-10-01",
        end_date: datetime | date | str = "2026-10-01",
    ) -> pd.DataFrame:
        """Synchronous wrapper for query_area."""
        return asyncio.run(
            self.query_area(source=source, bbox=bbox, start_date=start_date, end_date=end_date)
        )

    async def get_data_availability(self, source: str) -> dict[str, Any]:
        """Query data availability for a source via /api/data_availability/csv/<MAP_KEY>/<SOURCE>."""
        check_suomi_npp_cutoff(source, datetime.now(UTC))
        if not self.map_key:
            raise ValueError("FIRMS_MAP_KEY is required to query NASA FIRMS API")

        url = f"{DEFAULT_AVAILABILITY_BASE_URL}/{self.map_key}/{source}"
        if self._custom_resilient is not None:
            resp = await self._custom_resilient.get(url)
        else:
            async with create_client(connect_timeout=10.0, read_timeout=60.0) as client:
                resilient = ResilientClient(client, self.rate_budget, self.circuit_breaker)
                resp = await resilient.get(url)

        return {"source": source, "raw_response": resp.text}


def detections(
    source: str,
    bbox: tuple[float, float, float, float] | dict[str, float] | str | None = None,
    start: datetime | date | str = "2026-10-01",
    end: datetime | date | str = "2026-10-01",
    map_key: str | None = None,
) -> pd.DataFrame:
    """Convenience synchronous function matching the module docstring contract."""
    client = FirmsClient(map_key=map_key)
    return client.query_area_sync(source=source, bbox=bbox, start_date=start, end_date=end)
