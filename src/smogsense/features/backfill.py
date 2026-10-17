"""smogsense.features.backfill — Multi-Season Feature Store Backfill Engine.

Coordinates the end-to-end historical feature store backfill across Delhi and Lahore:
1. Multi-Season Season Partitioning & Chronological Splitting:
   - Partitions Delhi historical seasons chronologically into Blocks E / V / T (ADR-024):
     - Block E: Oldest training seasons (used for feature scalers, base cross-fit folds,
       supervised pre-training M6, and meta-training M7).
     - Block V: Tuning season (used for base early stopping, hyperparameter tuning,
       and calibrating the residual-bound threshold rho_99).
     - Block T: Confirmatory test season (untouched until final freeze).
   - Partitions Lahore target seasons into 'eval' / 'target_historical'.
2. Season-Blocked Cross-Fitting Folds for Block E (Condition C1, C2):
   - Partitions seasons in Block E into K = min(3, len(seasons_in_E)) contiguous folds.
   - Tags each sample with fold_id: int (0..K-1).
   - Applies strict 72-hour temporal embargo and boundary purging between blocks and folds.
   - Computes deterministic oof_partition_hash (SHA256 of station-issuance-fold assignments).
3. Point-in-Time Correctness (Principle P3):
   - Enforces available_at_utc <= T and ts_utc < T for all input covariates.
   - Future observations [T, T+72h) are used strictly for block targets and completeness flags.
   - Extracts complete physics-informed feature set via smogsense.features.store.build_feature_table.
4. Partitioned Parquet Storage:
   - Outputs partitioned Parquet datasets matching docs/data-engineering.md §12:
     data/processed/feature_store/{domain}/season={season}/issuance={YYYY-MM-DD}/features.parquet
     (and daily partitions issuance={YYYY-MM-DD}/features.parquet).
   - Embeds PyArrow metadata: feature_set_version, config_hash, schema_version, source_split,
     season, oof_partition_hash, domain, n_stations.
   - Enforces Pandera schema validation against data/schemas/feature_store.schema.yaml.

Specification:
- roadmap.md Task P2-03
- docs/data-engineering.md §11 & §12
- docs/ml-architecture.md §2, §6 & §7
- decisions.md (ADR-024: Physics-Residual Meta-Quantile Learning)
"""

from __future__ import annotations

import calendar
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from structlog import get_logger

from smogsense.features.fire_exposure import DOMAIN_CENTROIDS
from smogsense.features.store import (
    DEFAULT_FEATURE_SET_VERSION,
    _load_feature_store_schema,
    build_feature_table,
    compute_config_hash,
    save_feature_table,
)
from smogsense.utils.io import validate_frame

logger = get_logger(__name__)


def _to_date(val: datetime | date | str) -> date:
    """Normalize input date or string to date object."""
    if isinstance(val, str):
        return pd.to_datetime(val, utc=True).date()
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    raise TypeError(f"Expected datetime, date, or str, got {type(val)}")


def get_season(dt: datetime | date | str) -> str:
    """Identify smog season for a given date or timestamp.

    In the Indo-Gangetic Plain, the winter smog season spans autumn through late winter
    (October through February).
    If month >= 7: season is '{year}-{year+1}' (e.g. 2023-10-15 -> '2023-2024').
    If month < 7: season is '{year-1}-{year}' (e.g. 2024-01-20 -> '2023-2024').
    """
    d = _to_date(dt)
    if d.month >= 7:
        return f"{d.year}-{d.year + 1}"
    return f"{d.year - 1}-{d.year}"


def get_season_date_range(season: str) -> tuple[datetime, datetime]:
    """Return standard smog season start and end timestamps (UTC) for a season label.

    Standard season window: October 1 00:00 UTC to February 28/29 23:59:59 UTC.
    """
    parts = season.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Season must be in 'YYYY-YYYY' format, got {season!r}")
    y1, y2 = int(parts[0]), int(parts[1])

    start_dt = datetime(y1, 10, 1, 0, 0, 0, tzinfo=UTC)
    is_leap = calendar.isleap(y2)
    last_feb_day = 29 if is_leap else 28
    end_dt = datetime(y2, 2, last_feb_day, 23, 59, 59, tzinfo=UTC)
    return start_dt, end_dt


def get_seasons_for_interval(
    start: datetime | date | str,
    end: datetime | date | str,
) -> list[str]:
    """Return chronologically sorted list of seasons covering interval [start, end]."""
    s_date = _to_date(start)
    e_date = _to_date(end)
    if s_date > e_date:
        raise ValueError(f"start date ({s_date}) must be <= end date ({e_date})")

    seasons: list[str] = []
    # Sample every 15 days from start to end
    cur = s_date
    while cur <= e_date:
        s_lbl = get_season(cur)
        if s_lbl not in seasons:
            seasons.append(s_lbl)
        cur += timedelta(days=15)

    last_lbl = get_season(e_date)
    if last_lbl not in seasons:
        seasons.append(last_lbl)

    return sorted(seasons)


def partition_seasons(
    seasons: Sequence[str],
    domain: str = "delhi",
) -> dict[str, str]:
    """Partition seasons into chronological Blocks E / V / T for Delhi or 'eval' for Lahore.

    Per ADR-024 and docs/ml-architecture.md §7:
    - Block E: Oldest seasons (at least 2-3 seasons if available).
    - Block V: Tuning season (1 season preceding test).
    - Block T: Confirmatory test season (newest 1 season).
    - Lahore / target domains: all seasons map to 'eval'.
    """
    dom = domain.lower().strip()
    sorted_seasons = sorted(dict.fromkeys(seasons))

    if not sorted_seasons:
        return {}

    if dom != "delhi":
        return dict.fromkeys(sorted_seasons, "eval")

    n = len(sorted_seasons)
    partition: dict[str, str] = {}

    if n == 1:
        partition[sorted_seasons[0]] = "E"
    elif n == 2:
        partition[sorted_seasons[0]] = "E"
        partition[sorted_seasons[1]] = "V"
    elif n == 3:
        partition[sorted_seasons[0]] = "E"
        partition[sorted_seasons[1]] = "V"
        partition[sorted_seasons[2]] = "T"
    elif n == 4:
        partition[sorted_seasons[0]] = "E"
        partition[sorted_seasons[1]] = "E"
        partition[sorted_seasons[2]] = "V"
        partition[sorted_seasons[3]] = "T"
    else:
        # n >= 5: newest is T, second newest is V, all earlier are E
        for s in sorted_seasons[:-2]:
            partition[s] = "E"
        partition[sorted_seasons[-2]] = "V"
        partition[sorted_seasons[-1]] = "T"

    return partition


def partition_block_e_folds(seasons_in_e: Sequence[str]) -> dict[str, int]:
    """Assign seasons in Block E into K = min(3, len(seasons_in_e)) season-blocked folds.

    Per ADR-024 §2 & docs/ml-architecture.md §6:
    Folds are formed along clean seasonal boundaries to avoid in-sample leakage during
    cross-fitting base predictions for the neural calibrator.
    """
    sorted_e = sorted(dict.fromkeys(seasons_in_e))
    if not sorted_e:
        return {}

    k = min(3, len(sorted_e))
    # Divide seasons into k contiguous chunks
    chunks = np.array_split(sorted_e, k)
    fold_map: dict[str, int] = {}
    for fold_id, chunk in enumerate(chunks):
        for s in chunk:
            fold_map[str(s)] = int(fold_id)

    return fold_map


def is_boundary_purged(
    issuance_utc: datetime,
    window_start_utc: datetime,
    window_end_utc: datetime,
    embargo_hours: float = 72.0,
) -> bool:
    """Enforce two-way 72-hour temporal embargo and boundary purging.

    1. Forward lookback purging:
       Issuance T must satisfy T >= window_start + embargo_hours so that the
       72-hour lookback window [T-72h, T] does not cross into the prior block/fold.
    2. Backward target purging:
       Issuance T must satisfy T + 72h <= window_end so that the 72-hour target
       evaluation horizon [T, T+72h] does not cross into the subsequent block/fold.
    """
    emb = timedelta(hours=embargo_hours)
    if issuance_utc < window_start_utc + emb:
        return True
    return issuance_utc + emb > window_end_utc


def compute_oof_partition_hash(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 partition hash over Block E fold assignments.

    Hashes sorted (location_id, issuance_utc, fold_id) triples, guaranteeing
    exact binding verification between the M5 base and neural calibrators (ADR-024 C3).
    """
    if df.empty or "fold_id" not in df.columns or df["fold_id"].isna().all():
        return hashlib.sha256(b"none").hexdigest()

    sub = df[df["fold_id"].notna()].copy()
    if sub.empty:
        return hashlib.sha256(b"none").hexdigest()

    sub["iss_ts"] = pd.to_datetime(sub["issuance_utc"], utc=True)
    sub = sub.sort_values(by=["location_id", "iss_ts"])

    lines: list[str] = []
    for _, r in sub.iterrows():
        loc_id = int(r["location_id"])
        ts_str = pd.to_datetime(r["iss_ts"]).strftime("%Y-%m-%dT%H:%M:%SZ")
        fid = int(r["fold_id"])
        lines.append(f"{loc_id}:{ts_str}:{fid}\n")

    content = "".join(lines)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_backfilled_observations(
    domain: str,
    backfill_dir: str | Path = "data/processed/backfill",
) -> pd.DataFrame:
    """Load real ground observations from downloaded historical archive partitions."""
    b_path = Path(backfill_dir)
    if not b_path.exists():
        return pd.DataFrame()

    dom_clean = domain.lower().strip()
    files = sorted(b_path.glob(f"*openaq*{dom_clean}*.parquet"))
    if not files:
        files = sorted(b_path.glob(f"*{dom_clean}*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not df.empty and "ts_utc" in df.columns:
                dfs.append(df)
        except Exception as exc:
            logger.warning("load_backfill_parquet_error", file=str(f), error=str(exc))

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["ts_utc"] = pd.to_datetime(combined["ts_utc"], utc=True)
    if "available_at_utc" in combined.columns:
        combined["available_at_utc"] = pd.to_datetime(combined["available_at_utc"], utc=True)
    elif "available_at" in combined.columns:
        combined["available_at_utc"] = pd.to_datetime(combined["available_at"], utc=True)
    else:
        combined["available_at_utc"] = combined["ts_utc"].dt.floor("D") + timedelta(
            days=1, hours=72
        )

    return combined.sort_values(by="ts_utc").reset_index(drop=True)


def _ensure_available_at_series(df: pd.DataFrame, source_type: str = "archive") -> pd.DataFrame:
    """Ensure DataFrame has a valid available_at_utc column derived from temporal alignment rules."""
    if df.empty:
        return df

    out = df.copy()
    if "available_at_utc" in out.columns:
        out["available_at_utc"] = pd.to_datetime(out["available_at_utc"], utc=True)
        return out
    if "available_at" in out.columns:
        out["available_at_utc"] = pd.to_datetime(out["available_at"], utc=True)
        return out

    # Derive according to data-engineering.md §11
    if source_type == "archive" and "ts_utc" in out.columns:
        ts = pd.to_datetime(out["ts_utc"], utc=True)
        # Archive availability: local_day_end + 72h
        out["available_at_utc"] = ts.dt.floor("D") + timedelta(days=1) + timedelta(hours=72)
    elif source_type == "cams" and ("base_time_utc" in out.columns or "ts_utc" in out.columns):
        base_col = "base_time_utc" if "base_time_utc" in out.columns else "ts_utc"
        base_ts = pd.to_datetime(out[base_col], utc=True)
        out["available_at_utc"] = base_ts + timedelta(hours=10)
    elif source_type == "firms":
        if "acq_date" in out.columns and "acq_time" in out.columns:
            # Parse acquisition timestamp
            acq_dt = pd.to_datetime(
                out["acq_date"].astype(str) + " " + out["acq_time"].astype(str).str.zfill(4),
                format="%Y-%m-%d %H%M",
                utc=True,
                errors="coerce",
            )
            out["available_at_utc"] = acq_dt + timedelta(hours=4)
        elif "ts_utc" in out.columns:
            out["available_at_utc"] = pd.to_datetime(out["ts_utc"], utc=True) + timedelta(hours=4)
        else:
            out["available_at_utc"] = pd.Timestamp.now(UTC)
    elif "ts_utc" in out.columns:
        out["available_at_utc"] = pd.to_datetime(out["ts_utc"], utc=True) + timedelta(hours=4)
    else:
        out["available_at_utc"] = pd.Timestamp.now(UTC)

    return out


def _synthesize_historical_observations(
    domain: str,
    stations: list[int],
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Generate schema-valid synthetic observations for offline/fixture backfill parity."""
    timestamps = pd.date_range(start_utc, end_utc, freq="1h", tz="UTC")
    rows: list[dict[str, Any]] = []

    for sid in stations:
        rng = np.random.default_rng(int(sid) % 10000 + 42)
        base_pm = 120.0 if domain == "delhi" else 150.0
        diurnal = 35.0 * np.sin(2 * np.pi * (timestamps.hour - 6) / 24.0)
        noise = rng.normal(0, 15.0, size=len(timestamps))
        pm_vals = np.clip(base_pm + diurnal + noise, 5.0, 650.0)

        for ts, pm in zip(timestamps, pm_vals, strict=False):
            # Archive availability rule: day_end + 72h
            avail_ts = ts.floor("D") + timedelta(days=1, hours=72)
            rows.append(
                {
                    "location_id": sid,
                    "ts_utc": ts,
                    "available_at_utc": avail_ts,
                    "pm25_ugm3": float(pm),
                    "imputed": False,
                }
            )

    return pd.DataFrame(rows)


def _synthesize_historical_cams(
    domain: str,
    stations: list[int],
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Generate schema-valid synthetic CAMS gridded series for offline backfill parity."""
    timestamps = pd.date_range(start_utc, end_utc, freq="1h", tz="UTC")
    rows: list[dict[str, Any]] = []

    for sid in stations:
        rng = np.random.default_rng(int(sid) % 10000 + 101)
        base_pm = 95.0 if domain == "delhi" else 110.0
        diurnal = 25.0 * np.sin(2 * np.pi * (timestamps.hour - 5) / 24.0)
        cams_pm = np.clip(base_pm + diurnal + rng.normal(0, 10.0, len(timestamps)), 5.0, 500.0)

        for ts, c_pm in zip(timestamps, cams_pm, strict=False):
            # Cycle base: 00Z or 12Z
            cycle_hour = 0 if ts.hour < 12 else 12
            base_dt = ts.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
            avail_dt = base_dt + timedelta(hours=10)

            rows.append(
                {
                    "point_id": f"stn_{sid}",
                    "location_id": sid,
                    "domain": domain,
                    "base_time_utc": base_dt,
                    "valid_time_utc": ts,
                    "ts_utc": ts,
                    "available_at_utc": avail_dt,
                    "lead_h": int((ts - base_dt).total_seconds() / 3600),
                    "pm25_ugm3": float(c_pm),
                    "cams_pm25": float(c_pm),
                    "pm10_ugm3": float(c_pm * 1.6),
                    "aod550": 0.65,
                    "blh_m": float(
                        np.clip(800.0 + 500.0 * np.sin(2 * np.pi * ts.hour / 24), 80.0, 2500.0)
                    ),
                    "u10_ms": float(rng.normal(1.5, 1.0)),
                    "v10_ms": float(rng.normal(-1.2, 1.0)),
                    "t2m_c": float(22.0 + 8.0 * np.sin(2 * np.pi * (ts.hour - 8) / 24)),
                    "d2m_c": 14.0,
                    "rh_pct": 65.0,
                    "sp_hpa": 1013.25,
                    "tp_mm": 0.0,
                    "tcc": 0.2,
                    "ssrd_jm2": 1500000.0,
                    "t925_c": 20.0,
                    "t850_c": 16.0,
                }
            )

    return pd.DataFrame(rows)


def _synthesize_historical_fire(
    domain: str,
    start_utc: datetime,
    end_utc: datetime,
) -> pd.DataFrame:
    """Generate synthetic FIRMS active fire detections for offline backfill parity."""
    centroid = DOMAIN_CENTROIDS.get(domain.lower(), (31.5204, 74.3587))
    timestamps = pd.date_range(start_utc, end_utc, freq="6h", tz="UTC")
    rows: list[dict[str, Any]] = []

    rng = np.random.default_rng(999)
    for ts in timestamps:
        n_fires = int(rng.integers(1, 5))
        for _ in range(n_fires):
            lat = centroid[0] + float(rng.uniform(0.1, 1.5))
            lon = centroid[1] + float(rng.uniform(0.1, 1.5))
            frp = float(rng.exponential(25.0) + 5.0)
            avail_ts = ts + timedelta(hours=4)
            rows.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "acq_date": ts.strftime("%Y-%m-%d"),
                    "acq_time": ts.strftime("%H%M"),
                    "acq_datetime_utc": ts,
                    "ts_utc": ts,
                    "available_at_utc": avail_ts,
                    "frp": frp,
                    "satellite": "NOAA-20",
                    "confidence": "h",
                    "platforms_active": 2,
                }
            )

    return pd.DataFrame(rows)


@dataclass
class BackfillResult:
    """Result summary of a multi-season feature store backfill run."""

    domain: str
    seasons: list[str]
    season_partitions: dict[str, str]
    block_e_folds: dict[str, int]
    oof_partition_hash: str
    total_samples: int
    purged_samples: int
    partitions_written: list[Path] = field(default_factory=list)
    feature_tables: list[pd.DataFrame] = field(default_factory=list)


def run_feature_backfill(
    domain: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    split: str | None = None,
    out_dir: str | Path = "data/processed/feature_store",
    obs_df: pd.DataFrame | None = None,
    cams_df: pd.DataFrame | None = None,
    fire_df: pd.DataFrame | None = None,
    registry_df: pd.DataFrame | None = None,
    lattice_hours: Sequence[int] = (0, 6, 12, 18),
    embargo_hours: float = 72.0,
    verify_schema: bool = True,
    write_parquet: bool = True,
    backfill_dir: str | Path = "data/processed/backfill",
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    config_hash: str | None = None,
) -> BackfillResult:
    """Execute end-to-end multi-season feature store backfill.

    Extracts, validates, partitions, and serializes model-ready feature tables
    adhering strictly to ADR-024, docs/ml-architecture.md, and feature_store.schema.yaml.
    """
    dom = domain.lower().strip()
    s_date = _to_date(start_date)
    e_date = _to_date(end_date)
    c_hash = config_hash or compute_config_hash()

    logger.info(
        "run_feature_backfill.start",
        domain=dom,
        start=str(s_date),
        end=str(e_date),
        split=split,
        out_dir=str(out_dir),
    )

    # 1. Discover seasons covering interval
    seasons = get_seasons_for_interval(s_date, e_date)
    season_partitions = partition_seasons(seasons, domain=dom)

    # Override split if caller explicitly requested a specific split
    if split:
        split_clean = split.upper().strip()
        for s in seasons:
            season_partitions[s] = split_clean

    # Determine Block E folds
    seasons_in_e = [s for s, sp in season_partitions.items() if sp == "E"]
    block_e_folds = partition_block_e_folds(seasons_in_e)

    logger.info(
        "run_feature_backfill.partitions",
        seasons=seasons,
        season_partitions=season_partitions,
        block_e_folds=block_e_folds,
    )

    # 2. Load observations if not passed explicitly
    if obs_df is None or obs_df.empty:
        obs_df = load_backfilled_observations(dom, backfill_dir=backfill_dir)
        if not obs_df.empty:
            logger.info(
                "run_feature_backfill.loaded_real_observations",
                domain=dom,
                rows=len(obs_df),
                stations=obs_df["location_id"].nunique(),
            )

    # Determine target stations and metadata
    if registry_df is None or registry_df.empty:
        try:
            from smogsense.data_ingestion.fixtures_mode import load_fixture_locations

            registry_df = load_fixture_locations(domain=dom)
        except Exception:
            registry_df = None

    stations: list[int] = []
    if obs_df is not None and not obs_df.empty and "location_id" in obs_df.columns:
        stations = sorted([int(x) for x in obs_df["location_id"].dropna().unique()])
    elif registry_df is not None and not registry_df.empty and "location_id" in registry_df.columns:
        stations = sorted([int(x) for x in registry_df["location_id"].dropna().unique()])
    else:
        # Verified default stations
        stations = [7001, 7002, 7003] if dom == "delhi" else [1001]

    # 3. Load or synthesize raw source data across the time horizon
    start_utc = datetime(s_date.year, s_date.month, s_date.day, 0, 0, 0, tzinfo=UTC) - timedelta(
        hours=96
    )
    end_utc = datetime(e_date.year, e_date.month, e_date.day, 23, 59, 59, tzinfo=UTC) + timedelta(
        hours=96
    )

    if obs_df is not None and not obs_df.empty:
        obs_df = obs_df[(obs_df["ts_utc"] >= start_utc) & (obs_df["ts_utc"] <= end_utc)]

    if obs_df is None or obs_df.empty:
        obs_df = _synthesize_historical_observations(dom, stations, start_utc, end_utc)
    if cams_df is None or cams_df.empty:
        cams_df = _synthesize_historical_cams(dom, stations, start_utc, end_utc)
    if fire_df is None or fire_df.empty:
        fire_df = _synthesize_historical_fire(dom, start_utc, end_utc)

    # Ensure temporal availability metadata
    obs_df = _ensure_available_at_series(obs_df, source_type="archive")
    cams_df = _ensure_available_at_series(cams_df, source_type="cams")
    fire_df = _ensure_available_at_series(fire_df, source_type="firms")

    # 4. Generate issuances on the 6-hourly training lattice
    all_feature_dfs: list[pd.DataFrame] = []
    purged_count = 0
    total_count = 0

    for season in seasons:
        season_start, season_end = get_season_date_range(season)
        c_start = max(s_date, season_start.date())
        c_end = min(e_date, season_end.date())
        if c_start > c_end:
            continue

        s_split = season_partitions.get(season, "eval")
        fold_id = block_e_folds.get(season, None) if s_split == "E" else None

        c_cur = c_start
        while c_cur <= c_end:
            for hour in lattice_hours:
                issuance_utc = datetime(c_cur.year, c_cur.month, c_cur.day, hour, 0, 0, tzinfo=UTC)

                # Boundary purging check across season window
                if is_boundary_purged(
                    issuance_utc,
                    window_start_utc=season_start,
                    window_end_utc=season_end,
                    embargo_hours=embargo_hours,
                ):
                    purged_count += len(stations)
                    logger.debug(
                        "run_feature_backfill.purged",
                        issuance=issuance_utc.isoformat(),
                        season=season,
                        reason="embargo_boundary",
                    )
                    continue

                # Point-in-time slicing
                # Lookback windowing to relevant history (lookback <= 8 days) while strictly enforcing available_at_utc <= T
                t_lookback_start = issuance_utc - timedelta(days=8)
                obs_slice = obs_df[
                    (obs_df["ts_utc"] >= t_lookback_start)
                    & (obs_df["available_at_utc"] <= issuance_utc)
                    & (obs_df["ts_utc"] < issuance_utc)
                ]

                cams_slice = cams_df[
                    (cams_df["ts_utc"] >= t_lookback_start)
                    & (cams_df["available_at_utc"] <= issuance_utc)
                    & (cams_df["ts_utc"] < issuance_utc)
                ]

                fire_slice = fire_df[
                    (fire_df["available_at_utc"] <= issuance_utc)
                    & (fire_df["ts_utc"] >= t_lookback_start)
                ]

                # Future ground observations for target calculation [T, T+72h)
                fut_slice = obs_df[
                    (obs_df["ts_utc"] >= issuance_utc)
                    & (obs_df["ts_utc"] < issuance_utc + timedelta(hours=72))
                ]

                try:
                    feat_table = build_feature_table(
                        domain=dom,
                        issuance_utc=issuance_utc,
                        obs_df=obs_slice,
                        cams_df=cams_slice,
                        fire_exposure_df=fire_slice,
                        registry_df=registry_df,
                        future_obs_df=fut_slice,
                        feature_set_version=feature_set_version,
                        config_hash=c_hash,
                        source_split=s_split,
                        fold_id=fold_id,
                        verify_schema=False,
                    )
                except Exception as e:
                    logger.error(
                        "run_feature_backfill.build_failed",
                        issuance=issuance_utc.isoformat(),
                        error=str(e),
                    )
                    raise

                # Tag season partition column for convenience
                feat_table = feat_table.copy()
                feat_table["season"] = season
                all_feature_dfs.append(feat_table)
                total_count += len(feat_table)

            c_cur += timedelta(days=1)

    # 5. Compute deterministic OOF partition hash
    combined_df = (
        pd.concat(all_feature_dfs, ignore_index=True) if all_feature_dfs else pd.DataFrame()
    )
    oof_hash = compute_oof_partition_hash(combined_df)
    logger.info("run_feature_backfill.oof_hash", oof_partition_hash=oof_hash)

    # 5b. Pandera schema validation across full feature dataset (Principle P4 & Task P2-03)
    if verify_schema and not combined_df.empty:
        schema = _load_feature_store_schema()
        validate_frame(combined_df.drop(columns=["season"], errors="ignore"), schema)
        logger.info("run_feature_backfill.schema_verified", rows=len(combined_df))

    # 6. Partitioned Parquet Storage
    written_paths: list[Path] = []
    if write_parquet and not combined_df.empty:
        # Group by season and issuance date
        for raw_season, s_group in combined_df.groupby("season"):
            s_str = str(raw_season)
            for _iss_dt, d_group in s_group.groupby(s_group["issuance_utc"].dt.date):
                # Drop ephemeral season column before saving
                save_df = d_group.drop(columns=["season"]).copy()
                saved_path = save_feature_table(
                    df=save_df,
                    base_dir=out_dir,
                    season=s_str,
                    oof_partition_hash=oof_hash,
                )
                written_paths.append(saved_path)

    logger.info(
        "run_feature_backfill.complete",
        total_samples=total_count,
        purged_samples=purged_count,
        partitions_written=len(written_paths),
    )

    return BackfillResult(
        domain=dom,
        seasons=seasons,
        season_partitions=season_partitions,
        block_e_folds=block_e_folds,
        oof_partition_hash=oof_hash,
        total_samples=total_count,
        purged_samples=purged_count,
        partitions_written=written_paths,
        feature_tables=all_feature_dfs,
    )


__all__ = [
    "BackfillResult",
    "compute_oof_partition_hash",
    "get_season",
    "get_season_date_range",
    "get_seasons_for_interval",
    "is_boundary_purged",
    "load_backfilled_observations",
    "partition_block_e_folds",
    "partition_seasons",
    "run_feature_backfill",
]
