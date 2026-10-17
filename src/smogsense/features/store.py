"""smogsense.features.store — Versioned Parquet feature store and point-in-time snapshot manager.

Manages:
1. Re-exports InputSnapshot and daily input snapshot lifecycle for backward compatibility.
2. Canonical feature store construction adhering strictly to `data/schemas/feature_store.schema.yaml`:
   - Feature columns: `f_<group>_<name>` (float32)
   - Mask columns: `m_<group>_<name>` (bool)
   - Target columns: `y_h24`, `y_h48`, `y_h72` (float64) and completeness flags `valid_h24`, `valid_h48`, `valid_h72` (bool)
   - Metadata columns: `domain`, `location_id`, `issuance_utc`, `feature_set_version`, `config_hash`, `n_available_stations`, `source_split`
3. Partitioned Parquet storage under `data/processed/feature_store/{domain}/issuance=YYYY-MM-DD/features.parquet`
   with embedded PyArrow metadata (`feature_set_version`, `config_hash`, `schema_version`, `n_stations`).
4. Strict point-in-time guarantee (Principle P3):
   Rejects any feature record where `available_at_utc > T`.

Specification:
- docs/ml-architecture.md §2 ('Feature engineering (physics-informed)')
- docs/data-engineering.md §11 & §12 ('Temporal alignment' and 'Storage layout')
- configs/features.yaml
- data/schemas/feature_store.schema.yaml
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pyarrow as pa_arrow  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from smogsense.features.aerosol import extract_block_aerosol_features
from smogsense.features.calendar import extract_block_calendar_features
from smogsense.features.fire_exposure import DOMAIN_CENTROIDS, extract_block_fire_features
from smogsense.features.meteorology import extract_block_meteorology_features
from smogsense.features.snapshot import (
    InputSnapshot,
    build_input_snapshot,
    load_input_snapshot,
    save_input_snapshot,
)
from smogsense.features.static_covariates import extract_station_static_covariates
from smogsense.features.targets import block_targets
from smogsense.preprocessing.alignment import cams_lead_offset_hours
from smogsense.utils.io import validate_frame

SCHEMA_VERSION = "1.0.0"
DEFAULT_FEATURE_SET_VERSION = "fs1"


def _load_feature_store_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/feature_store.schema.yaml"),
        Path("/app/data/schemas/feature_store.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/feature_store.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(str(p))
    raise FileNotFoundError(f"feature_store.schema.yaml not found in candidates: {candidates}")


def compute_config_hash(features_yaml_path: str | Path | None = None) -> str:
    """Compute sha256 hash of the canonical feature configuration."""
    candidates = [
        Path("configs/features.yaml"),
        Path("/app/configs/features.yaml"),
        Path(__file__).resolve().parents[3] / "configs/features.yaml",
    ]
    if features_yaml_path is not None:
        candidates.insert(0, Path(features_yaml_path))

    for p in candidates:
        if p.exists():
            raw_bytes = p.read_bytes()
            return hashlib.sha256(raw_bytes).hexdigest()

    # Fallback deterministic hash
    return hashlib.sha256(b"features_yaml_default").hexdigest()


def compute_recent_cams_bias(
    obs_df: pd.DataFrame,
    cams_df: pd.DataFrame,
    location_id: int,
    issuance_utc: datetime | pd.Timestamp,
    window_days: int = 7,
) -> float:
    """Compute trailing 7-day mean CAMS bias: mean(obs_pm25 - cams_pm25) at the station.

    Enables local representativeness post-processing (40 km grid vs point monitor).
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    lookback_start = t_iss - timedelta(days=window_days)

    if (
        obs_df.empty
        or cams_df.empty
        or "ts_utc" not in obs_df.columns
        or "ts_utc" not in cams_df.columns
    ):
        return 0.0

    o_df = obs_df.copy()
    o_df["ts_utc"] = pd.to_datetime(o_df["ts_utc"], utc=True)
    if "location_id" in o_df.columns:
        o_df = o_df[o_df["location_id"] == location_id]
    o_slice = o_df[(o_df["ts_utc"] >= lookback_start) & (o_df["ts_utc"] < t_iss)]

    c_df = cams_df.copy()
    c_df["ts_utc"] = pd.to_datetime(c_df["ts_utc"], utc=True)
    if "location_id" in c_df.columns:
        c_sub = c_df[c_df["location_id"] == location_id]
        if not c_sub.empty:
            c_df = c_sub
    c_slice = c_df[(c_df["ts_utc"] >= lookback_start) & (c_df["ts_utc"] < t_iss)]

    if o_slice.empty or c_slice.empty:
        return 0.0

    cams_col = next(
        (c for c in ("cams_pm25", "pm25", "particulate_matter_2.5um") if c in c_slice.columns), None
    )
    if not cams_col or "pm25_ugm3" not in o_slice.columns:
        return 0.0

    # Join on ts_utc
    merged = (
        o_slice[["ts_utc", "pm25_ugm3"]]
        .dropna()
        .merge(
            c_slice[["ts_utc", cams_col]].dropna(),
            on="ts_utc",
            how="inner",
        )
    )
    if len(merged) < 6:
        return 0.0

    diff = merged["pm25_ugm3"].to_numpy(dtype=np.float64) - merged[cams_col].to_numpy(
        dtype=np.float64
    )
    return float(np.nanmean(diff))


def build_feature_table(
    domain: str,
    issuance_utc: datetime | pd.Timestamp,
    obs_df: pd.DataFrame,
    cams_df: pd.DataFrame,
    fire_exposure_df: pd.DataFrame | None = None,
    registry_df: pd.DataFrame | None = None,
    future_obs_df: pd.DataFrame | None = None,
    horizons: Sequence[int] = (24, 48, 72),
    feature_set_version: str = DEFAULT_FEATURE_SET_VERSION,
    config_hash: str | None = None,
    source_split: str | None = None,
    fold_id: int | None = None,
    degradation_level: int = 0,
) -> pd.DataFrame:
    """Construct model-ready feature store table conforming to feature_store.schema.yaml.

    Enforces strict point-in-time correctness:
    Raises ValueError if any input feature data has available_at_utc > T.
    """
    t_iss = pd.to_datetime(issuance_utc, utc=True)
    dom_clean = domain.lower().strip()
    c_hash = config_hash or compute_config_hash()

    # 1. Point-in-time verification
    for name, df_in in [
        ("obs_df", obs_df),
        ("cams_df", cams_df),
        ("fire_exposure_df", fire_exposure_df),
    ]:
        if df_in is not None and not df_in.empty:
            for col in ("available_at_utc", "available_at"):
                if col in df_in.columns:
                    leaks = df_in[pd.to_datetime(df_in[col], utc=True) > t_iss]
                    if not leaks.empty:
                        raise ValueError(
                            f"Point-in-time violation in {name}: {len(leaks)} records have "
                            f"{col} > issuance {t_iss}"
                        )

    # 2. Determine target stations
    stations: list[int] = []
    if registry_df is not None and not registry_df.empty and "location_id" in registry_df.columns:
        stations = [int(x) for x in registry_df["location_id"].dropna().unique()]
    elif obs_df is not None and not obs_df.empty and "location_id" in obs_df.columns:
        stations = [int(x) for x in obs_df["location_id"].dropna().unique()]
    else:
        # Default domain location fallback
        default_loc = 1001 if dom_clean == "lahore" else 7001
        stations = [default_loc]

    n_available_stations = len(stations)

    # 3. Compute block targets if future observations are provided
    targets_map: dict[int, dict[str, Any]] = {}
    if future_obs_df is not None and not future_obs_df.empty:
        tgts_df = block_targets(future_obs_df, issuance_utc=t_iss, horizons=horizons)
        if not tgts_df.empty:
            for _, t_row in tgts_df.iterrows():
                targets_map[int(t_row["location_id"])] = dict(t_row)

    # 4. Extract domain-level and block features
    # Calendar features
    cal_features = extract_block_calendar_features(
        issuance_utc=t_iss, domain=dom_clean, horizons=horizons
    )

    # CAMS lead offset
    cams_offset_h = float(cams_lead_offset_hours(t_iss, degradation_level=degradation_level))

    # Domain centroid
    centroid = DOMAIN_CENTROIDS.get(dom_clean, (31.5204, 74.3587))

    # Fire exposure features
    fire_features = extract_block_fire_features(
        detections_df=fire_exposure_df if fire_exposure_df is not None else pd.DataFrame(),
        target_lat=centroid[0],
        target_lon=centroid[1],
        wind_df=cams_df,
        issuance_utc=t_iss,
        horizons=horizons,
    )

    # Meteorology features
    met_features = extract_block_meteorology_features(
        cams_df=cams_df,
        issuance_utc=t_iss,
        horizons=horizons,
    )

    # Aerosol features
    aod_features = extract_block_aerosol_features(
        cams_df=cams_df,
        issuance_utc=t_iss,
        horizons=horizons,
    )

    rows: list[dict[str, Any]] = []

    for loc_id in stations:
        row: dict[str, Any] = {
            "domain": dom_clean,
            "location_id": np.int64(loc_id),
            "issuance_utc": t_iss,
            "feature_set_version": str(feature_set_version),
            "config_hash": str(c_hash),
            "n_available_stations": np.int32(n_available_stations),
            "source_split": source_split,
            "fold_id": int(fold_id) if fold_id is not None else None,
        }

        # Targets
        t_data = targets_map.get(loc_id, {})
        for h in (24, 48, 72):
            y_val = t_data.get(f"y_h{h}", t_data.get(f"y_{h}", np.nan))
            v_val = bool(t_data.get(f"valid_h{h}", t_data.get(f"valid_{h}", False)))
            row[f"y_h{h}"] = float(y_val) if pd.notna(y_val) else np.nan
            row[f"valid_h{h}"] = v_val

        # Static Station Covariates
        stn_meta: dict[str, Any] = {"location_id": loc_id}
        if (
            registry_df is not None
            and not registry_df.empty
            and "location_id" in registry_df.columns
        ):
            stn_slice = registry_df[registry_df["location_id"] == loc_id]
            if not stn_slice.empty:
                stn_meta = dict(stn_slice.iloc[0])

        stat_features = extract_station_static_covariates(stn_meta, domain=dom_clean)
        for k, v in stat_features.items():
            row[k] = np.float32(v)
            row[f"m_{k[2:]}"] = bool(pd.isna(v))

        # Add calendar, met, aerosol, fire features
        for feat_dict in (cal_features, met_features, aod_features, fire_features):
            for k, v in feat_dict.items():
                row[k] = np.float32(v) if pd.notna(v) else np.nan
                row[f"m_{k[2:]}"] = bool(pd.isna(v))

        # CAMS forecast age and recent bias
        row["f_cams_lead_offset_h"] = np.float32(cams_offset_h)
        row["m_cams_lead_offset_h"] = False

        cams_bias = compute_recent_cams_bias(
            obs_df=obs_df,
            cams_df=cams_df,
            location_id=loc_id,
            issuance_utc=t_iss,
            window_days=7,
        )
        row["f_cams_bias_trailing_7d"] = np.float32(cams_bias)
        row["m_cams_bias_trailing_7d"] = bool(pd.isna(cams_bias))

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Cast column dtypes strictly matching feature_store.schema.yaml
    df_out["domain"] = df_out["domain"].astype("string[python]")
    df_out["location_id"] = df_out["location_id"].astype("int64")
    df_out["issuance_utc"] = pd.to_datetime(df_out["issuance_utc"], utc=True)
    df_out["feature_set_version"] = df_out["feature_set_version"].astype("string[python]")
    df_out["config_hash"] = df_out["config_hash"].astype("string[python]")
    df_out["n_available_stations"] = df_out["n_available_stations"].astype("int32")

    for h in (24, 48, 72):
        df_out[f"y_h{h}"] = df_out[f"y_h{h}"].astype("float64")
        df_out[f"valid_h{h}"] = df_out[f"valid_h{h}"].astype("bool")

    for col in df_out.columns:
        if col.startswith("f_"):
            df_out[col] = df_out[col].astype("float32")
        elif col.startswith("m_"):
            df_out[col] = df_out[col].astype("bool")

    if "fold_id" in df_out.columns:
        df_out["fold_id"] = df_out["fold_id"].astype("Int32")

    schema = _load_feature_store_schema()
    return validate_frame(df_out, schema)


def save_feature_table(
    df: pd.DataFrame,
    base_dir: str | Path = "data/processed/feature_store",
    season: str | None = None,
    oof_partition_hash: str | None = None,
) -> Path:
    """Serialize feature store DataFrame into partitioned Parquet with PyArrow metadata.

    File layout:
    - If season is provided:
      {base_dir}/{domain}/season={season}/issuance={YYYY-MM-DD}/features.parquet
    - Otherwise:
      {base_dir}/{domain}/issuance={YYYY-MM-DD}/features.parquet
    """
    if df.empty:
        raise ValueError("Cannot save empty feature table")

    domain = str(df["domain"].iloc[0])
    iss_ts = pd.to_datetime(df["issuance_utc"].iloc[0], utc=True)
    iss_date_str = iss_ts.strftime("%Y-%m-%d")

    if season:
        target_dir = Path(base_dir) / domain / f"season={season}" / f"issuance={iss_date_str}"
    else:
        target_dir = Path(base_dir) / domain / f"issuance={iss_date_str}"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "features.parquet"

    # Embedded PyArrow metadata
    source_split_val = (
        str(df["source_split"].iloc[0])
        if "source_split" in df.columns and pd.notna(df["source_split"].iloc[0])
        else "none"
    )
    meta = {
        "feature_set_version": str(df["feature_set_version"].iloc[0]),
        "config_hash": str(df["config_hash"].iloc[0]),
        "schema_version": SCHEMA_VERSION,
        "n_stations": str(len(df)),
        "domain": domain,
        "issuance_utc": iss_ts.isoformat(),
        "source_split": source_split_val,
        "season": str(season or "none"),
        "oof_partition_hash": str(oof_partition_hash or "none"),
    }
    encoded_meta = {k.encode("utf-8"): v.encode("utf-8") for k, v in meta.items()}

    table = pa_arrow.Table.from_pandas(df)
    existing_meta = table.schema.metadata or {}
    combined_meta = {**existing_meta, **encoded_meta}
    table = table.replace_schema_metadata(combined_meta)

    pq.write_table(table, str(target_path), compression="snappy")
    return target_path


def load_feature_table(
    domain: str,
    issuance_date: str | date | datetime,
    base_dir: str | Path = "data/processed/feature_store",
    season: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load partitioned feature store table and associated PyArrow metadata.

    Returns (DataFrame, metadata_dict).
    """
    if isinstance(issuance_date, (datetime, date)):
        date_str = issuance_date.strftime("%Y-%m-%d")
    else:
        date_str = str(issuance_date)[:10]

    dom_clean = domain.lower().strip()
    if season:
        target_path = (
            Path(base_dir)
            / dom_clean
            / f"season={season}"
            / f"issuance={date_str}"
            / "features.parquet"
        )
    else:
        target_path = Path(base_dir) / dom_clean / f"issuance={date_str}" / "features.parquet"
        if not target_path.exists():
            candidates = sorted(
                (Path(base_dir) / dom_clean).glob(f"**/issuance={date_str}/features.parquet")
            )
            if candidates:
                target_path = candidates[0]

    if not target_path.exists():
        raise FileNotFoundError(f"Feature table not found at: {target_path}")

    parquet_file = pq.ParquetFile(str(target_path))
    raw_meta = parquet_file.schema_arrow.metadata or {}
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()}

    df = parquet_file.read().to_pandas()
    schema = _load_feature_store_schema()
    validated = validate_frame(df, schema)

    return validated, meta


__all__ = [
    "DEFAULT_FEATURE_SET_VERSION",
    "SCHEMA_VERSION",
    "InputSnapshot",
    "build_feature_table",
    "build_input_snapshot",
    "compute_config_hash",
    "compute_recent_cams_bias",
    "load_feature_table",
    "load_input_snapshot",
    "save_feature_table",
    "save_input_snapshot",
]
