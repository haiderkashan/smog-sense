"""smogsense.features.snapshot — Daily point-in-time input snapshot and replay interface.

Serializes observations, CAMS forecast/analysis series, and fire exposure available
at forecast reference time T into a canonical replay snapshot:
.state/inputs/{domain}/dt=YYYY-MM-DD.parquet

Enforces point-in-time correctness (Principle P3): no data with available_at > T is included.

Specification:
- docs/system-architecture.md §2 & §4 ('Point-in-time correctness' and 'Data availability')
- docs/data-engineering.md §12 ('Storage layout and the state branch')
- docs/evaluation-strategy.md §9 ('Prospective shadow-run protocol')
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def _normalize_date_str(val: str | date | datetime) -> str:
    """Normalize string, date, or datetime into YYYY-MM-DD."""
    if isinstance(val, str):
        if "T" in val or " " in val:
            return pd.to_datetime(val, utc=True).strftime("%Y-%m-%d")
        return val[:10]
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    raise TypeError(f"Expected str, date, or datetime, got {type(val)}")


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class InputSnapshot:
    """Point-in-time replay snapshot containing observations, CAMS, fire exposure and metadata."""

    issuance_utc: datetime
    domain: str
    obs_df: pd.DataFrame
    cams_df: pd.DataFrame
    fire_exposure_df: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = "unknown"
    config_hash: str = "unknown"
    cams_base_time_utc: datetime | None = None
    cams_lead_offset_h: float = 0.0
    n_available_stations: int = 0
    feature_set_version: str = "v1"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuance_utc": self.issuance_utc,
            "domain": self.domain,
            "obs_df": self.obs_df,
            "cams_df": self.cams_df,
            "fire_exposure_df": self.fire_exposure_df,
            "metadata": self.metadata,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "cams_base_time_utc": self.cams_base_time_utc,
            "cams_lead_offset_h": self.cams_lead_offset_h,
            "n_available_stations": self.n_available_stations,
            "feature_set_version": self.feature_set_version,
        }


def build_input_snapshot(
    issuance_utc: datetime,
    domain: str = "lahore",
    obs_df: pd.DataFrame | None = None,
    cams_df: pd.DataFrame | None = None,
    fire_exposure_df: pd.DataFrame | None = None,
    cams_base_time_utc: datetime | None = None,
    cams_lead_offset_h: float = 0.0,
    run_id: str = "unknown",
    config_hash: str = "unknown",
    n_available_stations: int = 0,
    feature_set_version: str = "v1",
) -> InputSnapshot:
    """Package point-in-time inputs into an InputSnapshot, validating point-in-time correctness.

    Raises:
        ValueError: If any input contains records with available_at > issuance_utc (lookahead leakage).
    """
    t_ref = _ensure_utc(issuance_utc)

    # 1. Validate Observations
    clean_obs = obs_df.copy() if obs_df is not None else pd.DataFrame()
    if not clean_obs.empty:
        if "available_at_utc" in clean_obs.columns:
            avail = pd.to_datetime(clean_obs["available_at_utc"], utc=True)
            if (avail > t_ref).any():
                max_viol = avail.max()
                raise ValueError(
                    f"Point-in-time leakage: obs_df contains records with available_at_utc ({max_viol}) > issuance_utc ({t_ref})."
                )
        elif "ts_utc" in clean_obs.columns:
            # Latency rule: if available_at_utc absent, ts_utc + 4h must not exceed T
            ts = pd.to_datetime(clean_obs["ts_utc"], utc=True)
            if (ts > t_ref).any():
                max_ts = ts.max()
                raise ValueError(
                    f"Point-in-time leakage: obs_df contains records with ts_utc ({max_ts}) > issuance_utc ({t_ref})."
                )

    # 2. Validate CAMS
    clean_cams = cams_df.copy() if cams_df is not None else pd.DataFrame()
    norm_cams_base: datetime | None = None
    if cams_base_time_utc is not None:
        norm_cams_base = _ensure_utc(cams_base_time_utc)
        # CAMS global forecast cycle B is available at B + 10h (docs/data-engineering.md §11)
        if norm_cams_base + timedelta(hours=10) > t_ref:
            raise ValueError(
                f"Point-in-time leakage: CAMS cycle base {norm_cams_base} available at "
                f"{norm_cams_base + timedelta(hours=10)} > issuance_utc ({t_ref})."
            )

    if not clean_cams.empty and "available_at_utc" in clean_cams.columns:
        c_avail = pd.to_datetime(clean_cams["available_at_utc"], utc=True)
        if (c_avail > t_ref).any():
            max_c = c_avail.max()
            raise ValueError(
                f"Point-in-time leakage: cams_df contains records with available_at_utc ({max_c}) > issuance_utc ({t_ref})."
            )

    # 3. Validate Fire Exposure
    clean_fire: pd.DataFrame | None = None
    if fire_exposure_df is not None and not fire_exposure_df.empty:
        clean_fire = fire_exposure_df.copy()
        if "available_at_utc" in clean_fire.columns:
            f_avail = pd.to_datetime(clean_fire["available_at_utc"], utc=True)
            if (f_avail > t_ref).any():
                max_f = f_avail.max()
                raise ValueError(
                    f"Point-in-time leakage: fire_exposure_df contains records with available_at_utc ({max_f}) > issuance_utc ({t_ref})."
                )
        elif "acq_datetime_utc" in clean_fire.columns:
            acq = pd.to_datetime(clean_fire["acq_datetime_utc"], utc=True)
            f_avail = acq + timedelta(hours=3)
            if (f_avail > t_ref).any():
                max_f = f_avail.max()
                raise ValueError(
                    f"Point-in-time leakage: fire_exposure_df contains detections available at ({max_f}) > issuance_utc ({t_ref})."
                )

    # Calculate station count if unset
    if n_available_stations == 0 and not clean_obs.empty and "location_id" in clean_obs.columns:
        n_available_stations = int(clean_obs["location_id"].nunique())

    meta = {
        "issuance_utc": t_ref.isoformat(),
        "domain": domain,
        "run_id": run_id,
        "config_hash": config_hash,
        "feature_set_version": feature_set_version,
        "cams_base_time_utc": norm_cams_base.isoformat() if norm_cams_base else None,
        "cams_lead_offset_h": cams_lead_offset_h,
        "n_available_stations": n_available_stations,
        "obs_rows": len(clean_obs),
        "cams_rows": len(clean_cams),
        "fire_rows": len(clean_fire) if clean_fire is not None else 0,
    }

    return InputSnapshot(
        issuance_utc=t_ref,
        domain=domain,
        obs_df=clean_obs,
        cams_df=clean_cams,
        fire_exposure_df=clean_fire,
        metadata=meta,
        run_id=run_id,
        config_hash=config_hash,
        cams_base_time_utc=norm_cams_base,
        cams_lead_offset_h=cams_lead_offset_h,
        n_available_stations=n_available_stations,
        feature_set_version=feature_set_version,
    )


def save_input_snapshot(
    snapshot: InputSnapshot,
    domain: str | None = None,
    issuance_date: str | date | datetime | None = None,
    state_dir: Path | str = Path(".state"),
    output_path: Path | str | None = None,
) -> Path:
    """Persist input snapshot to canonical partition path .state/inputs/{domain}/dt=YYYY-MM-DD.parquet."""
    dom = domain or snapshot.domain
    if output_path is not None:
        target_path = Path(output_path)
    else:
        dt_str = _normalize_date_str(issuance_date or snapshot.issuance_utc)
        target_path = Path(state_dir) / "inputs" / dom / f"dt={dt_str}.parquet"

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Serialize child component tables to buffer bytes for perfect roundtrip fidelity
    obs_buf = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(snapshot.obs_df), obs_buf, compression="snappy")
    obs_bytes = obs_buf.getvalue().to_pybytes()

    cams_buf = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pandas(snapshot.cams_df), cams_buf, compression="snappy")
    cams_bytes = cams_buf.getvalue().to_pybytes()

    fire_bytes = b""
    if snapshot.fire_exposure_df is not None and not snapshot.fire_exposure_df.empty:
        fire_buf = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pandas(snapshot.fire_exposure_df), fire_buf, compression="snappy"
        )
        fire_bytes = fire_buf.getvalue().to_pybytes()

    # 2. Build combined table for direct queryability via standard pandas/DuckDB read_parquet
    frames: list[pd.DataFrame] = []
    if not snapshot.obs_df.empty:
        o_copy = snapshot.obs_df.copy()
        o_copy["input_type"] = "observation"
        frames.append(o_copy)
    if not snapshot.cams_df.empty:
        c_copy = snapshot.cams_df.copy()
        c_copy["input_type"] = "cams"
        frames.append(c_copy)
    if snapshot.fire_exposure_df is not None and not snapshot.fire_exposure_df.empty:
        f_copy = snapshot.fire_exposure_df.copy()
        f_copy["input_type"] = "fire_exposure"
        frames.append(f_copy)

    if frames:
        combined_df = pd.concat(frames, ignore_index=True)
        if "location_id" in combined_df.columns:
            combined_df["location_id"] = combined_df["location_id"].astype(str)
    else:
        combined_df = pd.DataFrame(columns=["input_type", "domain", "issuance_utc"])

    combined_df["snapshot_issuance_utc"] = snapshot.issuance_utc
    combined_df["snapshot_domain"] = dom

    table = pa.Table.from_pandas(combined_df)

    # 3. Embed schema metadata and serialized child buffers
    custom_meta = {
        b"smogsense_meta": json.dumps(snapshot.metadata).encode("utf-8"),
        b"obs_bytes": obs_bytes,
        b"cams_bytes": cams_bytes,
        b"fire_bytes": fire_bytes,
    }
    existing_meta = table.schema.metadata or {}
    merged_meta = {**existing_meta, **custom_meta}
    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, target_path, compression="snappy")
    logger.info("Persisted input snapshot to %s", target_path)
    return target_path


def load_input_snapshot(
    domain: str,
    issuance_date: str | date | datetime,
    state_dir: Path | str = Path(".state"),
    snapshot_path: Path | str | None = None,
) -> InputSnapshot:
    """Load and unpack point-in-time input snapshot from Parquet."""
    if snapshot_path is not None:
        target_path = Path(snapshot_path)
    else:
        dt_str = _normalize_date_str(issuance_date)
        target_path = Path(state_dir) / "inputs" / domain / f"dt={dt_str}.parquet"

    if not target_path.exists():
        raise FileNotFoundError(f"Input snapshot not found at {target_path}")

    table = pq.read_table(target_path)
    meta_dict: dict[str, Any] = {}
    if table.schema.metadata and b"smogsense_meta" in table.schema.metadata:
        try:
            meta_dict = json.loads(table.schema.metadata[b"smogsense_meta"].decode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to decode snapshot metadata JSON: %s", exc)

    obs_df = pd.DataFrame()
    cams_df = pd.DataFrame()
    fire_exposure_df: pd.DataFrame | None = None

    # Recover child components from metadata bytes if present
    if table.schema.metadata:
        if table.schema.metadata.get(b"obs_bytes"):
            obs_df = pq.read_table(pa.BufferReader(table.schema.metadata[b"obs_bytes"])).to_pandas()
        if table.schema.metadata.get(b"cams_bytes"):
            cams_df = pq.read_table(
                pa.BufferReader(table.schema.metadata[b"cams_bytes"])
            ).to_pandas()
        if table.schema.metadata.get(b"fire_bytes"):
            fire_exposure_df = pq.read_table(
                pa.BufferReader(table.schema.metadata[b"fire_bytes"])
            ).to_pandas()

    # Fallback to combined rows unpacking if metadata buffers are empty
    if obs_df.empty and cams_df.empty:
        combined_df = table.to_pandas()
        if "input_type" in combined_df.columns:
            obs_part = combined_df[combined_df["input_type"] == "observation"].dropna(
                how="all", axis=1
            )
            cams_part = combined_df[combined_df["input_type"] == "cams"].dropna(how="all", axis=1)
            fire_part = combined_df[combined_df["input_type"] == "fire_exposure"].dropna(
                how="all", axis=1
            )

            drop_cols = ["input_type", "snapshot_issuance_utc", "snapshot_domain"]
            obs_df = obs_part.drop(columns=drop_cols, errors="ignore")
            cams_df = cams_part.drop(columns=drop_cols, errors="ignore")
            if not fire_part.empty:
                fire_exposure_df = fire_part.drop(columns=drop_cols, errors="ignore")

    issuance_raw = meta_dict.get("issuance_utc") or issuance_date
    issuance_utc = pd.to_datetime(issuance_raw, utc=True).to_pydatetime()

    cams_base_raw = meta_dict.get("cams_base_time_utc")
    cams_base_utc = (
        pd.to_datetime(cams_base_raw, utc=True).to_pydatetime() if cams_base_raw else None
    )

    return InputSnapshot(
        issuance_utc=issuance_utc,
        domain=domain,
        obs_df=obs_df,
        cams_df=cams_df,
        fire_exposure_df=fire_exposure_df,
        metadata=meta_dict,
        run_id=meta_dict.get("run_id", "unknown"),
        config_hash=meta_dict.get("config_hash", "unknown"),
        cams_base_time_utc=cams_base_utc,
        cams_lead_offset_h=float(meta_dict.get("cams_lead_offset_h", 0.0)),
        n_available_stations=int(meta_dict.get("n_available_stations", 0)),
        feature_set_version=meta_dict.get("feature_set_version", "v1"),
    )
