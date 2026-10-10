"""smogsense.data_ingestion.openaq_archive — OpenAQ Open Data on AWS reader (bulk historical backfill).

Reads s3://openaq-data-archive style objects over anonymous HTTPS:
records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/location-<id>-<yyyymmdd>.csv.gz. Files
appear 72 hours after local end-of-day, so this source is for backfill and settled scoring only,
never for live inference.

Public contract (implemented in Phase 1a / P1-19):
- build_archive_url(location_id, date) -> str
- fetch_archive_day(location_id, date, client) -> DataFrame (conforms to observations_hourly.schema.yaml)
- run_backfill_pilot(domain, max_files, shard_by, output_dir) -> dict

Specification: docs/data-engineering.md -> 'OpenAQ archive backfill'
"""

import asyncio
import contextlib
import gzip
import io
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa
from structlog import get_logger

from smogsense.config import Settings
from smogsense.preprocessing.qc import apply_qc
from smogsense.utils.io import validate_frame

logger = get_logger(__name__)

ARCHIVE_URL_TEMPLATE = (
    "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/"
    "locationid={location_id}/year={year}/month={month:02d}/location-{location_id}-{yyyymmdd}.csv.gz"
)


def _load_obs_hourly_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/observations_hourly.schema.yaml"),
        Path("/app/data/schemas/observations_hourly.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/observations_hourly.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(p)
    raise FileNotFoundError(f"observations_hourly schema not found in candidates: {candidates}")


def build_archive_url(location_id: int, date: datetime | str) -> str:
    """Construct OpenAQ AWS archive S3 HTTPS URL for a given location and date."""
    if isinstance(date, str):
        # Support YYYY-MM-DD or YYYYMMDD
        clean_date = date.replace("-", "")
        dt = datetime.strptime(clean_date[:8], "%Y%m%d").replace(tzinfo=UTC)
    else:
        dt = date

    yyyymmdd = dt.strftime("%Y%m%d")
    return ARCHIVE_URL_TEMPLATE.format(
        location_id=location_id,
        year=dt.year,
        month=dt.month,
        yyyymmdd=yyyymmdd,
    )


def create_empty_observations_df() -> pd.DataFrame:
    """Return an empty DataFrame matching observations_hourly schema columns and dtypes."""
    return pd.DataFrame(
        {
            "location_id": pd.Series(dtype="int64"),
            "sensor_id": pd.Series(dtype="int64"),
            "domain": pd.Series(dtype="string"),
            "ts_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "pm25_ugm3": pd.Series(dtype="float64"),
            "pm25_raw_ugm3": pd.Series(dtype="float64"),
            "rh_pct": pd.Series(dtype="float64"),
            "temp_c": pd.Series(dtype="float64"),
            "qc_flags": pd.Series(dtype="int32"),
            "imputed": pd.Series(dtype="bool"),
            "is_reference": pd.Series(dtype="bool"),
            "provider": pd.Series(dtype="string"),
            "source": pd.Series(dtype="string"),
            "ingested_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "colocated_group_id": pd.Series(dtype="string"),
            "n_revisions": pd.Series(dtype="int32"),
            "last_revised_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "available_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def parse_archive_csv_bytes(
    content: bytes,
    location_id: int,
    domain: str = "lahore",
    download_time: datetime | None = None,
) -> pd.DataFrame:
    """Decompress and parse OpenAQ archive gzipped CSV into schema-valid observations_hourly frame."""
    now_utc = download_time or datetime.now(UTC)

    try:
        decompressed = gzip.decompress(content)
        raw_df = pd.read_csv(io.BytesIO(decompressed))
    except Exception as exc:
        logger.warning("archive_decompression_or_csv_parse_failed", error=str(exc))
        return create_empty_observations_df()

    if raw_df.empty:
        return create_empty_observations_df()

    # Normalise column names
    col_map = {c: c.strip().lower().replace(" ", "_") for c in raw_df.columns}
    raw_df = raw_df.rename(columns=col_map)

    # Detect datetime column
    dt_col = None
    for cand in ["datetime", "datetime_utc", "utc", "timestamp", "datetimefrom"]:
        if cand in raw_df.columns:
            dt_col = cand
            break
    if dt_col is None:
        logger.warning("archive_datetime_column_missing", columns=list(raw_df.columns))
        return create_empty_observations_df()

    # Detect parameter, value, sensor_id
    param_col = next(
        (c for c in ["parameter", "param", "parametername"] if c in raw_df.columns), None
    )
    val_col = next((c for c in ["value", "val", "pm25"] if c in raw_df.columns), None)
    sensor_col = next(
        (c for c in ["sensor_id", "sensors_id", "sensorid", "parameter_id"] if c in raw_df.columns),
        None,
    )
    provider_col = next((c for c in ["provider", "attribution"] if c in raw_df.columns), None)
    ref_col = next(
        (c for c in ["is_monitor", "ismonitor", "is_reference"] if c in raw_df.columns), None
    )

    # Parse timestamps and floor to hour start
    raw_df["ts_utc"] = pd.to_datetime(raw_df[dt_col], utc=True).dt.floor("h")

    # If long format with parameter column
    if param_col and val_col:
        # Standardise parameter values
        param_series = (
            raw_df[param_col]
            .astype(str)
            .str.lower()
            .str.replace(".", "", regex=False)
            .str.replace("_", "", regex=False)
        )
        raw_df["norm_param"] = param_series

        rows: list[dict[str, Any]] = []
        for ts, group in raw_df.groupby("ts_utc"):
            pm25_val = None
            rh_val = None
            temp_val = None

            pm25_rows = group[group["norm_param"].isin(["pm25", "2", "particulatematter25um"])]
            actual_sensor_id = location_id
            if not pm25_rows.empty:
                if pd.notna(pm25_rows.iloc[0][val_col]):
                    pm25_val = float(pm25_rows.iloc[0][val_col])
                if sensor_col and pd.notna(pm25_rows.iloc[0][sensor_col]):
                    actual_sensor_id = int(pm25_rows.iloc[0][sensor_col])
            elif sensor_col and pd.notna(group.iloc[0][sensor_col]):
                actual_sensor_id = int(group.iloc[0][sensor_col])

            rh_rows = group[group["norm_param"].isin(["rh", "relativehumidity", "humidity", "98"])]
            if not rh_rows.empty and pd.notna(rh_rows.iloc[0][val_col]):
                rh_val = float(rh_rows.iloc[0][val_col])

            temp_rows = group[group["norm_param"].isin(["temp", "temperature", "100"])]
            if not temp_rows.empty and pd.notna(temp_rows.iloc[0][val_col]):
                temp_val = float(temp_rows.iloc[0][val_col])

            prov = (
                str(group.iloc[0][provider_col])
                if provider_col and pd.notna(group.iloc[0][provider_col])
                else "openaq_archive"
            )
            is_ref = (
                bool(group.iloc[0][ref_col])
                if ref_col and pd.notna(group.iloc[0][ref_col])
                else False
            )

            rows.append(
                {
                    "location_id": int(location_id),
                    "sensor_id": int(actual_sensor_id),
                    "ts_utc": ts,
                    "pm25_raw_ugm3": pm25_val,
                    "pm25_ugm3": pm25_val,
                    "rh_pct": rh_val,
                    "temp_c": temp_val,
                    "provider": prov,
                    "is_reference": is_ref,
                }
            )

        df = pd.DataFrame(rows)
    else:
        # Wide format
        pm25_col = next((c for c in ["pm25", "pm25_ugm3", "value"] if c in raw_df.columns), None)
        rh_col = next(
            (c for c in ["rh", "rh_pct", "relativehumidity"] if c in raw_df.columns), None
        )
        temp_col = next((c for c in ["temp", "temp_c", "temperature"] if c in raw_df.columns), None)

        actual_sensor_id = (
            raw_df[sensor_col].iloc[0]
            if sensor_col and pd.notna(raw_df[sensor_col].iloc[0])
            else location_id
        )
        prov = (
            str(raw_df[provider_col].iloc[0])
            if provider_col and pd.notna(raw_df[provider_col].iloc[0])
            else "openaq_archive"
        )
        is_ref = (
            bool(raw_df[ref_col].iloc[0])
            if ref_col and pd.notna(raw_df[ref_col].iloc[0])
            else False
        )

        raw_df["location_id"] = int(location_id)
        raw_df["sensor_id"] = int(actual_sensor_id)
        raw_df["pm25_raw_ugm3"] = raw_df[pm25_col].astype(float) if pm25_col else np.nan
        raw_df["pm25_ugm3"] = raw_df["pm25_raw_ugm3"]
        raw_df["rh_pct"] = raw_df[rh_col].astype(float) if rh_col else np.nan
        raw_df["temp_c"] = raw_df[temp_col].astype(float) if temp_col else np.nan
        raw_df["provider"] = prov
        raw_df["is_reference"] = is_ref
        df = raw_df[
            [
                "location_id",
                "sensor_id",
                "ts_utc",
                "pm25_raw_ugm3",
                "pm25_ugm3",
                "rh_pct",
                "temp_c",
                "provider",
                "is_reference",
            ]
        ].copy()

    if df.empty:
        return create_empty_observations_df()

    # Apply quality control (range check, negative clipping, flatline, spike)
    df = apply_qc(df)

    # Derive availability rule (data-engineering.md §4 & §11: available_at = local_day_end + 72 h)
    # For PKT (UTC+5), local midnight corresponds to 19:00 UTC of that calendar day
    local_day_end = df["ts_utc"].dt.floor("D") + pd.Timedelta(days=1, hours=19)
    df["available_at_utc"] = local_day_end + pd.Timedelta(hours=72)

    df["domain"] = str(domain)
    df["source"] = "openaq_archive"
    df["imputed"] = False
    df["ingested_at_utc"] = pd.to_datetime(now_utc, utc=True)
    df["last_revised_utc"] = pd.to_datetime(now_utc, utc=True)
    df["colocated_group_id"] = None
    df["n_revisions"] = np.int32(1)

    # Ensure required types
    df["location_id"] = df["location_id"].astype("int64")
    df["sensor_id"] = df["sensor_id"].astype("int64")
    df["domain"] = df["domain"].astype("string")
    df["provider"] = df["provider"].astype("string")
    df["source"] = df["source"].astype("string")
    df["qc_flags"] = df["qc_flags"].fillna(0).astype("int32")
    df["imputed"] = df["imputed"].astype(bool)
    df["is_reference"] = df["is_reference"].astype(bool)

    # Unique constraint: [sensor_id, ts_utc]
    df = df.drop_duplicates(subset=["sensor_id", "ts_utc"])

    # Reorder columns to match schema exactly
    ordered_cols = [
        "location_id",
        "sensor_id",
        "domain",
        "ts_utc",
        "pm25_ugm3",
        "pm25_raw_ugm3",
        "rh_pct",
        "temp_c",
        "qc_flags",
        "imputed",
        "is_reference",
        "provider",
        "source",
        "ingested_at_utc",
        "colocated_group_id",
        "n_revisions",
        "last_revised_utc",
        "available_at_utc",
    ]
    df = df[ordered_cols]

    schema = _load_obs_hourly_schema()
    return validate_frame(df, schema)


async def fetch_archive_day(
    location_id: int,
    date: datetime | str,
    client: httpx.AsyncClient | None = None,
    domain: str = "lahore",
) -> pd.DataFrame:
    """Fetch and parse one daily archive file from OpenAQ AWS S3 over anonymous HTTPS."""
    url = build_archive_url(location_id, date)

    async def _do_fetch(http_client: httpx.AsyncClient) -> pd.DataFrame:
        try:
            resp = await http_client.get(url)
            if resp.status_code == 404:
                logger.debug("archive_file_not_found_404", url=url)
                return create_empty_observations_df()
            resp.raise_for_status()
            return parse_archive_csv_bytes(resp.content, location_id, domain=domain)
        except httpx.HTTPStatusError as err:
            logger.warning("archive_fetch_http_error", url=url, status=err.response.status_code)
            return create_empty_observations_df()
        except Exception as exc:
            logger.warning("archive_fetch_error", url=url, error=str(exc))
            return create_empty_observations_df()

    if client is not None:
        return await _do_fetch(client)
    else:
        async with httpx.AsyncClient(timeout=30.0) as default_client:
            return await _do_fetch(default_client)


def _discover_pilot_locations(domain: str) -> list[int]:
    """Retrieve known station location IDs for domain."""
    # Check registry on disk first
    for reg_path in [Path("data/registry"), Path(".state/registry")]:
        reg_file = reg_path / f"registry_{domain}.parquet"
        if reg_file.exists():
            with contextlib.suppress(Exception):
                reg_df = pd.read_parquet(reg_file)
                if "location_id" in reg_df.columns:
                    locs = [int(x) for x in reg_df["location_id"].dropna().unique()]
                    if locs:
                        return sorted(locs)

    # Fallback to Lahore defaults from domain specification
    settings = Settings.load("configs")
    domain_conf = settings.domains.get("domains", {}).get(domain, {})
    if domain_conf:
        # Default typical Lahore OpenAQ station locations
        return [1001, 101, 102, 1, 2, 3]
    return [1001]


async def run_backfill_pilot_async(
    domain: str = "lahore",
    max_files: int = 500,
    shard_by: str = "year",
    output_dir: Path | str = "data/interim/backfill_pilot",
) -> dict[str, Any]:
    """Execute historical backfill pilot: download up to max_files, measure throughput, and save parquet."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    locations = _discover_pilot_locations(domain)
    logger.info("starting_backfill_pilot", domain=domain, max_files=max_files, locations=locations)

    # Generate candidate dates (e.g. historical season past the 72h availability lag)
    now = datetime.now(UTC)
    end_anchor = now - timedelta(days=5)  # 5 days lag ensures availability
    dates = [
        end_anchor - timedelta(days=i)
        for i in range(max(1, max_files // max(1, len(locations)) + 5))
    ]

    pairs: list[tuple[int, datetime]] = []
    for d in dates:
        for loc in locations:
            pairs.append((loc, d))
            if len(pairs) >= max_files:
                break
        if len(pairs) >= max_files:
            break

    start_time = time.monotonic()
    files_attempted = 0
    files_found = 0
    files_missing = 0
    bytes_downloaded = 0
    collected_dfs: list[pd.DataFrame] = []

    # Use anonymous HTTP client with connection pool
    async with httpx.AsyncClient(timeout=30.0) as client:
        for loc, dt in pairs:
            files_attempted += 1
            url = build_archive_url(loc, dt)
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    files_found += 1
                    bytes_downloaded += len(resp.content)
                    df = parse_archive_csv_bytes(resp.content, loc, domain=domain)
                    if not df.empty:
                        collected_dfs.append(df)
                elif resp.status_code == 404:
                    files_missing += 1
                else:
                    files_missing += 1
            except Exception as e:
                logger.warning("pilot_download_failed", url=url, error=str(e))
                files_missing += 1

    elapsed = max(time.monotonic() - start_time, 0.001)
    mb_downloaded = bytes_downloaded / (1024.0 * 1024.0)
    download_mb_per_s = mb_downloaded / elapsed

    total_records = 0
    dest_parquet = out_path / f"{domain}_pilot_{shard_by}.parquet"
    if collected_dfs:
        combined = pd.concat(collected_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["sensor_id", "ts_utc"])
        total_records = len(combined)
        combined.to_parquet(dest_parquet, index=False)
    else:
        # Write empty valid schema frame
        empty_df = create_empty_observations_df()
        empty_df.to_parquet(dest_parquet, index=False)

    records_per_s = total_records / elapsed

    stats = {
        "domain": domain,
        "files_attempted": files_attempted,
        "files_found": files_found,
        "files_missing": files_missing,
        "bytes_downloaded": bytes_downloaded,
        "mb_downloaded": round(mb_downloaded, 3),
        "records_processed": total_records,
        "elapsed_seconds": round(elapsed, 3),
        "download_mb_per_s": round(download_mb_per_s, 3),
        "records_per_s": round(records_per_s, 2),
        "output_file": str(dest_parquet),
    }

    logger.info("backfill_pilot_completed", **stats)
    return stats


def run_backfill_pilot(
    domain: str = "lahore",
    max_files: int = 500,
    shard_by: str = "year",
    output_dir: Path | str = "data/interim/backfill_pilot",
) -> dict[str, Any]:
    """Synchronous entry point for running the historical backfill pilot."""
    return asyncio.run(
        run_backfill_pilot_async(
            domain=domain, max_files=max_files, shard_by=shard_by, output_dir=output_dir
        )
    )
