"""smogsense.pipeline.orchestrator - Daily pipeline orchestrator.

Specification: docs/system-architecture.md
"""

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pandera.pandas as pa

from smogsense.config import Settings
from smogsense.data_ingestion.base import (
    CircuitBreaker,
    CircuitBreakerError,
    RateBudget,
    ResilientClient,
    create_client,
)
from smogsense.data_ingestion.copernicus import CamsClient
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations
from smogsense.errors import QuotaExceeded
from smogsense.inference.degradation import determine_mode, get_degradation_level
from smogsense.models.baselines import M0Persistence, M1Cams
from smogsense.models.distribution import QuantileFunction
from smogsense.preprocessing.alignment import asof_cams_run
from smogsense.preprocessing.gridded import extract_stations
from smogsense.preprocessing.qc import apply_qc
from smogsense.publishing.bulletin import generate_bulletin_json
from smogsense.publishing.site import generate_site
from smogsense.utils.io import get_git_sha, validate_frame

logger = logging.getLogger(__name__)


def generate_forecast_log(
    issuance_utc: datetime,
    cams_base_time: datetime | None,
    mode: str,
    baseline_used: str,
    stations_data: list[dict[str, Any]],
    city_forecasts: dict[int, list[float]],
    cams_lead_offset_h: float = 0.0,
    run_id: str = "unknown",
    git_sha: str = "unknown",
    config_hash: str = "unknown",
    data_cutoff_utc: datetime | None = None,
    n_available_stations: int = 0,
    is_rerun: bool = False,
    adaptation_status: str = "skipped_n0",
) -> pd.DataFrame:
    """Generate the forecast log dataframe validated against Pandera schema."""
    now_utc = datetime.now(UTC)
    rows = []

    for s in stations_data:
        point_id = f"station:{s['location_id']}"
        station_cutoff = s.get("data_cutoff_utc") or data_cutoff_utc
        for h, qf_vals in s["forecasts"].items():
            row = {
                "issuance_utc": issuance_utc,
                "generated_at_utc": now_utc,
                "domain": "lahore",
                "level": "station",
                "point_id": point_id,
                "horizon_h": h,
                "window_start_utc": issuance_utc + pd.Timedelta(hours=h - 24),
                "window_end_utc": issuance_utc + pd.Timedelta(hours=h),
                "method": baseline_used,
                "model_version": "baseline",
                "mode": mode,
                "cams_base_time_utc": cams_base_time,
                "data_cutoff_utc": station_cutoff,
                "git_sha": git_sha,
                "config_hash": config_hash,
                "calibrated": False,
                "run_id": run_id,
                "is_rerun": is_rerun,
                "adaptation_status": adaptation_status,
                "cams_lead_offset_h": cams_lead_offset_h,
                "n_available_stations": n_available_stations,
            }
            for q_idx, q_level in enumerate(
                [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
            ):
                row[f"q{q_level:02d}"] = float(qf_vals[q_idx])
            rows.append(row)

    for h, qf_vals in city_forecasts.items():
        row = {
            "issuance_utc": issuance_utc,
            "generated_at_utc": now_utc,
            "domain": "lahore",
            "level": "city",
            "point_id": "city:lahore",
            "horizon_h": h,
            "window_start_utc": issuance_utc + pd.Timedelta(hours=h - 24),
            "window_end_utc": issuance_utc + pd.Timedelta(hours=h),
            "method": baseline_used,
            "model_version": "baseline",
            "mode": mode,
            "cams_base_time_utc": cams_base_time,
            "data_cutoff_utc": data_cutoff_utc,
            "git_sha": git_sha,
            "config_hash": config_hash,
            "calibrated": False,
            "run_id": run_id,
            "is_rerun": is_rerun,
            "adaptation_status": adaptation_status,
            "cams_lead_offset_h": cams_lead_offset_h,
            "n_available_stations": n_available_stations,
        }
        for q_idx, q_level in enumerate(
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        ):
            row[f"q{q_level:02d}"] = float(qf_vals[q_idx])
        rows.append(row)

    df = pd.DataFrame(rows)
    schema = load_pandera_schema("forecast_log.schema.yaml")
    return validate_frame(df, schema)


def load_pandera_schema(schema_name: str) -> pa.DataFrameSchema:
    candidates = [
        Path(f"data/schemas/{schema_name}"),
        Path(f"/app/data/schemas/{schema_name}"),
        Path(__file__).resolve().parents[3] / f"data/schemas/{schema_name}",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(p)
    raise FileNotFoundError(f"Schema {schema_name} not found in candidate paths: {candidates}")


def record_obs_pull_log(
    raw_obs_df: pd.DataFrame,
    issuance_utc: datetime,
    pull_id: str,
    domain: str = "lahore",
) -> Path:
    """Record first-seen observations into partitioned Parquet log conforming to obs_pull_log.schema.yaml."""
    issuance_date = issuance_utc.strftime("%Y-%m-%d")
    log_dir = Path(".state/obs_pull_log")
    dest_path = log_dir / f"dt={issuance_date}.parquet"

    if raw_obs_df.empty:
        return dest_path

    rows: list[dict[str, Any]] = []
    for _, r in raw_obs_df.iterrows():
        val_raw = float(r["pm25_ugm3"]) if pd.notna(r.get("pm25_ugm3")) else None
        rh = float(r["rh_pct"]) if pd.notna(r.get("rh_pct")) else None
        temp = None
        if pd.notna(r.get("temperature_c")):
            temp = float(r["temperature_c"])
        elif pd.notna(r.get("temp_c")):
            temp = float(r["temp_c"])

        ts_utc = pd.to_datetime(r["ts_utc"], utc=True)
        hour_end_utc = (
            pd.to_datetime(r["hour_end_utc"], utc=True)
            if pd.notna(r.get("hour_end_utc"))
            else ts_utc + pd.Timedelta(hours=1)
        )
        first_seen_utc = (
            pd.to_datetime(r["first_seen_utc"], utc=True)
            if pd.notna(r.get("first_seen_utc"))
            else pd.to_datetime(datetime.now(UTC), utc=True)
        )

        rows.append(
            {
                "sensor_id": str(r.get("sensor_id") or r["location_id"]),
                "location_id": str(r["location_id"]),
                "domain": domain,
                "ts_utc": ts_utc,
                "hour_end_utc": hour_end_utc,
                "first_seen_utc": first_seen_utc,
                "pull_id": str(r.get("pull_id") or pull_id),
                "value_raw": val_raw,
                "rh_pct": rh,
                "temp_c": temp,
                "provider": str(r.get("provider") or "openaq"),
                "source": "openaq_api",
            }
        )

    log_df = pd.DataFrame(rows)
    # Deduplicate within batch
    log_df = log_df.drop_duplicates(subset=["sensor_id", "ts_utc", "pull_id"])

    if dest_path.exists():
        with contextlib.suppress(Exception):
            existing_df = pd.read_parquet(dest_path)
            log_df = pd.concat([existing_df, log_df], ignore_index=True)
            log_df = log_df.drop_duplicates(subset=["sensor_id", "ts_utc", "pull_id"])

    schema = load_pandera_schema("obs_pull_log.schema.yaml")
    validated = validate_frame(log_df, schema)
    log_dir.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(dest_path, index=False)
    return dest_path


def write_manifest(
    path: Path,
    run_id: str,
    issuance_utc: datetime,
    published: bool,
    is_rerun: bool,
    mode: str,
    level: int,
    exit_code: int,
    git_sha: str = "unknown",
    config_hash: str = "unknown",
    adaptation_status: str = "skipped_n0",
) -> None:
    manifest = {
        "run_id": run_id,
        "issued_at_utc": issuance_utc.isoformat(timespec="seconds"),
        "is_rerun": is_rerun,
        "published": published,
        "adaptation_status": adaptation_status,
        "observed_latency": 0.0,
        "degradation_mode": mode,
        "degradation_level": level,
        "exit_code": exit_code,
        "git_sha": git_sha,
        "config_hash": config_hash,
    }
    manifest_dir = path.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Timestamped run manifest
    run_file = manifest_dir / f"{run_id}.json"
    with run_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 2. Canonical date manifest (YYYY-MM-DD.json per docs/data-engineering.md §14)
    date_file = manifest_dir / f"{issuance_utc.strftime('%Y-%m-%d')}.json"
    with date_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # If original target path differs from both, write to it as well
    if path != run_file and path != date_file:
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


async def run_daily_pipeline_async(
    issuance_utc: datetime, force: bool = False, mode: str | None = None
) -> int:
    """Execute daily forecast pipeline for the given issuance time."""
    settings = Settings.load("configs")

    run_id = f"run_{issuance_utc.strftime('%Y%m%d_%H%M')}"
    manifest_dir = Path(".state/manifests")
    manifest_file = manifest_dir / f"{run_id}.json"
    canonical_manifest_file = manifest_dir / f"{issuance_utc.strftime('%Y-%m-%d')}.json"

    git_sha = get_git_sha()
    config_hash = settings.hash()

    is_rerun = False

    # Check for previous publication using both run_id and canonical date manifests
    for prev_candidate in [manifest_file, canonical_manifest_file]:
        if prev_candidate.exists():
            with contextlib.suppress(Exception):
                with prev_candidate.open("r", encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("published") is True:
                    if not force:
                        logger.info("Already published. Use --force.")
                        return 11
                    else:
                        is_rerun = True
                        run_id = f"run_{issuance_utc.strftime('%Y%m%d_%H%M')}_rerun_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
                        manifest_file = manifest_dir / f"{run_id}.json"
                        break

    has_obs = False
    obs_df = pd.DataFrame()
    stations_df = pd.DataFrame()
    has_cams = False
    cams_is_stale = False
    cams_base: datetime | None = None
    cams_df = pd.DataFrame()

    fixtures_mode = mode == "fixtures"

    if fixtures_mode:
        from smogsense.data_ingestion.fixtures_mode import (
            load_fixture_cams,
            load_fixture_locations,
            load_fixture_observations,
        )

        stations_df = load_fixture_locations("lahore")
        end_utc = issuance_utc - timedelta(hours=4)
        start_utc = end_utc - timedelta(hours=72)
        raw_obs_df = load_fixture_observations(
            stations_df, start_utc, end_utc, pull_id=run_id, domain="lahore"
        )
        if not raw_obs_df.empty:
            record_obs_pull_log(raw_obs_df, issuance_utc, run_id, domain="lahore")
            qc_df = apply_qc(raw_obs_df)
            invalid_mask = qc_df["qc_flags"].fillna(0).astype(int) & (1 | 4 | 8) > 0
            valid_obs = qc_df[(~invalid_mask) & qc_df["pm25_ugm3"].notna()].copy()
            if not valid_obs.empty:
                has_obs = True
                obs_df = valid_obs
                snap_dir = Path(".state/inputs/lahore")
                snap_dir.mkdir(parents=True, exist_ok=True)
                obs_df.to_parquet(
                    snap_dir / f"obs_snapshot_{issuance_utc.strftime('%Y%m%d_%H%M')}.parquet",
                    index=False,
                )

        cams_base = asof_cams_run(issuance_utc)
        cams_df = load_fixture_cams(stations_df, issuance_utc)
        cams_df["location_id"] = cams_df["location_id"].astype(str)
        has_cams = True
        snap_dir = Path(".state/inputs/lahore")
        snap_dir.mkdir(parents=True, exist_ok=True)
        cams_df.to_parquet(
            snap_dir / f"cams_snapshot_{issuance_utc.strftime('%Y%m%d_%H%M')}.parquet",
            index=False,
        )
    else:
        api_key = settings.openaq_api_key.get_secret_value() if settings.openaq_api_key else None
        ac = create_client(api_key=api_key)
        async with ac:
            rb = RateBudget(scope="operations")
            cb = CircuitBreaker()
            client = ResilientClient(ac, rb, cb)
            try:
                stations_df = await list_locations(client, "lahore")
                if not stations_df.empty:
                    # Point-in-time as-of rule (assumed 3h latency + 1h block end)
                    # Anchoring to T - 4h ensures 00:17, 02:47, and 05:47 retries see identical observations
                    end_utc = issuance_utc - timedelta(hours=4)
                    start_utc = end_utc - timedelta(hours=72)
                    raw_obs_df = await fetch_hourly(
                        client, stations_df, start_utc, end_utc, pull_id=run_id
                    )
                    if not raw_obs_df.empty:
                        # Task P1-18: Latency capture and obs_pull_log recording
                        record_obs_pull_log(raw_obs_df, issuance_utc, run_id, domain="lahore")

                        qc_df = apply_qc(raw_obs_df)
                        # Exclude invalid observations: RANGE_REJECT(1), FLATLINE(4), SPIKE(8)
                        # Keep NEGATIVE_CLIPPED(2) as valid 0.0 values
                        invalid_mask = qc_df["qc_flags"].fillna(0).astype(int) & (1 | 4 | 8) > 0
                        valid_obs = qc_df[(~invalid_mask) & qc_df["pm25_ugm3"].notna()].copy()
                        if not valid_obs.empty:
                            has_obs = True
                            obs_df = valid_obs
                            # Persist observation snapshot under .state/inputs/lahore
                            snap_dir = Path(".state/inputs/lahore")
                            snap_dir.mkdir(parents=True, exist_ok=True)
                            obs_df.to_parquet(
                                snap_dir
                                / f"obs_snapshot_{issuance_utc.strftime('%Y%m%d_%H%M')}.parquet",
                                index=False,
                            )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.error("OpenAQ quota exceeded (HTTP 429). Exiting with code 40.")
                    write_manifest(
                        manifest_file,
                        run_id,
                        issuance_utc,
                        False,
                        is_rerun,
                        "unknown",
                        4,
                        40,
                        git_sha=git_sha,
                        config_hash=config_hash,
                    )
                    return 40
                elif e.response.status_code in (401, 403):
                    logger.warning(f"OpenAQ authentication failed ({e.response.status_code}): {e}")
                else:
                    logger.warning(f"OpenAQ HTTP error: {e}")
            except (QuotaExceeded, CircuitBreakerError):
                logger.error("Rate budget / circuit breaker exhausted. Exiting with code 40.")
                write_manifest(
                    manifest_file,
                    run_id,
                    issuance_utc,
                    False,
                    is_rerun,
                    "unknown",
                    4,
                    40,
                    git_sha=git_sha,
                    config_hash=config_hash,
                )
                return 40
            except Exception as e:
                logger.warning(f"OpenAQ fetch failed: {e}")

        # Fallback to cached station registry if OpenAQ failed but registry exists
        if stations_df.empty:
            cached_registry_files = list(Path(".state/registry").glob("*.parquet")) + list(
                Path("data").glob("*registry*.parquet")
            )
            if cached_registry_files:
                with contextlib.suppress(Exception):
                    stations_df = pd.read_parquet(cached_registry_files[0])

        try:
            cams_base = asof_cams_run(issuance_utc)
            cams_conf = settings.sources if "ads" in settings.sources else settings.model_dump()
            cams_client = CamsClient(cams_conf)
            # CAMS GRIB files stored in data/raw/cams, outside the .state git branch
            cams_path = Path("data/raw/cams") / f"cams_{cams_base.strftime('%Y%m%d_%H')}.grib"

            if not cams_path.exists():
                cams_path.parent.mkdir(parents=True, exist_ok=True)
                # 24-hour block mean requires hourly leads covering all 3 horizons [T, T+72)
                all_target_hours = [issuance_utc + timedelta(hours=i) for i in range(72)]
                cams_leadtimes = sorted(
                    {int((t - cams_base).total_seconds() / 3600.0) for t in all_target_hours}
                )
                bbox = settings.domains["domains"]["lahore"]["station_bbox"]
                # 0.5° margin around station_bbox ensures proper grid coverage for 0.4° interpolation
                cams_area = [
                    bbox["north"] + 0.5,
                    bbox["west"] - 0.5,
                    bbox["south"] - 0.5,
                    bbox["east"] + 0.5,
                ]

                print(
                    f"[DEBUG-TRACE] orchestrator: calling cams_client.fetch_cams for {cams_path}",
                    flush=True,
                )
                cams_client.fetch_cams(
                    base_time=cams_base,
                    leadtime_hours=cams_leadtimes,
                    variables=["particulate_matter_2.5um"],
                    area=cams_area,
                    dest_path=cams_path,
                )
                print(
                    f"[DEBUG-TRACE] orchestrator: cams_client.fetch_cams completed, cams_path exists={cams_path.exists()}",
                    flush=True,
                )

            # Allow Level 2 (CAMS only / centroid fallback) even if stations_df is empty
            if cams_path.exists():
                print(
                    f"[DEBUG-TRACE] orchestrator: calling extract_stations on {cams_path} (size={cams_path.stat().st_size})",
                    flush=True,
                )
                logger.info(
                    "Extracting station series from CAMS GRIB %s (size %d bytes)",
                    cams_path,
                    cams_path.stat().st_size,
                )
                cams_df = extract_stations(
                    cams_path, stations_df, domain="lahore", settings=settings.model_dump()
                )
                print(
                    f"[DEBUG-TRACE] orchestrator: extract_stations returned {len(cams_df)} rows",
                    flush=True,
                )
                if not cams_df.empty:
                    logger.info(
                        "Extracted %d CAMS records across %d locations",
                        len(cams_df),
                        cams_df["location_id"].nunique(),
                    )
                    cams_df["location_id"] = cams_df["location_id"].astype(str)
                    has_cams = True
                    # Persist point-in-time input snapshot as Parquet under .state/inputs/
                    snap_dir = Path(".state/inputs/lahore")
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    cams_df.to_parquet(
                        snap_dir / f"cams_snapshot_{issuance_utc.strftime('%Y%m%d_%H%M')}.parquet",
                        index=False,
                    )
                    print(
                        "[DEBUG-TRACE] orchestrator: cams snapshot written to parquet", flush=True
                    )
                    import gc

                    print("[DEBUG-TRACE] orchestrator: running gc.collect()", flush=True)
                    gc.collect()
                    print(
                        "[DEBUG-TRACE] orchestrator: gc.collect() completed successfully",
                        flush=True,
                    )
        except Exception as e:
            print(
                f"[DEBUG-TRACE] orchestrator: CAMS fetch/extract failed with exception: {e}",
                flush=True,
            )
            logger.warning(f"CAMS fetch failed: {e}")

    print(
        f"[DEBUG-TRACE] orchestrator: STEP 1: determine_mode starting (has_obs={has_obs}, has_cams={has_cams}, cams_is_stale={cams_is_stale})",
        flush=True,
    )
    try:
        mode = determine_mode(has_obs, has_cams, cams_is_stale, False)
        degrad_level = get_degradation_level(mode)
        print(
            f"[DEBUG-TRACE] orchestrator: STEP 2: determine_mode done (mode={mode}, level={degrad_level})",
            flush=True,
        )
    except RuntimeError as e:
        print(f"[DEBUG-TRACE] orchestrator: determine_mode raised RuntimeError: {e}", flush=True)
        logger.error(str(e))
        write_manifest(
            manifest_file,
            run_id,
            issuance_utc,
            False,
            is_rerun,
            "unknown",
            4,
            20,
            git_sha=git_sha,
            config_hash=config_hash,
        )
        return 20

    baseline_used = "m1_cams_raw" if has_cams else "m0_persistence"
    stations_data: list[dict[str, Any]] = []
    cams_lead_offset_h = 0.0
    if has_cams and cams_base:
        cams_lead_offset_h = float((issuance_utc - cams_base).total_seconds() / 3600.0)

    # Real data cutoff and available station count
    real_data_cutoff: datetime | None = None
    if not obs_df.empty:
        raw_max = obs_df["ts_utc"].max()
        if hasattr(raw_max, "to_pydatetime"):
            real_data_cutoff = raw_max.to_pydatetime()
        elif isinstance(raw_max, datetime):
            real_data_cutoff = raw_max
    n_available_stations = len(obs_df["location_id"].unique()) if not obs_df.empty else 0

    try:
        seed_history: dict[str, Any] = {}
        m0_seed_history: dict[str, Any] = {}
        seed_path = Path(".state/artifacts/seed_history.json")
        if seed_path.exists():
            with seed_path.open("r", encoding="utf-8") as f:
                seed_data = json.load(f)
                seed_history = seed_data.get("quantiles", {})
                m0_seed_history = seed_data.get("m0_quantiles", {})

        print(
            f"[DEBUG-TRACE] orchestrator: STEP 3: starting inference (baseline={baseline_used}, stations={len(stations_df) if not stations_df.empty else 0})",
            flush=True,
        )
        if baseline_used == "m1_cams_raw":
            if has_cams and not cams_df.empty:
                for idx_s, (_, s) in enumerate(stations_df.iterrows()):
                    loc_id = s["location_id"]
                    print(
                        f"[DEBUG-TRACE] orchestrator: processing station {idx_s + 1}/{len(stations_df)}: loc_id={loc_id}",
                        flush=True,
                    )
                    s_name = s.get("name", str(loc_id))
                    lat = float(s["lat"])
                    lon = float(s["lon"])
                    is_ref = bool(s.get("is_monitor") is True)

                    # Calculate per-station data cutoff
                    station_cutoff = real_data_cutoff
                    if not obs_df.empty:
                        st_obs = obs_df[obs_df["location_id"].astype(str) == str(loc_id)]
                        if not st_obs.empty:
                            st_max = st_obs["ts_utc"].max()
                            if hasattr(st_max, "to_pydatetime"):
                                station_cutoff = st_max.to_pydatetime()
                            elif isinstance(st_max, datetime):
                                station_cutoff = st_max

                    cams_station = cams_df[cams_df["location_id"] == str(loc_id)]
                    if cams_station.empty:
                        continue

                    forecasts = {}
                    for h in [24, 48, 72]:
                        block_hours = [
                            issuance_utc + timedelta(hours=h - 24 + i) for i in range(24)
                        ]
                        if "target_hour_utc" in cams_station.columns:
                            match = cams_station[cams_station["target_hour_utc"].isin(block_hours)]
                            if match.empty:
                                target = issuance_utc + timedelta(hours=h)
                                match = cams_station[cams_station["target_hour_utc"] == target]
                        else:
                            match = pd.DataFrame()

                        if match.empty:
                            continue

                        # 24-hour block mean
                        val = float(match["pm25_ugm3"].mean())

                        q = seed_history.get(str(loc_id), {}).get(str(h))
                        if q is None:
                            # Fall back to centroid residual quantiles if station is missing
                            q = seed_history.get("centroid", {}).get(str(h))

                        m1 = M1Cams(q)
                        qf_out = m1.predict(val)
                        if isinstance(qf_out, QuantileFunction):
                            # Floor quantiles at 0.0 (ADR-012)
                            quantiles = np.maximum(0.0, qf_out.q).tolist()
                        else:
                            quantiles = [max(0.0, float(qf_out))] * 19
                        forecasts[h] = quantiles

                    # Drop station if fewer than 3 horizons per bulletin schema
                    if len(forecasts) == 3:
                        stations_data.append(
                            {
                                "location_id": loc_id,
                                "name": s_name,
                                "lat": lat,
                                "lon": lon,
                                "is_reference": is_ref,
                                "forecasts": forecasts,
                                "data_cutoff_utc": station_cutoff,
                            }
                        )

            # City pseudo-location forecast for M1 CAMS
            print(
                f"[DEBUG-TRACE] orchestrator: STEP 4: computing city forecast for M1 CAMS (cams_df rows={len(cams_df)})",
                flush=True,
            )
            city_cams = (
                cams_df[cams_df["location_id"] == "centroid"]
                if not cams_df.empty
                else pd.DataFrame()
            )
            city_forecasts = {}
            if not city_cams.empty:
                for h in [24, 48, 72]:
                    block_hours = [issuance_utc + timedelta(hours=h - 24 + i) for i in range(24)]
                    if "target_hour_utc" in city_cams.columns:
                        match = city_cams[city_cams["target_hour_utc"].isin(block_hours)]
                        if match.empty:
                            target = issuance_utc + timedelta(hours=h)
                            match = city_cams[city_cams["target_hour_utc"] == target]
                    else:
                        match = pd.DataFrame()
                    if match.empty:
                        continue
                    val = float(match["pm25_ugm3"].mean())
                    city_q = seed_history.get("centroid", {}).get(str(h))
                    city_m1 = M1Cams(city_q)
                    qf_out = city_m1.predict(val)
                    if isinstance(qf_out, QuantileFunction):
                        quantiles = np.maximum(0.0, qf_out.q).tolist()
                    else:
                        quantiles = [max(0.0, float(qf_out))] * 19
                    city_forecasts[h] = quantiles
        else:
            # Mode: observations_only (M0 Persistence)
            for _, s in stations_df.iterrows():
                loc_id = s["location_id"]
                s_obs = (
                    obs_df[obs_df["location_id"] == loc_id] if not obs_df.empty else pd.DataFrame()
                )
                if s_obs.empty:
                    continue

                s_obs = s_obs.sort_values("ts_utc")
                # 24-hour block ending at observation cutoff
                cutoff_ref = real_data_cutoff or issuance_utc
                recent_24h = s_obs[
                    (s_obs["ts_utc"] > (cutoff_ref - timedelta(hours=24)))
                    & (s_obs["ts_utc"] <= cutoff_ref)
                ]
                # Completeness rule: at least 18 of 24 valid hours (ADR-005)
                valid_recent = recent_24h["pm25_ugm3"].dropna()
                if len(valid_recent) < 18:
                    continue

                station_cutoff = real_data_cutoff
                st_max = s_obs["ts_utc"].max()
                if hasattr(st_max, "to_pydatetime"):
                    station_cutoff = st_max.to_pydatetime()
                elif isinstance(st_max, datetime):
                    station_cutoff = st_max

                forecasts = {}
                for h in [24, 48, 72]:
                    q_m0 = (
                        m0_seed_history.get(str(loc_id), {}).get(str(h))
                        or m0_seed_history.get("centroid", {}).get(str(h))
                        or seed_history.get("m0", {}).get(str(loc_id), {}).get(str(h))
                        or seed_history.get(str(loc_id), {}).get(str(h))
                        or seed_history.get("centroid", {}).get(str(h))
                    )
                    try:
                        m0 = M0Persistence(q_m0)
                        qf_out = m0.predict(valid_recent.to_numpy())
                        quantiles = np.maximum(0.0, qf_out.q).tolist()
                        forecasts[h] = quantiles
                    except ValueError:
                        continue

                if len(forecasts) == 3:
                    stations_data.append(
                        {
                            "location_id": loc_id,
                            "name": s.get("name", str(loc_id)),
                            "lat": float(s["lat"]),
                            "lon": float(s["lon"]),
                            "is_reference": bool(s.get("is_monitor") is True),
                            "forecasts": forecasts,
                            "data_cutoff_utc": station_cutoff,
                        }
                    )

            # City pseudo-location for M0 per FR-38 (requires >= 3 valid stations)
            city_forecasts = {}
            if len(stations_data) >= 3:
                for h in [24, 48, 72]:
                    station_means = [
                        float(obs_df[obs_df["location_id"] == s["location_id"]]["pm25_ugm3"].mean())
                        for s in stations_data
                    ]
                    city_mean_val = float(np.mean(station_means))
                    city_q_m0 = (
                        m0_seed_history.get("centroid", {}).get(str(h))
                        or seed_history.get("m0", {}).get("centroid", {}).get(str(h))
                        or seed_history.get("centroid", {}).get(str(h))
                    )
                    if city_q_m0 is not None:
                        try:
                            city_m0 = M0Persistence(city_q_m0)
                            qf_out = city_m0.predict([city_mean_val] * 24)
                            city_forecasts[h] = np.maximum(0.0, qf_out.q).tolist()
                        except ValueError:
                            pass
    except Exception as e:
        logger.error(f"Inference raised an error: {e}")
        write_manifest(
            manifest_file,
            run_id,
            issuance_utc,
            False,
            is_rerun,
            mode,
            4,
            50,
            git_sha=git_sha,
            config_hash=config_hash,
        )
        return 50

    print(
        f"[DEBUG-TRACE] orchestrator: STEP 5: inference completed (stations_data={len(stations_data)}, city_forecasts={list(city_forecasts.keys())})",
        flush=True,
    )

    # Ensure at least city forecast exists for publication
    if not city_forecasts:
        print("[DEBUG-TRACE] orchestrator: no city forecast, degrading to level 4", flush=True)
        logger.error("No city forecast could be computed. Degrading to level 4.")
        write_manifest(
            manifest_file,
            run_id,
            issuance_utc,
            False,
            is_rerun,
            mode,
            4,
            20,
            git_sha=git_sha,
            config_hash=config_hash,
        )
        return 20

    try:
        stations_meta_dict: dict[str, Any] = {}
        for s_meta in stations_data:
            sid = f"station:{s_meta['location_id']}"
            stations_meta_dict[sid] = {
                "name": s_meta["name"],
                "lat": s_meta["lat"],
                "lon": s_meta["lon"],
                "is_reference": s_meta["is_reference"],
            }

        sources_status = {
            "openaq": {"status": "ok" if has_obs else "missing"},
            "cams_global": {
                "status": "stale" if cams_is_stale else ("ok" if has_cams else "missing")
            },
        }
        if cams_base:
            sources_status["cams_global"]["as_of_utc"] = cams_base.isoformat(timespec="seconds")

        print("[DEBUG-TRACE] orchestrator: STEP 6: calling generate_forecast_log", flush=True)
        log_df = generate_forecast_log(
            issuance_utc,
            cams_base,
            mode,
            baseline_used,
            stations_data,
            city_forecasts,
            cams_lead_offset_h,
            run_id,
            git_sha=git_sha,
            config_hash=config_hash,
            data_cutoff_utc=real_data_cutoff,
            n_available_stations=n_available_stations,
            is_rerun=is_rerun,
        )
        print(
            f"[DEBUG-TRACE] orchestrator: STEP 7: generate_forecast_log completed (shape={log_df.shape})",
            flush=True,
        )

        print("[DEBUG-TRACE] orchestrator: STEP 8: calling generate_bulletin_json", flush=True)
        bulletin = generate_bulletin_json(
            log_df,
            issuance_utc,
            mode,
            baseline_used,
            real_data_cutoff,
            "lahore",
            sources_status,
            stations_meta_dict,
            git_sha=git_sha,
        )
        print("[DEBUG-TRACE] orchestrator: STEP 9: generate_bulletin_json completed", flush=True)

        # Write bulletin and site outputs under site/ (mounted writable volume)
        print("[DEBUG-TRACE] orchestrator: STEP 10: writing bulletin and site files", flush=True)
        site_dir = Path("site")
        site_dir.mkdir(parents=True, exist_ok=True)
        forecast_dir = site_dir / "forecast"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        issuance_date_str = issuance_utc.strftime("%Y-%m-%d")
        for f_name in [f"{run_id}.json", f"{issuance_date_str}.json", "latest.json"]:
            with (forecast_dir / f_name).open("w", encoding="utf-8") as f:
                json.dump(bulletin, f, indent=2)
            with (site_dir / f_name).open("w", encoding="utf-8") as f:
                json.dump(bulletin, f, indent=2)

        # Also write to gh-pages if writable (for local test parity)
        with contextlib.suppress(OSError):
            gh_pages_dir = Path("gh-pages")
            gh_pages_dir.mkdir(parents=True, exist_ok=True)
            gh_pages_forecast_dir = gh_pages_dir / "forecast"
            gh_pages_forecast_dir.mkdir(parents=True, exist_ok=True)
            for f_name in [f"{run_id}.json", f"{issuance_date_str}.json", "latest.json"]:
                with (gh_pages_forecast_dir / f_name).open("w", encoding="utf-8") as f:
                    json.dump(bulletin, f, indent=2)
                with (gh_pages_dir / f_name).open("w", encoding="utf-8") as f:
                    json.dump(bulletin, f, indent=2)

        # Write forecast log ONLY after bulletin successfully generated and validated
        print("[DEBUG-TRACE] orchestrator: STEP 11: writing forecast log to parquet", flush=True)
        Path(".state/forecasts").mkdir(parents=True, exist_ok=True)
        log_df.to_parquet(f".state/forecasts/forecast_{run_id}.parquet")
        print("[DEBUG-TRACE] orchestrator: STEP 12: forecast log written to parquet", flush=True)

    except Exception as e:
        print(
            f"[DEBUG-TRACE] orchestrator: bulletin/log generation failed with exception: {e}",
            flush=True,
        )
        logger.error(f"Failed to generate bulletin/log: {e}")
        write_manifest(
            manifest_file,
            run_id,
            issuance_utc,
            False,
            is_rerun,
            mode,
            4,
            50,
            git_sha=git_sha,
            config_hash=config_hash,
        )
        return 50

    try:
        print("[DEBUG-TRACE] orchestrator: STEP 13: calling generate_site", flush=True)
        generate_site(bulletin, site_dir)
        with contextlib.suppress(OSError):
            generate_site(bulletin, Path("gh-pages"))
        print("[DEBUG-TRACE] orchestrator: STEP 14: generate_site completed", flush=True)
    except Exception as e:
        print(f"[DEBUG-TRACE] orchestrator: generate_site failed with exception: {e}", flush=True)
        logger.error(f"Failed to generate site: {e}")
        write_manifest(
            manifest_file,
            run_id,
            issuance_utc,
            False,
            is_rerun,
            mode,
            4,
            50,
            git_sha=git_sha,
            config_hash=config_hash,
        )
        return 50

    exit_code = 10 if degrad_level in (1, 2, 3) else 0
    print(
        f"[DEBUG-TRACE] orchestrator: STEP 15: calling write_manifest (published=True, exit_code={exit_code})",
        flush=True,
    )
    write_manifest(
        manifest_file,
        run_id,
        issuance_utc,
        True,
        is_rerun,
        mode,
        degrad_level,
        exit_code,
        git_sha=git_sha,
        config_hash=config_hash,
    )
    print(
        f"[DEBUG-TRACE] orchestrator: STEP 16: write_manifest complete, returning {exit_code}",
        flush=True,
    )
    return exit_code


def run_daily_pipeline(issuance_utc: datetime, force: bool = False, mode: str | None = None) -> int:
    return asyncio.run(run_daily_pipeline_async(issuance_utc, force, mode=mode))
