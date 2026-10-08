"""smogsense.pipeline.orchestrator - Daily pipeline orchestrator.

Specification: docs/system-architecture.md
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pandera as pa

from smogsense.config import Settings
from smogsense.data_ingestion.base import ResilientClient
from smogsense.data_ingestion.copernicus import CamsClient
from smogsense.data_ingestion.openaq import fetch_hourly, list_locations
from smogsense.inference.degradation import determine_mode, get_degradation_level
from smogsense.models.baselines import M0Persistence, M1Cams
from smogsense.models.distribution import QuantileFunction
from smogsense.preprocessing.alignment import asof_cams_run
from smogsense.preprocessing.gridded import extract_stations
from smogsense.preprocessing.qc import apply_qc
from smogsense.publishing.bulletin import generate_bulletin_json
from smogsense.publishing.site import generate_site

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
) -> pd.DataFrame:
    """Generate the forecast log dataframe."""
    now_utc = datetime.now(UTC)
    rows = []

    for s in stations_data:
        point_id = f"station:{s['location_id']}"
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
                "data_cutoff_utc": s.get("data_cutoff_utc", issuance_utc),
                "git_sha": "unknown",
                "config_hash": "unknown",
                "calibrated": False,
                "run_id": run_id,
                "is_rerun": False,
                "adaptation_status": "unadapted",
                "cams_lead_offset_h": cams_lead_offset_h,
                "n_available_stations": len(stations_data),
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
            "data_cutoff_utc": issuance_utc,
            "git_sha": "unknown",
            "config_hash": "unknown",
            "calibrated": False,
            "run_id": run_id,
            "is_rerun": False,
            "adaptation_status": "unadapted",
            "cams_lead_offset_h": cams_lead_offset_h,
            "n_available_stations": len(stations_data),
        }
        for q_idx, q_level in enumerate(
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        ):
            row[f"q{q_level:02d}"] = float(qf_vals[q_idx])
        rows.append(row)

    df = pd.DataFrame(rows)
    schema = pa.DataFrameSchema.from_yaml("data/schemas/forecast_log.schema.yaml")
    return schema.validate(df)


def write_manifest(
    path: Path,
    run_id: str,
    issuance_utc: datetime,
    published: bool,
    is_rerun: bool,
    mode: str,
    level: int,
    exit_code: int,
) -> None:
    manifest = {
        "run_id": run_id,
        "issued_at_utc": issuance_utc.isoformat(timespec="seconds"),
        "is_rerun": is_rerun,
        "published": published,
        "adaptation_status": "unadapted",
        "observed_latency": 0.0,
        "git_sha": "unknown",
        "config_hash": "unknown",
        "issuance": issuance_utc.isoformat(timespec="seconds"),
        "degradation_level": level,
        "exit_code": exit_code,
    }
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


async def run_daily_pipeline_async(issuance_utc: datetime, force: bool = False) -> int:
    settings = Settings.load("configs")
    run_id = f"run_{issuance_utc.strftime('%Y%m%d_%H%M')}"
    manifest_dir = Path(".state/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"{run_id}.json"

    is_rerun = False

    if manifest_file.exists():
        try:
            with manifest_file.open("r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("published") is True:
                if not force:
                    logger.info("Already published. Use --force.")
                    return 11
                else:
                    is_rerun = True
        except Exception:
            pass

    from smogsense.data_ingestion.base import CircuitBreaker, RateBudget

    has_obs = False
    obs_df = pd.DataFrame()
    stations_df = pd.DataFrame()
    try:
        async with httpx.AsyncClient() as ac:
            rb = RateBudget()
            cb = CircuitBreaker()
            client = ResilientClient(ac, rb, cb)
            stations_df = await list_locations(client, "lahore")
            if not stations_df.empty:
                end_utc = issuance_utc
                start_utc = end_utc - timedelta(hours=72)
                raw_obs_df = await fetch_hourly(client, stations_df, start_utc, end_utc)
                if not raw_obs_df.empty:
                    qc_df = apply_qc(raw_obs_df)
                    valid_obs = qc_df[qc_df["qc_flags"] == 0].copy()
                    if not valid_obs.empty:
                        has_obs = True
                        obs_df = valid_obs
    except Exception as e:
        logger.warning(f"OpenAQ fetch failed: {e}")

    has_cams = False
    cams_is_stale = False
    cams_base = None
    cams_df = pd.DataFrame()

    try:
        cams_base = asof_cams_run(issuance_utc)
        cams_client = CamsClient(settings.model_dump())
        cams_path = Path(".state/inputs") / f"cams_{cams_base.strftime('%Y%m%d_%H')}.grib"

        if not cams_path.exists():
            cams_path.parent.mkdir(parents=True, exist_ok=True)
            # Calculate leadtimes required for targets T+24, T+48, T+72
            targets = [issuance_utc + timedelta(hours=h) for h in [24, 48, 72]]
            cams_leadtimes = [int((t - cams_base).total_seconds() / 3600.0) for t in targets]

            cams_client.fetch_cams(
                base_time=cams_base,
                leadtime_hours=cams_leadtimes,
                variables=["particulate_matter_2.5um"],
                area=[
                    settings.domains["domains"]["lahore"]["station_bbox"]["north"],
                    settings.domains["domains"]["lahore"]["station_bbox"]["west"],
                    settings.domains["domains"]["lahore"]["station_bbox"]["south"],
                    settings.domains["domains"]["lahore"]["station_bbox"]["east"],
                ],
                dest_path=cams_path,
            )

        if cams_path.exists() and not stations_df.empty:
            cams_df = extract_stations(
                cams_path, stations_df, domain="lahore", settings=settings.model_dump()
            )
            if not cams_df.empty:
                has_cams = True
    except Exception as e:
        logger.warning(f"CAMS fetch failed: {e}")

    try:
        mode = determine_mode(has_obs, has_cams, cams_is_stale, False)
        degrad_level = get_degradation_level(mode)
    except RuntimeError as e:
        logger.error(str(e))
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, "unknown", 4, 20)
        return 20

    baseline_used = "m1_cams_raw" if has_cams else "m0_persistence"
    stations_data = []
    cams_lead_offset_h = 0.0
    if has_cams and cams_base:
        cams_lead_offset_h = float((issuance_utc - cams_base).total_seconds() / 3600.0)

    try:
        seed_history = {}
        seed_path = Path(".state/artifacts/seed_history.json")
        if seed_path.exists():
            with seed_path.open("r", encoding="utf-8") as f:
                seed_history = json.load(f).get("quantiles", {})

        if has_cams:
            for _, s in stations_df.iterrows():
                loc_id = s["location_id"]
                station_cams = (
                    cams_df[cams_df["location_id"] == loc_id]
                    if not cams_df.empty
                    else pd.DataFrame()
                )
                if station_cams.empty:
                    continue

                s_name = s.get("name", str(loc_id))
                lat, lon = float(s["lat"]), float(s["lon"])
                forecasts = {}
                data_cutoff = issuance_utc

                for h in [24, 48, 72]:
                    target = issuance_utc + timedelta(hours=h)

                    if "target_hour_utc" in station_cams.columns:
                        match = station_cams[station_cams["target_hour_utc"] == target]
                    else:
                        match = pd.DataFrame()

                    if match.empty:
                        continue

                    val = float(match.iloc[0]["pm25_ugm3"])

                    q = seed_history.get(str(loc_id), {}).get(str(h))
                    m1 = M1Cams(q)
                    qf_out = m1.predict(val)
                    if isinstance(qf_out, QuantileFunction):
                        quantiles = qf_out.q.tolist()
                    else:
                        quantiles = [float(qf_out)] * 19
                    forecasts[h] = quantiles

                if forecasts:
                    stations_data.append(
                        {
                            "location_id": loc_id,
                            "name": s_name,
                            "lat": lat,
                            "lon": lon,
                            "is_reference": True,
                            "forecasts": forecasts,
                            "data_cutoff_utc": data_cutoff,
                        }
                    )

            # City aggregation for M1 CAMS
            city_cams = (
                cams_df[cams_df["location_id"] == "centroid"]
                if not cams_df.empty
                else pd.DataFrame()
            )
            city_forecasts = {}
            if not city_cams.empty:
                for h in [24, 48, 72]:
                    target = issuance_utc + timedelta(hours=h)
                    if "target_hour_utc" in city_cams.columns:
                        match = city_cams[city_cams["target_hour_utc"] == target]
                    else:
                        match = pd.DataFrame()
                    if match.empty:
                        continue
                    val = float(match.iloc[0]["pm25_ugm3"])
                    qf_out = m1.predict(val)
                    if isinstance(qf_out, QuantileFunction):
                        quantiles = qf_out.q.tolist()
                    else:
                        quantiles = [float(qf_out)] * 19
                    city_forecasts[h] = quantiles
        else:
            m0 = M0Persistence(None)
            for _, s in stations_df.iterrows():
                loc_id = s["location_id"]
                s_obs = (
                    obs_df[obs_df["location_id"] == loc_id] if not obs_df.empty else pd.DataFrame()
                )
                if s_obs.empty:
                    continue

                s_obs = s_obs.sort_values("ts_utc")
                recent_24h = s_obs[s_obs["ts_utc"] > (issuance_utc - timedelta(hours=24))]
                if len(recent_24h) == 0:
                    continue

                forecasts = {}
                data_cutoff = recent_24h["ts_utc"].max()

                for h in [24, 48, 72]:
                    try:
                        qf_out = m0.predict(recent_24h["pm25_ugm3"])
                        if isinstance(qf_out, QuantileFunction):
                            quantiles = qf_out.q.tolist()
                        else:
                            quantiles = [float(qf_out)] * 19
                        forecasts[h] = quantiles
                    except ValueError:
                        continue

                if forecasts:
                    stations_data.append(
                        {
                            "location_id": loc_id,
                            "name": s.get("name", str(loc_id)),
                            "lat": float(s["lat"]),
                            "lon": float(s["lon"]),
                            "is_reference": True,
                            "forecasts": forecasts,
                            "data_cutoff_utc": data_cutoff,
                        }
                    )

            # City aggregation for M0
            city_forecasts = {}
            for h in [24, 48, 72]:
                q_list = []
                q_list = [s["forecasts"][h] for s in stations_data if h in s["forecasts"]]
                if len(q_list) >= 1:
                    city_forecasts[h] = np.mean(q_list, axis=0).tolist()
    except Exception as e:
        logger.error(f"Inference raised an error: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, mode, 4, 50)
        return 50

    if not stations_data:
        logger.error("No station forecasts generated. Level 4 failure.")
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, mode, 4, 20)
        return 20

    try:
        Path(".state/forecasts").mkdir(parents=True, exist_ok=True)
        log_df = generate_forecast_log(
            issuance_utc,
            cams_base,
            mode,
            baseline_used,
            stations_data,
            city_forecasts,
            cams_lead_offset_h,
            run_id,
        )
        log_df.to_parquet(f".state/forecasts/forecast_{run_id}.parquet")
    except Exception as e:
        logger.error(f"Failed to generate forecast log: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, mode, 4, 50)
        return 50

    try:
        Path("gh-pages").mkdir(parents=True, exist_ok=True)
        stations_metadata = {}
        for s in stations_data:
            sid = f"station:{s['location_id']}"
            stations_metadata[sid] = {
                "name": s["name"],
                "lat": s["lat"],
                "lon": s["lon"],
                "is_reference": s["is_reference"],
            }

        sources_status = {
            "openaq": {"status": "ok" if has_obs else "missing"},
            "cams_global": {
                "status": "stale" if cams_is_stale else ("ok" if has_cams else "missing")
            },
        }
        if cams_base:
            sources_status["cams_global"]["as_of_utc"] = cams_base.isoformat(timespec="seconds")

        bulletin = generate_bulletin_json(
            log_df,
            issuance_utc,
            mode,
            baseline_used,
            issuance_utc,
            "lahore",
            sources_status,
            stations_metadata,
        )
        with Path(f"gh-pages/{run_id}.json").open("w", encoding="utf-8") as f:
            json.dump(bulletin, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to generate bulletin: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, mode, 4, 30)
        return 30

    try:
        generate_site(bulletin, Path("gh-pages"))
    except Exception as e:
        logger.error(f"Failed to generate site: {e}")
        write_manifest(manifest_file, run_id, issuance_utc, False, is_rerun, mode, 4, 50)
        return 50

    exit_code = 10 if degrad_level in (1, 2, 3) else 0
    write_manifest(
        manifest_file, run_id, issuance_utc, True, is_rerun, mode, degrad_level, exit_code
    )
    return exit_code


def run_daily_pipeline(issuance_utc: datetime, force: bool = False) -> int:
    import asyncio

    return asyncio.run(run_daily_pipeline_async(issuance_utc, force))
