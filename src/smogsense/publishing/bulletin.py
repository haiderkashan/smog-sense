"""smogsense.publishing.bulletin - Bulletin context and public JSON export.

Builds the Jinja2 context and the forecast/latest.json document validated against
data/schemas/bulletin.schema.json.

Specification: docs/dissemination-and-ui.md -> 'Jinja2 template contract'
"""

import json
import math
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from smogsense.models.distribution import QuantileFunction

PKT = timezone(timedelta(hours=5))
CATEGORIES = ["good", "moderate", "usg", "unhealthy", "very_unhealthy", "hazardous"]


def determine_aqi_category(pm25: float) -> str:
    """Return AQI category based on US EPA 2024 breaks for PM2.5 truncated to 0.1 ug/m3."""
    val = max(0.0, float(pm25))
    trunc = math.floor(round(val, 4) * 10) / 10
    if trunc <= 9.0:
        return "good"
    elif trunc <= 35.4:
        return "moderate"
    elif trunc <= 55.4:
        return "usg"
    elif trunc <= 125.4:
        return "unhealthy"
    elif trunc <= 225.4:
        return "very_unhealthy"
    return "hazardous"


def generate_bulletin_json(
    forecast_df: pd.DataFrame,
    issuance_utc: datetime,
    mode: str,
    model: str,
    data_cutoff_utc: datetime | None,
    domain: str,
    sources_status: dict[str, Any],
    stations_metadata: dict[str, dict[str, Any]],
    git_sha: str | None = None,
    mode_reason: str | None = None,
    model_version: str | None = None,
    calibrated: bool | None = None,
    adaptation_status: str | None = None,
) -> dict[str, Any]:
    """Generate the bulletin dictionary conforming to bulletin.schema.json."""

    now_utc = datetime.now(UTC)
    valid_until = issuance_utc + timedelta(hours=30)

    # Filter forecast_df by method == model if method column exists
    if "method" in forecast_df.columns:
        df_model = forecast_df[forecast_df["method"] == model]
        if df_model.empty:
            df_model = forecast_df
    else:
        df_model = forecast_df

    stations_data: dict[str, list[dict[str, Any]]] = {}
    city_data: dict[int, dict[str, Any]] = {}

    for _, row in df_model.iterrows():
        raw_point_id = str(row["point_id"])
        level = str(row.get("level", ""))
        horizon = int(row["horizon_h"])

        # Build quantiles dictionary
        q_dict = {
            "q05": float(row["q05"]),
            "q10": float(row["q10"]),
            "q25": float(row["q25"]),
            "q50": float(row["q50"]),
            "q75": float(row["q75"]),
            "q90": float(row["q90"]),
            "q95": float(row["q95"]),
        }

        # Determine categories
        cat_q10 = determine_aqi_category(row["q10"])
        cat_q50 = determine_aqi_category(row["q50"])
        cat_q90 = determine_aqi_category(row["q90"])

        # Build probabilities
        q_arr = [
            float(row[f"q{k:02d}"])
            for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        ]
        qf = QuantileFunction(q_arr)

        def get_prob(thresh: float, q_func: Any = qf) -> float:
            import numpy as np

            return float(np.asarray(q_func.prob_exceed(thresh)).item())

        prob_exceed = [
            {"threshold_ugm3": 55.5, "probability": get_prob(55.5)},
            {"threshold_ugm3": 125.5, "probability": get_prob(125.5)},
            {"threshold_ugm3": 225.5, "probability": get_prob(225.5)},
        ]

        # Category probabilities per US EPA 2024
        p_good = max(0.0, float(qf.cdf(9.0)))
        p_mod = max(0.0, float(qf.cdf(35.4) - qf.cdf(9.0)))
        p_usg = max(0.0, float(qf.cdf(55.4) - qf.cdf(35.4)))
        p_unh = max(0.0, float(qf.cdf(125.4) - qf.cdf(55.4)))
        p_very = max(0.0, float(qf.cdf(225.4) - qf.cdf(125.4)))
        p_haz = max(0.0, 1.0 - float(qf.cdf(225.4)))

        total_p = p_good + p_mod + p_usg + p_unh + p_very + p_haz
        if not (0.999 <= total_p <= 1.001) and total_p == 0.0:
            p_good = 1.0

        by_category = {
            "good": round(p_good, 4),
            "moderate": round(p_mod, 4),
            "usg": round(p_usg, 4),
            "unhealthy": round(p_unh, 4),
            "very_unhealthy": round(p_very, 4),
            "hazardous": round(p_haz, 4),
        }

        win_start_utc = pd.Timestamp(row["window_start_utc"]).to_pydatetime()
        win_end_utc = pd.Timestamp(row["window_end_utc"]).to_pydatetime()
        if win_start_utc.tzinfo is None:
            win_start_utc = win_start_utc.replace(tzinfo=UTC)
        if win_end_utc.tzinfo is None:
            win_end_utc = win_end_utc.replace(tzinfo=UTC)

        win_start_local = win_start_utc.astimezone(PKT)
        win_end_local = win_end_utc.astimezone(PKT)

        horizon_obj = {
            "lead_h": horizon,
            "window_start_utc": win_start_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "window_end_utc": win_end_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "window_start_local": win_start_local.isoformat(timespec="seconds"),
            "window_end_local": win_end_local.isoformat(timespec="seconds"),
            "quantiles_ugm3": q_dict,
            "category": {
                "lower_q10": cat_q10,
                "median_q50": cat_q50,
                "upper_q90": cat_q90,
            },
            "probabilities": {"by_category": by_category, "exceed": prob_exceed},
        }

        is_station = (level == "station") or raw_point_id.startswith("station:")
        if is_station and not raw_point_id.startswith("city"):
            sid = raw_point_id if raw_point_id.startswith("station:") else f"station:{raw_point_id}"
            if sid not in stations_data:
                stations_data[sid] = []

            stations_data[sid].append(
                {
                    "lead_h": horizon,
                    "q10": q_dict["q10"],
                    "q50": q_dict["q50"],
                    "q90": q_dict["q90"],
                    "category_median": cat_q50,
                }
            )
        else:
            city_data[horizon] = horizon_obj

    # Assemble horizons for city
    city_horizons = [city_data[h] for h in sorted(city_data.keys())]

    # Assemble stations - dropping any station with fewer than 3 horizons
    stations_list = []
    for sid, h_list in stations_data.items():
        if len(h_list) != 3:
            continue
        raw_id = sid.removeprefix("station:")
        candidates = [sid, raw_id, f"station:{raw_id}"]
        if raw_id.isdigit():
            candidates.append(int(raw_id))  # type: ignore[arg-type]
        meta: dict[str, Any] = {}
        for c in candidates:
            if c in stations_metadata:
                meta = stations_metadata[c]
                break

        stations_list.append(
            {
                "point_id": sid,
                "name": meta.get("name", sid),
                "lat": float(meta.get("lat", 0.0)),
                "lon": float(meta.get("lon", 0.0)),
                "is_reference": bool(meta.get("is_reference", False)),
                "horizons": sorted(h_list, key=lambda x: x["lead_h"]),
            }
        )

    provenance: dict[str, list[dict[str, Any]]] = {"sources": []}
    for src_id, status_info in sources_status.items():
        provenance["sources"].append(
            {
                "id": src_id,
                "status": status_info["status"],
                "as_of_utc": status_info.get("as_of_utc"),
            }
        )

    from smogsense.utils.io import get_git_sha

    resolved_git_sha = git_sha or get_git_sha()
    resolved_version = model_version
    if resolved_version is None:
        resolved_version = (
            str(df_model["model_version"].iloc[0])
            if "model_version" in df_model.columns and not df_model.empty
            else "baseline"
        )

    resolved_calibrated = calibrated
    if resolved_calibrated is None:
        resolved_calibrated = (
            bool(df_model["calibrated"].iloc[0])
            if "calibrated" in df_model.columns and not df_model.empty
            else False
        )

    resolved_adaptation = adaptation_status
    if resolved_adaptation is None:
        resolved_adaptation = (
            str(df_model["adaptation_status"].iloc[0])
            if "adaptation_status" in df_model.columns and not df_model.empty
            else "unadapted"
        )

    resolved_mode_reason = mode_reason or "Automated pipeline run"

    # Advisory policy: base category is day-1 (24h) median
    # Escalate 1 level if probability of next-worse category >= 0.35 (configs/bulletin.yaml)
    advisory_cat = "moderate"
    if city_horizons:
        day1_horizon = city_horizons[0]
        base_cat = str(day1_horizon["category"]["median_q50"])
        advisory_cat = base_cat
        if base_cat in CATEGORIES:
            idx = CATEGORIES.index(base_cat)
            if idx < len(CATEGORIES) - 1:
                next_cat = CATEGORIES[idx + 1]
                p_next = float(day1_horizon["probabilities"]["by_category"].get(next_cat, 0.0))
                if p_next >= 0.35:
                    advisory_cat = next_cat

    bulletin = {
        "schema_version": "1.0",
        "issuance_utc": issuance_utc.isoformat(timespec="seconds"),
        "generated_at_utc": now_utc.isoformat(timespec="seconds"),
        "data_cutoff_utc": data_cutoff_utc.isoformat(timespec="seconds")
        if data_cutoff_utc
        else None,
        "domain": {
            "id": domain,
            "name": {"en": "Lahore", "ur": "لاہور"},
            "centroid": {"lat": 31.5204, "lon": 74.3587},
            "timezone": "Asia/Karachi",
        },
        "mode": mode,
        "mode_reason": resolved_mode_reason,
        "model": {
            "id": model,
            "version": resolved_version,
            "git_sha": resolved_git_sha,
            "calibrated": resolved_calibrated,
            "adaptation_status": resolved_adaptation,
        },
        "aqi_scheme": "us_epa_2024_pm25_24h",
        "horizons": city_horizons,
        "stations": stations_list,
        "advisory": {
            "category": advisory_cat,
            "notes": ["mask_note"],
        },
        "provenance": provenance,
        "disclaimer_key": "research_only",
        "valid_until_utc": valid_until.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "basis": {
            "n_panel": len(stations_list),
            "n_reference": sum(1 for s in stations_list if s.get("is_reference")),
            "n_lowcost": sum(1 for s in stations_list if not s.get("is_reference")),
            "aggregation": "mean",
        },
    }

    # Validate against schema
    schema_path = Path("data/schemas/bulletin.schema.json")
    if not schema_path.exists():
        schema_path = Path(__file__).resolve().parents[3] / "data/schemas/bulletin.schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Bulletin schema not found at {schema_path}")

    import jsonschema

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(
        instance=bulletin,
        schema=schema,
        format_checker=jsonschema.FormatChecker(),
    )

    return bulletin
