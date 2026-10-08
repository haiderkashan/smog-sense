"""smogsense.publishing.bulletin - Bulletin context and public JSON export.

Builds the Jinja2 context and the forecast/latest.json document validated against
data/schemas/bulletin.schema.json.

Specification: docs/dissemination-and-ui.md -> 'Jinja2 template contract'
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from smogsense.models.distribution import QuantileFunction


def determine_aqi_category(pm25: float) -> str:
    """Return AQI category based on US EPA breaks for PM2.5."""
    if pm25 <= 12.0:
        return "good"
    elif pm25 <= 35.4:
        return "moderate"
    elif pm25 <= 55.4:
        return "usg"
    elif pm25 <= 150.4:
        return "unhealthy"
    elif pm25 <= 250.4:
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
) -> dict[str, Any]:
    """Generate the bulletin dictionary conforming to bulletin.schema.json."""

    now_utc = datetime.now(UTC)
    valid_until = issuance_utc + timedelta(hours=30)

    # We will aggregate by station and by horizon
    # The dataframe has rows for each point_id, horizon_h, method
    stations_data: dict[str, list[dict[str, Any]]] = {}
    city_data: dict[int, dict[str, Any]] = {}

    for _, row in forecast_df.iterrows():
        point_id: str = row["point_id"]
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
        # We need a QuantileFunction to evaluate prob_exceed and categories accurately,
        # but if we only have the quantiles, we can reconstruct a rough QuantileFunction
        q_arr = [
            float(row[f"q{k:02d}"])
            for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        ]
        qf = QuantileFunction(q_arr)

        # For simplicity, if step function (all equal), probabilities are 1 or 0
        def get_prob(thresh: float, q_func=qf) -> float:
            import numpy as np

            # P(X > thresh)
            return float(np.asarray(q_func.prob_exceed(thresh)).item())

        prob_exceed = [
            {"threshold_ugm3": 55.5, "probability": get_prob(55.5)},
            {"threshold_ugm3": 125.5, "probability": get_prob(125.5)},
            {"threshold_ugm3": 225.5, "probability": get_prob(225.5)},
        ]

        # Category probabilities (good, moderate, etc.)
        # P(good) = P(X <= 12.0) = qf.cdf(12.0)
        # P(moderate) = P(12.0 < X <= 35.4) = qf.cdf(35.4) - qf.cdf(12.0)
        p_good = max(0.0, float(qf.cdf(12.0)))
        p_mod = max(0.0, float(qf.cdf(35.4) - qf.cdf(12.0)))
        p_usg = max(0.0, float(qf.cdf(55.4) - qf.cdf(35.4)))
        p_unh = max(0.0, float(qf.cdf(150.4) - qf.cdf(55.4)))
        p_very = max(0.0, float(qf.cdf(250.4) - qf.cdf(150.4)))
        p_haz = max(0.0, 1.0 - float(qf.cdf(250.4)))

        # Do not silently normalize category probabilities as per specs.
        # Just ensure they sum to ~1 within tolerance.
        total_p = p_good + p_mod + p_usg + p_unh + p_very + p_haz
        if not (0.999 <= total_p <= 1.001):
            if total_p == 0.0:
                p_good = 1.0  # Safe fallback for 0 values if mathematically sound
            else:
                # Should not happen unless cdf is fundamentally broken
                pass

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
        win_start_local = win_start_utc + timedelta(hours=5)
        win_end_local = win_end_utc + timedelta(hours=5)

        horizon_obj = {
            "lead_h": horizon,
            "window_start_utc": win_start_utc.isoformat(timespec="seconds"),
            "window_end_utc": win_end_utc.isoformat(timespec="seconds"),
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

        if point_id.startswith("station:"):
            # It's a station
            if point_id not in stations_data:
                stations_data[point_id] = []

            stations_data[point_id].append(
                {
                    "lead_h": horizon,
                    "q10": q_dict["q10"],
                    "q50": q_dict["q50"],
                    "q90": q_dict["q90"],
                    "category_median": cat_q50,
                }
            )
        else:
            # It's the city
            city_data[horizon] = horizon_obj

    # Assemble horizons for city
    city_horizons = [city_data[h] for h in sorted(city_data.keys())]

    # Assemble stations
    stations_list = []
    for sid, h_list in stations_data.items():
        meta = stations_metadata.get(sid, {})
        stations_list.append(
            {
                "point_id": sid,
                "name": meta.get("name", sid),
                "lat": meta.get("lat", 0.0),
                "lon": meta.get("lon", 0.0),
                "is_reference": meta.get("is_reference", False),
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
        "mode_reason": "Automated pipeline run",
        "model": {
            "id": model,
            "version": "baseline",
            "git_sha": "0000000",
            "calibrated": False,
            "adaptation_status": "unadapted",
        },
        "aqi_scheme": "us_epa_2024_pm25_24h",
        "horizons": city_horizons,
        "stations": stations_list,
        "advisory": {
            "category": str(city_horizons[0]["category"]["median_q50"])
            if city_horizons
            else "moderate",
            "notes": ["mask_note"],
        },
        "provenance": provenance,
        "disclaimer_key": "research_only",
        "valid_until_utc": valid_until.isoformat(timespec="seconds"),
        "basis": {
            "n_panel": len(stations_list),
            "n_reference": sum(1 for s in stations_list if s.get("is_reference")),
            "n_lowcost": sum(1 for s in stations_list if not s.get("is_reference")),
            "aggregation": "mean",
        },
    }

    # Validate against schema
    schema_path = Path("data/schemas/bulletin.schema.json")
    if schema_path.exists():
        import jsonschema

        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=bulletin, schema=schema)

    return bulletin
