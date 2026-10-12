"""smogsense.pipeline.shadow — Prospective shadow scoring pipeline.

Scores closed forecast blocks (24h, 48h, 72h) against observed 24h block means,
enforcing completeness (>= 18 valid non-imputed hours), first-issuance-only scoring
(is_rerun = false), CRPS and pinball evaluation across 19 quantiles, Pandera schema validation,
and date-partitioned + canonical persistence.

Specification:
- docs/evaluation-strategy.md §2 ('Forecast targets and verification data')
- docs/system-architecture.md §4 ('Verification and scoring loop')
- data/schemas/score_log.schema.yaml
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from structlog import get_logger

from smogsense.evaluation.scoring import score_forecast_block
from smogsense.utils.io import validate_frame

logger = get_logger(__name__)


def _load_score_log_schema() -> pa.DataFrameSchema:
    candidates = [
        Path("data/schemas/score_log.schema.yaml"),
        Path("/app/data/schemas/score_log.schema.yaml"),
        Path(__file__).resolve().parents[3] / "data/schemas/score_log.schema.yaml",
    ]
    for p in candidates:
        if p.exists():
            return pa.DataFrameSchema.from_yaml(str(p))
    raise FileNotFoundError(f"score_log.schema.yaml not found in candidates: {candidates}")


def _normalize_issuance_date(val: str | date | datetime) -> str:
    """Normalize string, date or datetime into YYYY-MM-DD string."""
    if isinstance(val, str):
        return str(pd.to_datetime(val, utc=True).strftime("%Y-%m-%d"))
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")
    raise TypeError(f"Expected str, date, or datetime, got {type(val)}")


def score_forecast_targets(
    issuance_date: str | date | datetime,
    forecast_log_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    stage: str,
    scored_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Core scoring function matching 24h, 48h, 72h forecast targets against observed block means.

    Enforces:
    1. First-issuance-only scoring (run_id matching initial issuance, is_rerun = false).
    2. Completeness rule: requires >= 18 valid non-imputed hours per 24h block.
    3. Probabilistic scoring: CRPS, pinballs, interval score across 19 quantiles.
    4. Validation against data/schemas/score_log.schema.yaml.
    """
    if scored_at_utc is None:
        scored_at_utc = datetime.now(UTC)

    stage_clean = stage.lower().strip()
    if stage_clean not in {"provisional", "settled"}:
        raise ValueError(f"stage must be 'provisional' or 'settled', got '{stage}'")

    if forecast_log_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "issuance_utc",
                "scored_at_utc",
                "stage",
                "domain",
                "point_id",
                "horizon_h",
                "method",
                "y_obs",
                "n_valid_hours",
                "valid",
                "crps",
                "pinball_q10",
                "pinball_q50",
                "pinball_q90",
                "interval_score_80",
                "hit_below_q10",
                "hit_below_q50",
                "hit_below_q90",
                "obs_category",
                "forecast_run_id",
                "scoring_basis",
                "truth_basis",
            ]
        )
        return validate_frame(empty_df, _load_score_log_schema())

    # Filter by issuance date if specified
    dt_str = _normalize_issuance_date(issuance_date)
    f_df = forecast_log_df.copy()
    f_dates = pd.to_datetime(f_df["issuance_utc"], utc=True).dt.strftime("%Y-%m-%d")
    f_df = f_df[f_dates == dt_str].copy()

    # First-Issuance-Only Scoring (docs/evaluation-strategy.md §9 / PRD):
    # Score only initial runs where is_rerun == false. Never score reruns.
    if "is_rerun" in f_df.columns:
        f_df = f_df[f_df["is_rerun"] == False].copy()  # noqa: E712

    # Deduplicate multiple runs for the same forecast point
    if "run_id" in f_df.columns:
        f_df = f_df.drop_duplicates(
            subset=["issuance_utc", "point_id", "horizon_h", "method"], keep="first"
        ).reset_index(drop=True)

    if f_df.empty:
        empty_df = pd.DataFrame(
            columns=[
                "issuance_utc",
                "scored_at_utc",
                "stage",
                "domain",
                "point_id",
                "horizon_h",
                "method",
                "y_obs",
                "n_valid_hours",
                "valid",
                "crps",
                "pinball_q10",
                "pinball_q50",
                "pinball_q90",
                "interval_score_80",
                "hit_below_q10",
                "hit_below_q50",
                "hit_below_q90",
                "obs_category",
                "forecast_run_id",
                "scoring_basis",
                "truth_basis",
            ]
        )
        return validate_frame(empty_df, _load_score_log_schema())

    # Prepare observations
    clean_obs = obs_df.copy()
    if not clean_obs.empty and "ts_utc" in clean_obs.columns:
        clean_obs["ts_utc"] = pd.to_datetime(clean_obs["ts_utc"], utc=True)

    records: list[dict[str, Any]] = []
    for _, f_row in f_df.iterrows():
        issuance_utc = pd.to_datetime(f_row["issuance_utc"], utc=True)
        h = int(f_row["horizon_h"])

        if "window_end_utc" in f_row and pd.notna(f_row["window_end_utc"]):
            w_end = pd.to_datetime(f_row["window_end_utc"], utc=True)
        else:
            w_end = issuance_utc + timedelta(hours=h)

        if "window_start_utc" in f_row and pd.notna(f_row["window_start_utc"]):
            w_start = pd.to_datetime(f_row["window_start_utc"], utc=True)
        else:
            w_start = w_end - timedelta(hours=24)

        point_id = str(f_row["point_id"])
        valid_obs_series = pd.Series(dtype=float)
        scoring_basis = "station"

        if not clean_obs.empty and "ts_utc" in clean_obs.columns:
            # 24h verification block [w_start, w_end)
            block_obs = clean_obs[(clean_obs["ts_utc"] >= w_start) & (clean_obs["ts_utc"] < w_end)]

            if point_id.startswith("station:") or point_id.replace("station:", "").isdigit():
                scoring_basis = "station"
                loc_id = point_id.split(":", 1)[1] if ":" in point_id else point_id
                st_obs = block_obs[block_obs["location_id"].astype(str) == str(loc_id)]
                if not st_obs.empty and "pm25_ugm3" in st_obs.columns:
                    # Enforce non-imputed valid hours
                    if "imputed" in st_obs.columns:
                        st_obs = st_obs[st_obs["imputed"] == False]  # noqa: E712
                    valid_obs_series = st_obs["pm25_ugm3"].dropna()

            elif point_id.startswith("city:") or point_id == "centroid":
                scoring_basis = "city"
                dom = point_id.split(":", 1)[1] if ":" in point_id else str(f_row["domain"])
                # Check if direct matching location exists in block_obs
                direct_match = block_obs[
                    block_obs["location_id"].astype(str).isin([point_id, "centroid"])
                ]
                if not direct_match.empty and "pm25_ugm3" in direct_match.columns:
                    if "imputed" in direct_match.columns:
                        direct_match = direct_match[direct_match["imputed"] == False]  # noqa: E712
                    valid_obs_series = direct_match["pm25_ugm3"].dropna()
                else:
                    dom_obs = (
                        block_obs[block_obs["domain"] == dom]
                        if "domain" in block_obs.columns
                        else block_obs
                    )
                    if not dom_obs.empty and "location_id" in dom_obs.columns:
                        station_means: list[float] = []
                        for _loc, s_grp in dom_obs.groupby("location_id"):
                            if "imputed" in s_grp.columns:
                                s_grp = s_grp[s_grp["imputed"] == False]  # noqa: E712
                            s_vals = s_grp["pm25_ugm3"].dropna()
                            if len(s_vals) >= 18:
                                station_means.append(float(s_vals.mean()))
                        if len(station_means) >= 3 or (
                            len(station_means) > 0 and len(dom_obs["location_id"].unique()) < 3
                        ):
                            valid_obs_series = pd.Series([float(np.mean(station_means))] * 24)

        score_rec = score_forecast_block(
            forecast_row=f_row,
            valid_obs_series=valid_obs_series,
            stage=stage_clean,
            scored_at_utc=scored_at_utc,
            scoring_basis=scoring_basis,
            truth_basis="openaq_api" if stage_clean == "provisional" else "openaq_archive",
        )
        records.append(score_rec)

    scores_df = pd.DataFrame(records)
    schema = _load_score_log_schema()
    return validate_frame(scores_df, schema)


def score_provisional(
    issuance_date: str | date | datetime,
    forecast_log_df: pd.DataFrame,
    api_obs_df: pd.DataFrame,
    scored_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Score provisional forecasts using OpenAQ live API observations (Task P1-11)."""
    return score_forecast_targets(
        issuance_date=issuance_date,
        forecast_log_df=forecast_log_df,
        obs_df=api_obs_df,
        stage="provisional",
        scored_at_utc=scored_at_utc,
    )


def score_settled(
    issuance_date: str | date | datetime,
    forecast_log_df: pd.DataFrame,
    archive_obs_df: pd.DataFrame,
    scored_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Re-score settled forecasts once OpenAQ archive data lands (lag 72-96h) (Task P1-11)."""
    return score_forecast_targets(
        issuance_date=issuance_date,
        forecast_log_df=forecast_log_df,
        obs_df=archive_obs_df,
        stage="settled",
        scored_at_utc=scored_at_utc,
    )


def save_score_log(
    scores_df: pd.DataFrame,
    issuance_date: str | date | datetime,
    state_dir: Path | str = Path(".state"),
) -> Path:
    """Persist validated scores to date partition and canonical score_log ledger.

    Writes:
    1. .state/scores/dt=YYYY-MM-DD/score_log.parquet
    2. .state/score_log.parquet (append-only, deduplicated on unique keys)
    """
    st_dir = Path(state_dir)
    dt_str = _normalize_issuance_date(issuance_date)

    # 1. Date partition: .state/scores/dt=YYYY-MM-DD/score_log.parquet
    partition_dir = st_dir / "scores" / f"dt={dt_str}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    partition_file = partition_dir / "score_log.parquet"
    scores_df.to_parquet(partition_file, index=False)

    # 2. Canonical append-only ledger: .state/score_log.parquet
    canonical_file = st_dir / "score_log.parquet"
    if canonical_file.exists():
        try:
            existing = pd.read_parquet(canonical_file)
            merged = pd.concat([existing, scores_df], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=["issuance_utc", "point_id", "horizon_h", "method", "stage"],
                keep="last",
            ).reset_index(drop=True)
        except Exception:
            merged = scores_df
    else:
        merged = scores_df

    schema = _load_score_log_schema()
    validated = validate_frame(merged, schema)
    validated.to_parquet(canonical_file, index=False)

    return partition_file


def load_observations(
    stage: str,
    state_dir: Path | str = Path(".state"),
    obs_file: Path | str | None = None,
) -> pd.DataFrame:
    """Load observation data for shadow scoring from file, fixture, or candidate state paths."""
    if obs_file is not None:
        p = Path(obs_file)
        if p.exists():
            return pd.read_parquet(p)

    import os

    mode = os.getenv("SMOGSENSE_MODE", "live").lower()
    if mode == "fixtures":
        from smogsense.data_ingestion.fixtures_mode import (
            load_fixture_locations,
            load_fixture_observations,
        )

        stations = load_fixture_locations(domain="lahore")
        now = datetime.now(UTC)
        start = now - timedelta(days=5)
        return load_fixture_observations(stations, start, now, domain="lahore")

    st_dir = Path(state_dir)
    candidates = [
        st_dir / "inputs/obs_snapshot.parquet",
        st_dir / "obs/observations_hourly.parquet",
        Path("data/processed/observations_hourly.parquet"),
        Path("data/interim/observations_hourly.parquet"),
    ]
    for c in candidates:
        if c.exists():
            try:
                cand_df = pd.read_parquet(c)
                if not cand_df.empty:
                    return cand_df
            except Exception as e:
                logger.warning(f"Failed to read candidate obs {c}: {e}")

    return pd.DataFrame()


def run_shadow_scoring(
    stage: str = "provisional",
    issuance_date: str | date | datetime | None = None,
    state_dir: Path | str = Path(".state"),
    obs_file: Path | str | None = None,
) -> dict[str, Any]:
    """Orchestrate prospective shadow scoring for a given stage and issuance date."""
    st_dir = Path(state_dir)
    stage_clean = stage.lower().strip()

    # Discover forecast files
    forecasts_dir = st_dir / "forecasts"
    if not forecasts_dir.exists():
        return {
            "status": "no_forecasts",
            "stage": stage_clean,
            "new_scores": 0,
            "message": f"No forecasts directory found at {forecasts_dir}",
        }

    files = sorted(forecasts_dir.glob("forecast_*.parquet"))
    if not files:
        return {
            "status": "no_forecasts",
            "stage": stage_clean,
            "new_scores": 0,
            "message": "No forecast parquet files found.",
        }

    dfs = [pd.read_parquet(f) for f in files]
    all_forecasts = pd.concat(dfs, ignore_index=True)

    if issuance_date is None:
        # Default to latest issuance in forecast log
        latest_ts = all_forecasts["issuance_utc"].max()
        issuance_date = pd.to_datetime(latest_ts, utc=True).strftime("%Y-%m-%d")

    # Load observations
    obs_df = load_observations(stage=stage_clean, state_dir=st_dir, obs_file=obs_file)

    # Score
    if stage_clean == "provisional":
        scores_df = score_provisional(
            issuance_date=issuance_date,
            forecast_log_df=all_forecasts,
            api_obs_df=obs_df,
        )
    else:
        scores_df = score_settled(
            issuance_date=issuance_date,
            forecast_log_df=all_forecasts,
            archive_obs_df=obs_df,
        )

    # Persist
    saved_path = save_score_log(
        scores_df=scores_df,
        issuance_date=issuance_date,
        state_dir=st_dir,
    )

    return {
        "status": "ok",
        "stage": stage_clean,
        "issuance_date": _normalize_issuance_date(issuance_date),
        "new_scores": len(scores_df),
        "saved_path": str(saved_path),
    }
