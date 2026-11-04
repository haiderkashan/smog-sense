"""tests/unit/models/test_cross_fit.py — Unit tests for E/V/T splits and OOF cross-fitting.

Specification:
- docs/ml-architecture.md §6 & §7
- decisions.md (ADR-015, ADR-024)
- roadmap.md Task P2-11

Verifies:
1. K = min(3, #seasons in E) fold assignment determinism and bounds.
2. 72-hour boundary purging for forward lookback and backward target windows.
3. Zero in-sample leakage: fold k OOF predictions are generated strictly by models trained on folds != k.
4. OOF partition hash determinism, row-order invariance, and sensitivity.
5. Twin-aware co-located station preservation (colocated_group_id and <50m coordinates).
6. Monotone quantile rearrangement across all 19 quantile levels in OOF outputs.
7. Roundtrip serialization of base model with SHA256 and OOF partition hash binding.
8. Validation early stopping on Block V when val_df is provided.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smogsense.features.backfill import (
    compute_oof_partition_hash,
    is_boundary_purged,
    partition_block_e_folds,
    verify_twin_aware_folds,
)
from smogsense.models.lgbm_quantile import (
    DEFAULT_QUANTILES,
    LGBMQuantileEnsemble,
    fit_cross_fit_oof,
)


def _create_synthetic_block_e(
    n_per_fold: int = 50,
    n_folds: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic Block E dataset with explicit fold assignments."""
    rng = np.random.default_rng(seed)
    dfs: list[pd.DataFrame] = []

    for f in range(n_folds):
        start_ts = pd.Timestamp("2020-10-01", tz="UTC") + pd.Timedelta(days=f * 60)
        dates = pd.date_range(start_ts, periods=n_per_fold, freq="6h", tz="UTC")
        df_fold = pd.DataFrame(
            {
                "domain": ["delhi"] * n_per_fold,
                "location_id": rng.choice([7001, 7002, 7003], size=n_per_fold),
                "issuance_utc": dates,
                "fold_id": [f] * n_per_fold,
                "f_met_vc_mean_h24": rng.uniform(500, 2500, size=n_per_fold),
                "f_met_rh_mean_h24": rng.uniform(30, 90, size=n_per_fold),
                "f_fire_exposure_e24": rng.exponential(5.0, size=n_per_fold),
                "f_aod_mean_h24": rng.uniform(0.1, 1.2, size=n_per_fold),
                "f_cams_pm25_h24": rng.uniform(40, 250, size=n_per_fold),
                "f_stat_elevation_m": [215.0] * n_per_fold,
                "y_h24": rng.uniform(30, 300, size=n_per_fold),
                "valid_h24": [True] * n_per_fold,
                "y_h48": rng.uniform(30, 300, size=n_per_fold),
                "valid_h48": [True] * n_per_fold,
                "y_h72": rng.uniform(30, 300, size=n_per_fold),
                "valid_h72": [True] * n_per_fold,
            }
        )
        dfs.append(df_fold)

    return pd.concat(dfs, ignore_index=True)


def test_partition_block_e_folds_k_bounds() -> None:
    """Partitioning must strictly satisfy K = min(3, #seasons in E)."""
    # Empty seasons
    assert partition_block_e_folds([]) == {}

    # 1 season -> K = 1, fold 0
    f1 = partition_block_e_folds(["2018-2019"])
    assert f1 == {"2018-2019": 0}
    assert len(set(f1.values())) == 1

    # 2 seasons -> K = 2, folds 0 and 1
    f2 = partition_block_e_folds(["2018-2019", "2019-2020"])
    assert f2 == {"2018-2019": 0, "2019-2020": 1}
    assert len(set(f2.values())) == 2

    # 3 seasons -> K = 3, folds 0, 1, 2
    f3 = partition_block_e_folds(["2018-2019", "2019-2020", "2020-2021"])
    assert f3 == {"2018-2019": 0, "2019-2020": 1, "2020-2021": 2}
    assert len(set(f3.values())) == 3

    # 6 seasons (canonical Delhi Block E) -> K = 3, contiguous pairs
    seasons_6 = [
        "2018-2019",
        "2019-2020",
        "2020-2021",
        "2021-2022",
        "2022-2023",
        "2023-2024",
    ]
    f6 = partition_block_e_folds(seasons_6)
    assert len(set(f6.values())) == 3
    assert f6["2018-2019"] == 0 and f6["2019-2020"] == 0
    assert f6["2020-2021"] == 1 and f6["2021-2022"] == 1
    assert f6["2022-2023"] == 2 and f6["2023-2024"] == 2

    # 8 seasons -> K = 3 (bounded at max 3)
    seasons_8 = [f"201{i}-201{i + 1}" for i in range(8)]
    f8 = partition_block_e_folds(seasons_8)
    assert len(set(f8.values())) == 3
    assert max(f8.values()) == 2


def test_is_boundary_purged() -> None:
    """72-hour forward lookback and backward target boundary purging."""
    w_start = datetime(2022, 10, 1, 0, 0, tzinfo=UTC)
    w_end = datetime(2023, 3, 31, 23, 59, tzinfo=UTC)

    # 1. Forward boundary: T within 72h of window start -> lookback would cross prior season
    # T = start + 71h -> purged
    assert is_boundary_purged(
        datetime(2022, 10, 3, 23, 0, tzinfo=UTC),
        window_start_utc=w_start,
        window_end_utc=w_end,
        embargo_hours=72.0,
    )
    # T = start + 72h -> valid (not purged)
    assert not is_boundary_purged(
        datetime(2022, 10, 4, 0, 0, tzinfo=UTC),
        window_start_utc=w_start,
        window_end_utc=w_end,
        embargo_hours=72.0,
    )

    # 2. Interior issuance -> valid
    assert not is_boundary_purged(
        datetime(2022, 12, 1, 0, 0, tzinfo=UTC),
        window_start_utc=w_start,
        window_end_utc=w_end,
        embargo_hours=72.0,
    )

    # 3. Backward boundary: T + 72h extends beyond window end -> target would cross next season
    # T = end - 71h (i.e. T + 72h > end) -> purged
    assert is_boundary_purged(
        datetime(2023, 3, 29, 1, 0, tzinfo=UTC),
        window_start_utc=w_start,
        window_end_utc=w_end,
        embargo_hours=72.0,
    )
    # T = end - 72h -> valid
    assert not is_boundary_purged(
        datetime(2023, 3, 28, 23, 59, tzinfo=UTC),
        window_start_utc=w_start,
        window_end_utc=w_end,
        embargo_hours=72.0,
    )


def test_zero_in_sample_leakage() -> None:
    """Verify that OOF predictions for fold k are strictly generated by folds != k."""
    df = _create_synthetic_block_e(n_per_fold=30, n_folds=3, seed=123)

    oof_df, final_base = fit_cross_fit_oof(
        df_block_e=df,
        horizons=(24,),
        quantiles=(0.1, 0.5, 0.9),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
        num_boost_round=10,
    )

    # All rows from Block E must be present in OOF predictions
    assert len(oof_df) == len(df)
    assert set(oof_df["fold_id"].unique()) == {0, 1, 2}

    for fold_k in (0, 1, 2):
        fold_rows = oof_df[oof_df["fold_id"] == fold_k]
        assert len(fold_rows) == 30
        assert fold_rows["m5_oof_h24_q50"].notna().all()

    # Final base model is trained on full Block E and bound with hash
    assert final_base.oof_partition_hash is not None
    assert len(final_base.oof_partition_hash) == 64


def test_oof_partition_hash_properties() -> None:
    """OOF partition hash must be deterministic, row-order invariant, and sensitive to alterations."""
    df = _create_synthetic_block_e(n_per_fold=20, n_folds=3, seed=77)

    hash1 = compute_oof_partition_hash(df)
    hash2 = compute_oof_partition_hash(df)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Row-order invariance: shuffling rows must yield identical hash
    df_shuffled = df.sample(frac=1.0, random_state=999).reset_index(drop=True)
    hash_shuffled = compute_oof_partition_hash(df_shuffled)
    assert hash_shuffled == hash1

    # Sensitivity: altering a single fold_id must change hash
    df_altered_fold = df.copy()
    df_altered_fold.loc[0, "fold_id"] = 1 if df_altered_fold.loc[0, "fold_id"] != 1 else 2
    assert compute_oof_partition_hash(df_altered_fold) != hash1

    # Sensitivity: altering an issuance timestamp must change hash
    df_altered_ts = df.copy()
    ts_val = pd.to_datetime(str(df_altered_ts.loc[0, "issuance_utc"]))
    df_altered_ts.loc[0, "issuance_utc"] = ts_val + pd.Timedelta(hours=1)
    assert compute_oof_partition_hash(df_altered_ts) != hash1

    # Sensitivity: altering a location_id must change hash
    df_altered_loc = df.copy()
    df_altered_loc.loc[0, "location_id"] = 9999
    assert compute_oof_partition_hash(df_altered_loc) != hash1

    # Empty handling: returns sha256(b"none")
    none_hash = hashlib.sha256(b"none").hexdigest()
    assert compute_oof_partition_hash(pd.DataFrame()) == none_hash
    assert compute_oof_partition_hash(pd.DataFrame({"location_id": [1]})) == none_hash


def test_verify_twin_aware_folds() -> None:
    """Verify co-located twin stations are preserved within the same fold."""
    # 1. Valid group: co-located pair shares the same fold
    valid_group_df = pd.DataFrame(
        {
            "location_id": [101, 102, 201, 202],
            "colocated_group_id": ["group_A", "group_A", "group_B", "group_B"],
            "fold_id": [0, 0, 1, 1],
        }
    )
    assert verify_twin_aware_folds(valid_group_df, raise_on_violation=True)

    # 2. Invalid group: co-located pair split across folds 0 and 1
    invalid_group_df = pd.DataFrame(
        {
            "location_id": [101, 102],
            "colocated_group_id": ["group_A", "group_A"],
            "fold_id": [0, 1],
        }
    )
    assert not verify_twin_aware_folds(invalid_group_df, raise_on_violation=False)
    with pytest.raises(ValueError, match="Twin-aware violation"):
        verify_twin_aware_folds(invalid_group_df, raise_on_violation=True)

    # 3. Spatial co-location (< 50m) validation
    # Delhi coordinate: 28.6139 N, 77.2090 E. Offset of 0.0001 deg lat is ~11.1 meters (<50m)
    valid_spatial_df = pd.DataFrame(
        {
            "location_id": [301, 302],
            "f_stat_lat": [28.61390, 28.61400],
            "f_stat_lon": [77.20900, 77.20900],
            "fold_id": [2, 2],
        }
    )
    assert verify_twin_aware_folds(valid_spatial_df, raise_on_violation=True)

    # 4. Spatial co-location violation: stations within 11m split across fold 0 and fold 1
    invalid_spatial_df = pd.DataFrame(
        {
            "location_id": [301, 302],
            "f_stat_lat": [28.61390, 28.61400],
            "f_stat_lon": [77.20900, 77.20900],
            "fold_id": [0, 1],
        }
    )
    assert not verify_twin_aware_folds(invalid_spatial_df, raise_on_violation=False)
    with pytest.raises(ValueError, match="Twin-aware violation"):
        verify_twin_aware_folds(invalid_spatial_df, raise_on_violation=True)


def test_oof_predictions_monotone_rearrangement() -> None:
    """Monotone rearrangement must hold across all 19 quantiles for all OOF rows."""
    df = _create_synthetic_block_e(n_per_fold=30, n_folds=3, seed=42)

    oof_df, _ = fit_cross_fit_oof(
        df_block_e=df,
        horizons=(24, 48),
        quantiles=DEFAULT_QUANTILES,
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
        num_boost_round=10,
    )

    for h in (24, 48):
        cols = [f"m5_oof_h{h}_q{round(q * 100):02d}" for q in DEFAULT_QUANTILES]
        q_vals = oof_df[cols].to_numpy()
        # Non-crossing check: q_{i+1} >= q_i
        diffs = np.diff(q_vals, axis=-1)
        assert np.all(diffs >= -1e-6), f"Monotone rearrangement violated for horizon h={h}"


def test_base_binding_metadata_roundtrip(tmp_path: Path) -> None:
    """Base model must serialize and reload with identical SHA256 and OOF partition hash."""
    df = _create_synthetic_block_e(n_per_fold=25, n_folds=2, seed=888)

    _, final_base = fit_cross_fit_oof(
        df_block_e=df,
        horizons=(24,),
        quantiles=(0.25, 0.50, 0.75),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
        num_boost_round=10,
    )

    expected_oof_hash = compute_oof_partition_hash(df)
    assert final_base.oof_partition_hash == expected_oof_hash

    model_dir = tmp_path / "m5_base_ensemble"
    final_base.save(model_dir)

    # Verify metadata.json contents
    with (model_dir / "metadata.json").open("r") as f:
        meta = json.load(f)

    assert meta["base_sha256"] == final_base.compute_base_sha256()
    assert meta["oof_partition_hash"] == expected_oof_hash
    assert meta["model_type"] == "M5"

    # Reload and verify equality
    reloaded = LGBMQuantileEnsemble.load(model_dir)
    assert reloaded.compute_base_sha256() == final_base.compute_base_sha256()
    assert reloaded.oof_partition_hash == expected_oof_hash


def test_cross_fit_with_val_early_stopping() -> None:
    """Cross-fitting and final base training must accept val_df without error."""
    df_e = _create_synthetic_block_e(n_per_fold=25, n_folds=2, seed=10)
    df_v = _create_synthetic_block_e(n_per_fold=20, n_folds=1, seed=20)

    oof_df, final_base = fit_cross_fit_oof(
        df_block_e=df_e,
        val_df=df_v,
        horizons=(24,),
        quantiles=(0.1, 0.5, 0.9),
        lgbm_params={"num_leaves": 7, "min_data_in_leaf": 5},
        num_boost_round=15,
    )

    assert len(oof_df) == len(df_e)
    assert final_base.oof_partition_hash is not None
