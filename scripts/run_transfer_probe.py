"""scripts/run_transfer_probe.py — Early Transfer Probe (Phase 2 Task P2-04 / Gate G2).

Evaluates zero-shot transfer of the frozen LightGBM physics base (M5) trained
exclusively on Delhi source seasons (Block E) and tested on held-out Lahore ground truth
monitoring data (US Consulate 8664 reference monitor and network).

Gate G2 Pass Condition:
  CRPSS(M5, N=0) > 0 against operational baseline M2 (CAMS-BC) and M1 (Raw CAMS).

Specification: docs/ml-architecture.md §6 & docs/roadmap.md §Gate G2.
"""

from __future__ import annotations

import argparse

from smogsense.models.lgbm_quantile import run_early_transfer_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Early Transfer Probe (Gate G2 / P2-04).")
    parser.add_argument(
        "--feature-store-dir",
        default="data/processed/feature_store",
        help="Feature store base directory",
    )
    parser.add_argument(
        "--subsample-train",
        type=int,
        default=None,
        help="Subsample Delhi training rows for rapid execution",
    )
    parser.add_argument(
        "--location-id",
        type=int,
        default=8664,
        help="Target Lahore station ID to evaluate",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=120,
        help="Number of boosting iterations",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write JSON probe report",
    )
    args = parser.parse_args()

    run_early_transfer_probe(
        feature_store_dir=args.feature_store_dir,
        subsample_train=args.subsample_train,
        eval_location_id=args.location_id,
        num_boost_round=args.num_boost_round,
        out_json=args.output_json,
    )


if __name__ == "__main__":
    main()
