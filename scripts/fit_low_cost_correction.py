"""scripts/fit_low_cost_correction.py — Empirical low-cost sensor correction fitting and audit.

Fits a linear humidity-aware empirical correction model (Barkjohn et al., 2021)
against reference monitor observations in Lahore, evaluates 5-fold cross-validation,
applies the PRD FR-11 decision rule, and audits inverted corruption priors against
configs/model.yaml.

Usage:
    python scripts/fit_low_cost_correction.py --domain lahore --update-config --output-json reports/low_cost_correction.json
"""

from __future__ import annotations

import argparse

from smogsense.preprocessing.low_cost_correction import run_fitting_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit, validate, and audit empirical low-cost sensor correction."
    )
    parser.add_argument("--domain", default="lahore", help="Target domain (default: lahore)")
    parser.add_argument(
        "--backfill-dir",
        default="data/processed/backfill",
        help="Path to backfilled observation partitions",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Path to registry parquet file (default: auto)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=60,
        help="Window in days for pairing (default: 60)",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=8.0,
        help="Maximum distance in km from reference monitor (default: 8.0)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=500,
        help="Minimum required hourly pairs (default: 500)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Path to write JSON validation and audit report",
    )
    parser.add_argument(
        "--output-model",
        default=None,
        help="Path to save serialized model (JSON or Parquet)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update configs/preprocessing.yaml based on decision rule",
    )
    parser.add_argument(
        "--config-path",
        default="configs/preprocessing.yaml",
        help="Path to preprocessing.yaml",
    )
    parser.add_argument(
        "--model-config-path",
        default="configs/model.yaml",
        help="Path to model.yaml",
    )

    args = parser.parse_args()
    run_fitting_pipeline(
        domain=args.domain,
        backfill_dir=args.backfill_dir,
        registry_path=args.registry_path,
        window_days=args.window_days,
        max_distance_km=args.max_distance_km,
        min_pairs=args.min_pairs,
        output_json=args.output_json,
        output_model=args.output_model,
        update_config=args.update_config,
        config_path=args.config_path,
        model_config_path=args.model_config_path,
    )


if __name__ == "__main__":
    main()
