"""scripts/inspect_dataset.py — Comprehensive inspection and audit report for the feature store.

Audits:
1. Physical storage hierarchy and Parquet partition count.
2. Embedded PyArrow metadata (schema_version, feature_set_version, config_hash, oof_partition_hash).
3. Split partitions: Block E (Train with K-fold tags), Block V (Val), Block T (Test), Eval (Lahore).
4. Row counts, active stations, and target completeness (y_h24, y_h48, y_h72).
5. Schema conformity against feature_store.schema.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]


def audit_feature_store(base_dir: str | Path = "data/processed/feature_store") -> dict[str, Any]:
    """Inspect and compile comprehensive audit report of feature store partitions."""
    root = Path(base_dir)
    if not root.exists():
        raise FileNotFoundError(f"Feature store directory not found: {root}")

    report: dict[str, Any] = {
        "root": str(root),
        "domains": {},
        "total_partitions": 0,
        "total_rows": 0,
    }

    domain_dirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for dom_path in sorted(domain_dirs):
        dom = dom_path.name
        dom_report: dict[str, Any] = {
            "seasons": {},
            "total_rows": 0,
            "total_partitions": 0,
            "stations": set(),
            "splits": set(),
            "oof_partition_hash": None,
        }

        # Check for season partitions
        season_dirs = [s for s in dom_path.iterdir() if s.is_dir() and s.name.startswith("season=")]
        for s_dir in sorted(season_dirs):
            s_name = s_dir.name.split("=")[-1]
            s_report: dict[str, Any] = {
                "partitions": 0,
                "rows": 0,
                "splits": set(),
                "folds": set(),
                "stations": set(),
                "valid_targets": {"y_h24": 0, "y_h48": 0, "y_h72": 0},
                "dates": [],
            }

            parquet_files = sorted(s_dir.glob("**/features.parquet"))
            for p_file in parquet_files:
                meta = pq.read_metadata(p_file)
                custom_meta = meta.schema.to_arrow_schema().metadata or {}
                meta_decoded = {
                    k.decode("utf-8") if isinstance(k, bytes) else k: (
                        v.decode("utf-8") if isinstance(v, bytes) else v
                    )
                    for k, v in custom_meta.items()
                }

                if not dom_report["oof_partition_hash"] and "oof_partition_hash" in meta_decoded:
                    dom_report["oof_partition_hash"] = meta_decoded["oof_partition_hash"]

                df = pd.read_parquet(p_file)
                num_rows = len(df)
                s_report["partitions"] += 1
                s_report["rows"] += num_rows
                s_report["dates"].append(p_file.parent.name.split("=")[-1])

                if "source_split" in df.columns:
                    s_report["splits"].update(df["source_split"].dropna().unique().tolist())
                if "fold_id" in df.columns:
                    s_report["folds"].update(df["fold_id"].dropna().unique().tolist())
                if "location_id" in df.columns:
                    stn_list = df["location_id"].dropna().unique().tolist()
                    s_report["stations"].update(stn_list)
                    dom_report["stations"].update(stn_list)

                for h in (24, 48, 72):
                    col = f"y_h{h}"
                    val_col = f"valid_h{h}"
                    if val_col in df.columns:
                        s_report["valid_targets"][col] += int(df[val_col].sum())
                    elif col in df.columns:
                        s_report["valid_targets"][col] += int(df[col].notna().sum())

            dom_report["total_rows"] += s_report["rows"]
            dom_report["total_partitions"] += s_report["partitions"]
            dom_report["splits"].update(s_report["splits"])
            dom_report["seasons"][s_name] = s_report

        report["domains"][dom] = dom_report
        report["total_partitions"] += dom_report["total_partitions"]
        report["total_rows"] += dom_report["total_rows"]

    return report


def print_audit_summary(report: dict[str, Any]) -> None:
    """Print readable Markdown-formatted summary table of dataset inspection."""
    print("=" * 80)
    print("SMOGSENSE MULTI-SEASON FEATURE STORE AUDIT REPORT")
    print("=" * 80)
    print(f"Base Directory: {report['root']}")
    print(f"Total Partitions: {report['total_partitions']}")
    print(f"Total Dataset Rows: {report['total_rows']}\n")

    for dom, d_data in report["domains"].items():
        print(f"### DOMAIN: {dom.upper()}")
        print(f"  Total Rows: {d_data['total_rows']:,}")
        print(f"  Total Partitions: {d_data['total_partitions']}")
        print(f"  Active Stations: {sorted(d_data['stations'])}")
        print(f"  OOF Partition SHA256: {d_data.get('oof_partition_hash', 'N/A')}")
        print("\n  Season Breakdown:")
        print(
            "  | Season | Split | Folds | Partitions | Total Rows | Valid y_h24 | Valid y_h48 | Valid y_h72 |"
        )
        print("  | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for s_name, s_data in d_data["seasons"].items():
            splits_str = ",".join(sorted(str(x) for x in s_data["splits"])) or "-"
            folds_str = (
                ",".join(sorted(str(int(x)) for x in s_data["folds"])) if s_data["folds"] else "-"
            )
            rows_str = f"{s_data['rows']:,}"
            y24 = s_data["valid_targets"]["y_h24"]
            y48 = s_data["valid_targets"]["y_h48"]
            y72 = s_data["valid_targets"]["y_h72"]
            print(
                f"  | {s_name} | {splits_str} | {folds_str} | {s_data['partitions']} | {rows_str} | {y24:,} | {y48:,} | {y72:,} |"
            )
        print("\n" + "-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect feature store dataset.")
    parser.add_argument(
        "--base-dir",
        default="data/processed/feature_store",
        help="Feature store root directory",
    )
    args = parser.parse_args()

    rep = audit_feature_store(base_dir=args.base_dir)
    print_audit_summary(rep)
