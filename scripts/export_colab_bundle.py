"""scripts/export_colab_bundle.py — Package and publish feature store bundle for Colab.

Bundles data/processed/feature_store/ and data/registry/ into a single compressed
archive: data/processed/feature_store_bundle.tar.gz.
Optionally uploads the artifact to the GitHub release 'data-backfill'.
"""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path


def create_bundle(
    feature_store_dir: str | Path = "data/processed/feature_store",
    registry_dir: str | Path = "data/registry",
    output_bundle: str | Path = "data/processed/feature_store_bundle.tar.gz",
) -> Path:
    """Create compressed tar.gz bundle of feature store and station registries."""
    fs_path = Path(feature_store_dir)
    reg_path = Path(registry_dir)
    out_path = Path(output_bundle)

    if not fs_path.exists():
        raise FileNotFoundError(f"Feature store directory not found: {fs_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating bundle from {fs_path} and {reg_path} -> {out_path} ...")

    with tarfile.open(out_path, "w:gz") as tar:
        # Add feature store partitions
        tar.add(fs_path, arcname="feature_store")
        # Add registries if available
        if reg_path.exists():
            tar.add(reg_path, arcname="registry")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Bundle successfully created: {out_path} ({size_mb:.2f} MB)")
    return out_path


def upload_to_release(
    bundle_path: Path,
    release_tag: str = "data-backfill",
) -> None:
    """Upload bundle to GitHub release using the gh CLI."""
    print(f"Uploading {bundle_path} to GitHub release '{release_tag}' ...")
    gh_cmd = ["gh", "release", "upload", release_tag, str(bundle_path), "--clobber"]
    result = subprocess.run(gh_cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        print(f"Error uploading via gh: {result.stderr}")
        raise RuntimeError(f"gh release upload failed: {result.stderr}")
    print(f"Successfully uploaded {bundle_path.name} to release {release_tag}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feature store bundle for Google Colab.")
    parser.add_argument(
        "--feature-store-dir",
        default="data/processed/feature_store",
        help="Feature store root directory",
    )
    parser.add_argument(
        "--registry-dir",
        default="data/registry",
        help="Station registry directory",
    )
    parser.add_argument(
        "--output",
        default="data/processed/feature_store_bundle.tar.gz",
        help="Output tar.gz bundle path",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload bundle to GitHub release 'data-backfill'",
    )
    parser.add_argument(
        "--release-tag",
        default="data-backfill",
        help="GitHub release tag to upload to",
    )
    args = parser.parse_args()

    bundle = create_bundle(
        feature_store_dir=args.feature_store_dir,
        registry_dir=args.registry_dir,
        output_bundle=args.output,
    )

    if args.upload:
        upload_to_release(bundle, release_tag=args.release_tag)


if __name__ == "__main__":
    main()
