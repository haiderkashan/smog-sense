"""scripts/upload_to_release.py — Upload assets to GitHub release data-backfill.

Uses git credentials or GITHUB_TOKEN to authenticate against GitHub REST API.
Supports replacing existing assets (--clobber).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx


def get_github_token() -> str:
    """Retrieve GitHub token from environment or git credential helper."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],  # noqa: S607
            input="protocol=https\nhost=github.com\n",
            capture_output=True,
            text=True,
            check=True,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except Exception as e:
        print(f"Error fetching credentials via git: {e}", file=sys.stderr)
    raise RuntimeError("Could not find GitHub token in GITHUB_TOKEN or git credential helper.")


def upload_single_file(
    file_path: Path,
    client: httpx.Client,
    upload_url_base: str,
    headers: dict[str, str],
    existing_assets: dict[str, int],
    repo: str,
) -> None:
    """Upload a single file, deleting the existing asset if present."""
    if not file_path.exists():
        return

    name = file_path.name
    if name in existing_assets:
        asset_id = existing_assets[name]
        del_resp = client.delete(
            f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}",
            headers=headers,
        )
        if del_resp.status_code not in (204, 200):
            print(f"Warning: Failed to delete asset {name}: {del_resp.status_code}")

    size = file_path.stat().st_size
    upload_headers = {
        **headers,
        "Content-Type": "application/octet-stream",
        "Content-Length": str(size),
    }
    with file_path.open("rb") as f:
        file_data = f.read()

    up_resp = client.post(
        f"{upload_url_base}?name={name}",
        headers=upload_headers,
        content=file_data,
    )
    if up_resp.status_code == 201:
        print(f"Uploaded: {name} ({size / 1024:.1f} KB)")
    else:
        print(f"Error uploading {name}: {up_resp.status_code} {up_resp.text}", file=sys.stderr)
        raise RuntimeError(f"Failed to upload {name}")


def upload_assets(
    files: list[Path],
    repo: str = "haiderkashan/smog-sense",
    tag: str = "data-backfill",
    concurrency: int = 5,
) -> None:
    """Upload one or more files to the specified GitHub release, clobbering existing."""
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "SmogSense-Release-Uploader",
    }

    client = httpx.Client(timeout=120.0)

    # 1. Fetch release info
    rel_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    resp = client.get(rel_url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to get release {tag}: {resp.status_code} {resp.text}")

    rel_data: dict[str, Any] = resp.json()
    release_id = rel_data["id"]
    existing_assets = {a["name"]: a["id"] for a in rel_data.get("assets", [])}
    print(
        f"Connected to release '{tag}' (id={release_id}). Existing assets: {len(existing_assets)}"
    )

    # 2. Upload files concurrently
    upload_url_base = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets"
    print(f"Uploading {len(files)} file(s) with concurrency {concurrency} ...")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                upload_single_file,
                f,
                client,
                upload_url_base,
                headers,
                existing_assets,
                repo,
            ): f.name
            for f in files
        }
        for future in as_completed(futures):
            future.result()

    print(f"Successfully processed {len(files)} files for release '{tag}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload assets to GitHub release.")
    parser.add_argument("files", nargs="+", help="Files or glob patterns to upload")
    parser.add_argument("--repo", default="haiderkashan/smog-sense", help="GitHub repo")
    parser.add_argument("--tag", default="data-backfill", help="Release tag")
    parser.add_argument("--concurrency", type=int, default=5, help="Upload concurrency")
    args = parser.parse_args()

    file_paths: list[Path] = []
    for item in args.files:
        p = Path(item)
        if any(char in item for char in ["*", "?", "["]):
            matched = sorted(Path().glob(item))
            file_paths.extend(matched)
        elif p.exists():
            file_paths.append(p)
        else:
            print(f"Warning: pattern or file not found: {item}")

    upload_assets(file_paths, repo=args.repo, tag=args.tag, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
