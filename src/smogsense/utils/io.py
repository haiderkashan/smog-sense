"""smogsense.utils.io - I/O utilities including Pandera validation wrappers.

Specification: docs/data-engineering.md -> 'Data contracts'
"""

import contextlib
import os
import re
import shutil
import subprocess

import pandas as pd
import pandera.pandas as pa

from smogsense.errors import SchemaViolation


def get_git_sha() -> str:
    """Retrieve 7-character Git SHA from environment or git rev-parse HEAD.
    Falls back to a valid 7-character hex string if git is unavailable.
    """
    for env_var in ("GITHUB_SHA", "GIT_SHA"):
        val = os.environ.get(env_var, "").strip()
        if val and re.match(r"^[0-9a-f]{7,40}$", val):
            return val[:7]
    with contextlib.suppress(Exception):
        git_cmd = shutil.which("git") or "git"
        res = subprocess.run(  # noqa: S603
            [git_cmd, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        sha = res.stdout.strip()
        if sha and re.match(r"^[0-9a-f]{7,40}$", sha):
            return sha[:7]
    return "unknown"


def validate_frame(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Validates a DataFrame against a Pandera schema.
    Catches Pandera exceptions and duplicate indices, converting them to SchemaViolation.
    """
    if df.index.duplicated().any():
        raise SchemaViolation("Duplicate index found in dataframe before validation.")

    try:
        # Reset index internally to avoid Pandera index validation quirk (ADR-014 / memory.md §8)
        clean_df = df.reset_index(drop=True)
        return schema.validate(clean_df)
    except pa.errors.SchemaError as e:
        raise SchemaViolation(f"Schema violation: {e}") from e
    except pa.errors.SchemaErrors as e:
        raise SchemaViolation(f"Multiple schema violations: {e}") from e
