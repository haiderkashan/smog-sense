#!/usr/bin/env bash
# Rotate historical state journal partitions for a given year to a release bundle archive.
# Prunes archived year's partitions from the active state branch to enforce size budgets.
# Usage: scripts/rotate_state_branch.sh <archive_year> [state_dir] [bundle_out_dir]
set -euo pipefail

archive_year="${1:?archive_year required (e.g. 2025)}"
state_dir="${2:-.state}"
bundle_out_dir="${3:-data/processed/state_archives}"

python3 -c "
import sys
from pathlib import Path
from smogsense.pipeline.state_guard import rotate_state_year

year = int(sys.argv[1])
state_p = Path(sys.argv[2])
out_p = Path(sys.argv[3])

res = rotate_state_year(state_p, year, out_p)
print(f'Rotated year {year}: {res[\"files_archived\"]} files archived, {res[\"files_pruned\"]} files pruned.')
if res['bundle_path']:
    print(f'Archive bundle written to: {res[\"bundle_path\"]}')
" "${archive_year}" "${state_dir}" "${bundle_out_dir}"
