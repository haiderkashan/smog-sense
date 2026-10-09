#!/usr/bin/env bash
# Attach the orphan `state` branch as a separate clone in ./.state (creates the branch on first run).
# The state branch is an append-only journal of small Parquet files:
#   obs/ forecasts/ inputs/ scores/ manifests/   — see docs/system-architecture.md
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

STATE_DIR="${STATE_DIR:-.state}"
rm -rf "${STATE_DIR}"

# Check remote state branch existence treating ONLY exit 2 as "branch missing".
# Other exit codes (like 128) indicate auth or network failures and must not attempt branch creation.
rc=0
gh_git ls-remote --exit-code --heads "${REPO_URL}" state >/dev/null 2>&1 || rc=$?

if [ "${rc}" -eq 0 ]; then
  gh_git -c core.hooksPath=/dev/null -c core.fsmonitor=false clone --quiet --depth 1 --branch state "${REPO_URL}" "${STATE_DIR}"
  echo "state: attached existing branch ($(git -C "${STATE_DIR}" rev-parse --short HEAD))"
elif [ "${rc}" -eq 2 ]; then
  echo "state: branch not found, creating orphan branch"
  git init --quiet -b state "${STATE_DIR}"
  git -C "${STATE_DIR}" -c core.hooksPath=/dev/null -c core.fsmonitor=false remote add origin "${REPO_URL}"
  git_identity "${STATE_DIR}"
  printf '# SmogSense state journal\n\nAppend-only Parquet/JSON written by the daily pipeline. Do not edit by hand.\n' > "${STATE_DIR}/README.md"
  printf '*.grib\n*.grib2\n*.idx\n*.nc\n*.nc4\n*.bin\n' > "${STATE_DIR}/.gitignore"
  git -C "${STATE_DIR}" -c core.hooksPath=/dev/null -c core.fsmonitor=false add README.md .gitignore
  git -C "${STATE_DIR}" -c core.hooksPath=/dev/null -c core.fsmonitor=false commit --quiet -m "state: initialise orphan branch"
  gh_git -C "${STATE_DIR}" -c core.hooksPath=/dev/null -c core.fsmonitor=false push --quiet origin HEAD:state
else
  echo "state: git ls-remote failed with exit code ${rc} (auth or connection error)" >&2
  exit "${rc}"
fi

# Ensure .gitignore in state clone ignores binary files
if [ ! -f "${STATE_DIR}/.gitignore" ]; then
  printf '*.grib\n*.grib2\n*.idx\n*.nc\n*.nc4\n*.bin\n' > "${STATE_DIR}/.gitignore"
fi

git_identity "${STATE_DIR}"
