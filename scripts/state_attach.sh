#!/usr/bin/env bash
# Attach the orphan `state` branch as a separate clone in ./.state (creates the branch on first run).
# The state branch is an append-only journal of small Parquet files:
#   obs/ forecasts/ inputs/ scores/ manifests/   — see docs/system-architecture.md
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

STATE_DIR="${STATE_DIR:-.state}"
rm -rf "${STATE_DIR}"

if gh_git ls-remote --exit-code --heads "${REPO_URL}" state >/dev/null 2>&1; then
  gh_git clone --quiet --depth 1 --branch state "${REPO_URL}" "${STATE_DIR}"
  echo "state: attached existing branch ($(git -C "${STATE_DIR}" rev-parse --short HEAD))"
else
  echo "state: branch not found, creating orphan branch"
  git init --quiet -b state "${STATE_DIR}"
  git -C "${STATE_DIR}" remote add origin "${REPO_URL}"
  git_identity "${STATE_DIR}"
  printf '# SmogSense state journal\n\nAppend-only Parquet/JSON written by the daily pipeline. Do not edit by hand.\n' > "${STATE_DIR}/README.md"
  git -C "${STATE_DIR}" add README.md
  git -C "${STATE_DIR}" commit --quiet -m "state: initialise orphan branch"
  gh_git -C "${STATE_DIR}" push --quiet origin HEAD:state
fi
git_identity "${STATE_DIR}"
