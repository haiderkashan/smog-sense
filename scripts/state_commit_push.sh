#!/usr/bin/env bash
# Commit whatever the pipeline wrote under ./.state and push it, retrying on a non-fast-forward race.
# Usage: scripts/state_commit_push.sh "forecast 2026-11-05 (rc=0)"
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

msg="${1:?commit message required}"
dir="${STATE_DIR:-.state}"

# Guard: Ensure binary files are ignored so they never enter the state branch
if [ ! -f "${dir}/.gitignore" ]; then
  printf '*.grib\n*.grib2\n*.idx\n*.nc\n*.nc4\n*.bin\n' > "${dir}/.gitignore"
fi

# Run with hooks and fsmonitor disabled for host safety
git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false add -A
if git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false diff --cached --quiet; then
  echo "state: nothing to commit"
  exit 0
fi
git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false commit --quiet -m "${msg}"

for attempt in 1 2 3 4; do
  if gh_git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false push --quiet origin HEAD:state; then
    echo "state: pushed (attempt ${attempt})"
    exit 0
  fi
  echo "state: push rejected (attempt ${attempt}); rebasing and retrying" >&2
  if ! gh_git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false pull --rebase --quiet origin state; then
    echo "state: rebase failed, aborting rebase" >&2
    git -C "${dir}" -c core.hooksPath=/dev/null -c core.fsmonitor=false rebase --abort 2>/dev/null || true
  fi
  sleep $((attempt * 3))
done
echo "state: push failed after 4 attempts" >&2
exit 1
