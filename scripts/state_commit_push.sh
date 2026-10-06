#!/usr/bin/env bash
# Commit whatever the pipeline wrote under ./.state and push it, retrying on a non-fast-forward race.
# Usage: scripts/state_commit_push.sh "forecast 2026-11-05 (rc=0)"
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

msg="${1:?commit message required}"
dir="${STATE_DIR:-.state}"

git -C "${dir}" add -A
if git -C "${dir}" diff --cached --quiet; then
  echo "state: nothing to commit"
  exit 0
fi
git -C "${dir}" commit --quiet -m "${msg}"

for attempt in 1 2 3 4; do
  if gh_git -C "${dir}" push --quiet origin HEAD:state; then
    echo "state: pushed (attempt ${attempt})"
    exit 0
  fi
  echo "state: push rejected (attempt ${attempt}); rebasing and retrying" >&2
  gh_git -C "${dir}" pull --rebase --quiet origin state || true
  sleep $((attempt * 3))
done
echo "state: push failed after 4 attempts" >&2
exit 1
