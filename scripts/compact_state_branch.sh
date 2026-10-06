#!/usr/bin/env bash
# EMERGENCY TOOL: replace the `state` branch history with a single snapshot commit.
# Normally unnecessary (append-only daily files are ~150 KB/day). Use only if the repository nears
# the 1 GB recommendation or a large file was committed by mistake. History is DISCARDED, so the
# runbook (docs/runbooks/incident-response.md) requires a backup tarball to be attached to a Release first.
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

dir="${STATE_DIR:-.state}"
[[ "${CONFIRM_COMPACT:-}" == "yes" ]] || { echo "set CONFIRM_COMPACT=yes to proceed" >&2; exit 2; }

tree="$(git -C "${dir}" rev-parse 'HEAD^{tree}')"
commit="$(git -C "${dir}" commit-tree "${tree}" -m "state: compacted snapshot $(date -u +%F)")"
git -C "${dir}" reset --quiet --hard "${commit}"
gh_git -C "${dir}" push --force --quiet origin HEAD:state
echo "state: compacted to ${commit}"
