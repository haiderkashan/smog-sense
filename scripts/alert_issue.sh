#!/usr/bin/env bash
# Create or update a GitHub issue for an operational alert (free: no pager service needed).
# One open issue per title; repeated alerts become comments, so a flapping source does not spam.
# Usage: GH_TOKEN=... GITHUB_REPOSITORY=owner/repo scripts/alert_issue.sh "<title>" "<markdown body>" [label]
set -euo pipefail

title="${1:?title required}"
body="${2:?body required}"
label="${3:-pipeline-alert}"
: "${GH_TOKEN:?GH_TOKEN (the workflow GITHUB_TOKEN) must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

gh label create "${label}" --color D93F0B --description "Automated pipeline alert" >/dev/null 2>&1 || true

existing="$(gh issue list --state open --label "${label}" --search "in:title \"${title}\"" \
  --json number --jq '.[0].number // empty')"

if [[ -n "${existing}" ]]; then
  gh issue comment "${existing}" --body "${body}"
else
  gh issue create --title "${title}" --body "${body}" --label "${label}"
fi
