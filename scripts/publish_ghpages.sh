#!/usr/bin/env bash
# Publish ./site to the gh-pages branch as ONE orphan commit (force-push).
# Why: a daily commit of PNG cards would grow the repository without bound (GitHub recommends
# repositories under 1 GB). A single squashed commit keeps gh-pages at the size of one site (<= ~5 MB).
# The history of bulletins is preserved in the `state` branch (JSON/Parquet), not here.
# Usage: scripts/publish_ghpages.sh [site_dir]
set -euo pipefail
# shellcheck source=scripts/_git_auth.sh
source "$(dirname "$0")/_git_auth.sh"

site="${1:-site}"
[[ -f "${site}/index.html" ]] || { echo "refusing to publish: ${site}/index.html missing" >&2; exit 1; }
[[ -f "${site}/forecast/latest.json" ]] || { echo "refusing to publish: ${site}/forecast/latest.json missing" >&2; exit 1; }

size_mb="$(du -sm "${site}" | cut -f1)"
if (( size_mb > 200 )); then
  echo "refusing to publish: site is ${size_mb} MB (budget 200 MB; GitHub Pages hard limit 1 GB)" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT
cp -a "${site}/." "${tmp}/"
touch "${tmp}/.nojekyll"          # skip Jekyll processing; we ship plain static files

git init --quiet -b gh-pages "${tmp}"
git_identity "${tmp}"
git -C "${tmp}" remote add origin "${REPO_URL}"
git -C "${tmp}" add -A
git -C "${tmp}" commit --quiet -m "deploy $(date -u +%Y-%m-%dT%H:%MZ) from ${GITHUB_SHA:-local}"
gh_git -C "${tmp}" push --quiet --force origin HEAD:gh-pages
echo "gh-pages: published ${size_mb} MB"
