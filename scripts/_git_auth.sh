#!/usr/bin/env bash
# Sourced by the other scripts. Provides `gh_git`, a git wrapper that authenticates to github.com with
# $GITHUB_TOKEN via an HTTP extra-header (the token is never written to .git/config or a remote URL).
# shellcheck shell=bash
: "${GITHUB_TOKEN:?GITHUB_TOKEN must be set}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set (owner/name)}"

_auth_b64="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | base64 -w0)"
REPO_URL="https://github.com/${GITHUB_REPOSITORY}.git"
export REPO_URL

gh_git() {
  git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${_auth_b64}" "$@"
}

git_identity() {  # git_identity <dir>
  git -C "$1" config user.name "smogsense-bot"
  git -C "$1" config user.email "smogsense-bot@users.noreply.github.com"
}
