#!/usr/bin/env bash
# One-time local setup. Verifies prerequisites, prepares .env and bind-mount directories.
# Usage: ./scripts/bootstrap.sh
set -euo pipefail

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing prerequisite: $1 ($2)" >&2; exit 1; }; }
need docker "https://docs.docker.com/get-docker/"
need git "https://git-scm.com/downloads"
need make "install GNU Make (macOS: xcode-select --install)"
docker compose version >/dev/null 2>&1 || { echo "missing prerequisite: the 'docker compose' plugin (Compose v2)" >&2; exit 1; }

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "created .env from .env.example"
fi
# keep container file ownership aligned with the host user
sed -i.bak -E "s/^HOST_UID=.*/HOST_UID=$(id -u)/; s/^HOST_GID=.*/HOST_GID=$(id -g)/" .env && rm -f .env.bak

mkdir -p data/raw data/interim data/processed data/external models site .state

cat <<'MSG'

Next steps (all free):
  1. Get API credentials and put them in .env  ->  docs/deployment-and-ops.md  "Secrets and variables"
       OpenAQ key, ADS token (+ accept the CAMS licence), CDS token (+ accept the ERA5 licence), NASA FIRMS MAP_KEY
  2. make lock      # generate uv.lock (needs internet), then commit it
  3. make build     # build the runtime image
  4. make check     # lint + type-check + tests in the container (same as CI)
  5. make doctor    # verify credentials and configuration
MSG
