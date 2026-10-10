#!/usr/bin/env bash
# Container entrypoint (runs under tini). Responsibilities:
#   1. make HOME and cache dirs writable even when the container runs as an arbitrary host UID
#      (GitHub Actions runner UID is 1001, not the image's default 1000);
#   2. refuse to start in live mode when no credentials are present at all (fail fast, clear message);
#   3. exec the requested command so signals reach the process.
set -euo pipefail

export HOME="${HOME:-/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/.mpl}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# cdsapi reads ~/.cdsapirc; we avoid writing secrets to disk and rely on CDSAPI_URL/CDSAPI_KEY set per call
# by the application (the two Copernicus data stores use different tokens). Remove stale rc files if present.
rm -f "$HOME/.cdsapirc"

if [[ "${SMOGSENSE_MODE:-live}" == "live" && "${1:-}" == "smogsense" ]]; then
  case "${2:-}" in
    ""|--help|-h|doctor|state|site|bulletin|eval|train|features) : ;;  # commands that do not need source credentials
    ingest|forecast|score|publish)
      if [[ -z "${OPENAQ_API_KEY:-}${ADS_API_KEY:-}${CDS_API_KEY:-}${FIRMS_MAP_KEY:-}" ]]; then
        echo "error: SMOGSENSE_MODE=live but no source credentials are set (see .env.example)." >&2
        echo "       Use SMOGSENSE_MODE=fixtures for an offline run." >&2
        exit 20
      fi ;;
  esac
fi

exec "$@"
