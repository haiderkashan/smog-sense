"""smogsense.config — Typed settings: YAML configuration plus environment-provided secrets.

Loads configs/*.yaml into validated pydantic models and merges environment variables. Secrets
are accepted only from the environment (never from YAML) and are wrapped in SecretStr.

Public contract (implemented in Phase 0):
- Settings.load(path_dir) -> Settings, deterministic and side-effect free.
- Config hash (sha256 of canonicalised YAML) is embedded in every run manifest and forecast_log
  row.
- Unknown keys are errors, not warnings.

Specification: configs/*.yaml and docs/system-architecture.md
"""
