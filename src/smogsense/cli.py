"""smogsense.cli — Typer command-line entry point (the only supported way to run pipeline stages).

Every pipeline stage is a CLI command so that local Docker, Docker Compose and GitHub Actions
execute byte-identical entry points. Commands are thin: parse arguments, load typed settings,
call library code, emit a run manifest.

Command surface (Phase 0 fixes the names; Phases 1-4 implement them):

    smogsense doctor [--online]
    smogsense ingest   live | openaq | openaq-archive | cams | era5 | firms | maiac
    smogsense features build | backfill
    smogsense train    source | meta | lgbm | calibrate | bundle
    smogsense forecast run
    smogsense score    shadow --stage provisional|settled
    smogsense eval     ladder | sparsity | report
    smogsense bulletin render
    smogsense site     build | validate
    smogsense publish  telegram
    smogsense state    exists | verify
    smogsense run      daily | backfill | score

Cross-cutting contract:
- Every command accepts --issuance (ISO-8601 UTC or 'latest') and --run-id, and writes
  data/processed/run_manifests/<run_id>.json plus latest_summary.md (rendered into the GitHub
  job summary by the workflow).
- Exit codes: 0 ok; 10 degraded-but-published; 11 issuance already published (idempotent no-op);
  20 input unavailable; 30 contract violation; 40 quota/rate-limit; 50 internal error.
- `run daily` is idempotent: it consults `state exists` first and exits 11 unless
  SMOGSENSE_FORCE=1. Under GitHub Actions it emits ::group:: markers per stage.
- Publishing to gh-pages/Cloudflare is NOT a CLI concern: scripts/publish_ghpages.sh runs on the
  host runner so that the git token never enters the (read-only, capability-dropped) container.

Specification: docs/system-architecture.md → 'Component responsibilities and CLI contract'
"""
