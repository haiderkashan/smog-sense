# Test Specifications

Contract tests to implement for Phase 0 and Phase 1.

## Phase 0

- **As-of parity.** On one recorded snapshot, the rule-reconstructed feature inputs equal the snapshot, and the eligible set is identical at 00:17, 02:47 and 05:47.
- **Clip.** No eligible row has hour_end > T.
- **Lookback table.** The corrected rule returns the offsets in the table above at all four lattice points.
- **Bearing.** The worked example gives a ≈ 0.97 (easterly) and 0 (westerly) within 0.01.
- **Dedup.** Re-ingesting an overlapping 72-h pull leaves exactly one canonical row per (sensor, hour) and `n_revisions` increments.
- **Co-location.** {reference + 2 low-cost} resolves to the reference value and records the basis.
- **Schema negatives.** Duplicate (sensor_id, ts_utc, pull_id) is rejected; a rerun with a new run_id is accepted.
- **Config cross-consistency.** As above (steps equality, partitions, 10 methods, γ in the grid).
- **i18n.** 73-key parity; the 6 new keys' placeholders match.
- **Workflow structure.** Daily, scoring, keep-alive and the watchdog share the concurrency group; every downstream step is guarded by the skip flag.
- **Docs.** No reference to the git-ignored files; anchors resolve; hours reconcile (a local-only test that skips when `roadmap.md` is absent).
- **Pending experiments (protocols only).** Latent-shift and stacker-frame test; in-sample vs out-of-sample latents; pinball vs quantile-Huber with and without clipping; RH-mask audit; peak-RSS loop; state-branch growth and compaction/rebase exercise; API-vs-archive diff; backfill pilot; ACI γ backtest.
