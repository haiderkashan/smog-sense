# Incident Response Runbook

> **Use this when** an alert issue appears (`pipeline-alert`, `degraded`), the watchdog fires, or someone reports a wrong or missing forecast.
> Principle: **protect the public first** (wrong or stale information is worse than none), **then** fix, **then** learn.

## Severity and response targets

| Level | Definition | Respond within | Examples |
|---|---|---|---|
| **S1** | A *wrong* forecast is live, or a secret is exposed | 30 min | implausible numbers published; token committed |
| **S2** | No fresh bulletin by 06:00 UTC / site down | 2 h | workflow disabled; ADS outage + no fallback |
| **S3** | Degraded but published; scoring or backfill failures | 24 h | `stale_cams` mode; scoring job failed |
| **S4** | Cosmetic or documentation | next working day | typo; broken doc link |

## First ten minutes

1. Open the alert issue and the linked **Actions run**; read the *job summary* (it embeds `latest_summary.md`: stage durations, freshness, request counters, degradation level, validation gates).
2. Decide the level above. If **S1**: stop publication immediately (Settings → Pages → *Unpublish site*, or disable `daily-forecast.yml`), post a short notice on the Telegram channel, then continue.
3. Note the exit code: `20` inputs unavailable · `30` contract violation · `40` quota · `50` internal (see [system architecture §5](../system-architecture.md#5-component-responsibilities-and-cli-contract)).

## Symptom → action

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| Exit **20**, ADS stage | CAMS run not yet published / ADS delay | summary: `cams_base_time_utc`; ECMWF/ADS status pages and forum | Wait for the 02:47 or 05:47 UTC retry. If > 24 h, level 3 (`observations_only`) is automatic |
| `403 required licences not accepted` | Dataset licence not accepted for this account | open the dataset page while logged in | Click *Accept*; re-run via *workflow_dispatch* |
| `401/403` from OpenAQ | Key missing, revoked or rotated | `make doctor` locally | Create a new key; update `OPENAQ_API_KEY`; re-run |
| Repeated `429` from OpenAQ, circuit open | Quota exhausted (another consumer on the same key?) | manifest: requests used vs 48/min, 1,600/h | Stop ad-hoc queries on that key; wait for the window; never loop retries (bans are possible) |
| Exit **30** | Output failed a hard gate (schema, monotonicity, bounds, budgets) | summary lists the failing gate | Reproduce with `SMOGSENSE_MODE=fixtures`; fix code; add a regression test; previous bulletin stays live |
| Implausible median (soft gate) | Sensor fault or CAMS glitch | `qc_flags` histogram; compare with CAMS-BC and persistence | If wrong *and published*: **S1** (unpublish), flag stations, re-run with `SMOGSENSE_FORCE=1` |
| `degraded` issue: `stale_cams` | Earlier CAMS cycle used | provenance `cams_global.as_of_utc` | None if it clears the next day; else treat as ADS incident |
| `degraded`: `baseline_only` | Bundle missing/invalid or `promoted_bundle: none` | `models/registry` message | If unintended: verify the Release asset sha256 and `feature_set_version`; roll back the config line |
| Watchdog: *no forecast for today's issuance* | Workflow never started | Actions tab: is `daily-forecast` disabled? any run at 00:17? | `gh workflow enable daily-forecast.yml`; run it manually; check repo inactivity (60-day rule) |
| Image pull denied in Actions | GHCR package is private | package settings | Make public; the job falls back to a local build meanwhile |
| `state` push rejected repeatedly | Concurrent writer | `smogsense-state` concurrency queue | Re-run; scripts already retry four times with rebase |
| Site unchanged after success | Pages not publishing `gh-pages` | Settings → Pages; the *pages build and deployment* run | Re-select the branch; check `.nojekyll` and size < 1 GB |
| Satellite/product retired (FIRMS, MODIS, CAMS cycle) | Provider change | provider notice | Update platform table in `configs/sources.yaml`; add a `verification-log` entry; confirm features are platform-normalised |
| Repository nearing 1 GB | Something large was committed | `git count-objects -vH` | Find/remove the object; for `state`, last resort `scripts/compact_state_branch.sh` **after** attaching a backup bundle to a Release |

## If a secret was exposed

1. **Rotate first** in the provider UI (OpenAQ, ADS/CDS, FIRMS, Telegram, Cloudflare) — assume it is already compromised.
2. Replace the GitHub secret. 3. Remove it from history (`git filter-repo`, then force-push; GitHub support can purge cached views). 4. Check provider logs for misuse.
5. Add the pattern to `gitleaks` if it slipped through; document in the incident issue.

## Public communication

* **Wrong forecast was live:** post a bilingual correction on the Telegram channel and the site's methodology/accuracy page; state what was wrong, for how long, and what changed.
* **Planned or long outage:** unpublish or leave the staleness banner (appears automatically after 36 h) and post a short notice. Do not backfill missing days with guesses.

## After the incident

Close the alert issue with: *what happened · root cause · detection time · time to mitigate · permanent fix*. Add a regression test or a new validation gate, update the [verification log](../verification-log.md) if a provider fact changed, and run the monthly forced-failure drill again (disable a secret on a branch and confirm that an alert issue appears).
