# Deployment and Operations

> **Audience:** the person who owns the repository on 1 October 2026 and must have a live pipeline before the peak of the smog season. All limits below were verified in October 2026
> (sources in the [verification log](verification-log.md)); recheck before the season starts ([season-start checklist](runbooks/season-start-checklist.md)).

## 1. Zero-cost resource budget

| Service | Permanent free-tier limit (verified) | SmogSense usage | Headroom | What happens at the limit |
|---|---|---|---|---|
| **GitHub Actions** (public repo) | Standard Linux runners **free and unlimited**; **4 vCPU / 16 GB RAM / 14 GB SSD**; 6 h per job | ≈ 1,100 min/month (daily ≈ 25 min + two ~2-min no-op retries; scoring ≈ 5 min; keep-alive) + CI | unlimited | n/a |
| GitHub Actions (*private* repo) | **2,000 min/month** on the Free plan; Linux **2 vCPU, 7–8 GB** (the docs disagree) | same ≈ 1,100 min + CI | ≈ 800 min for CI | Jobs stop; **budget to the smaller runner anyway** — compose emulates 2 CPU / 7 GB |
| **GitHub Pages** | Source repo ≤ 1 GB recommended; published site ≤ 1 GB; **≈ 100 GB/month soft bandwidth**; ≈ 10 builds/hour soft (does not apply to custom Actions workflows); 10-min deploy timeout. **Available only for public repos on the Free plan** | site ≈ 3–50 MB; ≤ 3 deploys/day | 20×+ | Soft limits: GitHub may email you; the Cloudflare mirror absorbs traffic |
| **GitHub Container Registry** | Free and unmetered for **public** packages; a private package on the Free plan gets only ~500 MB | runtime image ≈ 3 GB | n/a if public | Private image would not fit ⇒ keep repo + package **public** |
| **GitHub Releases** | 2 GiB per asset; no total limit | model bundles (tens of MB), `data-backfill` (≈ GBs) | large | n/a |
| **Repository size** | ≈ 1 GB recommended, 100 MiB hard per file | code + docs ≪ 10 MB; `state` ≈ 55 MB/year | ≫ | `scripts/compact_state_branch.sh` (emergency) |
| **Actions cache** | 10 GB per repo, oldest evicted | raw GRIB/CSV per run < 100 MB | ≫ | eviction only; jobs re-download |
| **Artifacts / logs** | retained 90 days by default | failure diagnostics only, 14 days | ≫ | auto-expiry |
| **Cloudflare Pages Free** (optional mirror) | 500 builds/month (Git-integrated builds), 20,000 files/site, 25 MiB/file, unlimited static bandwidth | direct upload from CI, ≤ 3 deploys/day, ~100 files | ≫ | Deploy step is `continue-on-error` |
| **OpenAQ API v3** | 60 req/min **and** 2,000 req/h per key; key mandatory | ≈ 60–200 req/day | ≫ (budgeted at 80 %) | 429 → backoff → circuit breaker |
| **OpenAQ AWS archive** | anonymous public bucket, no account, no quota | backfill only | n/a | n/a |
| **Copernicus ADS / CDS** | free, queued, per-dataset licence | 1–2 requests/day (+ backfill) | ≫ | Queue delay → degradation ladder |
| **NASA FIRMS** | free MAP_KEY; 5,000 transactions / 10 min | ≈ 3 requests/day | ≫ | 429 → mask fire group |
| **Telegram Bot API** (optional) | free | 1–2 posts/day | ≫ | step is `continue-on-error` |
| WhatsApp Business Platform | **billed per delivered template message** outside a user-initiated window | **not used** | – | see below |

**Red lines (each would silently introduce a cost):** a *private* repository beyond 2,000 minutes; GitHub Pages in a private repo (needs a paid plan); a private GHCR package above its quota; **Git LFS** (1 GB storage / 1 GB bandwidth then billed — never use it);
GitHub *larger runners*, Codespaces, Copilot seats; any cloud account with a card on file; any LLM or SaaS API. `tests/unit/test_repo_hygiene.py` fails the build if a paid-SDK dependency is added.

**Why WhatsApp is not automated.** Meta's WhatsApp Business Platform charges per delivered template message (per-message pricing since 1 July 2025); free-form messages are free only inside a 24-hour window opened by the user. A daily broadcast to subscribers is outside that window, hence billable.
Unofficial automation libraries violate WhatsApp's terms and risk number bans. SmogSense therefore generates **WhatsApp-ready cards** with stable URLs and *Share* links (`wa.me`) for people and channel admins to forward, and mirrors the card to a free **Telegram** channel.

## 2. Repository settings (one-time)

1. **Create a public repository** under your personal GitHub account (Free plan). Public ⇒ unlimited Actions minutes, GitHub Pages, and free GHCR.
2. **Actions → General:** allow GitHub-authored and Marketplace actions *from verified creators* (all third-party actions are pinned by SHA); *Workflow permissions* may stay "Read repository contents and packages" (jobs request more explicitly).
3. **Pages → Build and deployment:** *Deploy from a branch* → `gh-pages` / `/ (root)`. (The branch is created by the first run; re-open the page once afterwards.) Note the URL; set it as variable `SITE_BASE_URL`.
4. **Packages:** after the first `docker-publish` run, open the package → *Package settings* → set visibility **Public**, and link it to the repository.
5. **Branch protection on `main`:** require pull request, require status checks `hygiene`, `container-checks`, `runtime-image`; disallow force-push.
6. **Security:** enable *Private vulnerability reporting*, *Dependabot alerts and security updates*, and *Secret scanning with push protection* (free for public repositories).
7. **Labels** `pipeline-alert`, `degraded` are created automatically by the alert script.

## 3. Secrets and variables

Create under *Settings → Secrets and variables → Actions*. Obtain each credential once; all are free.

| Name | Kind | How to obtain | Notes |
|---|---|---|---|
| `OPENAQ_API_KEY` | secret | Register at the OpenAQ Explorer and create an API key | mandatory in v3 |
| `ADS_API_KEY` | secret | Create an ECMWF/Copernicus account → Atmosphere Data Store → profile page → personal access token. **Open the CAMS global forecasts dataset page and accept the licence once** | without acceptance: HTTP 403 |
| `CDS_API_KEY` | secret | Climate Data Store profile page → token. Accept the ERA5 licence once | hindcast/backfill only |
| `FIRMS_MAP_KEY` | secret | NASA FIRMS "Get MAP_KEY" page (e-mail only) | |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | secret, optional | Create a bot with @BotFather; add it as admin of a channel | |
| `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | secret, optional | Cloudflare dashboard → API token with *Cloudflare Pages: Edit* | |
| `SITE_BASE_URL` | **variable** | e.g. the Pages URL | used for canonical URLs, Open Graph tags, sitemap |
| `CLOUDFLARE_PAGES_PROJECT` | **variable**, optional | Pages project name | |
| `EARTHDATA_TOKEN` | secret, optional | NASA Earthdata (research-only MAIAC) | tokens expire; rotate |

Rotation: replace the secret in the UI; no code change. If any secret is ever committed, treat it as compromised, rotate first, then clean history ([incident runbook](runbooks/incident-response.md)).

## 4. Workflow catalogue

| Workflow | Trigger | Purpose | Permissions (job) | Budget |
|---|---|---|---|---|
| `ci.yml` | PR, push to `main` | pre-commit (ruff, yaml/toml/json, shellcheck, hadolint, actionlint, gitleaks, schemas); lock-file sync; lint + strict mypy + tests in the dev image; runtime image smoke test (CPU-only torch, Raqm, non-root, Nastaliq font, size ≤ 5 GB) | `contents: read` | ≈ 15–25 min |
| `docker-publish.yml` | push to `main`, **weekly** (Mon 03:23 UTC), manual | build `runtime`, push `ghcr.io/<owner>/smogsense:sha-<commit>` and `:latest` (weekly run picks up Ubuntu security patches) | `packages: write` | ≈ 10 min (cached) |
| `daily-forecast.yml` | **00:17, 02:47, 05:47 UTC**, manual | the heartbeat: checkout → prepare → attach state → pre-check → login/pull (skipped when pre-check says skip) → run → publish. All downstream conditions test `skip != 'true'`. Image tag: `img-<hash of Dockerfile, pyproject.toml, uv.lock, src/**, docker/entrypoint.sh>` | `contents: write`, `packages: read`, `issues: write` | ≈ 25 min, cap 40 |
| `shadow-scoring.yml` | 06:07 UTC | **watchdog** (alerts if today's forecast is missing, then publishes the static expiry notice from the last good bulletin; shared concurrency group); provisional and settled scoring; push score journals | as above | ≈ 5 min |
| `historical-backfill.yml` | manual | resumable backfill to the rolling `data-backfill` release. API-using sources run under quota partitions. | `contents: write` | cap 345 min |
| `model-retrain.yml` | manual | five-stage training + ladder/sparsity evaluation → `models-vX.Y.Z` pre-release | `contents: write` | cap 350 min |
| `repo-keepalive.yml` | monthly | re-enable schedules; heartbeat; **git-bundle backup of `state`** to the `state-backups` release | `contents: write`, `actions: write` | ≈ 3 min |

All third-party actions are pinned by commit SHA (Dependabot, weekly, keeps them current); every job has `timeout-minutes`; no workflow uses `pull_request_target`.

## 5. Scheduling reliability

* **Off-the-hour minutes.** GitHub documents that scheduled runs may be delayed, particularly at the start of every hour. We trigger at `:17` / `:47`. A test fails the build if a cron uses minute 0.
* **Three idempotent triggers** (00:17, 02:47, 05:47 UTC): the retries exit 11 in ~2 minutes if the primary succeeded, and rescue a dropped event or a late CAMS publication (the 00 UTC CAMS run is guaranteed only by 10:00 UTC; the **previous** 12 UTC run, guaranteed by 22:00 UTC, is what the 00:17 job consumes).
* **Dead-man's switch.** `shadow-scoring.yml` (06:07 UTC) checks that the day's issuance exists and opens an issue otherwise — covering the case where `daily-forecast.yml` never started (disabled, dropped, expired token).
* **60-day auto-disable.** GitHub disables scheduled workflows in a public repository after 60 days without repository activity. The claim that daily `state` pushes count as activity is unverified. Real mitigations: GitHub e-mails the owner when it disables one, plus a monthly manual check of the Actions tab. `repo-keepalive.yml` also runs monthly to re-enable workflows.
* **Concurrency.** Group `smogsense-state` (cancel-in-progress: false) serialises every job that writes `state`. A pending job replaced by a newer one is acceptable because the triggers are idempotent.
* **Default-branch rule.** Schedules run on the latest commit of the default branch; keep `main` deployable.

## 6. Container strategy

* **Content-hash tagging**: `ghcr.io/<owner>/smogsense:img-<hash>`; the daily job pulls the tag matching its content hash, so code, config and image always agree. If the image is missing it builds locally with BuildKit cache (slower but safe).
* **Why a container on a runner at all.** Parity: the same bytes run on a laptop, in CI and in production; native GRIB/GDAL/Pillow-Raqm stacks make "works on my machine" the main reproducibility risk otherwise.
* **Hardening.** Non-root UID 1000 (host UID mapped via `HOST_UID/HOST_GID`), read-only root filesystem, `/tmp` tmpfs, all capabilities dropped, `no-new-privileges`, **the git token is never passed into the container**.
* **Size and speed.** Budget ≈ 3 GB (warn > 3.5 GB, fail > 5 GB in CI); `--build-arg INSTALL_GDAL=false` removes system GDAL if the research-only MAIAC path is dropped.
* **Reproducibility.** `uv.lock` is mandatory (`make lock`; CI runs `uv lock --check`); wheels only (`--no-build`). CPU-only torch comes from the PyTorch CPU index.

## 7. Release and model promotion

* **Versioning.** `models-vMAJOR.MINOR.PATCH`: MAJOR = feature-set or output-schema change; MINOR = retrain with the same features; PATCH = calibration refresh.
* **Bundle** (`models/README.md`): weights + `manifest.json` (sha256 of every file, git SHA, config hash, CAMS cycles in training window, library versions) + model card + evaluation outputs. The loader refuses a bundle whose hashes or `feature_set_version` do not match.
* **Promotion = reviewed PR** changing `promoted_bundle` in `configs/model.yaml` (initially `none` ⇒ baseline-only, level 2 of the ladder). The PR must attach the ladder table and sparsity curve, tick the guardrail checklist, and show that H-tests were not tuned on test blocks.
* **Challenger mode** logs a candidate beside the published model before promotion.
* **Rollback = revert the one line.** Previous bundles stay in Releases.
* **Retrain triggers:** a CAMS model-cycle change; CRPSS vs CAMS-BC < 0.05 for 14 days; coverage of the 80 % interval outside 0.70–0.90 for 14 days; a sensor-network change that alters $N$ materially.

## 8. Alerting

Free, GitHub-native, de-duplicated (one open issue per title, repeats become comments):

| Condition | Channel | Severity |
|---|---|---|
| Daily job failed (exit ≥ 20) | Issue "SmogSense daily pipeline failed" + GitHub notification e-mail to watchers + diagnostics artifact | high |
| Published in degraded mode (exit 10) | Issue "published in DEGRADED mode" (label `degraded`) | medium |
| No forecast by 06:07 UTC (watchdog) | Issue "no forecast published for today's issuance" | high |
| Shadow scoring failed | Issue | medium |
| Soft validation gates (implausible median, stale CAMS/FIRMS, > 40 % stations rejected) | Annotation in job summary + issue | low |
| Optional | Telegram message to the operator chat | – |

**Setup:** *Watch → Custom → Workflows* and *Issues* so failures reach your inbox/phone. Configure the GitHub mobile app for push. An on-call rota of one student is a risk; add a second maintainer before peak season.

## 9. Artifact persistence and retention

| Artefact | Home | Retention | Size |
|---|---|---|---|
| Forecast log, scores, input snapshots, manifests, bulletin JSON | `state` branch | permanent (append-only) | ≈ 150 KB/day |
| Model bundles | Release `models-v…` | permanent | tens of MB |
| Backfilled Parquet | Release `data-backfill` | permanent, overwritten per partition | GBs |
| State backups | Release `state-backups` (monthly git bundle) | permanent | ≈ tens of MB |
| Raw GRIB/CSV | Actions cache / runner disk | ≤ 7 days / run | < 100 MB/run |
| Failure diagnostics | Actions artifact | 14 days | KBs |
| Published site | `gh-pages` (single orphan commit) | current only; **cards kept for 14 days**, JSON for all days (rebuilt from `state`) | ≈ 3–50 MB |

## 10. Cost and quota monitoring

Monthly (first Monday; part of the season checklist): Actions usage page (private repos: stay < 1,500 of 2,000 minutes); repository size and `state` branch size (State size guard: warn at 400 MB, alert at 700 MB via GitHub API repo size; clone over 300 MB triggers sparse partial clone; yearly rotation to a Release bundle); Pages bandwidth (soft 100 GB; if cards are widely hot-linked, move heavy traffic to the Cloudflare mirror);
GHCR package visibility still public; Release asset sizes; the run-manifest request counters against `configs/sources.yaml` budgets. Memory budget is a measured requirement (peak-RSS tests), and API-using sources run under quota partitions. CI itself enforces: image ≤ 5 GB, site ≤ 200 MB (`publish_ghpages.sh`), per-page and card budgets (`site validate`).

## 11. Disaster recovery

| Scenario | Recovery |
|---|---|
| `state` branch damaged or force-pushed | `git clone state-YYYY-MM.bundle` from the `state-backups` release (≤ 1 month old), then re-ingest the gap from the OpenAQ archive; forecasts after the backup are lost *unless* a local `.state/` clone exists |
| Bad bulletin published | Re-run `daily-forecast` via *workflow_dispatch* with `issuance` and `force`, or `scripts/publish_ghpages.sh` with a prior site rebuilt by `make daily ISSUANCE=…`. Emergency: disable Pages |
| Secret leaked | Rotate in the provider UI first; update the GitHub secret; purge history; document in the incident log |
| Provider outage (OpenAQ/ADS/FIRMS) | Degradation ladder handles it; alert issue; no action unless > 24 h |
| ADS publishes late after a model upgrade | Level 1 (previous cycle) and the 02:47 / 05:47 retries |
| GitHub outage | Static site stays up (Pages/Cloudflare); resume when Actions return; idempotent triggers catch up the missed issuance |
| Repository deleted/locked | Clones + Release assets + monthly bundles; keep a personal clone of `state` |
| A satellite or product retires (e.g. Suomi-NPP on 2026-11-01) | Config-driven platform table; platform-normalised features; `docs/runbooks/incident-response.md` |
