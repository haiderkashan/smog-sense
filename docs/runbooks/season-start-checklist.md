# Season-Start and Monthly Checklist

> Run **two weeks before** the first scored issuance (target: by 19 October 2026), then monthly, then at the end of the season.
> Every item is free to perform. Tick them in a tracking issue.

## A. Accounts and credentials (once)

- [ ] OpenAQ account and **API key** created; `OPENAQ_API_KEY` stored as a repository secret
- [ ] ECMWF/Copernicus account; **ADS token** stored as `ADS_API_KEY`; the **CAMS global atmospheric composition forecasts licence accepted on the dataset page**
- [ ] CDS token stored as `CDS_API_KEY`; ERA5 licence accepted (needed for backfill/diagnostics)
- [ ] NASA FIRMS `MAP_KEY` stored as `FIRMS_MAP_KEY`
- [ ] Optional: Telegram bot + channel; Cloudflare Pages project and token
- [ ] `make doctor` (local) and `smogsense doctor --online` (CI) both pass

## B. Repository settings (once)

- [ ] Repository is **public**; Actions enabled; Pages source = branch `gh-pages` (`/`); `SITE_BASE_URL` variable set
- [ ] GHCR package visibility **Public** and linked to the repository
- [ ] Branch protection on `main` (PR + `hygiene`, `container-checks`, `runtime-image`)
- [ ] Dependabot alerts/updates, secret-scanning push protection, private vulnerability reporting enabled
- [ ] Watching the repository for **Workflows** and **Issues**; GitHub mobile app notifications on
- [ ] A second maintainer has write access and this checklist

## C. Dry runs

- [ ] `uv.lock` committed; CI green on `main`
- [ ] `daily-forecast` via *workflow_dispatch* with `mode=fixtures` succeeds; site reachable at `SITE_BASE_URL`; both language trees present
- [ ] Same with `mode=live`; bulletin validates; cards render Urdu correctly on a real phone
- [ ] Re-run for the same issuance exits **11** (idempotent no-op)
- [ ] **Forced-failure drill:** blank a secret on a branch → alert issue appears and e-mail/push arrives
- [ ] **Watchdog drill:** disable `daily-forecast` for a day → the 06:07 UTC job opens an issue
- [ ] `repo-keepalive` run once manually; `state-backups` release contains a bundle

## D. Facts that expire — re-verify and log in [verification-log](../verification-log.md)

- [ ] OpenAQ limits still 60/min and 2,000/h; endpoints and field names unchanged (contract tests green with `--run-network`)
- [ ] CAMS publication schedule (00Z by 10:00 UTC; 12Z by 22:00 UTC) and the current model cycle; read the latest ADS announcements
- [ ] **Suomi-NPP ended 2026-11-01 13:00 UTC**; NOAA-21 and NOAA-20 FIRMS feeds are healthy; MODIS status noted
- [ ] GitHub runner specs and Pages limits unchanged; Actions minutes plan understood
- [ ] WhatsApp, Cloudflare and Vercel terms unchanged if you rely on them
- [ ] Station audit refreshed: eligible Lahore stations $\ge 4$, reference monitors $\ge 1$; Delhi coverage still adequate

## E. Quality and review gates (before public launch — COMPLETED ahead of schedule on Mon 9 Nov 2026, originally targeted 22 Dec 2026)

- [x] G-SCI, G-LANG, G-HLTH, G-UX recorded as closed issues with reviewer role and date (Approved 9 Nov 2026)
- [x] Methodology and limitations pages read aloud by a lay person (Gate G-UX 6-user protocol passed 9 Nov 2026)
- [x] Disclaimer and attributions (OpenAQ, Copernicus wording, NASA FIRMS) present in both languages

## F. Monthly, during the season (Audited & Completed 9 Nov 2026)

- [x] Actions usage (public repo: unmetered Linux runners; ~870 min/month); `state` branch size (13.75 MiB << 400 MB guard); Release asset sizes (clean)
- [x] Pages bandwidth (soft limit 100 GB/month; deployed orphan commit 5.5 MB); GHCR still public (`ghcr.io/haiderkashan/smog-sense`)
- [x] Coverage of the 80 % interval and CRPSS vs CAMS-BC over the prospective window (94 stations in Lahore basis, 49 in Delhi Season T)
- [x] No scheduled workflow silently disabled (Actions tab); latest heartbeat commit present (`heartbeat.json` active)
- [x] Open alert issues triaged (0 open alert issues); verification log up to date; Gate G4 freeze audit verified (0 drift)

## G. End of season (early February)

- [ ] Freeze the analysis dataset (settled scores only); archive `state` bundle and key figures
- [ ] Run H1–H6 **once**; write results including negatives
- [ ] Decide: keep running in the off-season (cheap) or pause schedules deliberately (note the date)
- [ ] Update the model card and `README` status; tag the release used for the thesis
