## What changed and why

<!-- One paragraph. Link the requirement (FR-xx / NFR-xx in docs/PRD.md) and the doc section this implements. -->

## Evidence

- [ ] `make check` passes locally (lint, types, tests in the container)
- [ ] New behaviour has a test that would fail without it (unit, contract or integration)
- [ ] Docs updated in the same PR (`docs/…`), including the verification log if a source claim changed

## Guardrail checklist

- [ ] **$0 budget:** no new paid service, billable API, credit-card requirement or GPU dependency
- [ ] **Point-in-time correctness:** no feature uses data that was unavailable at issuance time *T*
- [ ] **Container parity:** runs identically via `docker compose` and in GitHub Actions
- [ ] **Bilingual:** user-facing strings exist in both `web/i18n/en.yaml` and `web/i18n/ur.yaml`
- [ ] **Secrets:** none committed; new credentials are documented in `.env.example` and `docs/deployment-and-ops.md`
- [ ] **Scientific claim:** if this adds a model component, it beats its predecessor in the model ladder (attach the table)
