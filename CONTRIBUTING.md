# Contributing to SmogSense

SmogSense is a research-engineering project with a strict **USD 0.00 budget** and a public-health purpose. Contributions are
welcome when they respect both constraints.

## Ground rules

1. **No paid services.** Nothing may require a credit card, a billable API key, a paid cloud VM, or a GPU cluster. If a change
   needs one, it will not be merged. Free tiers must be *permanent* free tiers, and their limits must be documented in
   `docs/deployment-and-ops.md`.
2. **Container parity.** Anything that runs in GitHub Actions must run identically via `docker compose` locally.
3. **Scientific honesty.** A new model component must beat its simpler predecessor in the model ladder
   (`docs/ml-architecture.md → Model ladder and ablations`) on the pre-registered metrics, or it does not ship.
4. **Point-in-time correctness.** Features at issuance time *T* may only use data that was available at *T*
   (`docs/data-engineering.md → Temporal alignment`). Leakage is treated as a bug of the highest severity.
5. **Bilingual by default.** User-facing strings go into `web/i18n/en.yaml` **and** `web/i18n/ur.yaml`; CI fails on key mismatch.

## Workflow

```bash
git switch -c feat/<short-description>
make build          # build the dev image
make check          # ruff + mypy + pytest inside the container (same as CI)
git commit          # pre-commit hooks run: ruff, yaml/json checks, gitleaks, shellcheck
git push -u origin feat/<short-description>
```

Open a pull request; fill in the template (what changed, which doc section it implements, which test proves it).

## Commit style

Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`). Reference the requirement ID (e.g. `FR-12`,
`NFR-04`) from `docs/PRD.md` where applicable.

## Data and licences

Do not commit data. Respect each provider's licence and attribution requirements (see README → *Data sources and
attribution*). Model bundles go to GitHub Releases, never into Git.
