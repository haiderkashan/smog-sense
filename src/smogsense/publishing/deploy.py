"""smogsense.publishing.deploy — Delivery preparation for static hosting.

Validates the built site against its budgets (page weight, card size, language-tree parity, JSON schema)
and emits the host-specific files: `.nojekyll`, `404.html`, `sitemap.xml`, `robots.txt` and, for the
optional Cloudflare Pages mirror, `_headers` (CSP and cache policy). The Git push to the `gh-pages`
branch is performed by scripts/publish_ghpages.sh on the host runner as a single orphan commit, so
the repository token never enters the application container.

Public contract (implemented in Phase 4):
- validate_site(site_dir) -> list[Violation]; any violation makes `smogsense site validate` exit 30.

Specification: docs/deployment-and-ops.md → 'Workflow catalogue'
"""
