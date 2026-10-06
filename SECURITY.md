# Security Policy

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** (Security tab → *Report a vulnerability*). Please do not open a public
issue for anything that could expose credentials or allow tampering with published advisories.

## Scope that matters for this project

SmogSense publishes **public-health information**. Integrity of the published forecast matters more than confidentiality:

- Tampering with `forecast/latest.json`, the cards, or the bilingual advisory text is the highest-impact attack.
- Compromise of a repository secret (OpenAQ, ADS/CDS, FIRMS, Telegram, Cloudflare) is the most likely path to it.

## Rules for contributors

1. Never commit secrets. `.env`, `.cdsapirc`, `*.pem`, and `credentials*.json` are git-ignored and scanned by pre-commit (`gitleaks`) and CI.
2. If a secret is ever committed, treat it as compromised: rotate it first, then remove it from history.
3. Third-party GitHub Actions are pinned by commit SHA and updated through Dependabot.
4. Workflows use least-privilege `permissions:`; only the publishing jobs receive `contents: write`.
5. Fork pull requests never receive secrets; scheduled workflows run only on the default branch.
