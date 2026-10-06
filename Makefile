# =============================================================================
# SmogSense developer & operator shortcuts. Everything runs inside containers so that
# your laptop, a classmate's laptop and the GitHub runner behave identically.
# Requires: Docker (with the Compose plugin) and GNU Make. Nothing else.
# =============================================================================
SHELL := /bin/bash
.DEFAULT_GOAL := help

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
COMPOSE  ?= docker compose
ISSUANCE ?= latest

.PHONY: help dirs lock lockcheck build build-dev shell lint fmt test check docs-check doctor daily demo serve hooks clean distclean

help: ## List targets
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make <target> [ISSUANCE=2026-11-05T00:00:00Z]\n\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

dirs: ## Create bind-mount directories as YOU (prevents root-owned dirs created by Docker)
	@mkdir -p data/raw data/interim data/processed data/external models site .state

lock: dirs ## (Re)generate uv.lock — needs internet (PyPI + download.pytorch.org). Commit the result.
	$(COMPOSE) --profile dev run --rm dev uv lock

lockcheck: ## Fail if pyproject.toml and uv.lock disagree (CI runs this)
	$(COMPOSE) --profile ci run --rm lockcheck

build: dirs ## Build the lean production image (target: runtime)
	$(COMPOSE) --profile ops build pipeline

build-dev: dirs ## Build the developer image (target: dev)
	$(COMPOSE) --profile dev build dev

shell: dirs ## Interactive shell in the dev container (repo bind-mounted at /workspace)
	$(COMPOSE) --profile dev run --rm dev

lint: ## ruff (lint + format check) and mypy
	$(COMPOSE) --profile ci run --rm lint

fmt: ## Auto-format and auto-fix
	$(COMPOSE) --profile dev run --rm dev bash -c "ruff check --fix . && ruff format ."

test: ## pytest in offline fixtures mode (no credentials, no network)
	$(COMPOSE) --profile ci run --rm test

check: lint test ## Everything CI runs on every pull request

docs-check: ## Repo-hygiene tests only: doc links, schemas, i18n parity, gitignore rules
	$(COMPOSE) --profile ci run --rm test pytest -q tests/unit/test_repo_hygiene.py tests/contract tests/unit/test_i18n.py

doctor: dirs ## Check environment, credentials presence and config parse
	$(COMPOSE) --profile ops run --rm doctor

daily: dirs ## Run the full daily cycle (needs .env credentials). Example: make daily ISSUANCE=2026-11-05T00:00:00Z
	ISSUANCE=$(ISSUANCE) $(COMPOSE) --profile ops run --rm daily

demo: dirs ## Offline end-to-end demo on recorded fixtures, served at http://localhost:8080
	$(COMPOSE) --profile demo up --build

serve: dirs ## Serve ./site at http://localhost:8080
	$(COMPOSE) --profile serve up

hooks: ## Install pre-commit hooks (requires pre-commit on the host: pipx install pre-commit)
	pre-commit install --install-hooks

clean: ## Remove generated outputs and caches (keeps data/, models/, .state/)
	rm -rf site output .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage

distclean: clean ## ALSO delete local data and models (asks first)
	@read -r -p "Delete everything under data/{raw,interim,processed,external} and models/ ? [y/N] " a; \
	if [ "$$a" = "y" ]; then \
	  find data/raw data/interim data/processed data/external models -type f ! -name .gitkeep ! -name README.md ! -name model_card_template.md -delete; \
	  echo "done"; else echo "aborted"; fi
