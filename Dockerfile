# syntax=docker/dockerfile:1.7
# =============================================================================
# SmogSense — one Dockerfile, three consumable targets
#
#   runtime : lean production image. Used by GitHub Actions and `docker compose --profile ops`.
#   dev     : runtime + test/lint tooling. Source is bind-mounted by docker-compose (PYTHONPATH wins).
#   (internal stages: base, uv-base, deps-prod, deps-dev)
#
# Design notes
#   * Base is Ubuntu 24.04 — the same distribution as the `ubuntu-24.04` GitHub-hosted runner — so
#     "works on my laptop" and "works on the runner" share a libc, a Python (3.12) and a GDAL (3.8).
#   * CPU only. torch is resolved from the PyTorch CPU index (see pyproject.toml [tool.uv.sources]);
#     no CUDA libraries are ever downloaded.
#   * ecCodes arrives as a wheel (eccodes -> eccodeslib), so no libeccodes apt package is needed.
#   * Pillow's wheel bundles Raqm/HarfBuzz/FriBiDi, giving correct Urdu Nastaliq shaping without a browser.
#   * fonts-noto-core provides Noto Nastaliq Urdu and Noto Sans (OFL-licensed) for card rendering.
#   * Wheels only (`--no-build`): a dependency without a wheel fails the build loudly instead of
#     silently compiling for ten minutes.
#   * Reproducibility: if uv.lock is present it is honoured with --frozen. If it is missing the build
#     still succeeds but prints a warning; CI (ci.yml) fails on a missing or stale lock.
#
# Build:   docker build --target runtime -t smogsense:local .
# Slimmer: docker build --build-arg INSTALL_GDAL=false ...   (drops system GDAL; only needed for the
#          optional research-only MAIAC/HDF4 path)
# =============================================================================

ARG UBUNTU_VERSION=24.04
ARG UV_VERSION=0.11.7

# -----------------------------------------------------------------------------
# base: OS runtime libraries shared by every later stage
# -----------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS base
ARG INSTALL_GDAL=true
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      ca-certificates \
      tzdata \
      tini \
      curl \
      git \
      libgomp1 \
      fonts-noto-core \
 && if [ "${INSTALL_GDAL}" = "true" ]; then \
      apt-get install -y --no-install-recommends gdal-bin libgdal34t64; \
    fi \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# uv-base: the dependency resolver/installer (official multi-arch image, pinned)
# -----------------------------------------------------------------------------
FROM base AS uv-base
ARG UV_VERSION
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
# README.md is required by hatchling (pyproject `readme`); uv.lock may legitimately be absent on first build.
COPY pyproject.toml uv.lock* README.md ./

# -----------------------------------------------------------------------------
# deps-prod: third-party runtime dependencies + the project (non-editable)
# -----------------------------------------------------------------------------
FROM uv-base AS deps-prod
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
      uv sync --frozen --no-dev --no-install-project --no-build; \
    else \
      echo "WARNING: uv.lock not found; resolving fresh (NOT reproducible). Run 'make lock' and commit uv.lock." >&2; \
      uv sync --no-dev --no-install-project --no-build; \
    fi
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
      uv sync --frozen --no-dev --no-editable --no-build; \
    else \
      uv sync --no-dev --no-editable --no-build; \
    fi

# -----------------------------------------------------------------------------
# deps-dev: same, plus the `dev` dependency group (pytest, ruff, mypy, ...)
# -----------------------------------------------------------------------------
FROM uv-base AS deps-dev
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
      uv sync --frozen --no-install-project --no-build; \
    else \
      echo "WARNING: uv.lock not found; resolving fresh (NOT reproducible). Run 'make lock' and commit uv.lock." >&2; \
      uv sync --no-install-project --no-build; \
    fi

# -----------------------------------------------------------------------------
# runtime: lean production image
# -----------------------------------------------------------------------------
FROM base AS runtime
ARG USER_UID=1000
ARG USER_GID=1000
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# Ubuntu 24.04 images ship a default `ubuntu` user at UID/GID 1000; replace it with `smog`.
RUN set -eux; \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if [ -n "${existing_user}" ]; then userdel -r "${existing_user}" || true; fi; \
    existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if [ -n "${existing_group}" ]; then groupdel "${existing_group}" || true; fi; \
    groupadd --gid "${USER_GID}" smog; \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash smog

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    SMOGSENSE_HOME=/app \
    HOME=/home/smog \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.mpl \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

COPY --from=deps-prod /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
COPY web ./web
COPY data/schemas ./data/schemas
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh \
 && mkdir -p /app/data/raw /app/data/interim /app/data/processed /app/data/external /app/models /app/site /app/.state \
 && chown -R smog:smog /app /home/smog

USER smog
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["smogsense", "--help"]

# -----------------------------------------------------------------------------
# dev: runtime + tooling. The repository is bind-mounted at /workspace and shadows the installed
# package through PYTHONPATH, so edits are live without reinstalling.
# -----------------------------------------------------------------------------
FROM runtime AS dev
USER root
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends make jq less \
 && rm -rf /var/lib/apt/lists/*
COPY --from=deps-dev /opt/venv /opt/venv
# uv is kept in the dev image for `make lock` and `uv lock --check`
COPY --from=uv-base /usr/local/bin/uv /usr/local/bin/uv
ENV SMOGSENSE_HOME=/workspace \
    PYTHONPATH=/workspace/src \
    UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /workspace
USER smog
CMD ["bash"]
