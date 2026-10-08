# SmogSense: Phase 0 Study Guide (Scaffolding & Infrastructure)

> **Note**: This file is git-ignored and serves as a detailed academic and engineering reference for everything accomplished during Phase 0 of the SmogSense project.

## 1. The Objectives of Phase 0

Phase 0 was fundamentally about establishing the **"Zero-Cost, Ephemeral Architecture"** before writing any complex machine learning or data ingestion logic. The objective was to create a robust, production-ready skeleton that enforces strict code quality, guarantees container parity (meaning it runs exactly the same on a laptop as it does on GitHub Actions), and establishes the foundational contracts (schemas, logging, configuration) for the rest of the project.

By passing Gate 0 (G0), we proved that the foundational infrastructure works automatically and reliably.

---

## 2. Core Infrastructure & Dependency Management

### 2.1 Lightning-Fast Dependencies with `uv`
Instead of using standard `pip` or `poetry`, the project was configured to use `uv`, an extremely fast Rust-based Python package installer.
* We created a `pyproject.toml` file that explicitly lists all dependencies (like `torch`, `lightgbm`, `pandera`, `typer`).
* We generated a frozen `uv.lock` file. This guarantees reproducible builds across environments.
* **Why it matters**: In serverless CI/CD environments (like GitHub Actions), saving 3-4 minutes on dependency installation on every single run is crucial for staying within the free tier budget.

### 2.2 Dockerization and Container Parity
We built a multi-stage `Dockerfile` and a `compose.yml` to enforce container parity.
* **`base` stage**: Installs OS-level requirements (`libgomp1` for LightGBM, `libraqm0` for text shaping, `fonts-noto-core` for UI, and `git`).
* **`runtime` stage**: The stripped-down production image that runs as a non-root user (`smog`). This enforces security best practices.
* **`dev` stage**: The development image used locally. It mounts your local repository directly into the container so you can edit code and run tests without rebuilding the image.

### 2.3 Shell Automation
We created several bash scripts in the `scripts/` directory to automate GitHub Actions tasks (like `state_attach.sh` for cloning the bitemporal data state, and `publish_ghpages.sh` for deploying the frontend). These scripts include a shared `_git_auth.sh` to securely configure Git with GitHub tokens.

---

## 3. The Core Application Skeleton (Python)

We laid out the `src/smogsense/` package. Even though there is no real machine learning logic yet, the "interfaces" (contracts) are fully defined.

### 3.1 CLI Framework (`cli.py`)
We used **Typer** to build the command-line interface.
* We scaffolded all the commands the system will eventually need: `ingest`, `train`, `forecast`, `site build`, etc.
* We implemented the `doctor` command. Running `smogsense doctor --online` tests the application's configuration and network connectivity, which was a strict requirement for passing Gate 0.

### 3.2 Configuration Management (`config.py`)
We used **Pydantic** to manage configurations. The system reads YAML files from the `configs/` directory (e.g., `configs/models.yaml`, `configs/targets.yaml`) and validates them against strongly-typed Python schemas. This prevents the application from crashing halfway through a run due to a typo in a config file.

### 3.3 Structured Logging (`logging.py`)
We implemented structured JSON logging. Instead of just printing text, the system logs structured JSON objects containing timestamps, severity levels, and event data. Crucially, the logger includes a **Secret Redaction** feature to ensure API keys (like the OpenAQ or CAMS tokens) are never accidentally printed in the GitHub Actions logs.

### 3.4 Data Validation & Contracts (`contracts.py` & `io.py`)
We used **Pandera** to define strict schemas for Pandas DataFrames. Whenever the system loads CSV or Parquet files, it verifies that every expected column exists and that the data types are correct. We also defined the JSON schema for the final forecasting bulletin (`docs/bulletin.schema.json`).

---

## 4. Code Quality Guardrails (Pre-commit & CI)

To ensure the codebase never degrades in quality, we configured a massive suite of tools to run automatically via `pre-commit` and GitHub Actions.

### 4.1 The Linter Suite (`.pre-commit-config.yaml`)
* **Ruff**: An extremely fast Python linter and formatter. We configured it to enforce rigorous standards and sort imports.
* **Mypy (Strict)**: Static type checking. We configured it strictly, meaning every function must declare what types of arguments it accepts and what it returns (e.g., `def my_func(a: int) -> str:`). We fixed **37 strict typing errors** during Phase 0 to achieve compliance.
* **Shellcheck**: Lints all our `.sh` bash scripts for common bash pitfalls.
* **Hadolint**: Analyzes the `Dockerfile` for best practices.
* **Actionlint**: Verifies the syntax of our `.github/workflows/` YAML files.
* **Yamllint**: Ensures all `.yaml` files (including translation files) are perfectly formatted.

### 4.2 The Test Suite (`tests/`)
We wrote 124 passing tests using **Pytest**. This includes Unit tests, Integration tests, and Contract tests (verifying that our Pandera schemas round-trip perfectly). We also implemented a `--run-network` marker so that tests that ping live APIs don't run automatically and consume rate limits.

---

## 5. The Hardest Challenges & Fixes ("The Gotchas")

Getting Phase 0 fully green on the CI server required debugging several complex environmental discrepancies between Windows (your local machine) and Ubuntu (the CI server). Here is a detailed breakdown of what broke and how we fixed it:

### 5.1 The `ecCodes` C-Library Missing Error
* **The Problem**: The Python `eccodes` library is required to read GRIB format weather data from the Copernicus satellite service. However, the Python library is just a wrapper; it requires the actual C-library `libeccodes.so` installed on the host OS. When the smoke test ran in the GitHub Actions Ubuntu container, it crashed with `RuntimeError: Cannot find the ecCodes library`.
* **The Fix**: We didn't want to install compiling tools to build it from source. Instead, we discovered a third-party PyPI package called `eccodeslib` that packages pre-compiled binaries for Linux. We added `eccodeslib; sys_platform != 'win32'` to the `pyproject.toml` file, instantly resolving the error without bloating the Docker image.

### 5.2 The Urdu Nastaliq Text Shaping Issue (Pillow / Raqm)
* **The Problem**: The project generates WhatsApp-ready image cards with text in Urdu. The Python imaging library `Pillow` requires a C-library called `libraqm` to correctly render complex right-to-left scripts with ligatures (Nastaliq). A comment in the Dockerfile assumed that the Linux `Pillow` wheel bundled this library natively. It did not. The CI smoke test crashed with: `AssertionError: Pillow lacks Raqm: Urdu Nastaliq would not shape`.
* **The Fix**: We explicitly added `libraqm0` to the `apt-get install` block in the `base` stage of our `Dockerfile`. Pillow dynamically linked against it at runtime, fixing the Urdu rendering engine.

### 5.3 Cross-Platform File Quirks (Windows vs. Linux)
* **Executable Bits**: Git tracks whether a file is executable (`chmod +x`). Because you cloned the repo on Windows, the `.sh` scripts lost their executable permissions, causing the CI to fail. We fixed this by manually updating the Git index: `git update-index --chmod=+x scripts/*.sh`.
* **Mixed Line Endings**: Windows uses `\r\n` (CRLF) for line breaks, while Linux uses `\n` (LF). A Python script we ran locally accidentally wrote CRLF into the `Dockerfile`. The strict `pre-commit` hook caught this "mixed line ending" on the CI server and failed the build. We normalized the file to LF and pushed the fix.

### 5.4 Shellcheck Missing Source (`SC1091`)
* **The Problem**: Shellcheck failed our bash scripts because they contained `source "$(dirname "$0")/_git_auth.sh"`. Shellcheck couldn't "follow" this source because it was running in isolation via `pre-commit`.
* **The Fix**: We updated `.pre-commit-config.yaml` to pass the `-x` flag to Shellcheck (`args: ["-x"]`), which explicitly tells it to trace external file imports.

### 5.5 Yamllint Duplicate Keys
* **The Problem**: The translation files `web/i18n/en.yaml` and `ur.yaml` accidentally contained duplicate root keys (like `methodology` or `uncertainty` declared twice in the same file). `yamllint` caught this and crashed.
* **The Fix**: We carefully merged the duplicate keys at the bottom of the files to form a single, valid YAML tree.

### 5.6 Docker `COPY --from` Syntax Error & Package Building
* **The Problem**: The `Dockerfile` contained an invalid string interpolation syntax: `COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION}` which docker's buildkit rejected. Also, `uv sync --no-build` crashed because it was trying to build our local package without a pre-built wheel.
* **The Fix**: We hardcoded the `uv` image pull path or adjusted the build arguments, and we adjusted the `Dockerfile` to allow building the local project from source while strictly enforcing `--no-build` for external dependencies.

---

### Conclusion
By meticulously solving every platform discrepancy, typing error, and dependency quirk, Phase 0 is now a rock-solid foundation. The architecture currently standing is completely bulletproof, highly automated, and rigorously tested, setting up Phase 1 (Data Engineering) for immediate success.

## 6. Phase 1 — Zero-cost ETL, baselines, shadow ledger

This section acts as the source of truth for all implementations, architectural invariants, and discovered failure modes during Phase 1. A future engineer should read this to understand *why* the code is written the way it is.

### 6.1. Subphase 1a.1 — Resilient HTTP Foundation (P1-01)

The Resilient HTTP Foundation provides the core networking layer for the entire data ingestion pipeline, guaranteeing rate-limit compliance, automatic retries, circuit breaking, and complete auditability.

**Architecture and Core Mechanisms:**
*   **Async I/O (`httpx.AsyncClient`)**: Chosen for high-throughput, non-blocking network calls required when fetching thousands of sensor hours concurrently.
*   **Explicit Timeouts**:
    *   Connect: 10s. Read: 60s. Write/Pool: Standard fallback bounds.
    *   *Why*: Cloud APIs often drop connections. Infinite hang on a `read` operation freezes the daily workflow, burning Action minutes silently.
*   **RateBudget**:
    *   Enforces a project budget of 48 requests/minute and 1,600 requests/hour with a `safety_margin` of 0.8 (effectively 38/min and 1280/hr).
    *   Uses dual token-bucket refill mathematics independently tracking minute and hour limits.
    *   Protects against concurrent callers overdrawing tokens via an `asyncio.Lock()`.
*   **Deterministic Rate-Limit Reconciliation**:
    *   APIs return `x-ratelimit-remaining`. We reconcile our internal token bucket with this header to ensure we do not overspend if another process uses the same API key.
    *   *Invariant*: We only reconcile if the API's `x-ratelimit-limit` exactly matches our known `per_minute` or `per_hour` published capacity.
    *   *Why heuristics were rejected*: If we assumed any reset > 60 seconds meant the hourly bucket, or any limit * 5 meant the minute bucket, a non-standard API (like FIRMS' 10-minute window) could trick us into collapsing the hourly bucket down to 2 tokens, destroying throughput.
    *   *Ambiguous Windows*: If the window is ambiguous, we completely ignore the header. This mathematically bounds the error to our local token budget. Copernicus polling APIs do not even map cleanly to token buckets, reinforcing the need to ignore unknown headers.
*   **Circuit Breaker**:
    *   Opens after exactly 3 consecutive `429` (Too Many Requests) responses.
    *   Stays open for exactly 900 seconds (15 minutes).
    *   *Why*: APIs will ban keys for repeated 429 violations.
    *   *Recovery*: The breaker recovers automatically, and a single successful request immediately resets the consecutive 429 counter.
*   **Tenacity Retry Policy**:
    *   Exactly 6 *maximum total attempts* (1 initial + 5 retries). It is critical to distinguish "6 attempts" from "6 retries" (which means 7 attempts).
    *   *Retryable statuses*: 408, 429, 500, 502, 503, 504.
    *   *Timeout/Network Exceptions*: `httpx.TimeoutException` and `httpx.NetworkError` are retried exactly like 502s. They do **not** increment the 429 circuit breaker.
    *   *Early Termination*: The circuit breaker can forcefully terminate retries before attempt 6 if 3 consecutive 429s hit.
    *   *Delay Math*: Exponential full jitter `U(0,1) * min(120, 2 * 2^n)`.
*   **HTTP-Date Retry-After Handling**:
    *   The standard allows `Retry-After: Wed, 21 Oct 2015 07:28:00 GMT`.
    *   *Why*: Naive integer parsing throws exceptions and crashes the pipeline. We use `email.utils.parsedate_to_datetime` to parse HTTP dates strictly against UTC.
*   **Audit Logging**:
    *   Every request emits structured JSON with a SHA-256 hash of the response body.
    *   *Why*: Cryptographic guarantee of reproducibility and provenance.
    *   *Failure Isolation*: If audit JSON encoding crashes, the HTTP request silently succeeds. It is unacceptable to let an observability failure destroy a successful data fetch. We assume standard stdout/stderr persistence handles the durable storage.
*   **Credential Sanitization**:
    *   Query parameters: explicitly scrubs keys matching `token`, `secret`, `api_key`, `password`, `map_key`, etc.
    *   NASA FIRMS Path Redaction: FIRMS embeds its `MAP_KEY` directly inside the URL path (`/api/area/csv/<MAP_KEY>/...`). Redaction explicitly scrubs the path segment, as query parameter filtering alone is insufficient.
    *   All project-specific credentials discovered: OpenAQ API key, ADS API key, CDS API key, FIRMS MAP_KEY, Earthdata token.

### 6.1.1. P1-01 Failure Modes and Lessons

During implementation, several critical bugs were written, identified during strict verification, and fixed.

1.  **Sleeping while holding the RateBudget lock**
    *   *Original*: The code locked the token bucket, realized it needed to wait 2 seconds, called `await asyncio.sleep(2)`, and then released the lock.
    *   *Why it looked reasonable*: It logically serialized token consumption.
    *   *Why it was wrong*: It blocked all 100+ concurrent coroutines from even evaluating their states. Fetching 100 stations effectively became completely synchronous.
    *   *Production Failure*: Destroys concurrency.
    *   *Fix*: Compute the required sleep, release the lock, sleep, then `while True` loop to reacquire the lock and re-evaluate.
    *   *Regression test*: `test_concurrent_refill` executes 150 tasks concurrently.
2.  **Heuristic rate-limit-window classification**
    *   *Original*: Assumed `reset_seconds > 60` implied the hourly bucket.
    *   *Why it looked reasonable*: It is a common fallback heuristic when APIs omit window metadata.
    *   *Why it was wrong*: An API returning a 2-minute reset for a 2-minute limit would incorrectly obliterate the 1,600-token hourly budget.
    *   *Production Failure*: Destroys throughput by incorrectly zeroing out hourly buckets.
    *   *Fix*: Deterministic matching only against known capacities.
    *   *Regression test*: `test_ambiguous_ratelimit_header_ignored`.
3.  **Ambiguous x-ratelimit-remaining**
    *   *Original*: Applied the remaining token count to whichever bucket it guessed.
    *   *Why wrong*: Destroys unrelated independent buckets.
    *   *Production Failure*: Destroys throughput.
    *   *Fix*: Only applies if the window limit explicitly matches known bounds.
    *   *Regression test*: `test_ambiguous_ratelimit_header_ignored`.
4.  **HTTP-date Retry-After parsing**
    *   *Original*: Casted `Retry-After` to `int()`.
    *   *Why wrong*: Crashes on HTTP-dates.
    *   *Production Failure*: Unhandled exception crashes ETL loop.
    *   *Fix*: `email.utils` parsing.
    *   *Regression test*: `test_resilient_client_retry_after_http_date`.
5.  **Missing network/timeout retries**
    *   *Original*: Only retried HTTP status codes.
    *   *Why wrong*: Network layer drops (Connection Reset) crash the pipeline instantly.
    *   *Production Failure*: Network instability halts ingestion.
    *   *Fix*: Added `httpx.RequestError` to Tenacity retry targets.
    *   *Regression test*: `test_network_error_retried_does_not_trip_circuit_breaker`.
6.  **Secret leakage in query parameters and URL paths**
    *   *Original*: `url.copy_with(username=None)` which only strips HTTP Basic Auth.
    *   *Why wrong*: NASA FIRMS `MAP_KEY` leaked in the path, OpenAQ leaked in `?api_key=`.
    *   *Production Failure*: Credentials permanently written to audit logs.
    *   *Fix*: Explicit list of `SENSITIVE_KEYS` and path-segment masking.
    *   *Regression test*: `test_audit_logger_redacts_firms_path_secret`.
7.  **Audit logging crashing successful requests**
    *   *Original*: Audit logger was directly in the return path without a `try/except`.
    *   *Why wrong*: Observability failures (e.g. non-serializable objects) destroyed valid payloads.
    *   *Production Failure*: Data loss due to logging bugs.
    *   *Fix*: Broad `Exception` catch inside `log_request`.
    *   *Regression test*: `test_audit_logger_failure_isolation`.
8.  **Retry/circuit-breaker attempt semantics**
    *   *Original*: 6 retries.
    *   *Why wrong*: This means 7 attempts, violating the spec.
    *   *Production Failure*: API terms violation, prolonged blocked jobs.
    *   *Fix*: 6 total attempts. Circuit breaker can early-terminate on 3rd attempt.
    *   *Regression test*: `test_resilient_client_retry_matrix`.

### 6.1.3. P1-01 Tests as Study Material

The test suite must be understood as an immovable architectural guardrail.
*   **`test_concurrent_refill`**: Spawns 150 tasks. Proves tasks can overlap correctly without driving the token count below 0 and without blocking sequentially. Protects against lock-sleep bugs.
*   **`test_resilient_client_retry_after_http_date`**: Monkeypatches `asyncio.sleep` to ensure a parsed 3600-second date wait executes instantly while proving the arithmetic. Protects against string-cast crashes.
*   **`test_circuit_breaker_opens_on_429s`**: Yields exactly three 429s, verifies the circuit opens, yields a 200, verifies it fails fast, advances time by 900s, verifies it allows traffic. Protects the 3-strike / 15-minute constants.
*   **`test_resilient_client_retry_matrix`**: Proves exactly 6 attempts for a 502, exactly 3 for a 429, exactly 1 for a 403. Protects the max-attempt logic.
*   **`test_network_error_retried_does_not_trip_circuit_breaker`**: Yields 4 network timeouts. Proves it retries, but importantly verifies the circuit breaker state remains CLOSED.
*   **`test_audit_logger_redacts_firms_path_secret`**: Feeds a FIRMS URI. Proves the output log string contains `***REDACTED***` in the specific path segment.
*   **`test_ambiguous_ratelimit_header_ignored`**: Proves that an unclassified reset threshold completely aborts bucket reconciliation rather than guessing incorrectly.
*   **`test_audit_logger_failure_isolation`**: Proves that an internal exception raised inside `log_request` does not crash the upstream HTTP payload delivery.

**Final Quality Gates**:
*   `pytest`: Runs all deterministic scenarios.
*   `ruff check`: Syntax, formatting, and complexity gating.
*   `mypy --strict`: Proves type consistency across the network boundaries.

---

### 6.2. Subphase 1a.2 — OpenAQ Live Client & Station Registry (P1-02)

#### 6.2.1. OpenAQ Client Implementation
**API Mechanics:**
*   `/v3/locations`: Fetches metadata for all stations in a bounding box (from `domains.yaml`).
*   `/v3/sensors/{id}/hours`: Fetches the actual hourly measurements for a specific sensor.
*   **Pagination Termination**: We strictly calculate `limit * page >= found`. We never use empty-page guessing, but we do use `if not data.get("results"): break` to prevent catastrophic infinite loops if the metadata lies.
*   **Parameter Identification**:
    *   *Authoritative*: PM2.5 = `2`, Temperature = `19`, RH = `100`. We never use string matching like `name == "pm25"` because provider strings vary.
*   **Sensor Selection**:
    *   If a location has multiple PM2.5 sensors, we employ a deterministic policy: highest `coverage.percentComplete` wins. If tied, lowest `sensor_id` wins.
*   **Concurrent Hourly Retrieval**:
    *   `fetch_hourly` uses `asyncio.gather` over the sensors using the shared P1-01 rate limited client to manage throughput gracefully.
*   **`list_locations()` Contract**:
    *   Returns a DataFrame with columns: `location_id`, `name`, `provider`, `lat`, `lon`, `is_monitor`, `sensor_id_pm25`, `sensor_id_rh`, `sensor_id_temp`, `first_datetime`, `last_datetime`, `lifecycle_uptime`.
    *   `is_monitor` strictly preserves explicit True, explicit False, or `None` (missing).
*   **`fetch_hourly()` Contract**:
    *   Receives the registry mapping dataframe.
    *   Spawns concurrent tasks utilizing the P1-01 `ResilientClient` to fetch each individual parameter sensor.
    *   **Pivoting**: Pivots the independent results into a wide format.
    *   Returns: `location_id | ts_utc | pm25_ugm3 | rh_pct | temperature_c`.
    *   All timestamps explicitly mapped to `UTC` timezone awareness.
    *   Returns *raw* data. Quality control flags are applied in later pipeline phases.

#### 6.2.2. Station Registry Implementation
**Registry Mechanics:**
*   **Output**: Saved as a versioned Parquet file.
*   **30-Day History**: Evaluates the literal calendar span (`last_datetime - first_datetime >= 30 days`).
*   **Co-location Mathematics**:
    *   Rule: Stations <= 50m apart with *different providers* are grouped.
    *   Calculated using `pyproj.Geod(ellps="WGS84")`.
    *   Uses a Union-Find algorithm to enforce transitive co-location (if A=B and B=C, then A=C).
    *   **Deterministic IDs**: The `colocated_group_id` is assigned as the `min(location_id)` of the group. This guarantees reproducibility regardless of dataframe row order.
    *   **Singleton Rule**: Isolated stations receive `pd.NA`. They MUST NOT receive a fake group ID. A fake group ID mathematically obscures the difference between an isolated station and a duplicate set, making downstream auditing impossible.
*   **Registry Versioning**:
    *   Uses a SHA-256 hash of sorted `location_id` and `last_datetime` strings. Deterministic, unlike `datetime.now()`.
*   **Empty Behavior**: Gracefully initializes the exact schema even if zero locations are found.

---

### 6.3. CRITICAL Semantic Decision — Lifecycle Coverage vs 90-Day Uptime

**THIS IS A PERMANENT ENGINEERING INVARIANT. DO NOT VIOLATE IT.**

The project specification requires a station to have a specific **90-day PM2.5 uptime** to be eligible for machine learning.

The OpenAQ `/v3/locations` metadata payload provides a field named `coverage.percentComplete`.

*   **The Difference**: `coverage.percentComplete` reflects the data density over the sensor's *entire historical lifecycle*. It is NOT a rolling 90-day metric.
*   **The Invariant Rule**:
    1.  The OpenAQ coverage field is retained and renamed explicitly to `lifecycle_uptime`.
    2.  It MUST NOT be renamed or interpreted as `pm25_uptime_90d`.
    3.  `lifecycle_uptime` alone **cannot satisfy** the 90-day requirement.
    4.  If the column `pm25_uptime_90d` is missing, the station **strictly fails** the eligibility gate.
*   **Where Calculation Occurs**: The actual `pm25_uptime_90d` must be mathematically calculated from the appropriate historical observation window. The **historical backfill pipelines (P1-07 and P1-13)** are exclusively responsible for fetching that data, computing the true ratio, and injecting `pm25_uptime_90d` into the registry pipeline.
*   **Testing**:
    *   `test_registry_fails_all_if_90d_uptime_missing`: Explicitly asserts that the raw `list_locations` dataframe (which lacks the 90-day column) fails the eligibility check for 100% of stations, regardless of their `lifecycle_uptime`.
    *   `test_registry_colocation_and_eligibility`: Adds an assertion that explicitly testing missing `pm25_uptime_90d` alongside high `lifecycle_uptime` yields False.

---

### 6.4. P1-02 Failure Modes and Lessons

During independent verification, the following severe flaws were discovered and corrected:

1.  **Lifecycle uptime incorrectly treated as 90-day uptime**
    *   *Original*: `percentComplete` passed directly into `pm25_uptime_90d`.
    *   *Why wrong*: It was a false proxy violating scientific semantics.
    *   *Corrected*: Renamed to `lifecycle_uptime`. Strict 90-day check enforced.
    *   *Regression Test*: `test_registry_colocation_and_eligibility` (Loc 8 missing 90d uptime fails).
2.  **Missing/null isMonitor**
    *   *Original*: Defaulted missing values to `False`.
    *   *Why wrong*: Conflates explicit low-cost classifications with unknown metadata.
    *   *Corrected*: Uses `.get("isMonitor")` to preserve `pd.NA`.
    *   *Regression Test*: `test_list_locations_semantics`.
3.  **Incorrect parameter assumptions**
    *   *Original*: Checked if parameter name string matched "pm25".
    *   *Why wrong*: Upstream string representations vary wildly.
    *   *Corrected*: Authoritative integer IDs (`PARAM_ID_PM25 = 2`, `PARAM_ID_TEMP = 19`, `PARAM_ID_RH = 100`).
    *   *Regression Test*: `test_list_locations_semantics`.
4.  **Incorrect hourly long-form shape**
    *   *Original*: `fetch_hourly` appended a row per sensor.
    *   *Why wrong*: The project schema requires one row per `station_hour` containing all parameter columns.
    *   *Corrected*: Grouped by `location_id` and `ts_utc`, taking the `.first()` non-null value per column.
    *   *Regression Test*: `test_fetch_hourly_pivot`.
5.  **Co-location singleton grouping**
    *   *Original*: Every station received a unique group ID.
    *   *Why wrong*: Violates singleton semantics.
    *   *Corrected*: Filtered `root_sizes > 1`, leaving singletons as `pd.NA`.
    *   *Regression Test*: `test_registry_colocation_and_eligibility`.
6.  **Nondeterministic group IDs**
    *   *Original*: Iteration order dictated the group integer.
    *   *Why wrong*: Run the pipeline twice, get different IDs.
    *   *Corrected*: `colocated_group_id` = `min(location_id)` of the group.
    *   *Regression Test*: `test_registry_colocation_determinism`.
7.  **Sensor-selection ambiguity**
    *   *Original*: Silently overwrote the PM2.5 sensor if a second one was found.
    *   *Why wrong*: Discards data unpredictably.
    *   *Corrected*: Highest coverage wins.
    *   *Regression Test*: `test_list_locations_semantics`.

---

### 6.5. P1-02 Verification Checklist
Before approving future modifications to the registry or ingestion clients, the reviewer must check:

- [ ] **API Contract**: Are URL paths and parameters exact?
- [ ] **Pagination**: Is infinite loop protection in place (`limit * page >= found`)?
- [ ] **Parameter IDs**: Are explicit integer IDs used?
- [ ] **Null Semantics**: Are explicit `None` / `pd.NA` values preserved?
- [ ] **Sensor Selection**: Is selection deterministic for duplicates?
- [ ] **Timestamp Semantics**: Are all bounds strictly `utc=True`?
- [ ] **Output Schema**: Does `fetch_hourly` pivot exactly to one row per station-hour?
- [ ] **Co-location Mathematics**: Are distance rules applied via WGS84 Geodesic?
- [ ] **Provider Rule**: Are identical providers excluded from co-location?
- [ ] **Singleton Rule**: Do isolated stations strictly receive `pd.NA`?
- [ ] **Deterministic IDs**: Are group and registry version IDs strictly reproducible?
- [ ] **30-Day History**: Is the calendar span calculated correctly?
- [ ] **Lifecycle Uptime**: Is it cleanly separated from the 90-day proxy?
- [ ] **90-Day Uptime**: Is eligibility strictly failed if actual 90-day uptime is absent?
- [ ] **Missing-History Behavior**: Is it properly defined?
- [ ] **Tests**: Are boolean arrays cast correctly (`bool() is False`)?
- [ ] **Quality Gates**: Pytest passes? Ruff passes? Mypy `--strict` passes?

---

### 6.6. Cross-Phase Architectural Invariants

The following invariants apply to all future phases:
1.  **Never use a proxy metric while naming it as the real metric.**
2.  **Never infer undocumented API semantics from convenient thresholds.**
3.  **Never introduce future information into historical/as-of calculations.**
4.  **Never let logging expose credentials.** (Check paths, headers, and query strings).
5.  **Never let observability failures destroy successful data ingestion.**
6.  **Never rely solely on happy-path tests for external APIs.**
7.  **Determinism matters for reproducibility.** (Always sort before hashing).
8.  **Ambiguous external metadata must be handled conservatively.**
9.  **Scientific semantics take precedence over implementation convenience.**

---

### 6.7. Reviewer / Human Approval Guide

**What the AI / Automated Gates Can Verify:**
*   Python syntax and style (`ruff`).
*   Type correctness (`mypy --strict`).
*   Deterministic transformations (hashing, math).
*   Pagination mechanics and loop termination.
*   Data shapes and schema presence.
*   Execution of unit tests.

**What the Human Reviewer MUST Verify:**
*   **Semantic Reality**: Whether an API field actually means what the code claims it means (e.g., `coverage` vs `90-day uptime`).
*   **Scientific Validity**: Whether distance thresholds, imputation algorithms, and statistical definitions match atmospheric science reality.
*   **Project Specification Alignment**: Whether an architectural fallback (e.g., guessing rate limit windows) silently degrades throughput or correctness.
*   **API Contract Truth**: Whether the external API documentation has been interpreted correctly, not just whether the mocked unit test passes.
*   **Test Intent**: Whether a test proves the *intended requirement* (e.g., parsing an HTTP-date correctly) rather than merely exercising code branches to hit 100% coverage.
*   **Fallback semantics**: Whether a fallback silently changes semantics.



## 6.8 P1-03: CAMS Ingestion & Temporal Alignment (Resilient HTTP Foundation Layer)
### Overview
Subphase 1a.3 implements CAMS forecast fetching, strict bitemporal alignment, and precise GRIB extraction avoiding positional-dimension hacks.

### Bitemporal Contract & Alignment (Contract B)
1. **Ambiguity Resolution:** The documentation was ambiguous about whether the `<12h` lead constraint intentionally produced 10-hour gaps in meteorology or was an error. We established Contract B: continuous bitemporal stitching where `B = max { cycle : B + 10h <= t }` with no `<12h` gap restriction.
2. **Implementation:** `asof_cams_run` implements the explicit lattice (00Z -> previous 12Z, 06Z -> previous 12Z, 12Z -> same-day 00Z). `stitch_cams_series` utilizes this to sweep backwards, continuously mapping `target_hour_utc` to the maximum *knowable* `cams_cycle_utc` at that exact hour `t`.
3. **Leakage Protection:** No cycle is ever selected if it requires future knowledge relative to `t`. Normalization natively pushes all timestamps (aware/naive) directly to UTC.

### CAMS/ADS Ingestion
1. **Authentication/Credentials:** Purely driven by environment variables.
2. **Licence Faults:** Explicitly intercepted `cdsapi` exceptions containing "accept the terms/licence". This raises `SourceUnavailable("Accept the dataset licence on the ADS website once")`, structurally enforcing the CLI's exit code 20 handling.
3. **Polling Budget:** Polling loop explicitly calculates elapsed `time.monotonic()` against a rigid 20-minute operational wall-clock budget. Sleep intervals dynamically shrink.
4. **Atomic Concurrency:** Unique `tempfile.NamedTemporaryFile` landing directly into `dest_path.parent` before executing `os.replace` guarantees zero concurrency overwrites.

### Explicit GRIB/Xarray Processing
1. **Positional Safety:** Replaced positional index slicing `[v].values` with explicitly queried `lat_dim` and `lon_dim`. Non-spatial axes are `stacked` into an explicit sample array, ensuring exact matching of `valid_time` regardless of dataset shape `(time, step, lat, lon)` vs `(step, lat, lon)`.
2. **Units Auditing:** Checks `da.attrs.get("units")` before mutating variables. Maps `kg m**-3` correctly to `* 1e9` for PM2.5, and evaluates `K` and `C` conversions dynamically for temp/dewpoint using defined alias subsets.
3. **Bilinear Spatial Extraction:**
   - Detects and strictly flips ascending/descending orientation before providing coordinate pairs to `scipy`'s `RegularGridInterpolator(method='linear')`.
   - Modulo arithmetic securely wraps `-180/180` and `0/360` domains.
4. **Centroids and Geodesics:** WGS84 great-circle distance `grid_distance_km` calculates exact range from the point target to the physically *nearest* grid node, avoiding interpolation-distance confusion. `outside_grid` is recorded natively.

### Distribution and CRPS (Phase 1a.5)
1. **Tail formulations:** The lower tail uses a log-linear shape from the lowest two percentiles (`0.05`, `0.10`) capped at 0. The upper tail projects linearly on a transformed space `-\ln(1-	au)` using the top two percentiles (`0.90`, `0.95`). Vectorizing inverse evaluation properly clamps values directly to prevent `log(0)` overflows on extremes.
2. **CRPS Numeric Integrity:** Numerical pinball integration exactly aligns mathematically to the `scoringrules` implementation without relying on R code integrations. The 19-knot trapezoidal approximation precisely meets `< 0.5%` absolute error targets compared to exact Normal distributions evaluated across extreme endpoints.
3. **Monotonicity:** The `QuantileFunction` intrinsically applies ascending sorts (`np.sort`) on init to guarantee submodular ordering of probabilities and crossing boundaries, minimizing potential losses before any metric queries.
