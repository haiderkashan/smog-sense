"""Repository guardrails as executable checks: $0 budget, supply-chain pinning, least privilege, link integrity."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

AI_CONTEXT_FILES = ["memory.md", "decisions.md", "roadmap.md", "AGENTS.md", "CLAUDE.md"]
PUBLIC_DOC_GLOBS = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/**/*.md",
    "data/README.md",
    "models/*.md",
    "docker/README.md",
    "web/**/*.md",
    "notebooks/README.md",
    "tests/fixtures/README.md",
    ".github/*.md",
]
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "site",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".state",
}
# Packages whose use would break the strict zero-cost constraint (paid inference or billable clouds)
PAID_OR_BILLABLE = {
    "openai",
    "anthropic",
    "boto3",
    "botocore",
    "google-cloud-storage",
    "google-cloud-bigquery",
    "google-cloud-aiplatform",
    "azure-storage-blob",
    "azure-identity",
    "sagemaker",
    "vertexai",
    "cohere",
    "replicate",
    "twilio",
    "pinecone-client",
}
SECRET_PATTERNS = [
    r"ghp_[A-Za-z0-9]{36}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
]


def _public_docs(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in PUBLIC_DOC_GLOBS:
        files.update(
            p for p in root.glob(pattern) if p.is_file() and p.name not in AI_CONTEXT_FILES
        )
    return sorted(files)


def _text_files(root: Path) -> list[Path]:
    suffixes = {
        ".py",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".json",
        ".sh",
        ".css",
        ".js",
        ".j2",
        ".cff",
        ".txt",
        "",
    }
    out = []
    for p in root.rglob("*"):
        if (
            p.is_file()
            and not (set(p.relative_to(root).parts) & SKIP_DIRS)
            and p.suffix in suffixes
            and p.name not in AI_CONTEXT_FILES
        ):
            if p.name in {"uv.lock", ".env"}:
                continue
            out.append(p)
    return out


def test_gitignore_keeps_ai_context_secrets_and_weights_out_of_git(repo_root: Path) -> None:
    lines = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    for required in [
        *AI_CONTEXT_FILES,
        ".env",
        ".cdsapirc",
        "models/*",
        "*.nc",
        "*.grib2",
        "*.pt",
        "site/",
        ".state/",
    ]:
        assert required in lines, f".gitignore must contain {required!r}"
    assert "!.env.example" in lines


def test_env_example_has_no_secret_values(repo_root: Path) -> None:
    for line in (repo_root / ".env.example").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            if key.endswith(("_KEY", "_TOKEN", "_ID", "_SECRET")) and key not in {
                "HOST_UID",
                "HOST_GID",
            }:
                assert value.strip() == "", f"{key} must be empty in .env.example"


def test_public_docs_never_reference_git_ignored_ai_files(repo_root: Path) -> None:
    pattern = re.compile("|".join(re.escape(n) for n in AI_CONTEXT_FILES))
    offenders = [
        str(p.relative_to(repo_root))
        for p in _public_docs(repo_root)
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"public docs must not link git-ignored files: {offenders}"


def test_markdown_relative_links_resolve(repo_root: Path) -> None:
    link = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    broken: list[str] = []
    for doc in _public_docs(repo_root):
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        for target in link.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = target.split("#", 1)[0]
            if path and not (doc.parent / path).resolve().exists():
                broken.append(f"{doc.relative_to(repo_root)} -> {target}")
    assert not broken, broken


def test_no_obvious_secrets_in_repository(repo_root: Path) -> None:
    compiled = [re.compile(p) for p in SECRET_PATTERNS]
    hits = []
    for path in _text_files(repo_root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits += [f"{path.relative_to(repo_root)}: {c.pattern}" for c in compiled if c.search(text)]
    assert not hits, hits


def _workflows(root: Path) -> dict[str, dict[str, Any]]:
    return {
        p.name: yaml.safe_load(p.read_text(encoding="utf-8"))
        for p in sorted((root / ".github/workflows").glob("*.yml"))
    }


def test_every_third_party_action_is_pinned_to_a_commit_sha(repo_root: Path) -> None:
    unpinned = []
    for name, wf in _workflows(repo_root).items():
        for job in wf["jobs"].values():
            for step in job.get("steps", []):
                uses = step.get("uses")
                if (
                    uses
                    and not uses.startswith(("./", "docker://"))
                    and not re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", uses)
                ):
                    unpinned.append(f"{name}: {uses}")
    assert not unpinned, unpinned


def test_workflows_are_least_privilege_and_bounded(repo_root: Path) -> None:
    for name, wf in _workflows(repo_root).items():
        triggers = wf.get("on", wf.get(True, {}))
        assert "pull_request_target" not in triggers, (
            f"{name}: pull_request_target exposes secrets to forks"
        )
        assert "permissions" in wf, f"{name}: declare top-level permissions"
        assert wf["permissions"] in (
            {"contents": "read"},
            {"contents": "read", "packages": "write"},
        ), f"{name}: default must be read-only"
        for job_id, job in wf["jobs"].items():
            assert "timeout-minutes" in job, f"{name}/{job_id}: needs timeout-minutes"


def test_scheduled_workflows_use_off_peak_minutes(repo_root: Path) -> None:
    for name, wf in _workflows(repo_root).items():
        for entry in wf.get("on", wf.get(True, {})).get("schedule", []):
            assert entry["cron"].split()[0] != "0", (
                f"{name}: avoid minute 0 (GitHub delays top-of-hour schedules)"
            )


def test_dockerfile_is_pinned_and_runs_non_root(repo_root: Path) -> None:
    text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG UBUNTU_VERSION=24\.04$", text, re.M)
    assert "FROM ubuntu:${UBUNTU_VERSION}" in text
    assert ":latest" not in text
    runtime = text.split("FROM base AS runtime", 1)[1].split("FROM runtime AS dev", 1)[0]
    assert re.search(r"^USER smog$", runtime, re.M)
    assert "--no-build" in text, "wheels-only installs keep builds fast and reproducible"


def test_compose_never_passes_the_git_token_into_containers(repo_root: Path) -> None:
    compose = yaml.safe_load((repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "GITHUB_TOKEN" not in compose["x-env"]
    ops = compose["x-ops"]
    assert (
        ops["cap_drop"] == ["ALL"]
        and ops["read_only"] is True
        and "no-new-privileges:true" in ops["security_opt"]
    )


def test_no_paid_or_billable_dependencies(repo_root: Path) -> None:
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(pyproject["project"]["dependencies"])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        specs += extra
    specs += pyproject.get("dependency-groups", {}).get("dev", [])
    names = {re.split(r"[<>=!~\[ ;]", s, maxsplit=1)[0].lower().replace("_", "-") for s in specs}
    assert not (names & PAID_OR_BILLABLE), names & PAID_OR_BILLABLE


def test_configs_are_consistent_with_each_other(repo_root: Path) -> None:
    load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))  # noqa: E731
    model, feats, evaluation, bulletin = (
        load("model.yaml"),
        load("features.yaml"),
        load("evaluation.yaml"),
        load("bulletin.yaml"),
    )
    grid = model["quantiles"]["grid"]
    assert len(grid) == 19 and all(abs(grid[i] - (0.05 + 0.05 * i)) < 1e-9 for i in range(19))
    assert set(model["quantiles"]["public"]) <= set(grid)
    assert (
        model["target"]["horizons_h"]
        == feats["targets"]["horizons_h"]
        == evaluation["horizons_h"]
        == [24, 48, 72]
    )
    assert model["target"]["min_valid_hours"] == feats["targets"]["min_valid_hours"] == 18
    lowers = [c["lower"] for c in bulletin["aqi"]["categories"]]
    assert lowers == sorted(lowers) and lowers[0] == 0.0
    assert evaluation["methods"][0] == "m0_persistence" and len(evaluation["methods"]) == 9


# ----------------------------------------------------------------------------- documentation integrity (anchors, spec references)
def _github_slug(heading: str) -> str:
    """GitHub's heading anchor: strip markdown, lowercase, drop punctuation except hyphen/underscore, spaces to hyphens."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_]{1,2}(\S(?:.*?\S)?)[*_]{1,2}", r"\1", text)
    text = text.strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _headings(path: Path) -> list[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    return [line.lstrip("#").strip() for line in text.splitlines() if re.match(r"^#{1,6}\s", line)]


def test_markdown_anchor_links_resolve(repo_root: Path) -> None:
    link = re.compile(r"\[[^\]]*\]\(([^)\s#]+\.md)#([^)\s]+)\)")
    broken: list[str] = []
    for doc in _public_docs(repo_root):
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        for path, anchor in link.findall(text):
            target = (doc.parent / path).resolve()
            if target.exists() and anchor not in {_github_slug(h) for h in _headings(target)}:
                broken.append(f"{doc.relative_to(repo_root)} -> {path}#{anchor}")
    assert not broken, broken


def test_stub_specification_references_resolve_to_real_headings(repo_root: Path) -> None:
    ref = re.compile(r"Specification: (docs/[\w\-]+\.md) → '([^']+)'")
    broken: list[str] = []
    for source in sorted((repo_root / "src").rglob("*.py")):
        for doc_path, heading in ref.findall(source.read_text(encoding="utf-8")):
            doc = repo_root / doc_path
            normalised = (
                [re.sub(r"^\d+\.\s*", "", h).replace("`", "").lower() for h in _headings(doc)]
                if doc.exists()
                else []
            )
            if not any(heading.lower() == h for h in normalised):
                broken.append(f"{source.relative_to(repo_root)} -> {doc_path} '{heading}'")
    assert not broken, broken


def test_markdown_tables_are_well_formed(repo_root: Path) -> None:
    """A stray unescaped '|' (even inside a code span or $|x|$ math) splits a GFM table cell and garbles the table."""
    problems: list[str] = []
    for doc in _public_docs(repo_root):
        text = re.sub(
            r"```.*?```",
            lambda m: "\n" * m.group(0).count("\n"),
            doc.read_text(encoding="utf-8"),
            flags=re.S,
        )
        expected: int | None = None
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.lstrip().startswith("|"):
                expected = None
                continue
            cells = len(re.split(r"(?<!\\)\|", line.strip())) - 2
            if expected is None:
                expected = cells
            elif cells != expected:
                problems.append(
                    f"{doc.relative_to(repo_root)}:{number}: {cells} cells, header has {expected}"
                )
    assert not problems, problems
