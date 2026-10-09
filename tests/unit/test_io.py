import pandas as pd
import pandera.pandas as pa
import pytest

from smogsense.errors import SchemaViolation
from smogsense.utils.io import validate_frame


def test_validate_frame_duplicate_index() -> None:
    df = pd.DataFrame({"val": [1, 2]}, index=[0, 0])
    schema = pa.DataFrameSchema(  # type: ignore[no-untyped-call]
        {"val": pa.Column(int)}
    )
    with pytest.raises(SchemaViolation, match="Duplicate index"):
        validate_frame(df, schema)


def test_validate_frame_schema_error() -> None:
    df = pd.DataFrame({"val": ["a", "b"]}, index=[0, 1])
    schema = pa.DataFrameSchema(  # type: ignore[no-untyped-call]
        {"val": pa.Column(int)}
    )
    with pytest.raises(SchemaViolation, match="Schema violation"):
        validate_frame(df, schema)


def test_validate_frame_success() -> None:
    df = pd.DataFrame({"val": [1, 2]}, index=[0, 1])
    schema = pa.DataFrameSchema(  # type: ignore[no-untyped-call]
        {"val": pa.Column(int)}
    )
    validated = validate_frame(df, schema)
    assert validated.equals(df)


def test_validate_frame_resets_index() -> None:
    df = pd.DataFrame({"val": [10, 20]}, index=[5, 9])
    schema = pa.DataFrameSchema({"val": pa.Column(int)})  # type: ignore[no-untyped-call]
    validated = validate_frame(df, schema)
    assert list(validated.index) == [0, 1]


def test_get_git_sha_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from smogsense.utils.io import get_git_sha

    def mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", mock_run)
    assert get_git_sha() == "unknown"
