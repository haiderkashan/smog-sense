import pandas as pd
import pandera as pa
import pytest

from smogsense.errors import SchemaViolation
from smogsense.utils.io import validate_frame


def test_validate_frame_duplicate_index():
    df = pd.DataFrame({"val": [1, 2]}, index=[0, 0])
    schema = pa.DataFrameSchema({"val": pa.Column(int)})
    with pytest.raises(SchemaViolation, match="Duplicate index"):
        validate_frame(df, schema)


def test_validate_frame_schema_error():
    df = pd.DataFrame({"val": ["a", "b"]}, index=[0, 1])
    schema = pa.DataFrameSchema({"val": pa.Column(int)})
    with pytest.raises(SchemaViolation, match="Schema violation"):
        validate_frame(df, schema)


def test_validate_frame_success():
    df = pd.DataFrame({"val": [1, 2]}, index=[0, 1])
    schema = pa.DataFrameSchema({"val": pa.Column(int)})
    validated = validate_frame(df, schema)
    assert validated.equals(df)
