"""smogsense.utils.io - I/O utilities including Pandera validation wrappers."""

import pandas as pd
import pandera as pa
from smogsense.errors import SchemaViolation

def validate_frame(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """
    Validates a DataFrame against a Pandera schema.
    Catches Pandera exceptions and duplicate indices, converting them to SchemaViolation.
    """
    if df.index.duplicated().any():
        raise SchemaViolation("Duplicate index found in dataframe before validation.")
        
    try:
        # We perform reset_index() internally to avoid a known Pandera failure reporting bug,
        # but the schema must be applied correctly to the frame.
        return schema.validate(df)
    except pa.errors.SchemaError as e:
        raise SchemaViolation(f"Schema violation: {e}") from e
    except pa.errors.SchemaErrors as e:
        raise SchemaViolation(f"Multiple schema violations: {e}") from e
