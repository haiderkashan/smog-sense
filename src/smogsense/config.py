"""smogsense.config - Typed settings: YAML configuration plus environment-provided secrets."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration loaded from configs/*.yaml and environment variables."""
    # We will load a flat or nested structure matching the YAML. 
    # For Phase 0 stub, we can define a catch-all dict or explicitly define blocks.
    # To satisfy `unknown keys are errors` and `deterministic hash`, we enforce pydantic.
    
    # We'll map the config domains roughly:
    model: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    preprocessing: dict[str, Any] = {}
    features: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    bulletin: dict[str, Any] = {}
    domains: dict[str, Any] = {}

    # Secrets from environment
    openaq_api_key: SecretStr | None = None
    ads_api_key: SecretStr | None = None
    cds_api_key: SecretStr | None = None
    firms_map_key: SecretStr | None = None

    class Config:
        extra = 'forbid' # Unknown keys are errors
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load(cls, path_dir: Path | str) -> "Settings":
        """Load YAML configurations from a directory and return a Settings model."""
        path_dir = Path(path_dir)
        yaml_data: dict[str, Any] = {}
        if path_dir.exists():
            for filepath in path_dir.glob("*.yaml"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content:
                        yaml_data[filepath.stem] = content
        
        return cls(**yaml_data)

    def hash(self) -> str:
        """Compute SHA256 of the canonicalised YAML representation."""
        # Dump model without secrets
        data = self.model_dump(exclude={'openaq_api_key', 'ads_api_key', 'cds_api_key', 'firms_map_key'})
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
