from pathlib import Path

import pytest
from pydantic import ValidationError

from smogsense.config import Settings


def test_settings_unknown_keys_forbid(tmp_path: Path) -> None:
    conf_dir = tmp_path / "configs"
    conf_dir.mkdir()

    # Write a valid stub
    with (conf_dir / "model.yaml").open("w") as f:
        f.write("learning_rate: 0.01\n")

    # Valid load
    s = Settings.load(conf_dir)
    assert s.model == {"learning_rate": 0.01}

    # Write an unknown key to trigger extra=forbid
    with (conf_dir / "unknown.yaml").open("w") as f:
        f.write("foo: bar\n")

    with pytest.raises(ValidationError):
        Settings.load(conf_dir)


def test_settings_hash(tmp_path: Path) -> None:
    conf_dir = tmp_path / "configs"
    conf_dir.mkdir()

    s = Settings()
    h1 = s.hash()
    assert isinstance(h1, str)
    assert len(h1) == 64
