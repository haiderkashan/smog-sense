import pytest
from smogsense.config import Settings
from pydantic import ValidationError

def test_settings_unknown_keys_forbid(tmp_path):
    conf_dir = tmp_path / "configs"
    conf_dir.mkdir()
    
    # Write a valid stub
    with open(conf_dir / "model.yaml", "w") as f:
        f.write("learning_rate: 0.01\n")
    
    # Valid load
    s = Settings.load(conf_dir)
    assert s.model == {"learning_rate": 0.01}
    
    # Write an unknown key to trigger extra=forbid
    with open(conf_dir / "unknown.yaml", "w") as f:
        f.write("foo: bar\n")
        
    with pytest.raises(ValidationError):
        Settings.load(conf_dir)

def test_settings_hash(tmp_path):
    conf_dir = tmp_path / "configs"
    conf_dir.mkdir()
    
    s = Settings()
    h1 = s.hash()
    assert isinstance(h1, str)
    assert len(h1) == 64
