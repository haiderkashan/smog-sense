import re

with open('tests/unit/test_repo_hygiene.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('def _workflows(repo_root: Path) -> dict[str, dict[str, Any]]:', 'def _workflows(repo_root: Path) -> dict[str, Any]:')
text = text.replace('triggers = wf.get(\"on\", wf.get(True, {}))', 'triggers = wf.get(\"on\", wf.get(True, {}))  # type: ignore')
text = text.replace('for entry in wf.get(\"on\", wf.get(True, {})).get(\"schedule\", []):', 'for entry in wf.get(\"on\", wf.get(True, {})).get(\"schedule\", []):  # type: ignore')
text = text.replace('model, feats, evaluation, bulletin = (', 'model, feats, evaluation, bulletin = (  # type: ignore')
with open('tests/unit/test_repo_hygiene.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/contract/test_schemas_roundtrip.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('return dict(row)', 'from typing import cast\n        return cast(dict[str, Any], dict(row))')
with open('tests/contract/test_schemas_roundtrip.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_contracts.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('def test_quantile_grid_validation():', 'def test_quantile_grid_validation() -> None:')
with open('tests/unit/test_contracts.py', 'w', encoding='utf-8') as f: f.write(text)

with open('src/smogsense/logging.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('def redact_secrets(logger: object, log_method: str, event_dict: dict) -> dict:', 'def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:')
text = text.replace('def _redact(obj: object) -> object:', 'def _redact(obj: Any) -> Any:')
with open('src/smogsense/logging.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_config.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('def test_settings_unknown_keys_forbid(tmp_path: object) -> None:', 'from pathlib import Path\ndef test_settings_unknown_keys_forbid(tmp_path: Path) -> None:')
text = text.replace('def test_settings_hash(tmp_path: object) -> None:', 'def test_settings_hash(tmp_path: Path) -> None:')
with open('tests/unit/test_config.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_io.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('pa.DataFrameSchema(', 'pa.DataFrameSchema(  # type: ignore[no-untyped-call]')
with open('tests/unit/test_io.py', 'w', encoding='utf-8') as f: f.write(text)
