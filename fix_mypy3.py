import re

with open('src/smogsense/logging.py', 'r', encoding='utf-8') as f: text = f.read()
if 'from typing import Any' not in text:
    text = text.replace('import os', 'import os\nfrom typing import Any, cast')
text = text.replace('def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:', 'def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:\n    # type: ignore[no-any-return]')
text = text.replace('processors=[', 'processors=cast(Any, [')
text = text.replace('],', ']),')
with open('src/smogsense/logging.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_repo_hygiene.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('triggers = wf.get(\"on\", wf.get(True, {}))  # type: ignore', 'triggers = wf.get(\"on\") or wf.get(True) or {}')
text = text.replace('load = lambda n: yaml.safe_load((repo_root / \"configs\" / n).read_text(encoding=\"utf-8\"))  # noqa: E731', 'load = lambda n: yaml.safe_load((repo_root / \"configs\" / n).read_text(encoding=\"utf-8\"))  # type: ignore')
with open('tests/unit/test_repo_hygiene.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/contract/test_schemas_roundtrip.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('from typing import cast\n        return cast(dict[str, Any], dict(row))', 'from typing import cast, Any\n        return cast(dict[str, Any], dict(row))')
with open('tests/contract/test_schemas_roundtrip.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_contracts.py', 'r', encoding='utf-8') as f: text = f.read()
text = re.sub(r'def test_quantile_grid_validation\(\):', 'def test_quantile_grid_validation() -> None:', text)
with open('tests/unit/test_contracts.py', 'w', encoding='utf-8') as f: f.write(text)
