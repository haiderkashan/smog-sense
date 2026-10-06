with open('tests/unit/test_repo_hygiene.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re

text = text.replace('def _workflows(repo_root: Path) -> dict[str, dict[str, Any]]:', 'def _workflows(repo_root: Path) -> dict[str, Any]:')

old_load = 'load = lambda n: yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8"))  # noqa: E731'
new_load = 'def load(n: str) -> dict[str, Any]:\n        from typing import cast, Any\n        return cast(dict[str, Any], yaml.safe_load((repo_root / "configs" / n).read_text(encoding="utf-8")))'
text = text.replace(old_load, new_load)

text = text.replace('wf.get("on", wf.get(True, {}))', 'wf.get("on", wf.get(True, {}))  # type: ignore')
text = text.replace('wf.get("on", wf.get(True, {})).get("schedule", [])', 'wf.get("on", wf.get(True, {})).get("schedule", [])  # type: ignore')

with open('tests/unit/test_repo_hygiene.py', 'w', encoding='utf-8') as f:
    f.write(text)
