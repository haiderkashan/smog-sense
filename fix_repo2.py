with open('tests/unit/test_repo_hygiene.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('wf.get("on", wf.get(True, {}))  # type: ignore.get("schedule", []):', 'wf.get("on", wf.get(True, {})).get("schedule", []):  # type: ignore')

with open('tests/unit/test_repo_hygiene.py', 'w', encoding='utf-8') as f:
    f.write(text)
