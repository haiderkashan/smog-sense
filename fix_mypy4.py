with open('src/smogsense/logging.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:\n    # type: ignore[no-any-return]', 'def redact_secrets(logger: Any, log_method: str, event_dict: dict[str, Any]) -> dict[str, Any]:')
text = text.replace('return _redact(event_dict)', 'return cast(dict[str, Any], _redact(event_dict))')
with open('src/smogsense/logging.py', 'w', encoding='utf-8') as f: f.write(text)

with open('tests/unit/test_repo_hygiene.py', 'r', encoding='utf-8') as f: text = f.read()
text = text.replace('assert "pull_request_target" not in triggers, (', 'assert triggers is not None and "pull_request_target" not in triggers, (')
with open('tests/unit/test_repo_hygiene.py', 'w', encoding='utf-8') as f: f.write(text)
