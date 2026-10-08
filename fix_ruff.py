import re
from pathlib import Path

files_to_fix = [
    "src/smogsense/cli.py",
    "src/smogsense/pipeline/orchestrator.py",
    "src/smogsense/publishing/bulletin.py",
    "src/smogsense/publishing/site.py",
]

for file_path in files_to_fix:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace('with open(manifest_file) as f:', 'with manifest_file.open("r", encoding="utf-8") as f:')
    content = content.replace('with open(path, "w") as f:', 'with Path(path).open("w", encoding="utf-8") as f:')
    content = content.replace('with open(f"gh-pages/{run_id}.json", "w") as f:', 'with Path(f"gh-pages/{run_id}.json").open("w", encoding="utf-8") as f:')
    content = content.replace('with open(schema_path) as f:', 'with schema_path.open("r", encoding="utf-8") as f:')
    content = content.replace('with open(json_path, "w", encoding="utf-8") as f:', 'with json_path.open("w", encoding="utf-8") as f:')
    content = content.replace('with open(html_path, "w", encoding="utf-8") as f:', 'with html_path.open("w", encoding="utf-8") as f:')

    content = content.replace('def get_prob(thresh: float) -> float:', 'def get_prob(thresh: float, q_func=qf) -> float:')
    content = content.replace('qf.prob_exceed(thresh)', 'q_func.prob_exceed(thresh)')

    content = content.replace('''                for s in stations_data:
                    if h in s["forecasts"]:
                        q_list.append(s["forecasts"][h])''', '''                q_list = [s["forecasts"][h] for s in stations_data if h in s["forecasts"]]''')

    content = content.replace('''        except Exception:
            pass''', '''        except Exception:
            logger.debug("Ignored exception")''')

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
