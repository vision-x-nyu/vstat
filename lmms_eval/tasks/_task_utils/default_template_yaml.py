import os
from pathlib import Path

import yaml

# QA JSON resolution order:
#   1. $VSTAT_QA_PATH (absolute file path) — overrides every split.
#   2. YAML-given absolute path — passes through.
#   3. YAML-given relative path — resolved against <repo_root>/data/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _REPO_ROOT / "data"


def load_default_template_yaml(task_file):
    with open(Path(task_file).parent / "_default_template_yaml", "r") as f:
        safe_data = [line for line in f if "!function" not in line]
    config = yaml.safe_load("".join(safe_data))

    qa_override = os.environ.get("VSTAT_QA_PATH")
    data_files = config.get("dataset_kwargs", {}).get("data_files")
    if isinstance(data_files, dict):
        for split, rel in data_files.items():
            if qa_override:
                data_files[split] = qa_override
            elif isinstance(rel, str) and not Path(rel).is_absolute():
                data_files[split] = str(_DATA_ROOT / rel)

    return config
