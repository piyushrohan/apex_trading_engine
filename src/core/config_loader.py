import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
