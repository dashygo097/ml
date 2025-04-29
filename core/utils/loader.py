import json
import os
import sys
from typing import Dict

import yaml


def load_json(file_path: str) -> Dict:
    script_dir = sys.path[0]
    file_path = os.path.join(script_dir, file_path)
    with open(file_path, "r") as f:
        file = json.load(f)

    return file


def load_yaml(file_path: str) -> Dict:
    script_dir = sys.path[0]
    file_path = os.path.join(script_dir, file_path)
    with open(file_path, "r") as f:
        file = yaml.safe_load(f)

    return file
