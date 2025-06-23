import json
import os
import sys
from typing import Dict
import pandas as pd

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


def load_csv(file_path: str) -> pd.DataFrame:
    script_dir = sys.path[0]
    file_path = os.path.join(script_dir, file_path)
    df = pd.read_csv(file_path)
    return df


def load_excel(file_path: str) -> pd.DataFrame:
    script_dir = sys.path[0]
    file_path = os.path.join(script_dir, file_path)
    df = pd.read_excel(file_path)
    return df


def load_txt(file_path: str) -> str:
    script_dir = sys.path[0]
    file_path = os.path.join(script_dir, file_path)
    with open(file_path, "r") as f:
        content = f.read()
    return content
