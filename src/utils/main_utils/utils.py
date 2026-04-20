"""YAML helpers used by the rule engine for optional config loading."""
from __future__ import annotations

import os
import sys

import yaml

from src.exception.exception import BPException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001
        raise BPException(exc, sys) from exc


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as exc:  # noqa: BLE001
        raise BPException(exc, sys)
