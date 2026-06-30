from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries without mutating inputs."""

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with optional relative `extends` support."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    extends = data.pop("extends", None)
    if not extends:
        return data

    parent_paths = extends if isinstance(extends, list) else [extends]
    merged: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        merged = deep_merge(merged, load_yaml_config(parent_path))
    return deep_merge(merged, data)

