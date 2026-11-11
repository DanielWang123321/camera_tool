"""Packaged read-only resources and user-selected writable data paths."""
from __future__ import annotations

import os
from pathlib import Path

RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CAMERA_TOOL_HOME", Path.cwd())).expanduser().resolve()


def config_path(name: str | Path) -> Path:
    value = Path(name).expanduser()
    if value.is_absolute():
        return value.resolve()
    for base in (DATA_ROOT, REPO_ROOT):
        for candidate in (base / value, base / "configs" / value):
            if candidate.is_file():
                return candidate.resolve()
    for candidate in (RESOURCE_ROOT / "configs" / value, RESOURCE_ROOT / value):
        if candidate.is_file():
            return candidate.resolve()
    return (RESOURCE_ROOT / "configs" / value).resolve()


def resource_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    for candidate in (DATA_ROOT / value, REPO_ROOT / value, RESOURCE_ROOT / value):
        if candidate.exists():
            return candidate.resolve()
    return (DATA_ROOT / value).resolve()


def output_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else DATA_ROOT / value).resolve()


project_path = output_path
