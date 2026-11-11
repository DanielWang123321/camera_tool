"""Exceptions, atomic JSON/YAML serialization, and calibration session logging."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from camera_tool.paths import output_path, resource_path


class CalibrationError(RuntimeError):
    """Fatal calibration, configuration, or hardware state error."""


class RejectedSample(CalibrationError):
    """Single frame or candidate pose rejected by quality or safety gates."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = resource_path(path)
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CalibrationError(f"配置文件格式无效: {path}")
    return data


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Any) -> Path:
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_hash(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    from camera_tool.core.geometry import valid_transform

    return [[float(v) for v in row] for row in valid_transform(matrix).tolist()]


class Session:
    """Structured event-logged session directory for multi-stage calibration runs."""

    def __init__(
        self,
        parent: str | Path,
        mode: str,
        on_event: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        base = output_path(parent)
        self.path = base / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.path.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event
        self.manifest: dict[str, Any] = {
            "schema": 1,
            "mode": mode,
            "status": "running",
            "samples": [],
            "fixed_samples": [],
            "started_at": datetime.now().astimezone().isoformat(),
        }
        self.save()

    def save(self) -> None:
        write_json(self.path / "session.json", self.manifest)

    def event(self, event: str, **fields: Any) -> None:
        row = {"time": datetime.now().astimezone().isoformat(), "event": event, **fields}
        with (self.path / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
        print(event + (f"：{fields['reason']}" if "reason" in fields else ""), flush=True)
        if self.on_event:
            self.on_event(row, self.manifest)

    def image(self, name: str, array: np.ndarray) -> str:
        target = output_path(self.path / name)
        if not cv2.imwrite(str(target), array):
            raise CalibrationError(f"图像保存失败：{target}")
        return name

    def finish(self, status: str, **fields: Any) -> None:
        self.manifest.update(status=status, finished_at=datetime.now().astimezone().isoformat(), **fields)
        self.save()
