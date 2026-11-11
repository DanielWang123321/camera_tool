"""Load precomputed wrist camera undistortion maps without requiring vendor SDK at runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from camera_tool.core import CalibrationError
from camera_tool.paths import resource_path


def load_wrist_intrinsics(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    resolved = resource_path(path)
    if not resolved.is_file():
        raise CalibrationError(f"找不到腕部相机内参文件：{resolved}（请先运行 wrist-model 生成本地内参）")
    metadata = json.loads(resolved.read_text(encoding="utf-8"))
    mapping_name = metadata.get("mapping_file")
    if not mapping_name:
        raise CalibrationError(f"{resolved} 缺少 mapping_file")
    mapping_path = (resolved.parent / mapping_name).resolve()
    if not mapping_path.is_relative_to(resolved.parent.resolve()):
        raise CalibrationError("去畸变映射路径超出内参目录")
    if not mapping_path.is_file():
        raise CalibrationError(f"找不到腕部去畸变映射：{mapping_path}")
    with np.load(mapping_path, allow_pickle=False) as data:
        if "mapping" not in data:
            raise CalibrationError("去畸变文件缺少 mapping")
        mapping = np.asarray(data["mapping"], dtype=np.float32)
    width, height = int(metadata["width"]), int(metadata["height"])
    if mapping.shape != (height, width, 2) or not np.isfinite(mapping).all():
        raise CalibrationError("腕部去畸变映射尺寸或数值无效")
    if metadata.get("model") not in ("pinhole", "brown_conrady"):
        raise CalibrationError("腕部内参必须是去畸变后的针孔模型")
    metadata = dict(metadata)
    metadata["intrinsics_json"] = str(resolved)
    metadata["mapping_path"] = str(mapping_path)
    return metadata, mapping
