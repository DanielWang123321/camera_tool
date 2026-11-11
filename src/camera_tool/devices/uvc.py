"""Generic UVC / V4L2 color camera driver isolated in a subprocess via ProcessCamera."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from camera_tool.devices.process_camera import ProcessCamera

BY_ID_DIR = Path("/dev/v4l/by-id")
DEFAULT_BY_ID_PREFIX = "usb-XVisio_Technology_XVisio_vSLAM_"
DEFAULT_BY_ID_SUFFIX = "-video-index0"


def fourcc_to_str(value: int) -> str:
    if value <= 0:
        return "?"
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def list_uvc_devices(
    prefix: str = DEFAULT_BY_ID_PREFIX,
    suffix: str = DEFAULT_BY_ID_SUFFIX,
) -> list[tuple[str, Path, Path]]:
    """Return `(serial, by_id_path, resolved_video_node)` for matching V4L2 by-id links."""
    found: list[tuple[str, Path, Path]] = []
    if not BY_ID_DIR.is_dir():
        return found
    for path in sorted(BY_ID_DIR.iterdir()):
        name = path.name
        if prefix and not name.startswith(prefix):
            continue
        if suffix and not name.endswith(suffix):
            continue
        end = -len(suffix) if suffix else None
        serial = name[len(prefix) : end]
        if serial:
            found.append((serial, path, path.resolve()))
    return found


def open_uvc_capture(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import cv2

    serial = str(config.get("serial", "")).strip()
    prefix = str(config.get("by_id_prefix", DEFAULT_BY_ID_PREFIX))
    suffix = str(config.get("by_id_suffix", DEFAULT_BY_ID_SUFFIX))
    width = int(config.get("width", 1280))
    height = int(config.get("height", 1280))
    fps = int(config.get("fps", 100))
    fourcc = str(config.get("fourcc", "YU12"))
    warmup_frames = int(config.get("warmup_frames", 8))

    devices = list_uvc_devices(prefix, suffix)
    if not serial:
        if not devices:
            raise RuntimeError(f"未检测到匹配的 UVC 相机节点（{BY_ID_DIR}）。")
        if len(devices) > 1:
            found = ", ".join(sn for sn, _, _ in devices)
            raise RuntimeError(f"检测到多个 UVC 相机（{found}），请指定序列号。")
        serial, path, _ = devices[0]
    else:
        path = BY_ID_DIR / f"{prefix}{serial}{suffix}"
        if not path.exists():
            extra = (
                f" 当前已连接：{', '.join(sn for sn, _, _ in devices)}"
                if devices
                else f" 未检测到任何匹配的 by-id 节点（{BY_ID_DIR}）。"
            )
            raise RuntimeError(f"未找到序列号 {serial} 的 UVC 相机：{path} 不存在。{extra}")

    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开 {path}（解析为 {path.resolve()}）。"
            "请确认相机未被其他程序占用，且节点仍绑定 uvcvideo。"
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if actual_w != width or actual_h != height:
        cap.release()
        raise RuntimeError(f"无法以 {width}x{height} 开流，设备回报 {actual_w}x{actual_h}。")

    frame = None
    for _ in range(max(warmup_frames, 1)):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"预热取流失败：{path} 在 {warmup_frames} 帧内没有有效画面。")
    if frame.ndim != 3 or frame.shape[2] != 3:
        cap.release()
        raise RuntimeError(f"预热帧格式异常：shape={getattr(frame, 'shape', None)}，期望 BGR 三通道。")

    metadata = {
        "serial": serial,
        "name": str(config.get("name", "UVC Camera")),
        "width": actual_w,
        "height": actual_h,
        "fps": actual_fps,
        "actual_stream": [actual_w, actual_h, actual_fps],
        "fourcc": fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC))),
        "sdk_depth_unit_m": None,
        "depth_scale_correction": 1.0,
        "K": None,
    }
    return cap, metadata


class UvcSession:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.cap: Any = None
        self.metadata: dict[str, Any] = {}
        self.sequence = 0

    def start(self) -> dict[str, Any]:
        self.cap, self.metadata = open_uvc_capture(self.config)
        self.sequence = 0
        return self.metadata

    def read(self) -> dict[str, Any]:
        if self.cap is None:
            raise RuntimeError("UVC 相机未启动")
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError("UVC 相机断流")
        self.sequence += 1
        return {
            "color_bgr": image,
            "timestamp_mono": time.monotonic(),
            "frame_number": self.sequence,
            "depth_m": None,
            "metadata": dict(self.metadata),
        }

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    close = stop


class UvcCamera(ProcessCamera):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("camera_tool.devices.uvc:UvcSession", config)
