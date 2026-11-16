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
            raise RuntimeError(f"No matching UVC camera node found in {BY_ID_DIR}.")
        if len(devices) > 1:
            found = ", ".join(sn for sn, _, _ in devices)
            raise RuntimeError(f"Multiple UVC cameras detected ({found}); specify a serial number.")
        serial, path, _ = devices[0]
    else:
        path = BY_ID_DIR / f"{prefix}{serial}{suffix}"
        if not path.exists():
            extra = (
                f" Connected: {', '.join(sn for sn, _, _ in devices)}"
                if devices
                else f" No matching by-id nodes found in {BY_ID_DIR}."
            )
            raise RuntimeError(f"UVC camera with serial {serial} not found: {path} does not exist.{extra}")

    cap = cv2.VideoCapture(str(path), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open {path} (resolved to {path.resolve()}). "
            "Ensure the camera is not in use by another process and remains bound to uvcvideo."
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
        raise RuntimeError(f"Failed to open stream at {width}x{height}; device reported {actual_w}x{actual_h}.")

    frame = None
    for _ in range(max(warmup_frames, 1)):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Warmup stream failed: {path} produced no valid frame within {warmup_frames} reads.")
    if frame.ndim != 3 or frame.shape[2] != 3:
        cap.release()
        raise RuntimeError(f"Unexpected warmup frame format: shape={getattr(frame, 'shape', None)}, expected 3-channel BGR.")

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
            raise RuntimeError("UVC camera is not started")
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError("UVC camera stream interrupted")
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


class UvcCamera(ProcessCamera):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("camera_tool.devices.uvc:UvcSession", config)
