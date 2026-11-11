"""Generic camera backend registry and isolated SDK device discovery."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import Any

BACKENDS: dict[str, tuple[str, str | None]] = {
    "realsense": ("RealSense RGB-D", "pyrealsense2"),
    "orbbec": ("Orbbec RGB-D", "pyorbbecsdk"),
    "uvc": ("UVC Camera", None),
    "sim": ("Simulated RGB-D", None),
}
KINDS = BACKENDS

PROFILES: dict[str, list[tuple[int, int, int]]] = {
    "realsense": [(1280, 720, 30), (1920, 1080, 30), (848, 480, 30), (640, 480, 60), (640, 480, 30)],
    "orbbec": [(1280, 800, 30), (640, 400, 30)],
    "uvc": [(1280, 1280, 100), (1280, 720, 30), (640, 480, 30)],
    "sim": [(1280, 720, 30), (1280, 800, 30), (640, 480, 30)],
}


def availability() -> dict[str, bool]:
    return {
        backend: not sdk or importlib.util.find_spec(sdk) is not None
        for backend, (_, sdk) in BACKENDS.items()
    }


def create_camera(backend: str, config: dict[str, Any]) -> Any:
    if backend not in BACKENDS:
        raise ValueError(f"未知设备后端 {backend}")
    if not availability().get(backend):
        raise RuntimeError(f"缺少 {BACKENDS[backend][1]}，该相机后端不可用")
    if backend == "realsense":
        from camera_tool.devices.realsense import RealSenseCamera

        return RealSenseCamera(config)
    if backend == "orbbec":
        from camera_tool.devices.orbbec import OrbbecCamera

        return OrbbecCamera(config)
    if backend == "uvc":
        from camera_tool.devices.uvc import UvcCamera

        return UvcCamera(config)
    if backend == "sim":
        from camera_tool.devices.sim import SimulatedRgbdCamera

        return SimulatedRgbdCamera(config)
    raise ValueError(f"未知设备后端 {backend}")


def device_resource(backend: str, serial: str) -> str:
    from camera_tool.services.tasks import resource

    return resource(backend, serial)


def discover() -> dict[str, Any]:
    result: list[dict[str, Any]] = []
    errors: list[str] = []
    available = availability()
    for vendor in ("realsense", "orbbec"):
        if available.get(vendor):
            try:
                process = subprocess.run(
                    [sys.executable, "-m", "camera_tool.devices.probe", vendor],
                    capture_output=True,
                    text=True,
                    timeout=8,
                )
                if process.returncode:
                    detail = (process.stderr or process.stdout).strip().splitlines()
                    raise RuntimeError(detail[-1] if detail else f"SDK 退出 {process.returncode}")
                result.extend(json.loads(process.stdout.strip().splitlines()[-1]))
            except Exception as exc:
                errors.append(f"{vendor}: {exc}")
    from camera_tool.devices.uvc import list_uvc_devices

    result.extend({"kind": "uvc", "name": "UVC Camera", "serial": serial} for serial, _, _ in list_uvc_devices())
    return {"devices": result, "errors": errors, "availability": available}
