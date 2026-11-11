"""Internal subprocess entry for SDK enumeration; SDK native exits stay isolated."""
from __future__ import annotations

import json
import sys
from typing import Any


def probe(vendor: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if vendor == "realsense":
        import pyrealsense2 as rs

        for dev in rs.context().query_devices():
            result.append(
                {
                    "kind": "realsense",
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                }
            )
    elif vendor == "orbbec":
        from camera_tool.devices.orbbec import create_context

        context = create_context()
        devices = context.query_devices()
        for index in range(devices.get_count()):
            info = devices.get_device_by_index(index).get_device_info()
            result.append(
                {
                    "kind": "orbbec",
                    "name": info.get_name(),
                    "serial": info.get_serial_number(),
                }
            )
    return result


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1]), ensure_ascii=False))
