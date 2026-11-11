"""USB link speed and descriptor inspection via Linux sysfs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

USB3_MIN_SPEED_MBPS = 4000  # USB 3.x >= 5000 Mbps, USB 2.0 = 480 Mbps
VENDOR_IDS_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "realsense": ("8086",),
    "orbbec": ("2bc5",),
}


class UsbSpeed:
    """USB link speed and serial read from Linux sysfs."""

    __slots__ = ("path", "speed_mbps", "serial")

    def __init__(self, path: str, speed_mbps: int, serial: str = "") -> None:
        self.path = path
        self.speed_mbps = speed_mbps
        self.serial = serial

    def is_usb3(self) -> bool:
        return self.speed_mbps >= USB3_MIN_SPEED_MBPS


def normalize_usb_type(text: str | None) -> str:
    """Normalize USB descriptor text into `3.x`, `2.x`, or `unknown`."""
    if not text:
        return "unknown"
    t = text.strip().lower()
    if "3.2" in t or "3.1" in t or "3.0" in t or "3" in t:
        return "3.x"
    if "2.0" in t or "2" in t:
        return "2.x"
    return text.strip() or "unknown"


def _extract_usb_port_name(physical_port: str | None) -> str | None:
    if not physical_port:
        return None
    raw = physical_port.strip()
    if not raw:
        return None
    if "/" not in raw:
        return raw
    p = Path(raw)
    for node in (p, *p.parents):
        if (node / "idVendor").is_file() and (node / "speed").is_file():
            return node.name
    for part in reversed(p.parts):
        base = part.split(":")[0]
        if "-" in base and base[0].isdigit():
            return base
    return None


def find_usb_sysfs(
    serial: str | None = None,
    physical_port: str | None = None,
    asic_serial: str | None = None,
    sysfs_root: Path | None = None,
    vendor_ids: tuple[str, ...] = ("8086", "2bc5"),
) -> UsbSpeed | None:
    """Locate a matching USB camera in Linux sysfs and return its actual negotiated link speed."""
    root = sysfs_root if sysfs_root is not None else Path("/sys/bus/usb/devices")
    if not root.is_dir():
        return None

    allowed_vids = {v.lower() for v in vendor_ids}
    candidates: list[UsbSpeed] = []
    for dev_dir in sorted(root.iterdir()):
        vid_file = dev_dir / "idVendor"
        pid_file = dev_dir / "idProduct"
        if not vid_file.exists() or not pid_file.exists():
            continue
        try:
            vid = vid_file.read_text().strip().lower()
        except OSError:
            continue
        if allowed_vids and vid not in allowed_vids:
            continue

        speed_file = dev_dir / "speed"
        try:
            speed_mbps = int(float(speed_file.read_text().strip())) if speed_file.exists() else 0
        except (OSError, ValueError):
            speed_mbps = 0

        serial_file = dev_dir / "serial"
        try:
            dev_serial = serial_file.read_text().strip() if serial_file.exists() else ""
        except OSError:
            dev_serial = ""

        candidates.append(UsbSpeed(dev_dir.name, speed_mbps, dev_serial))

    if not candidates:
        return None

    port_name = _extract_usb_port_name(physical_port)
    if port_name:
        for c in candidates:
            if c.path == port_name or c.path.startswith(port_name + "."):
                return c

    valid_serials = {s.strip() for s in (serial, asic_serial) if s and s.strip()}
    if valid_serials:
        for c in candidates:
            if c.serial in valid_serials:
                return c
        return None

    return candidates[0]


def find_realsense_sysfs(
    serial: str | None = None,
    physical_port: str | None = None,
    asic_serial: str | None = None,
    sysfs_root: Path | None = None,
) -> UsbSpeed | None:
    """Locate a RealSense USB device in Linux sysfs (matches any Intel 8086 camera PID)."""
    return find_usb_sysfs(
        serial=serial,
        physical_port=physical_port,
        asic_serial=asic_serial,
        sysfs_root=sysfs_root,
        vendor_ids=("8086",),
    )


def read_current_usb_info(
    device: Any, serial: str, physical_port: str | None = None
) -> tuple[str, UsbSpeed | None]:
    import pyrealsense2 as rs

    try:
        descriptor = device.get_info(rs.camera_info.usb_type_descriptor)
    except Exception:
        descriptor = "unknown"
    port = physical_port
    if not port:
        try:
            if device.supports(rs.camera_info.physical_port):
                port = device.get_info(rs.camera_info.physical_port)
        except Exception:
            port = None
    asic_serial = None
    try:
        if device.supports(rs.camera_info.asic_serial_number):
            asic_serial = device.get_info(rs.camera_info.asic_serial_number)
    except Exception:
        asic_serial = None
    speed = find_realsense_sysfs(serial, port, asic_serial=asic_serial)
    return descriptor, speed
