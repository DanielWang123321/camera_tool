"""USB link stability measurements consuming every SDK frame independent of UI refresh."""
from __future__ import annotations

import csv
from datetime import datetime
import time

from camera_tool.devices.registry import create_camera, device_resource
from camera_tool.paths import output_path
from camera_tool.services.tasks import DEVICES


class UsbStatistics:
    def __init__(self, min_fps=29.0, max_gap=0.1):
        self.min_fps, self.gap_threshold = min_fps, max_gap
        self.first = self.last = self.previous_number = None
        self.window_first = self.window_last = None
        self.count = self.window_count = self.missing = self.big_gaps = self.stalls = self.resets = 0
        self.max_gap = self.window_max_gap = 0.0
        self.low_windows = 0
        self.min_usb_speed = None
        self.usb_bad = False

    def record(self, timestamp, number):
        if self.last is not None:
            gap = timestamp - self.last
            self.max_gap = max(self.max_gap, gap)
            self.window_max_gap = max(self.window_max_gap, gap)
            self.big_gaps += gap > self.gap_threshold
            if number > self.previous_number:
                self.missing += number - self.previous_number - 1
            else:
                self.resets += 1
        else:
            self.first = timestamp
        if self.window_first is None:
            self.window_first = timestamp
        self.last = self.window_last = timestamp
        self.previous_number = number
        self.count += 1
        self.window_count += 1

    def link(self, descriptor, speed):
        if speed is not None:
            self.min_usb_speed = speed if self.min_usb_speed is None else min(self.min_usb_speed, speed)
            self.usb_bad |= speed < 4000
        self.usb_bad |= str(descriptor).startswith("2")

    def snapshot(self, now, descriptor="unknown", speed=None, final=False):
        self.link(descriptor, speed)
        count = self.count if final else self.window_count
        first = self.first if final else self.window_first
        last = self.last if final else self.window_last
        span = (last - first) if first is not None and last is not None else 0
        trailing = max(0.0, now - last) if last is not None else 0.0
        if trailing > self.gap_threshold:
            span += trailing
        fps = (count - 1) / span if count >= 2 and span > 0 else 0.0
        gap = max(self.max_gap if final else self.window_max_gap, trailing)
        low = fps < self.min_fps
        return {
            "window_start": datetime.now().astimezone().isoformat(timespec="seconds"),
            "fps": fps,
            "frame_count": count,
            "usb_descriptor": descriptor,
            "usb_speed_mbps": speed,
            "max_gap_s": gap,
            "missing_frame_numbers": self.missing,
            "low_fps": low,
            "stalls": self.stalls,
            "counter_resets": self.resets,
            "elapsed_s": now - self.first if self.first is not None else 0.0,
            "min_usb_speed_mbps": self.min_usb_speed,
            "stable": bool(
                count >= 2
                and not low
                and not self.low_windows
                and not self.missing
                and not self.big_gaps
                and not self.stalls
                and not self.resets
                and not self.usb_bad
                and gap <= self.gap_threshold
            ),
        }

    def reset_window(self, low=False):
        self.low_windows += bool(low)
        self.window_count = 0
        self.window_first = self.window_last = None
        self.window_max_gap = 0.0


def export_csv(rows, path):
    if not rows:
        raise ValueError("没有可导出的 USB 测试数据")
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def usb_test(
    context,
    config,
    duration=3600,
    interval=5,
    min_fps=29.0,
    max_gap=0.1,
    csv_path=None,
    camera_factory=None,
    link_reader=None,
):
    if duration <= 0 or interval <= 0 or max_gap <= 0 or min_fps <= 0:
        raise ValueError("测试时长、统计间隔、帧率与间隔阈值必须大于 0")
    backend = str(config.get("backend") or config.get("kind") or "realsense").strip().lower()
    if backend not in ("realsense", "orbbec"):
        raise ValueError("USB 稳定性测试仅支持 realsense 或 orbbec 后端")
    stats, rows = UsbStatistics(min_fps, max_gap), []
    camera = (camera_factory or create_camera)(backend, {**config, "enable_depth": False})
    summary = None
    with DEVICES.lease([device_resource(backend, config.get("serial", ""))], "USB 稳定性测试"):
        try:
            context.check()
            metadata = camera.start()
            context.emit("metadata", metadata)

            def read_link():
                if link_reader:
                    return link_reader()
                from camera_tool.diagnostics.usb_link import find_usb_sysfs, normalize_usb_type, read_current_usb_info

                if backend == "realsense":
                    device = camera.pipeline.get_active_profile().get_device()
                    descriptor, link = read_current_usb_info(device, metadata["serial"])
                    return normalize_usb_type(descriptor), link.speed_mbps if link else None
                link = find_usb_sysfs(serial=metadata.get("serial"), vendor_ids=("2bc5",))
                if link is None:
                    return "unknown", None
                return ("3.x" if link.is_usb3() else "2.x"), link.speed_mbps

            descriptor, speed = read_link()
            stats.link(descriptor, speed)
            start = time.monotonic()
            next_report = start + interval
            while time.monotonic() - start < duration and not context.cancelled.is_set():
                try:
                    frame = camera.read()
                    stats.record(frame["timestamp_mono"], frame["frame_number"])
                except RuntimeError as exc:
                    stats.stalls += 1
                    context.emit("status", f"无帧：{exc}")
                now = time.monotonic()
                if now >= next_report:
                    descriptor, speed = read_link()
                    row = stats.snapshot(now, descriptor, speed)
                    rows.append(row)
                    context.emit("usb", row)
                    stats.reset_window(row["low_fps"])
                    next_report = now + interval
            descriptor, speed = read_link()
            summary = stats.snapshot(time.monotonic(), descriptor, speed, final=True)
            summary["stopped"] = context.cancelled.is_set()
            if stats.window_count or not rows:
                rows.append(stats.snapshot(time.monotonic(), descriptor, speed))
            context.emit("usb_summary", summary)
            return {"summary": summary, "rows": rows}
        finally:
            camera.stop()
            if csv_path and rows:
                export_csv(rows, csv_path)
