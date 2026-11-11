"""Calibration packet adapters wrapping generic UVC and RGB-D camera drivers."""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from camera_tool.calibration.eye_to_hand.wrist_model import load_wrist_intrinsics
from camera_tool.core import CalibrationError, write_json
from camera_tool.devices.uvc import DEFAULT_BY_ID_PREFIX, DEFAULT_BY_ID_SUFFIX, UvcCamera, list_uvc_devices


class WristCamera:
    def __init__(self, config: dict[str, Any], session: Any) -> None:
        self.config = config
        self.session = session
        self.serial = str(config.get("serial") or "").strip()
        self.cap: UvcCamera | None = None
        self.mapping: np.ndarray | None = None
        self.metadata: dict[str, Any] | None = None
        self.raw_shape: tuple[int, int] | None = None
        self.last_seq = 0

    def start(self) -> dict[str, Any]:
        if not self.serial:
            prefix = str(self.config.get("by_id_prefix", DEFAULT_BY_ID_PREFIX))
            suffix = str(self.config.get("by_id_suffix", DEFAULT_BY_ID_SUFFIX))
            devices = list_uvc_devices(prefix, suffix)
            if not devices:
                raise CalibrationError("未检测到腕部 UVC 相机设备，请检查连接或指定序列号")
            if len(devices) > 1:
                found = ", ".join(sn for sn, _, _ in devices)
                raise CalibrationError(f"检测到多个腕部 UVC 相机（{found}），请指定序列号")
            self.serial = devices[0][0]

        intrinsics_path = str(self.config.get("intrinsics") or "").strip() or f"intrinsics/wrist_{self.serial}.json"
        self.metadata, self.mapping = load_wrist_intrinsics(intrinsics_path)
        if self.metadata.get("serial") != self.serial:
            raise CalibrationError(
                f"腕部内参序列号 {self.metadata.get('serial')} 与当前相机 {self.serial} 不一致"
            )
        raw_w, raw_h = self.config["capture_size"]
        rect_w, rect_h = self.config["rectified_size"]
        if (
            self.metadata["width"] != rect_w
            or self.metadata["height"] != rect_h
            or self.metadata.get("raw_width", raw_w) != raw_w
            or self.metadata.get("raw_height", raw_h) != raw_h
        ):
            raise CalibrationError("腕部内参分辨率与配置不一致")
        np.savez_compressed(self.session.path / "wrist_rectification.npz", mapping=self.mapping)
        write_json(self.session.path / "wrist_camera.json", self.metadata)
        self.cap = UvcCamera({**self.config, "serial": self.serial, "width": raw_w, "height": raw_h})
        info = self.cap.start()
        self.raw_shape = (raw_h, raw_w)
        self.metadata.update(
            backend="v4l2",
            frame_clock="host_monotonic_after_read",
            image_size_verified=True,
            actual_stream=info["actual_stream"],
            sdk_depth_unit_m=None,
            depth_scale_correction=1.0,
        )
        write_json(self.session.path / "wrist_camera.json", self.metadata)
        return self.metadata

    def frame(self, after_mono: float = 0.0, timeout: float = 3.0, rectify: bool = True) -> dict[str, Any]:
        if self.cap is None or self.mapping is None:
            raise CalibrationError("腕部相机未打开")
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            frame = self.cap.read()
            raw, host = frame["color_bgr"], frame["timestamp_mono"]
            if raw.shape[:2] != self.raw_shape:
                raise CalibrationError("腕部相机运行中分辨率改变，停止标定")
            self.last_seq += 1
            last = {"sequence": self.last_seq, "host_timestamp": host}
            if host <= after_mono:
                time.sleep(0.005)
                continue
            color = raw if not rectify else cv2.remap(raw, self.mapping, None, cv2.INTER_LINEAR)
            return {
                "raw": raw,
                "color": color,
                "host_timestamp": host,
                "source_timestamp_us": int(host * 1e6),
                "frame_number": self.last_seq,
            }
        reason = "未收到任何图像" if last is None else "没有停稳后的新图像"
        raise CalibrationError(f"腕部相机断流：{reason}")

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.stop()
            self.cap = None


class BufferedRgbdCamera:
    """Background-buffered fixed RGB-D camera supporting RealSense and Orbbec backends."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = str(config.get("backend", "realsense")).strip().lower()
        self.width = int(config["width"])
        self.height = int(config["height"])
        self.fps = int(config["fps"])
        self.serial = str(config.get("serial") or "").strip()
        self.depth_scale = 0.001
        self.full_metadata: dict[str, Any] = {}
        self.calibration_metadata: dict[str, Any] = {}
        self.last_error: str | None = None
        self.driver: Any = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] | None = None
        self._latest_mono = 0.0

    def start(self) -> dict[str, Any]:
        self.last_error = None
        if self.backend == "orbbec":
            from camera_tool.devices.orbbec import OrbbecCamera

            self.driver = OrbbecCamera(self.config)
        else:
            from camera_tool.devices.realsense import RealSenseCamera

            self.driver = RealSenseCamera(self.config)
        try:
            metadata = self.driver.start()
            self.full_metadata = dict(metadata)
            self.calibration_metadata = metadata.get("native") or dict(metadata)
            self.depth_scale = float(metadata.get("sdk_depth_unit_m") or metadata.get("depth_scale") or 0.001)
            self.serial = str(metadata.get("serial") or self.serial)
            self._stop.clear()
            self._thread = threading.Thread(target=self._grab_loop, name="fixed-rgbd-calibration", daemon=True)
            self._thread.start()
            return metadata
        except BaseException:
            self.driver.stop()
            raise

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(3)
        self._thread = None
        if self.driver is not None:
            self.driver.stop()
            self.driver = None
        with self._lock:
            self._latest = None

    def _grab_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.driver.read()
                packet = {
                    **frame,
                    "color": frame["color_bgr"],
                    "depth": frame.get("depth_raw"),
                    "native_depth": frame.get("native_depth") if frame.get("native_depth") is not None else frame.get("depth_raw"),
                    "received_mono": frame["timestamp_mono"],
                }
                with self._lock:
                    self._latest, self._latest_mono = packet, time.monotonic()
            except Exception as exc:
                self.last_error = str(exc)
                break

    def get_frame(
        self,
        max_age_s: float = 0.3,
        wait_s: float = 0.0,
        after_frame_number: int | None = None,
        after_mono: float = 0.0,
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + wait_s
        while True:
            if self.last_error:
                raise CalibrationError(f"固定相机取流失败：{self.last_error}")
            with self._lock:
                packet = self._latest
                age = time.monotonic() - self._latest_mono
            if (
                packet is not None
                and age <= max_age_s
                and packet["received_mono"] > after_mono
                and (after_frame_number is None or packet["frame_number"] > after_frame_number)
            ):
                return packet
            if time.monotonic() >= deadline or self._stop.wait(0.01):
                return None

    def read(self) -> dict[str, Any]:
        packet = self.get_frame(wait_s=2.0)
        if packet is None:
            raise CalibrationError("固定相机取流超时")
        return packet


RealSenseD435 = BufferedRgbdCamera
