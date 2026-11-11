"""Generic Intel RealSense RGB-D camera driver (pyrealsense2)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from camera_tool.core import CalibrationError, CameraModel


class RealSenseCamera:
    """Hardware wrapper for Intel RealSense RGB-D cameras with aligned color + depth streams."""

    def __init__(self, camera_cfg: dict[str, Any]) -> None:
        self.cfg = camera_cfg
        self.width = int(camera_cfg.get("width", 1280))
        self.height = int(camera_cfg.get("height", 720))
        self.fps = int(camera_cfg.get("fps", 30))
        self.serial = str(camera_cfg.get("serial", "")).strip()
        self.pipeline: Any = None
        self.align: Any = None
        self.depth_scale: float = 0.001
        self.color_intrinsics: Any = None
        self.metadata: dict[str, Any] = {}
        self.connected = False
        self._pipeline_started = False
        self._pointcloud_rays: dict[int, np.ndarray] = {}

    def start(self) -> dict[str, Any]:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CalibrationError("未安装 pyrealsense2，无法连接 RealSense 相机") from exc

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise CalibrationError("未检测到 RealSense 相机设备，请检查 USB 3.x 连接")

        model_filter = str(self.cfg.get("device_model", "")).strip().lower()
        dev = next(
            (
                d
                for d in devices
                if (not self.serial or d.get_info(rs.camera_info.serial_number) == self.serial)
                and (not model_filter or model_filter in d.get_info(rs.camera_info.name).lower())
            ),
            None,
        )
        if dev is None:
            raise CalibrationError(f"未检测到指定 RealSense 相机: {self.serial}")
        selected_serial = dev.get_info(rs.camera_info.serial_number)
        device_name = dev.get_info(rs.camera_info.name)

        self.pipeline = rs.pipeline()
        requested = (self.width, self.height, self.fps)
        profiles: dict[Any, set[tuple[int, int, int]]] = {rs.stream.color: set(), rs.stream.depth: set()}
        for sensor in dev.query_sensors():
            for stream in sensor.get_stream_profiles():
                kind = stream.stream_type()
                fmt = rs.format.bgr8 if kind == rs.stream.color else rs.format.z16
                if kind in profiles and stream.format() == fmt and stream.is_video_stream_profile():
                    video = stream.as_video_stream_profile()
                    profiles[kind].add((video.width(), video.height(), stream.fps()))
        candidates = sorted(
            profiles[rs.stream.color] & profiles[rs.stream.depth]
            if self.cfg.get("enable_depth", True)
            else profiles[rs.stream.color],
            key=lambda p: (p != requested, abs(p[2] - requested[2]), abs(p[0] * p[1] - requested[0] * requested[1])),
        )
        config = None
        for width, height, fps in candidates:
            candidate = rs.config()
            candidate.enable_device(selected_serial)
            candidate.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            if self.cfg.get("enable_depth", True):
                candidate.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            if candidate.can_resolve(rs.pipeline_wrapper(self.pipeline)):
                config = candidate
                break
        if config is None:
            raise CalibrationError("相机没有可同时启用的彩色与深度流配置")
        try:
            profile = self.pipeline.start(config)
            self._pipeline_started = True
            return self._initialize_stream(profile, rs, device_name, selected_serial, requested)
        except Exception:
            self.close()
            raise

    def _initialize_stream(
        self,
        profile: Any,
        rs: Any,
        device_name: str,
        serial: str,
        requested: tuple[int, int, int],
    ) -> dict[str, Any]:
        enabled_depth = bool(self.cfg.get("enable_depth", True))
        self.align = rs.align(rs.stream.color) if enabled_depth else None
        if enabled_depth:
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.color_intrinsics = intr
        self.width, self.height, self.fps = int(intr.width), int(intr.height), int(color_stream.fps())

        for _ in range(10):
            self.pipeline.wait_for_frames(2000)

        coeffs = [float(x) for x in intr.coeffs]
        model_str = str(intr.model).split(".")[-1]
        if all(abs(c) < 1e-9 for c in coeffs) or model_str == "none":
            model_name = "pinhole"
        else:
            model_name = model_str
            if model_name not in ("brown_conrady", "inverse_brown_conrady", "modified_brown_conrady"):
                raise CalibrationError(f"不支持的 RealSense 畸变模型: {model_name}")

        self.metadata = {
            "name": device_name,
            "serial": serial,
            "width": int(intr.width),
            "height": int(intr.height),
            "fps": self.fps,
            "requested_stream": list(requested),
            "depth_scale": self.depth_scale,
            "sdk_depth_unit_m": self.depth_scale if enabled_depth else None,
            "depth_scale_correction": 1.0,
            "camera_frame": "color_optical_frame",
            "depth_aligned_to_color": enabled_depth,
            "model": model_name,
            "K": [
                [float(intr.fx), 0.0, float(intr.ppx)],
                [0.0, float(intr.fy), float(intr.ppy)],
                [0.0, 0.0, 1.0],
            ],
            "distortion": coeffs,
            "actual_stream": [self.width, self.height, self.fps],
        }
        if enabled_depth:
            dp = profile.get_stream(rs.stream.depth)
            di = dp.as_video_stream_profile().get_intrinsics()
            ext = dp.get_extrinsics_to(color_stream)
            x = np.eye(4)
            x[:3, :3] = np.asarray(ext.rotation).reshape(3, 3, order="F")
            x[:3, 3] = ext.translation

            def native(i: Any) -> dict[str, Any]:
                return {key: getattr(i, key) for key in ("width", "height", "fx", "fy", "ppx", "ppy")} | {
                    "model": str(i.model),
                    "coeffs": list(i.coeffs),
                }

            self.metadata["native"] = {
                "serial": serial,
                "color": native(intr),
                "native_depth": native(di),
                "depth_scale": self.depth_scale,
                "X_color_depth": x.tolist(),
            }
        self.connected = True
        self._pointcloud_rays.clear()
        return self.metadata

    def read_frame(self, compute_pointcloud: bool = False) -> dict[str, Any]:
        if not self.connected or self.pipeline is None:
            raise CalibrationError("RealSense 相机未启动")
        frames = self.pipeline.wait_for_frames(2000)
        aligned = self.align.process(frames) if self.align is not None else frames
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or (self.cfg.get("enable_depth", True) and not depth_frame):
            raise CalibrationError("RealSense 取流超时或帧不完整")

        host_ts = time.monotonic()
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_raw = np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None
        depth_m = depth_raw.astype(np.float32) * self.depth_scale if depth_raw is not None else None

        points_cam = None
        colors_rgb = None
        if compute_pointcloud and depth_m is not None:
            points_cam, colors_rgb = self._deproject_subsampled(color_bgr, depth_m, step=6)

        return {
            "timestamp_mono": host_ts,
            "timestamp": color_frame.get_timestamp() / 1000.0,
            "frame_number": color_frame.get_frame_number(),
            "metadata": self.metadata,
            "color_bgr": color_bgr,
            "depth_raw": depth_raw,
            "native_depth": np.asanyarray(frames.get_depth_frame().get_data()).copy() if depth_frame else None,
            "depth_m": depth_m,
            "points_cam": points_cam,
            "colors_rgb": colors_rgb,
        }

    def _deproject_subsampled(
        self, color_bgr: np.ndarray, depth_m: np.ndarray, step: int = 6
    ) -> tuple[np.ndarray, np.ndarray]:
        return CameraModel(self.metadata).deproject_depth_pointcloud(
            depth_m,
            color_bgr,
            step=step,
            min_depth_m=float(self.cfg.get("min_depth_m", 0.07)),
            max_depth_m=float(self.cfg.get("max_depth_m", 0.80)),
            ray_cache=self._pointcloud_rays,
        )

    def close(self) -> None:
        if self.pipeline is not None and (self._pipeline_started or self.connected):
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.connected = False
        self._pipeline_started = False

    stop = close
    read = read_frame
