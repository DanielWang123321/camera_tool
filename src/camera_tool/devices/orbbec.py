"""Orbbec RGB-D camera driver using Orbbec SDK v2 in an isolated ProcessCamera worker."""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from camera_tool.devices.process_camera import ProcessCamera
from camera_tool.paths import RESOURCE_ROOT

UDEV_SCRIPT = RESOURCE_ROOT / "install_orbbec_udev_rules.sh"
ORBBEC_VID = "2bc5"


@dataclass(frozen=True)
class SysfsUsbInfo:
    sysfs_name: str
    vid: str
    pid: str
    product: str
    serial: str
    speed_mbps: int
    devnode: Path | None
    writable: bool


def _read_sysfs_attr(dev_dir: Path, name: str) -> str:
    path = dev_dir / name
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except OSError:
        return ""


def scan_sysfs_orbbec() -> list[SysfsUsbInfo]:
    """Read connected Orbbec USB devices and libusb device node permissions from Linux sysfs."""
    root = Path("/sys/bus/usb/devices")
    results: list[SysfsUsbInfo] = []
    if not root.is_dir():
        return results

    for dev_dir in sorted(root.iterdir()):
        vid_file = dev_dir / "idVendor"
        pid_file = dev_dir / "idProduct"
        if not vid_file.is_file() or not pid_file.is_file():
            continue
        try:
            vid = vid_file.read_text(encoding="utf-8").strip().lower()
            if vid != ORBBEC_VID:
                continue
            pid = pid_file.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue

        product = _read_sysfs_attr(dev_dir, "product") or "Orbbec Camera"
        serial = _read_sysfs_attr(dev_dir, "serial")
        try:
            speed_mbps = int(float(_read_sysfs_attr(dev_dir, "speed") or "0"))
        except ValueError:
            speed_mbps = 0

        devnode: Path | None = None
        writable = True
        bus_str = _read_sysfs_attr(dev_dir, "busnum")
        dev_str = _read_sysfs_attr(dev_dir, "devnum")
        if bus_str.isdigit() and dev_str.isdigit():
            candidate = Path(f"/dev/bus/usb/{int(bus_str):03d}/{int(dev_str):03d}")
            if candidate.exists():
                devnode = candidate
                writable = os.access(candidate, os.R_OK | os.W_OK)

        results.append(
            SysfsUsbInfo(
                sysfs_name=dev_dir.name,
                vid=vid,
                pid=pid,
                product=product,
                serial=serial,
                speed_mbps=speed_mbps,
                devnode=devnode,
                writable=writable,
            )
        )
    return results


def format_permission_error(sysfs_devices: list[SysfsUsbInfo]) -> str:
    unwritable = [d for d in sysfs_devices if not d.writable]
    lines = ["无法打开 Orbbec USB 设备：当前用户缺少 libusb 节点写权限。"]
    for dev in unwritable:
        node_str = str(dev.devnode) if dev.devnode else "未知节点"
        sn_str = f"SN {dev.serial}" if dev.serial else "SN 未知"
        lines.append(
            f"  检测到 {dev.product} ({dev.vid}:{dev.pid}, {sn_str}, {dev.speed_mbps} Mbps) -> {node_str} (不可写)"
        )
    lines.append("请先在终端执行一次 udev 权限安装脚本：")
    lines.append(f"  sudo bash {UDEV_SCRIPT}")
    return "\n".join(lines)


def create_context() -> Any:
    import pyorbbecsdk as ob

    ob.Context.set_logger_to_file(ob.OBLogLevel.NONE, "")
    ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)
    return ob.Context()


def _summarize_profiles(profile_list: Any) -> list[str]:
    grouped: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for idx in range(profile_list.get_count()):
        sp = profile_list.get_stream_profile_by_index(idx)
        if not sp.is_video_stream_profile():
            continue
        vp = sp.as_video_stream_profile()
        key = (vp.get_width(), vp.get_height(), vp.get_fps())
        fmt_name = vp.get_format().name
        if fmt_name not in grouped[key]:
            grouped[key].append(fmt_name)
    return [
        f"{w}x{h}@{fps} [{'/'.join(fmts)}]"
        for (w, h, fps), fmts in sorted(
            grouped.items(), key=lambda item: (item[0][0] * item[0][1], item[0][2]), reverse=True
        )
    ]


def _find_device(ctx: Any, serial: str) -> Any:
    import pyorbbecsdk as ob

    sysfs_devices = scan_sysfs_orbbec()
    if sysfs_devices and any(not d.writable for d in sysfs_devices):
        raise RuntimeError(format_permission_error(sysfs_devices))

    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count == 0:
        raise RuntimeError("未检测到 Orbbec 设备，请检查 USB 连接。")

    opened: list[tuple[str, Any]] = []
    for idx in range(count):
        try:
            dev = device_list.get_device_by_index(idx)
        except (ob.OBError, RuntimeError) as exc:
            msg = str(exc)
            if sysfs_devices and any(not d.writable for d in sysfs_devices):
                raise RuntimeError(format_permission_error(sysfs_devices)) from exc
            if "openUsbDevice failed" in msg or "permission" in msg.lower():
                raise RuntimeError(
                    f"打开 Orbbec USB 设备失败（{msg}）。\n"
                    f"请确认未被其他程序占用，并已执行：sudo bash {UDEV_SCRIPT}"
                ) from exc
            raise RuntimeError(f"打开 Orbbec 设备失败：{msg}") from exc
        sn = dev.get_device_info().get_serial_number()
        if not serial or sn == serial:
            return dev
        opened.append((sn, dev))

    found = ", ".join(sn for sn, _ in opened) or "无"
    raise RuntimeError(f"未找到序列号 {serial} 的 Orbbec 设备。当前已连接：{found}")


def _select_video_profile(
    profile_list: Any,
    width: int,
    height: int,
    fps: int,
    preferred_formats: list[Any],
    stream_label: str,
) -> Any:
    import pyorbbecsdk as ob

    last_err: Exception | None = None
    for fmt in preferred_formats:
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except (ob.OBError, RuntimeError) as exc:
            last_err = exc
    try:
        return profile_list.get_video_stream_profile(width, height, ob.OBFormat.UNKNOWN_FORMAT, fps)
    except (ob.OBError, RuntimeError) as exc:
        available = ", ".join(_summarize_profiles(profile_list)[:10])
        detail = f" ({last_err})" if last_err is not None else ""
        raise RuntimeError(
            f"无法以 {width}x{height}@{fps} 开启 {stream_label} 流{detail}。 当前设备支持：{available}"
        ) from exc


def _frame_to_bgr(color_frame: Any) -> np.ndarray | None:
    import cv2
    import pyorbbecsdk as ob

    width = color_frame.get_width()
    height = color_frame.get_height()
    fmt = color_frame.get_format()
    raw = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == ob.OBFormat.BGR:
        expected = height * width * 3
        return raw[:expected].reshape((height, width, 3)).copy() if raw.size >= expected else None
    if fmt == ob.OBFormat.RGB:
        expected = height * width * 3
        return cv2.cvtColor(raw[:expected].reshape((height, width, 3)), cv2.COLOR_RGB2BGR) if raw.size >= expected else None
    if fmt == ob.OBFormat.BGRA:
        expected = height * width * 4
        return cv2.cvtColor(raw[:expected].reshape((height, width, 4)), cv2.COLOR_BGRA2BGR) if raw.size >= expected else None
    if fmt == ob.OBFormat.RGBA:
        expected = height * width * 4
        return cv2.cvtColor(raw[:expected].reshape((height, width, 4)), cv2.COLOR_RGBA2BGR) if raw.size >= expected else None
    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size > 0 else None
    if fmt in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
        expected = height * width * 2
        return cv2.cvtColor(raw[:expected].reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUYV) if raw.size >= expected else None
    if fmt == ob.OBFormat.UYVY:
        expected = height * width * 2
        return cv2.cvtColor(raw[:expected].reshape((height, width, 2)), cv2.COLOR_YUV2BGR_UYVY) if raw.size >= expected else None
    if fmt == ob.OBFormat.NV12:
        expected = height * width * 3 // 2
        return cv2.cvtColor(raw[:expected].reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12) if raw.size >= expected else None
    if fmt == ob.OBFormat.NV21:
        expected = height * width * 3 // 2
        return cv2.cvtColor(raw[:expected].reshape((height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV21) if raw.size >= expected else None
    if fmt in (ob.OBFormat.I420, ob.OBFormat.YV12):
        expected = height * width * 3 // 2
        if raw.size < expected:
            return None
        code = cv2.COLOR_YUV2BGR_I420 if fmt == ob.OBFormat.I420 else cv2.COLOR_YUV2BGR_YV12
        return cv2.cvtColor(raw[:expected].reshape((height * 3 // 2, width)), code)
    raise RuntimeError(f"暂不支持的彩色像素格式：{fmt.name}")


def _extract_depth_meters(depth_frame: Any) -> tuple[np.ndarray, np.ndarray]:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale_mm = float(depth_frame.get_depth_scale())
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    expected = height * width
    if raw.size < expected:
        raise RuntimeError(f"深度帧数据长度异常：收到 {raw.size} 像素，期望 {expected} ({width}x{height})")
    raw_2d = raw[:expected].reshape((height, width))
    return raw_2d, raw_2d.astype(np.float32) * (scale_mm / 1000.0)


class OrbbecSession:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.ctx: Any = None
        self.device: Any = None
        self.pipeline: Any = None
        self.align_filter: Any = None
        self.color_profile: Any = None
        self.depth_profile: Any = None
        self.metadata: dict[str, Any] = {}

    def _pull_frames(self, timeout_ms: int = 1000) -> tuple[Any, Any]:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None
        if self.align_filter is None:
            return frames.get_color_frame(), None
        raw_color = frames.get_color_frame()
        aligned = self.align_filter.process(frames)
        if aligned is None:
            return None, None
        aligned_set = aligned.as_frame_set()
        return aligned_set.get_color_frame() or raw_color, aligned_set.get_depth_frame()

    def start(self) -> dict[str, Any]:
        import pyorbbecsdk as ob

        serial = str(self.config.get("serial", "")).strip()
        width = int(self.config.get("width", 1280))
        height = int(self.config.get("height", 800))
        fps = int(self.config.get("fps", 30))
        enable_depth = bool(self.config.get("enable_depth", True))

        try:
            self.ctx = create_context()
            self.device = _find_device(self.ctx, serial)
            info = self.device.get_device_info()
            used_serial = info.get_serial_number()
            device_name = info.get_name()

            self.pipeline = ob.Pipeline(self.device)
            ob_config = ob.Config()
            color_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            self.color_profile = _select_video_profile(
                color_profiles,
                width,
                height,
                fps,
                preferred_formats=[
                    ob.OBFormat.BGR,
                    ob.OBFormat.RGB,
                    ob.OBFormat.MJPG,
                    ob.OBFormat.YUYV,
                    ob.OBFormat.YUY2,
                    ob.OBFormat.UYVY,
                    ob.OBFormat.NV12,
                    ob.OBFormat.I420,
                    ob.OBFormat.BGRA,
                    ob.OBFormat.RGBA,
                ],
                stream_label="Color",
            )
            ob_config.enable_stream(self.color_profile)

            if enable_depth:
                depth_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                self.depth_profile = _select_video_profile(
                    depth_profiles,
                    width,
                    height,
                    fps,
                    preferred_formats=[ob.OBFormat.Y16, ob.OBFormat.Z16],
                    stream_label="Depth",
                )
                ob_config.enable_stream(self.depth_profile)
                ob_config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self.pipeline.enable_frame_sync()
                self.align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)

            try:
                self.pipeline.start(ob_config)
            except (ob.OBError, RuntimeError) as exc:
                msg = str(exc)
                if "busy" in msg.lower():
                    raise RuntimeError("相机被占用，请关闭其他预览程序后再试。") from exc
                raise RuntimeError(f"无法启动取流：{msg}") from exc

            intr = self.color_profile.get_intrinsic()
            distortion = self.color_profile.get_distortion()
            coeffs = [float(getattr(distortion, k)) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
            raw_model = str(distortion.model).split(".")[-1].lower()
            model_name = "pinhole" if (raw_model == "none" or all(abs(c) < 1e-9 for c in coeffs)) else "brown_conrady"
            depth_unit_m = None
            warmed = 0
            for _ in range(25):
                c_frame, d_frame = self._pull_frames(timeout_ms=1000)
                if c_frame is not None and (not enable_depth or d_frame is not None):
                    if d_frame is not None and depth_unit_m is None:
                        depth_unit_m = float(d_frame.get_depth_scale()) / 1000.0
                    warmed += 1
                    if warmed >= 5:
                        break

            self.metadata = {
                "serial": used_serial,
                "name": device_name,
                "width": self.color_profile.get_width(),
                "height": self.color_profile.get_height(),
                "fps": self.color_profile.get_fps(),
                "actual_stream": [
                    self.color_profile.get_width(),
                    self.color_profile.get_height(),
                    self.color_profile.get_fps(),
                ],
                "K": [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
                "model": model_name,
                "distortion": coeffs,
                "depth_scale": depth_unit_m,
                "sdk_depth_unit_m": depth_unit_m,
                "depth_scale_correction": 1.0,
                "camera_frame": "color_optical_frame",
                "depth_aligned_to_color": bool(enable_depth),
            }
            return self.metadata
        except BaseException:
            self.stop()
            raise

    def read(self) -> dict[str, Any]:
        color = depth = None
        for _ in range(5):
            color, depth = self._pull_frames(timeout_ms=1000)
            if color is not None and (self.depth_profile is None or depth is not None):
                break
        if color is None:
            raise RuntimeError("Orbbec 取流超时或缺少彩色帧")
        image = _frame_to_bgr(color)
        if image is None:
            raise RuntimeError("Orbbec 彩色帧格式异常")
        depth_m = depth_raw = None
        if self.depth_profile is not None:
            if depth is None:
                raise RuntimeError("Orbbec 缺少深度帧")
            depth_raw, depth_m = _extract_depth_meters(depth)
            self.metadata["sdk_depth_unit_m"] = float(depth.get_depth_scale()) / 1000.0
            self.metadata["depth_scale"] = self.metadata["sdk_depth_unit_m"]
        return {
            "color_bgr": image,
            "depth_m": depth_m,
            "depth_raw": depth_raw,
            "timestamp_mono": time.monotonic(),
            "timestamp": color.get_timestamp_us() / 1e6,
            "frame_number": color.get_index(),
            "metadata": dict(self.metadata),
        }

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.device = None
        self.ctx = None

    close = stop


class OrbbecCamera(ProcessCamera):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("camera_tool.devices.orbbec:OrbbecSession", config)
        self._pointcloud_rays: dict[int, np.ndarray] = {}

    def start(self) -> dict[str, Any]:
        self._pointcloud_rays.clear()
        return super().start()

    def read(self, compute_pointcloud: bool = False, **kwargs: Any) -> dict[str, Any]:
        frame = super().read(fresh=compute_pointcloud)
        if compute_pointcloud and frame.get("depth_m") is not None:
            from camera_tool.core.camera_model import CameraModel

            pts, colors = CameraModel(self.metadata).deproject_depth_pointcloud(
                frame["depth_m"],
                frame["color_bgr"],
                step=6,
                min_depth_m=float(self.config.get("min_depth_m", 0.07)),
                max_depth_m=float(self.config.get("max_depth_m", 0.80)),
                ray_cache=self._pointcloud_rays,
            )
            frame = {**frame, "points_cam": pts, "colors_rgb": colors}
        return frame
