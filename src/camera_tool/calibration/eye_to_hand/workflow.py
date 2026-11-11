"""Eye-to-Hand calibration workflow (live acquisition, read-only check, and offline replay)."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from camera_tool.calibration.eye_to_hand.cameras import BufferedRgbdCamera, WristCamera
from camera_tool.calibration.eye_to_hand.solver import (
    check_samples,
    fit_hand_eye,
    solve_fixed_camera,
    solve_wrist,
)
from camera_tool.core import (
    CalibrationError,
    CameraModel,
    MarkerDetector,
    RejectedSample,
    Session,
    image_hash,
    inv,
    marker_points,
    matrix_list,
    pose_error,
    read_json,
    transform,
    transform_points,
    valid_transform,
    write_json,
)
from camera_tool.devices.robot import TcpCalibrationRobot
from camera_tool.paths import DATA_ROOT, project_path, resource_path


def load_settings(path: str | Path) -> dict[str, Any]:
    config_file = resource_path(path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise CalibrationError("标定配置为空")

    fixed_cfg = config.get("fixed_camera") or config.get("realsense")
    wrist_cfg = config.get("wrist_camera") or config.get("wrist")
    if not isinstance(fixed_cfg, dict) or not isinstance(wrist_cfg, dict):
        raise CalibrationError("标定配置缺少 fixed_camera 或 wrist_camera 配置块")
    config["fixed_camera"] = fixed_cfg
    config["realsense"] = fixed_cfg
    config["wrist_camera"] = wrist_cfg
    config["wrist"] = wrist_cfg

    raw_intrinsics = str(wrist_cfg.get("intrinsics") or "").strip()
    if raw_intrinsics:
        intrinsics = Path(raw_intrinsics)
        if not intrinsics.is_absolute():
            local = config_file.parent / intrinsics
            wrist_cfg["intrinsics"] = str(local.resolve() if local.is_file() else resource_path(intrinsics))
    else:
        wrist_cfg["intrinsics"] = ""

    motion = config["motion"]
    for key in (
        "joint_margin_deg",
        "joint_step_deg",
        "path_sample_deg",
        "translation_step_m",
        "rotation_step_deg",
        "speed_deg_s",
        "acceleration_deg_s2",
        "settle_s",
        "settle_position_mm",
        "settle_rotation_deg",
        "move_timeout_s",
    ):
        if not np.isfinite(motion[key]) or motion[key] <= 0:
            raise CalibrationError(f"运动参数 {key} 必须为正数")
    if (
        not 60 <= motion["max_attempts"] <= 90
        or motion["speed_deg_s"] > 15
        or motion["acceleration_deg_s2"] > 50
        or motion["path_sample_deg"] > 2
        or motion["joint_step_deg"] > 10
        or motion["joint_margin_deg"] < 5
        or motion["settle_s"] < 0.3
        or motion["translation_step_m"] > 0.02
        or motion["rotation_step_deg"] > 6
    ):
        raise CalibrationError("自动标定运动参数超过实现允许的保守范围")

    frames = int(fixed_cfg["frames"])
    config["fixed_frames"] = frames
    config["realsense_frames"] = frames
    if not (5 <= frames <= 60):
        raise CalibrationError("固定相机需要 5～60 帧")
    if not (0.01 <= config["marker"]["side_m"] <= 0.5):
        raise CalibrationError("标定码边长单位为米，当前值不合理")
    width, height = wrist_cfg["rectified_size"]
    if not (128 <= width <= 1920 and 128 <= height <= 1920 and 30 <= wrist_cfg["rectified_fov_deg"] <= 120):
        raise CalibrationError("腕部相机去畸变尺寸或视野范围无效")
    raw_w, raw_h = wrist_cfg["capture_size"]
    if not (128 <= raw_w <= 4096 and 128 <= raw_h <= 2160):
        raise CalibrationError("腕部相机原图尺寸无效")
    for key, value in config["checks"].items():
        if not np.isfinite(value) or (key != "table_z_m" and value <= 0):
            raise CalibrationError(f"验收参数无效：{key}")
    return config


def color_metadata(native: dict[str, Any]) -> dict[str, Any]:
    if "color" in native:
        ci = native["color"]
        coefficients = np.asarray(ci["coeffs"], dtype=float)
        if np.all(coefficients == 0) or ci["model"] == "distortion.none":
            model = "pinhole"
        elif ci["model"] == "distortion.brown_conrady":
            model = "brown_conrady"
        else:
            raise CalibrationError(f"固定相机彩色畸变模型尚不支持：{ci['model']}，不能忽略畸变")
        return {
            "serial": native["serial"],
            "width": ci["width"],
            "height": ci["height"],
            "model": model,
            "distortion": coefficients.tolist(),
            "K": [[ci["fx"], 0.0, ci["ppx"]], [0.0, ci["fy"], ci["ppy"]], [0.0, 0.0, 1.0]],
            "native": native,
        }
    coefficients = np.asarray(native.get("distortion", [0.0] * 5), dtype=float)
    model = str(native.get("model", "pinhole"))
    if np.all(coefficients == 0):
        model = "pinhole"
    elif model not in ("pinhole", "brown_conrady"):
        raise CalibrationError(f"固定相机彩色畸变模型尚不支持：{model}，不能忽略畸变")
    return {
        "serial": str(native.get("serial", "")),
        "width": int(native["width"]),
        "height": int(native["height"]),
        "model": model,
        "distortion": coefficients.tolist(),
        "K": native["K"],
        "native": {
            "serial": str(native.get("serial", "")),
            "depth_scale": float(native.get("sdk_depth_unit_m") or native.get("depth_scale") or 0.001),
            "X_color_depth": np.eye(4).tolist(),
        },
    }


def make_depth_geometry(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    native = metadata.get("native", {})
    if "native_depth" in native:
        import pyrealsense2 as rs

        ci = native["native_depth"]
        if ci["model"] not in ("distortion.none", "distortion.brown_conrady", "distortion.inverse_brown_conrady"):
            raise CalibrationError(f"原始深度反投影暂不支持此畸变模型：{ci['model']}")
        if not np.isfinite(native["depth_scale"]) or native["depth_scale"] <= 0:
            raise CalibrationError("原始深度的单位比例无效")
        intr = rs.intrinsics()
        for key in ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs"):
            setattr(intr, key, ci[key])
        intr.model = getattr(rs.distortion, ci["model"].split(".")[-1])
        y, x = np.mgrid[0 : ci["height"] : 3, 0 : ci["width"] : 3]
        pixels = np.column_stack([x.ravel(), y.ravel()])
        rays = np.asarray([rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0) for u, v in pixels])
        if not np.isfinite(rays).all() or not np.allclose(rays[:, 2], 1.0):
            raise CalibrationError("原始深度反投影失败")
        return {"pixels": pixels, "rays": rays}

    cam = CameraModel(metadata)
    y, x = np.mgrid[0 : cam.height : 3, 0 : cam.width : 3]
    pixels = np.column_stack([x.ravel(), y.ravel()])
    rays = cam.deproject_pixels(pixels, np.ones(len(pixels), dtype=float))
    return {"pixels": pixels, "rays": rays}


def depth_points(native_depth: np.ndarray | None, geometry: Any, metadata: dict[str, Any]) -> np.ndarray:
    native = metadata.get("native", {})
    if "native_depth" in native:
        di = native["native_depth"]
        expected_shape = (di["height"], di["width"])
    else:
        expected_shape = (int(metadata["height"]), int(metadata["width"]))
    if native_depth is None or native_depth.shape != expected_shape:
        raise CalibrationError("原始深度尺寸与内参不一致")
    pixels, rays = geometry["pixels"], geometry["rays"]
    if pixels.shape != (len(rays), 2) or rays.shape[1:] != (3,) or not np.isfinite(rays).all():
        raise CalibrationError("保存的深度射线数据无效")
    scale = float(native.get("depth_scale", 0.001))
    depth = native_depth[pixels[:, 1], pixels[:, 0]] * scale
    x_color_depth = valid_transform(native.get("X_color_depth", np.eye(4)))
    return transform_points(x_color_depth, rays[depth > 0] * depth[depth > 0, None])


def capture_consistency(
    before: dict[str, Any],
    after: dict[str, Any],
    host_timestamp: float,
    motion: dict[str, Any],
) -> None:
    mm, deg = pose_error(valid_transform(before["X_base_tcp"]), valid_transform(after["X_base_tcp"]))
    if (
        before["state"] == 1
        or after["state"] == 1
        or mm > motion["settle_position_mm"]
        or deg > motion["settle_rotation_deg"]
        or np.rad2deg(np.abs(np.asarray(before["joints_rad"]) - after["joints_rad"])).max() > 0.1
    ):
        raise RejectedSample("曝光前后机械臂有移动")
    if not before["read_end_mono"] <= host_timestamp <= after["read_start_mono"]:
        raise CalibrationError("图像曝光时间不在两次 TCP 读取之间，停止不同步采样")


def clamp_workspace(pose: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    out = np.asarray(pose, dtype=float).copy()
    low = np.asarray(config["workspace_min_m"], dtype=float)
    high = np.asarray(config["workspace_max_m"], dtype=float)
    out[:3, 3] = np.clip(out[:3, 3], low, high)
    return out


def search_candidate(index: int, config: dict[str, Any]) -> np.ndarray | None:
    offsets = [
        (0, 0, 0),
        (-0.06, 0, 0),
        (0.06, 0, 0),
        (0, -0.06, 0),
        (0, 0.06, 0),
        (-0.06, -0.06, 0.04),
        (0.06, -0.06, 0.04),
        (-0.06, 0.06, 0.04),
        (0.06, 0.06, 0.04),
    ]
    if index >= len(offsets):
        return None
    return transform(
        Rotation.from_euler("xyz", config["search_rpy_deg"], degrees=True).as_matrix(),
        np.asarray(config["search_center_m"]) + offsets[index],
    )


def sample_candidate(anchor: np.ndarray, index: int, preliminary: dict[str, Any] | None) -> np.ndarray:
    angles = [
        (8, 0),
        (0, 8),
        (-8, 0),
        (0, -8),
        (12, 8),
        (-12, 8),
        (-8, -12),
        (8, -12),
        (20, 0),
        (0, 20),
        (-20, 0),
        (0, -20),
        (22, 15),
        (-22, 15),
        (-15, -22),
        (15, -22),
    ]
    a, b = angles[index % len(angles)]
    if index >= len(angles):
        a, b = a * 1.1, b * 1.1
    rotation = Rotation.from_euler("xyz", [a, b, (index % 3 - 1) * 4], degrees=True).as_matrix()
    offset = np.array([(index % 3 - 1) * 0.025, ((index // 3) % 3 - 1) * 0.025, ((index // 9) % 3 - 1) * 0.025])
    if preliminary is None:
        return transform(anchor[:3, :3] @ rotation, anchor[:3, 3] + offset)
    wrist, board = preliminary["X_tcp_wrist"], preliminary["X_base_board"]
    original = anchor @ wrist
    distance = float((inv(original) @ board)[2, 3])
    if not 0.12 < distance < 1.5:
        raise CalibrationError("初步解中的拍摄距离异常")
    r = original[:3, :3] @ rotation
    camera = transform(r, board[:3, 3] - r @ np.array([offset[0], offset[1], distance + offset[2]]))
    return camera @ inv(wrist)


def predicted_visible(tcp: np.ndarray, solution: dict[str, Any], camera: CameraModel, side: float) -> bool:
    points = transform_points(inv(tcp @ solution["X_tcp_wrist"]) @ solution["X_base_board"], marker_points(side))
    if points[:, 2].min() < 0.08:
        return False
    pixels = camera.project(points)
    return bool(
        np.isfinite(pixels).all()
        and pixels.min() > 12
        and np.all(pixels[:, 0] < camera.width - 12)
        and np.all(pixels[:, 1] < camera.height - 12)
    )


class LiveCalibration:
    def __init__(self, config: dict[str, Any], session: Session, context: Any = None) -> None:
        self.config = config
        self.session = session
        wrist_cfg = config.get("wrist_camera") or config["wrist"]
        fixed_cfg = config.get("fixed_camera") or config["realsense"]
        self.wrist = WristCamera(wrist_cfg, session)
        self.fixed = BufferedRgbdCamera(fixed_cfg)
        self.robot = TcpCalibrationRobot(config["robot"], config["motion"], session.event)
        self.seen_hashes: set[str] = set()
        self.fixed_number = -1
        self.fixed_initial_corners: np.ndarray | None = None
        self.context = context
        self.robot.cancel_check = context.check if context else lambda: None

    def start(self) -> None:
        if self.context:
            self.context.check()
        self.session.event("读取腕部相机内参并开流")
        self.wrist.start()
        self.session.manifest["wrist_camera"] = self.wrist.metadata
        self.session.save()
        self.wrist_model = CameraModel(self.wrist.metadata)
        self.wrist_detector = MarkerDetector(self.config["marker"], self.wrist_model)

        self.fixed.start()
        self.fixed_meta = color_metadata(self.fixed.calibration_metadata)
        self.fixed_model = CameraModel(self.fixed_meta)
        self.fixed_detector = MarkerDetector(self.config["marker"], self.fixed_model)
        self.geometry = make_depth_geometry(self.fixed_meta)
        np.savez_compressed(self.session.path / "depth_geometry.npz", **self.geometry)
        write_json(self.session.path / "fixed_camera.json", self.fixed_meta)
        self.session.manifest["fixed_camera"] = self.fixed_meta
        self.session.manifest["realsense_camera"] = self.fixed_meta
        self.session.save()

        self.session.manifest["robot"] = self.robot.start()
        self.session.save()
        self.session.event("设备参数和机械臂只读检查通过")

    def fixed_frame(self, after_mono: float = 0.0) -> dict[str, Any]:
        if self.context:
            self.context.check()
        frame = self.fixed.get_frame(after_frame_number=self.fixed_number, after_mono=after_mono, wait_s=2.0, max_age_s=0.3)
        if frame is None:
            raise CalibrationError(f"固定相机断流或没有新帧：{self.fixed.last_error or '等待超时'}")
        self.fixed_number = frame["frame_number"]
        if self.context:
            self.context.frame("fixed", frame)
        return frame

    def health(self) -> None:
        if self.context:
            self.context.check()
        wrist = self.wrist.frame(timeout=1.0)
        if self.context:
            self.context.frame("wrist", wrist)
        self.fixed_frame()

    def capture_wrist(self, settled: float, name: str) -> dict[str, Any]:
        assert self.wrist.metadata is not None
        before = self.robot.read()
        packet = self.wrist.frame(after_mono=max(settled, before["read_end_mono"]))
        if self.context:
            self.context.check()
            self.context.frame("wrist", packet)
        after = self.robot.read()
        sample = {
            "raw_image": self.session.image(f"{name}_raw.png", packet["raw"]),
            "image": self.session.image(f"{name}.png", packet["color"]),
            "before": before,
            "after": after,
            "X_base_tcp": after["X_base_tcp"],
            "host_timestamp": packet["host_timestamp"],
            "frame_number": packet["frame_number"],
            "source_timestamp_us": packet["source_timestamp_us"],
            "frame_clock": self.wrist.metadata.get("frame_clock"),
            "sha256": image_hash(packet["raw"]),
        }
        write_json(self.session.path / f"{name}.json", sample)
        capture_consistency(before, after, packet["host_timestamp"], self.config["motion"])
        sample.update(self.wrist_detector.detect(packet["color"]))
        if sample["sha256"] in self.seen_hashes:
            raise RejectedSample("重复图像，不能当作新姿态")
        self.seen_hashes.add(sample["sha256"])
        return sample

    def check_only(self) -> dict[str, Any]:
        settled = self.robot.settle(self.health)
        wrist = self.capture_wrist(settled, "check_wrist")
        frame = self.fixed_frame()
        self.session.image("check_fixed.png", frame["color"])
        try:
            detection = self.fixed_detector.detect(frame["color"])
            fixed_visibility = {"visible": True, **detection}
        except RejectedSample as exc:
            fixed_visibility = {"visible": False, "reason": str(exc)}
        self.session.manifest["check"] = {
            "wrist": wrist,
            "fixed_camera": fixed_visibility,
            "realsense": fixed_visibility,
            "motion_commands_sent": False,
            "calibration_published": False,
        }
        self.session.save()
        marker_id = self.config["marker"]["id"]
        print(f"腕部相机已识别码 {marker_id}，角点投影 {wrist['pnp_rms_px']:.3f} px", flush=True)
        if fixed_visibility["visible"]:
            print(f"固定相机已识别码 {marker_id}，角点投影 {fixed_visibility['pnp_rms_px']:.3f} px", flush=True)
        else:
            print(f"固定相机当前未看到码：{fixed_visibility['reason']}", flush=True)
        return self.session.manifest["check"]

    def monitor_static_board(self, name: str) -> None:
        frame = self.fixed_frame()
        try:
            detection = self.fixed_detector.detect(frame["color"])
        except RejectedSample:
            return
        corners = np.asarray(detection["corners"])
        if self.fixed_initial_corners is None:
            self.fixed_initial_corners = corners
        elif (
            np.linalg.norm(corners - self.fixed_initial_corners, axis=1).max()
            > self.config["checks"]["static_corner_drift_px"]
        ):
            self.session.image(f"{name}_board_moved.png", frame["color"])
            raise CalibrationError("固定相机发现标定板或相机移动，停止整次标定")

    def collect_wrist(self) -> list[dict[str, Any]]:
        samples = self.session.manifest["samples"]
        initial = self.robot.read()["X_base_tcp"]
        anchor, preliminary = None, None
        local_index, search_index = 0, 0
        last_preliminary_count = 0
        for attempt in range(self.config["motion"]["max_attempts"]):
            if self.context:
                self.context.check()
            name = f"attempt_{attempt:02d}"
            if attempt == 0:
                try:
                    self.robot.check_workspace(initial)
                    target = initial
                except RejectedSample:
                    target = search_candidate(0, self.config["motion"])
                    search_index = 1
            elif anchor is None:
                target = search_candidate(search_index, self.config["motion"])
                search_index += 1
                if target is None:
                    raise CalibrationError("有限搜索结束，未找到完整标定码")
            else:
                target = sample_candidate(anchor, local_index, preliminary)
                local_index += 1
            assert target is not None
            target = clamp_workspace(target, self.config["motion"])
            self.session.event("检查候选姿态", attempt=attempt, target=target)
            try:
                if preliminary is not None and not predicted_visible(
                    target, preliminary, self.wrist_model, self.config["marker"]["side_m"]
                ):
                    raise RejectedSample("预计标定板超出腕部相机视野")
                current = self.robot.read()["X_base_tcp"]
                mm, deg = pose_error(current, target)
                if mm < 2.0 and deg < 1.0:
                    settled = self.robot.settle(self.health)
                else:
                    settled = self.robot.move(target, self.health)
                sample = self.capture_wrist(settled, name)
                for old in samples:
                    mm, deg = pose_error(np.asarray(old["X_base_tcp"]), sample["X_base_tcp"])
                    if mm < 8.0 and deg < 3.0:
                        raise RejectedSample("姿态与已有样本太接近")
            except RejectedSample as exc:
                self.session.event("跳过候选", attempt=attempt, reason=str(exc))
                continue
            sample.update(id=len(samples), attempt=attempt, role="check" if len(samples) % 5 == 4 else "train")
            samples.append(sample)
            self.session.save()
            self.session.event("保存有效姿态", count=len(samples), role=sample["role"])
            self.monitor_static_board(name)
            if anchor is None:
                anchor = sample["X_base_tcp"]
            training = [s for i, s in enumerate(samples) if i % 5 != 4]
            if len(training) in (8, 12, 16) and len(training) != last_preliminary_count:
                last_preliminary_count = len(training)
                try:
                    rough = fit_hand_eye(
                        training, self.wrist_model, self.config["marker"]["side_m"], minimum_rotation_deg=2.0
                    )
                    if rough["fit_rms_px"] < 2.0:
                        preliminary = rough
                        self.session.event("已有初步解，扩大拍摄角度", training_count=len(training))
                except CalibrationError as exc:
                    self.session.event("继续小范围采样", reason=str(exc))
            if len(samples) == 30:
                return samples
        raise CalibrationError(f"{self.config['motion']['max_attempts']} 个候选已用完，仅采到 {len(samples)} 个有效姿态")

    def capture_fixed_samples(self) -> list[dict[str, Any]]:
        samples = self.session.manifest["fixed_samples"]
        required = int(self.config.get("fixed_frames") or self.config["realsense_frames"])
        hashes: set[str] = set()
        for attempt in range(required * 3):
            if self.context:
                self.context.check()
            frame = self.fixed_frame(after_mono=time.monotonic() - 0.05)
            name = f"fixed_{attempt:02d}"
            image_name = self.session.image(name + ".png", frame["color"])
            depth_name = self.session.image(name + "_depth.png", frame["depth"])
            native_name = self.session.image(name + "_native_depth.png", frame["native_depth"])
            try:
                detection = self.fixed_detector.detect(frame["color"])
                digest = image_hash(frame["color"])
                if digest in hashes:
                    raise RejectedSample("固定相机重复图像")
            except RejectedSample as exc:
                self.session.event("跳过固定相机图像", image=image_name, reason=str(exc))
                continue
            hashes.add(digest)
            sample = {
                **detection,
                "image": image_name,
                "depth_image": depth_name,
                "native_depth_image": native_name,
                "frame_number": frame["frame_number"],
                "timestamp": frame["timestamp"],
                "sha256": digest,
            }
            if (
                self.fixed_initial_corners is not None
                and np.linalg.norm(np.asarray(sample["corners"]) - self.fixed_initial_corners, axis=1).max()
                > self.config["checks"]["static_corner_drift_px"]
            ):
                raise CalibrationError("板或固定相机在腕部采样后移动")
            samples.append(sample)
            self.session.manifest["realsense"] = samples
            self.session.save()
            if len(samples) == required:
                return [
                    {
                        **s,
                        "points_color": depth_points(
                            cv2.imread(str(self.session.path / s["native_depth_image"]), cv2.IMREAD_UNCHANGED),
                            self.geometry,
                            self.fixed_meta,
                        ),
                    }
                    for s in samples
                ]
        raise CalibrationError("机械臂让开后，固定相机仍未取得足够有效图像")

    def run(self) -> dict[str, Any]:
        assert self.wrist.metadata is not None
        self.monitor_static_board("initial")
        samples = self.collect_wrist()
        solution = solve_wrist(samples, self.wrist.metadata, self.config)
        write_json(self.session.path / "wrist_result.json", solution)
        self.session.event("腕部标定和 6 个独立姿态检查通过", report=solution["independent_check"])

        witness_pose = np.asarray(samples[-1]["X_base_tcp"])
        park = transform(
            Rotation.from_euler("xyz", self.config["motion"]["park_rpy_deg"], degrees=True).as_matrix(),
            self.config["motion"]["park_tcp_m"],
        )
        self.robot.move(park, self.health)
        fixed_samples = self.capture_fixed_samples()
        result = {**solution, **solve_fixed_camera(fixed_samples, self.fixed_meta, solution, self.config)}
        if not result["depth_ok"]:
            self.session.event(
                "深度核对未达门槛，仍发布刚体外参",
                report=result["depth_checks"][:1],
                depth_scale=result["depth_scale"],
                warnings=result["depth_warnings"],
            )
        settled = self.robot.move(witness_pose, self.health)
        witness = self.capture_wrist(settled, "final_witness")
        self.session.manifest["witness"] = witness
        self.session.save()
        result["witness_check"] = check_samples(
            [witness],
            self.wrist_model,
            self.config["marker"]["side_m"],
            solution,
            self.config["checks"],
            "交接后腕部复核",
        )
        self.robot.move(park, self.health)
        final = self.fixed_frame(after_mono=time.monotonic())
        self.session.image("final_fixed.png", final["color"])
        final_corners = self.fixed_detector.detect(final["color"])["corners"]
        if (
            np.linalg.norm(np.asarray(final_corners) - fixed_samples[0]["corners"], axis=1).max()
            > self.config["checks"]["static_corner_drift_px"]
        ):
            raise CalibrationError("收尾时标定板或固定相机移动")
        final_record = {"image": "final_fixed.png", "corners": final_corners}
        self.session.manifest["final_fixed"] = final_record
        self.session.manifest["final_realsense"] = final_record
        self.session.save()
        result.update(
            passed=True,
            units="m",
            tcp_definition="controller_actual_tcp",
            tcp_offset_controller=self.session.manifest["robot"]["tcp_offset_controller"],
            wrist_serial=self.wrist.serial,
            fixed_serial=self.fixed_meta["serial"],
            realsense_serial=self.fixed_meta["serial"],
        )
        return result

    def close(self, failed: bool) -> None:
        if failed:
            self.robot.stop()
        for name, close in (("机械臂", self.robot.close), ("固定相机", self.fixed.stop), ("腕部相机", self.wrist.stop)):
            try:
                close()
            except Exception as exc:
                self.session.event(f"关闭{name}失败", reason=str(exc))


def read_image(directory: Path, name: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    path = project_path(directory / name)
    if not path.is_relative_to(directory.resolve()):
        raise CalibrationError("回放图像路径超出运行目录")
    image = cv2.imread(str(path), flags)
    if image is None:
        raise CalibrationError(f"回放图像缺失或损坏：{path}")
    return image


def replay(source: str | Path, session: Session) -> dict[str, Any]:
    source = project_path(source)
    if source.is_file():
        source = source.parent
    manifest = read_json(source / "session.json")
    if manifest.get("schema") != 1:
        raise CalibrationError("不支持此记录版本")
    config = manifest["config"]
    fixed_cfg = config.get("fixed_camera") or config.get("realsense") or {}
    config["fixed_frames"] = int(
        config.get("fixed_frames") or config.get("realsense_frames") or fixed_cfg.get("frames", 12)
    )
    config["realsense_frames"] = config["fixed_frames"]

    camera = CameraModel(manifest["wrist_camera"])
    detector = MarkerDetector(config["marker"], camera)
    with np.load(source / "wrist_rectification.npz", allow_pickle=False) as data:
        mapping = data["mapping"]
    if mapping.shape != (camera.height, camera.width, 2) or not np.isfinite(mapping).all():
        raise CalibrationError("保存的腕部相机去畸变映射无效")

    seen: set[str] = set()
    frames: set[Any] = set()
    samples: list[dict[str, Any]] = []
    for item in manifest["samples"]:
        raw = read_image(source, item["raw_image"])
        digest = image_hash(raw)
        if digest != item["sha256"] or digest in seen or item["frame_number"] in frames:
            raise CalibrationError("回放中存在重复帧或原始图像被更改")
        seen.add(digest)
        frames.add(item["frame_number"])
        image = cv2.remap(raw, mapping, None, cv2.INTER_LINEAR)
        capture_consistency(item["before"], item["after"], item["host_timestamp"], config["motion"])
        samples.append({**item, **detector.detect(image)})

    solution = solve_wrist(samples, manifest["wrist_camera"], config)
    fixed_meta = manifest.get("fixed_camera") or manifest["realsense_camera"]
    fixed_model = CameraModel(fixed_meta)
    fixed_detector = MarkerDetector(config["marker"], fixed_model)
    raw_fixed_samples = manifest.get("fixed_samples") or manifest["realsense"]
    with np.load(source / "depth_geometry.npz", allow_pickle=False) as geometry:
        fixed_samples: list[dict[str, Any]] = []
        seen_fixed: set[str] = set()
        numbers: set[Any] = set()
        for item in raw_fixed_samples:
            image = read_image(source, item["image"])
            digest = image_hash(image)
            if digest != item["sha256"] or digest in seen_fixed or item["frame_number"] in numbers:
                raise CalibrationError("固定相机回放中存在重复帧或图像被更改")
            seen_fixed.add(digest)
            numbers.add(item["frame_number"])
            depth = read_image(source, item["native_depth_image"], cv2.IMREAD_UNCHANGED)
            fixed_samples.append(
                {
                    **item,
                    **fixed_detector.detect(image),
                    "points_color": depth_points(depth, geometry, fixed_meta),
                }
            )
    result = {**solution, **solve_fixed_camera(fixed_samples, fixed_meta, solution, config)}

    witness = manifest.get("witness")
    final_record = manifest.get("final_fixed") or manifest.get("final_realsense")
    if witness is None or final_record is None:
        raise CalibrationError("记录缺少相机交接后的腕部复核或收尾图像，不能验收完整标定")
    raw = read_image(source, witness["raw_image"])
    witness = {**witness, **MarkerDetector(config["marker"], camera).detect(cv2.remap(raw, mapping, None, cv2.INTER_LINEAR))}
    capture_consistency(witness["before"], witness["after"], witness["host_timestamp"], config["motion"])
    result["witness_check"] = check_samples(
        [witness], camera, config["marker"]["side_m"], solution, config["checks"], "交接后腕部复核"
    )
    final = MarkerDetector(config["marker"], fixed_model).detect(read_image(source, final_record["image"]))
    if (
        np.linalg.norm(np.asarray(final["corners"]) - fixed_samples[0]["corners"], axis=1).max()
        > config["checks"]["static_corner_drift_px"]
    ):
        raise CalibrationError("回放收尾图像发现板或相机移动")
    result.update(passed=True, published=False, source=str(source), units="m")
    session.manifest.update(source=str(source), config=config)
    session.save()
    return result


def publish_camera(path: str | Path, result: dict[str, Any], session: Session) -> str:
    target = project_path(path)
    if session.manifest["mode"] != "live" or result.get("passed") is not True:
        raise CalibrationError("只有完整实机流程通过几何检查后才能发布外参")
    fixed_serial = result.get("fixed_serial") or result.get("realsense_serial")
    if fixed_serial is None:
        raise CalibrationError("发布外参缺少必要检查/设备信息：fixed_serial")
    for name in ("independent_check", "witness_check", "wrist_serial", "X_base_camera"):
        if result.get(name) is None:
            raise CalibrationError(f"发布外参缺少必要检查/设备信息：{name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    original = target.read_bytes() if target.is_file() else b""
    raw = yaml.safe_load(original) or {} if original else {}
    if not isinstance(raw, dict):
        raw = {}
    fixed_meta = session.manifest.get("fixed_camera") or session.manifest["realsense_camera"]
    for name in ("X_base_camera", "X_tcp_wrist", "X_base_board"):
        raw[name] = matrix_list(result[name])
    raw.update(
        serial=fixed_serial,
        width=fixed_meta["width"],
        height=fixed_meta["height"],
        depth_scale=float(result["depth_scale"]),
    )
    for key, default in (
        ("fps", 30),
        ("num_points", 1024),
        ("bbox_min", [0.20, -0.22, -0.03]),
        ("bbox_max", [0.61, 0.45, 0.40]),
    ):
        raw.setdefault(key, default)
    marker = session.manifest["config"]["marker"]
    raw.update(marker_length_m=marker["side_m"], marker_id=marker["id"], aruco_dict=marker["dictionary"])
    raw["calibration"] = {
        "method": "wrist_aruco_chain",
        "session": str(session.path.relative_to(DATA_ROOT) if session.path.is_relative_to(DATA_ROOT) else session.path),
        "units": "m",
        "camera_frame": "color_optical_frame",
        "depth_aligned_to": "color",
        "wrist_serial": result["wrist_serial"],
        "tcp_offset_controller": result["tcp_offset_controller"],
        "independent_check": result["independent_check"],
        "depth_ok": bool(result.get("depth_ok")),
        "sdk_depth_unit_m": fixed_meta.get("native", {}).get("depth_scale"),
        "depth_scale_correction": float(result["depth_scale"]),
        "depth_scale_note": "把对齐到彩色的深度（米）乘以此系数后再变到基座；不乘则桌面高度可能有数毫米偏差。",
    }
    if original:
        backup = session.path / "camera.before.yaml"
        with backup.open("xb") as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
    tmp = project_path(target.with_name(target.name + f".autocalib.{session.path.name}.tmp"))
    created = False
    try:
        with tmp.open("x", encoding="utf-8") as f:
            created = True
            f.write("# Eye-to-Hand 标定结果。点_base = X_base_camera @ 点_camera（彩色光心），位置单位米。\n")
            f.write("# 点云已对齐到彩色，不要再乘出厂 depth→color。TCP 为控制器实际工具坐标。\n")
            yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        if target.is_file() and target.read_bytes() != original:
            raise CalibrationError("写入前发现输出文件改变，取消覆盖")
        os.replace(tmp, target)
    finally:
        if created and tmp.exists():
            tmp.unlink()
    return str(target)
