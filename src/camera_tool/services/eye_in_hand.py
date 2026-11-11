"""Eye-in-Hand calibration service shared by desktop UI and headless CLI."""
from __future__ import annotations

import os
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from camera_tool.calibration.eye_in_hand.session import CalibrationSession
from camera_tool.calibration.eye_in_hand.solver import (
    evaluate_board_geometry,
    evaluate_depth_consistency,
    solve_hand_eye_all,
)
from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner
from camera_tool.core import (
    CalibrationError,
    CameraModel,
    MarkerDetector,
    RejectedSample,
    load_yaml,
    pose_error,
    pose_from_mm_deg,
)
from camera_tool.paths import config_path
from camera_tool.services.tasks import DEVICES, resource

PRESET_CONFIGS: dict[str, str] = {
    "default": "eye_in_hand/default.yaml",
    "medium_range": "eye_in_hand/medium_range.yaml",
    "orbbec": "eye_in_hand/orbbec.yaml",
}


def detect_usb_camera_backend(usb_root: Path = Path("/sys/bus/usb/devices")) -> str:
    """Detect attached RGB-D camera backend via Linux sysfs without importing hardware SDKs."""
    has_realsense = has_orbbec = False
    if usb_root.is_dir():
        for dev_dir in usb_root.iterdir():
            try:
                vid = (dev_dir / "idVendor").read_text(encoding="utf-8").strip().lower()
            except OSError:
                continue
            if vid == "8086":
                has_realsense = True
            elif vid == "2bc5":
                has_orbbec = True
    if has_realsense:
        return "realsense"
    if has_orbbec:
        return "orbbec"
    return "realsense"


def resolve_preset_config(
    preset: str | None = None,
    backend: str | None = None,
    usb_root: Path = Path("/sys/bus/usb/devices"),
) -> Path:
    """Resolve an Eye-in-Hand YAML config path from preset name and/or camera backend."""
    preset_key = str(preset or "auto").strip().lower()
    if preset_key in PRESET_CONFIGS:
        return config_path(PRESET_CONFIGS[preset_key])
    backend_key = str(backend or "auto").strip().lower()
    if backend_key in ("", "auto"):
        backend_key = detect_usb_camera_backend(usb_root)
    if backend_key == "orbbec":
        return config_path(PRESET_CONFIGS["orbbec"])
    return config_path(PRESET_CONFIGS["default"])


def config_for_camera_metadata(
    camera_metadata: dict[str, Any] | None,
    explicit_config: str | Path | None = None,
    robot_metadata: dict[str, Any] | None = None,
) -> str | Path:
    """Select the matching Eye-in-Hand YAML config for a recorded or live session."""
    if explicit_config:
        return explicit_config
    preset = str((robot_metadata or {}).get("preset", "")).strip().lower()
    if preset in PRESET_CONFIGS:
        return config_path(PRESET_CONFIGS[preset])
    meta = camera_metadata or {}
    backend = str(meta.get("backend", "")).strip().lower()
    name = str(meta.get("name", "")).lower()
    if backend == "orbbec" or "orbbec" in name or (meta.get("width"), meta.get("height")) == (1280, 800):
        return config_path(PRESET_CONFIGS["orbbec"])
    return config_path(PRESET_CONFIGS["default"])


def _tool_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    robot_section = cfg.get("robot", {})
    return robot_section.get("tool") or robot_section.get("gripper") or {}


def solve_session(
    session: CalibrationSession,
    cfg: dict[str, Any],
    depth_state: dict[str, Any] | None = None,
    tcp_offset_override_mm_rad: list[float] | None = None,
) -> dict[str, Any]:
    session.last_solution = None
    model = CameraModel(session.camera_metadata or {})
    tcp_offset = (
        tcp_offset_override_mm_rad
        if tcp_offset_override_mm_rad is not None
        else (session.robot_metadata or {}).get("tcp_offset_mm_rad", [0.0] * 6)
    )
    checks = cfg.get("checks", {})
    marker = cfg.get("marker", {})
    cam_cfg = cfg.get("camera", {})
    tool_cfg = _tool_cfg(cfg)
    table_z_m = float(checks.get("table_z_m", tool_cfg.get("table_z_m", 0.0)))
    board_thickness_m = float(checks.get("board_thickness_m", marker.get("thickness_m", 0.0)))

    solution = solve_hand_eye_all(
        session.samples,
        model,
        float(marker["side_m"]),
        float(checks["min_rotation_deg"]),
        tcp_offset_mm_rad=tcp_offset,
    )
    solution["board_check"] = evaluate_board_geometry(
        solution["X_base_board"],
        table_z_m=table_z_m,
        board_thickness_m=board_thickness_m,
        max_tilt_deg=float(checks.get("board_max_tilt_deg", 12.0)),
        height_tolerance_m=float(checks.get("board_height_tolerance_m", 0.04)),
    )
    if depth_state is None:
        depth_state = next(
            (s for s in reversed(session.samples) if s.get("enabled", True) and s.get("points_cam") is not None),
            None,
        )
    if depth_state is not None and depth_state.get("points_cam") is not None:
        solution["depth_check"] = evaluate_depth_consistency(
            depth_state["points_cam"],
            depth_state["X_base_flange"],
            solution["X_flange_cam"],
            solution["X_base_board"],
            table_z_m=table_z_m,
            board_thickness_m=board_thickness_m,
            board_side_m=float(marker.get("side_m", 0.15)),
            min_depth_m=float(cam_cfg.get("min_depth_m", 0.07)),
            max_depth_m=float(cam_cfg.get("max_depth_m", 1.20)),
        )
    session.last_solution = solution
    return solution


class EyeInHand:
    def __init__(
        self,
        config: str | Path | None = None,
        sim: bool = False,
        output: str | Path = "runs/eye_in_hand",
        profile: str | None = None,
        model: str | None = None,
        ip: str | None = None,
        backend: str | None = None,
        preset: str | None = None,
        serial: str | None = None,
    ) -> None:
        req_backend = str(backend or "").strip().lower()
        self.sim = bool(sim or req_backend == "sim")
        self._explicit_backend = req_backend if req_backend in ("realsense", "orbbec") else None

        if config is not None and str(config).strip():
            self.config_path: str | Path = config
        elif self.sim and not preset:
            self.config_path = config_path(PRESET_CONFIGS["default"])
        else:
            self.config_path = resolve_preset_config(preset=preset, backend=self._explicit_backend)

        resolved_cfg_path = Path(config_path(self.config_path)).resolve()
        self.preset: str | None = next(
            (k for k, rel in PRESET_CONFIGS.items() if config_path(rel).resolve() == resolved_cfg_path),
            None,
        )

        self.cfg = load_yaml(self.config_path)
        resolved_ip = str(ip or os.environ.get("XARM_IP", "")).strip()
        if resolved_ip:
            self.cfg.setdefault("robot", {})["ip"] = resolved_ip
        if serial is not None and str(serial).strip():
            self.cfg.setdefault("camera", {})["serial"] = str(serial).strip()
        if self._explicit_backend:
            self.cfg.setdefault("camera", {})["backend"] = self._explicit_backend

        self.planner = WaypointPlanner(default_cfg_path=str(self.config_path))
        selected_profile = profile or model
        if selected_profile:
            self.planner.select_model(selected_profile)
        self.session = CalibrationSession(output)
        self.camera: Any = None
        self.robot: Any = None
        self.detector: MarkerDetector | None = None
        self.frame: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None
        self.evaluation: dict[str, Any] | None = None
        self.index = -1
        self.commands: Queue[tuple[str, Any]] = Queue()
        self._resolved_backend: str | None = None

    def _resolve_backend(self) -> str:
        if self.sim:
            return "sim"
        if self._resolved_backend is not None:
            return self._resolved_backend
        if self._explicit_backend is not None:
            self._resolved_backend = self._explicit_backend
            return self._resolved_backend
        cam_cfg = self.cfg.get("camera", {})
        configured = str(cam_cfg.get("backend") or cam_cfg.get("kind") or "").strip().lower()
        if configured in ("realsense", "orbbec"):
            self._resolved_backend = configured
        else:
            self._resolved_backend = detect_usb_camera_backend()
        return self._resolved_backend

    @property
    def resources(self) -> list[str]:
        if self.sim:
            return []
        backend = self._resolve_backend()
        return [
            resource(backend, self.cfg.get("camera", {}).get("serial")),
            resource("robot", self.cfg.get("robot", {}).get("ip")),
        ]

    def open(self, context=None, persist: bool = True) -> None:
        backend = self._resolve_backend()
        if self.sim:
            from camera_tool.devices.sim import SimulatedRgbdCamera, SimulatedRobotController

            self.camera = SimulatedRgbdCamera(self.cfg["camera"], self.cfg["marker"])
        elif backend == "orbbec":
            from camera_tool.devices.orbbec import OrbbecCamera

            cam_cfg = dict(self.cfg["camera"])
            if str(cam_cfg.get("backend", "")).lower() != "orbbec":
                if (cam_cfg.get("width"), cam_cfg.get("height")) == (1280, 720):
                    cam_cfg["width"], cam_cfg["height"] = 1280, 800
            self.camera = OrbbecCamera(cam_cfg)
        else:
            from camera_tool.devices.realsense import RealSenseCamera

            self.camera = RealSenseCamera(self.cfg["camera"])

        metadata = self.camera.start()
        if not self.sim and not str(self.cfg.get("robot", {}).get("ip", "")).strip():
            raise CalibrationError("未指定机械臂 IP，请通过 --robot-ip、环境变量 XARM_IP 或配置文件设置")

        tool_cfg = _tool_cfg(self.cfg)
        if self.sim:
            from camera_tool.devices.sim import SimulatedRobotController

            self.robot = SimulatedRobotController(self.planner.robot_cfg, tool_cfg)
        else:
            from camera_tool.devices.robot import XArmController

            self.robot = XArmController(self.cfg["robot"]["ip"], self.planner.robot_cfg, tool_cfg)

        robot_metadata = self.robot.connect()
        if context and not self.sim:
            self.robot.cancel_check = context.check
        robot_metadata["profile"] = self.planner.current_model
        robot_metadata["model"] = self.planner.current_model
        robot_metadata["simulated"] = self.sim
        if self.preset:
            robot_metadata["preset"] = self.preset
        self.detector = MarkerDetector(self.cfg["marker"], CameraModel(metadata))
        self.session.start_new_session(metadata, robot_metadata, persist=persist and not self.sim)

    def close(self, failed: bool = False) -> None:
        try:
            if self.robot:
                if failed:
                    self.robot.stop()
                self.robot.close()
        finally:
            if self.camera:
                self.camera.stop()

    def poll(self, allow_unready: bool = False) -> dict[str, Any]:
        before = self.robot.read_state()
        if before.get("error_code") or (
            not allow_unready and (before.get("warn_code") or before["state"] not in (0, 1, 2))
        ):
            raise CalibrationError(
                f"机械臂未就绪或异常，停止任务：状态 {before['state']}（正式标定前请先在控制器端使能机械臂至状态 0/2）"
            )
        kwargs: dict[str, Any] = {"compute_pointcloud": True}
        if self.sim:
            kwargs["x_base_flange"] = before["X_base_flange"]
        frame = self.camera.read(**kwargs)
        after = self.robot.read_state()
        if after.get("error_code") or (
            not allow_unready and (after.get("warn_code") or after["state"] not in (0, 1, 2))
        ):
            raise CalibrationError(
                f"机械臂曝光后状态异常，停止任务：状态 {after['state']}（正式标定前请先使能机械臂）"
            )
        mm, deg = pose_error(before["X_base_flange"], after["X_base_flange"])
        stationary = not before["is_moving"] and not after["is_moving"] and mm <= 0.3 and deg <= 0.1
        self.frame, self.state = frame, after
        assert self.detector is not None
        self.evaluation = self.detector.evaluate(frame["color_bgr"])
        if not stationary:
            self.evaluation = {**self.evaluation, "passed": False, "reason": "曝光期间机械臂未停稳"}
        return {
            "frame": frame,
            "evaluation": self.evaluation,
            "state": after,
            "annotated": self.detector.draw_overlay(frame["color_bgr"], self.evaluation),
        }

    def capture(self, name: str = "manual") -> dict[str, Any]:
        self.poll()
        assert self.evaluation is not None and self.state is not None and self.frame is not None
        if not self.evaluation["passed"] or not self.state["safety"].get("safe", False):
            raise CalibrationError(self.evaluation.get("reason") or self.state["safety"].get("message"))
        self.session.last_solution = None
        return self.session.add_sample(
            self.frame["color_bgr"],
            self.evaluation,
            self.state,
            points_cam=self.frame.get("points_cam"),
            waypoint_name=name,
        )

    def move_next(self) -> dict[str, Any]:
        if not self.planner.waypoints:
            raise CalibrationError("当前构型配置没有预设点位")
        index = (self.index + 1) % len(self.planner.waypoints)
        waypoint = self.planner.waypoints[index]
        self.robot.move_to_flange_pose(waypoint["X_base_flange"])
        self.index = index
        return waypoint

    def recenter(self) -> list[float]:
        self.poll()
        assert self.evaluation is not None and self.state is not None
        if not self.evaluation["passed"]:
            raise CalibrationError("标定板尚未通过质量检查")
        nominal = pose_from_mm_deg(
            self.cfg.get("camera", {}).get("nominal_flange_cam_mm_deg", [45.0, 0.0, 62.0, 0.0, 0.0, 0.0])
        )
        solution = self.session.last_solution
        board = (
            self.state["X_base_flange"]
            @ (solution["X_flange_cam"] if solution else nominal)
            @ self.evaluation["pose_cam_board"]
        )
        checks = self.cfg.get("checks", {})
        tool_cfg = _tool_cfg(self.cfg)
        marker_cfg = self.cfg.get("marker", {})
        expected_board_z = float(checks.get("table_z_m", tool_cfg.get("table_z_m", 0.0))) + float(
            checks.get("board_thickness_m", marker_cfg.get("thickness_m", 0.0))
        )
        target = [float(board[0, 3]), float(board[1, 3]), expected_board_z]
        self.planner.recenter_waypoints(target)
        self.index = -1
        return target

    def solve(self) -> dict[str, Any]:
        depth = (
            {**self.state, "points_cam": self.frame.get("points_cam")}
            if self.state and self.frame and not self.state["is_moving"]
            else None
        )
        return solve_session(self.session, self.cfg, depth)

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "eye-in-hand",
            "samples": [dict(s) for s in self.session.samples],
            "session": self.session,
            "solution": self.session.last_solution,
            "index": self.index,
            "waypoints": self.planner.waypoints,
            "metadata": self.session.camera_metadata,
            "directory": str(self.session.session_dir or "仿真会话"),
        }

    def submit(self, command: str, value: Any = None) -> None:
        self.commands.put((command, value))

    def run(
        self,
        context,
        check: bool = False,
        automatic: bool = False,
        result: str | Path = "result/eye_in_hand.yaml",
    ) -> dict[str, Any]:
        failed = True
        label = {
            "sim": "仿真 RGB-D",
            "orbbec": "Orbbec RGB-D",
            "realsense": "RealSense RGB-D",
        }.get(self._resolve_backend(), "RGB-D")
        with DEVICES.lease(self.resources, f"{label} 手眼标定"):
            try:
                context.check()
                context.emit("status", f"连接 {label} 与机械臂")
                self.open(context, persist=not check)
                context.emit("connected", self.snapshot())
                context.frame("calibration", self.poll(allow_unready=check))
                if check:
                    failed = False
                    cam_meta = self.session.camera_metadata or {}
                    rob_meta = self.session.robot_metadata or {}
                    assert self.state is not None
                    return {
                        "motion_commands_sent": False,
                        "camera": {
                            "name": cam_meta.get("name"),
                            "serial": cam_meta.get("serial"),
                            "actual_stream": cam_meta.get("actual_stream"),
                            "camera_frame": cam_meta.get("camera_frame"),
                            "depth_aligned_to_color": cam_meta.get("depth_aligned_to_color"),
                        },
                        "robot": {
                            "ip": rob_meta.get("ip"),
                            "dof": rob_meta.get("dof"),
                            "firmware": rob_meta.get("firmware"),
                            "state": self.state["state"],
                            "ready_for_motion": self.state["state"] in (0, 2) and not self.state.get("warn_code"),
                            "tcp_offset_mm_rad": rob_meta.get("tcp_offset_mm_rad"),
                            "flange_mm_deg": self.state.get("flange_mm_deg"),
                            "table_clearance_mm": self.state.get("safety", {}).get("table_clearance_mm"),
                            "safety_message": self.state.get("safety", {}).get("message"),
                        },
                        "evaluation": self.evaluation,
                    }
                if automatic:
                    if not self.sim and self.evaluation and self.evaluation.get("passed"):
                        target = self.recenter()
                        context.emit(
                            "status",
                            f"已根据当前标定板位置重定位点位至 ({target[0]*1000:.1f}, {target[1]*1000:.1f}, {target[2]*1000:.1f}) mm",
                        )
                    self._automatic(context)
                    context.check()
                    assert self.session.last_solution is not None
                    output = self.session.export_result(self.session.last_solution, result, cfg=self.cfg)
                    failed = False
                    return {"output": str(output), **self.snapshot()}
                while True:
                    context.check()
                    try:
                        command, value = self.commands.get_nowait()
                    except Empty:
                        command = None
                    if command:
                        context.emit("busy", True)
                        try:
                            if command == "auto":
                                self._automatic(context)
                            elif command == "capture":
                                self.capture()
                            elif command == "next":
                                self.move_next()
                            elif command == "solve":
                                self.solve()
                            elif command == "toggle":
                                row, enabled = value
                                self.session.samples[row]["enabled"] = enabled
                                self.session.last_solution = None
                                self.session._save_samples_index()
                            elif command == "clear":
                                self.session.samples.clear()
                                self.index = -1
                                self.session.last_solution = None
                                self.session._save_samples_index()
                            elif command == "export":
                                if self.session.last_solution is None:
                                    raise CalibrationError("样本变化后须重新求解")
                                output = self.session.export_result(
                                    self.session.last_solution, value or result, cfg=self.cfg
                                )
                                context.emit("status", f"已导出 {output}")
                            elif command == "recenter":
                                target = self.recenter()
                                context.emit(
                                    "status",
                                    f"已重定位点位至 ({target[0]*1000:.1f}, {target[1]*1000:.1f}, {target[2]*1000:.1f}) mm",
                                )
                        except CalibrationError as exc:
                            if command in ("next", "auto"):
                                raise
                            context.emit("status", str(exc))
                        context.emit("snapshot", self.snapshot())
                        context.emit("busy", False)
                    context.frame("calibration", self.poll())
                    context.sleep(0.04)
            finally:
                self.close(failed and not check)

    def _automatic(self, context) -> None:
        self.index = -1
        total = len(self.planner.waypoints)
        for step_num in range(1, total + 1):
            context.check()
            next_idx = (self.index + 1) % total
            wp_name = self.planner.waypoints[next_idx]["name"]
            context.emit("status", f"[{step_num}/{total}] 前往点位 {wp_name}")
            try:
                waypoint = self.move_next()
            except RejectedSample as exc:
                self.index = next_idx
                context.emit("status", f"[{step_num}/{total}] 跳过不可达点位 {wp_name}：{exc}")
                continue
            context.check()
            try:
                sample = self.capture(waypoint["name"])
                context.emit(
                    "status",
                    f"[{step_num}/{total}] 已采集 #{sample['id']} {waypoint['name']} (PnP RMS={sample['pnp_rms_px']:.3f}px)",
                )
            except CalibrationError as exc:
                context.emit("status", f"[{step_num}/{total}] 跳过样本 {waypoint['name']}：{exc}")
            context.frame("calibration", self.poll())
            context.emit("snapshot", self.snapshot())
        context.check()
        context.emit("status", "采集结束，正在求解")
        self.solve()
        context.check()
