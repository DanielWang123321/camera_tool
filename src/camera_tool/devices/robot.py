"""Unified robot controller adapters for flange-space and TCP-space calibration workflows."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from camera_tool.core import (
    CalibrationError,
    RejectedSample,
    check_tool_safety,
    flange_from_tcp,
    interpolate_poses,
    nearest_continuous_joints,
    pose_error,
    pose_from_mm_rad,
    pose_to_mm_deg,
    pose_to_mm_rad,
    tcp_from_flange,
    valid_transform,
)


class XArmController:
    """Flange-decoupled robot controller with end-effector tool envelope protection."""

    def __init__(
        self,
        ip: str,
        robot_cfg: dict[str, Any],
        tool_cfg: dict[str, Any] | None = None,
        event_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.ip = str(ip or "").strip()
        self.robot_cfg = robot_cfg
        self.tool_cfg = tool_cfg or robot_cfg.get("tool") or robot_cfg.get("gripper") or {}
        self.event_cb = event_cb or (lambda msg: None)
        self.arm: Any = None
        self.connected = False
        self.cancel_check = lambda: None
        self.moving_enabled = False
        self.dof = int(robot_cfg.get("dof", 6))
        self.tcp_offset_mm_rad = np.zeros(6, dtype=float)
        self.limits_rad = np.column_stack(
            [np.full(self.dof, -np.deg2rad(350.0)), np.full(self.dof, np.deg2rad(350.0))]
        )

    def _log(self, msg: str) -> None:
        self.event_cb(msg)

    @staticmethod
    def _unwrap(reply: tuple[int, Any], action: str) -> Any:
        code, val = reply
        if code != 0 or val is None:
            raise CalibrationError(f"Failed to {action} (code={code})")
        return val

    def connect(self) -> dict[str, Any]:
        if not self.ip:
            raise CalibrationError(
                "Robot IP is not configured; specify via --robot-ip, XARM_IP env var, or YAML config"
            )
        try:
            from xarm.core.config.x_config import XCONF
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise CalibrationError("xarm-python-sdk is not installed; cannot connect to robot") from exc

        self.arm = XArmAPI(self.ip, is_radian=True, do_not_open=True)
        self.arm.connect()
        if not self.arm.connected:
            raise CalibrationError(f"Failed to connect to robot ({self.ip})")
        time.sleep(0.25)

        self.dof = int(self.arm.axis)
        try:
            self.limits_rad = np.asarray(
                XCONF.Robot.JOINT_LIMITS[self.arm.axis][self.arm.device_type], dtype=float
            )
        except Exception:
            pass

        raw_offset = np.asarray(self.arm.tcp_offset, dtype=float)
        if raw_offset.shape != (6,) or not np.isfinite(raw_offset).all():
            raise CalibrationError("Failed to read controller TCP tool offset")
        self.tcp_offset_mm_rad = raw_offset.copy()

        err, warn = self._unwrap(self.arm.get_err_warn_code(), "read error/warning codes")
        state = self._unwrap(self.arm.get_state(), "read robot state")
        if err != 0:
            raise CalibrationError(f"Robot controller hardware error: error={err} (warn={warn}, state={state})")
        if warn != 0:
            self._log(f"Controller warning on connect (warn={warn}, state={state})")

        self.connected = True
        info = self.read_state()
        flange_xyz = [round(float(v), 1) for v in info["flange_mm_deg"][:3]]
        tcp_xyz = [round(float(v), 1) for v in info["tcp_mm_deg"][:3]]
        offset_xyz = [round(float(v), 1) for v in self.tcp_offset_mm_rad[:3]]
        if np.any(np.abs(self.tcp_offset_mm_rad) > 1e-6):
            pose_summary = f"flange_xyz={flange_xyz} mm (tcp_xyz={tcp_xyz} mm, tcp_offset_xyz={offset_xyz} mm)"
        else:
            pose_summary = f"flange_xyz={flange_xyz} mm"
        self._log(
            f"Connected to robot ({self.ip}, {self.dof}-axis); {pose_summary}, "
            f"table_clearance={info['safety']['table_clearance_mm']:.1f} mm"
        )
        if not info["safety"].get("safe", False):
            self._log(f"Warning: Current robot start pose fails preset safety guard: {info['safety']['message']}")
        return {
            "ip": self.ip,
            "dof": self.dof,
            "device_type": int(getattr(self.arm, "device_type", self.dof)),
            "firmware": str(getattr(self.arm, "version", "")),
            "tcp_offset_mm_rad": self.tcp_offset_mm_rad.tolist(),
            **info,
        }

    def read_state(self) -> dict[str, Any]:
        self.cancel_check()
        if not self.connected or self.arm is None:
            raise CalibrationError("Robot is not connected")
        t_start = time.monotonic()
        tcp_mm_rad = self._unwrap(self.arm.get_position(is_radian=True), "read TCP pose")
        joints = self._unwrap(self.arm.get_servo_angle(is_radian=True, is_real=True), "read actual joint angles")
        state = self._unwrap(self.arm.get_state(), "read robot state")
        err, warn = self._unwrap(self.arm.get_err_warn_code(), "read controller error/warning codes")
        t_end = time.monotonic()

        x_base_tcp = pose_from_mm_rad(tcp_mm_rad)
        x_base_flange = flange_from_tcp(x_base_tcp, self.tcp_offset_mm_rad)
        try:
            safety = check_tool_safety(x_base_flange, self.tool_cfg)
            safety["safe"] = True
            safety["message"] = "OK"
        except RejectedSample as exc:
            flange_z = float(x_base_flange[2, 3] * 1000.0)
            tip_z = flange_z - float(self.tool_cfg.get("length_m", 0.185)) * 1000.0
            table_z = float(self.tool_cfg.get("table_z_m", 0.0)) * 1000.0
            safety = {
                "safe": False,
                "message": str(exc),
                "flange_z_mm": flange_z,
                "tool_tip_z_mm": tip_z,
                "table_clearance_mm": tip_z - table_z,
            }
        if err or (self.moving_enabled and int(state) not in (0, 1, 2)):
            safety["safe"] = False
            safety["message"] = f"Controller state={state}, error={err}, warn={warn}"

        return {
            "X_base_flange": x_base_flange,
            "X_base_tcp": x_base_tcp,
            "flange_mm_deg": pose_to_mm_deg(x_base_flange),
            "tcp_mm_deg": pose_to_mm_deg(x_base_tcp),
            "joints_rad": [float(x) for x in joints[: self.dof]],
            "state": int(state),
            "error_code": int(err),
            "warn_code": int(warn),
            "is_moving": int(state) == 1,
            "read_start_mono": t_start,
            "read_end_mono": t_end,
            "safety": safety,
        }

    def enable_motion(self) -> None:
        if not self.connected or self.arm is None:
            raise CalibrationError("Robot is not connected")
        pre = self.read_state()
        self._assert_pre_enable_state(pre)
        if pre.get("warn_code", 0):
            self._log(f"Controller warning before motion enable (warn={pre['warn_code']})")
            if hasattr(self.arm, "clean_warn"):
                self.arm.clean_warn()

        for action, command in (
            ("enable motion", lambda: self.arm.motion_enable(enable=True)),
            ("set position mode", lambda: self.arm.set_mode(0)),
            ("enable self-collision detection", lambda: self.arm.set_self_collision_detection(True)),
            ("set ready state", lambda: self.arm.set_state(0)),
        ):
            code = command()
            if code != 0:
                raise CalibrationError(f"Failed to {action} (code={code})")

        time.sleep(0.2)
        post = self.read_state()
        self._assert_motion_state(post, allow_moving=False)
        self.moving_enabled = True

    @staticmethod
    def _assert_pre_enable_state(state: dict[str, Any]) -> None:
        if state.get("error_code", 0):
            raise CalibrationError(
                f"Controller hardware error (error={state['error_code']}, "
                f"warn={state.get('warn_code', 0)}, state={state['state']})"
            )
        if state.get("is_moving", False) or state["state"] == 1:
            raise CalibrationError("Robot is currently in motion")
        if not state["safety"].get("safe", False):
            raise CalibrationError(state["safety"].get("message", "Tool safety check failed"))

    @staticmethod
    def _assert_motion_state(state: dict[str, Any], allow_moving: bool) -> None:
        allowed = (0, 1, 2) if allow_moving else (0, 2)
        if state.get("error_code", 0) or state["state"] not in allowed:
            raise CalibrationError(
                f"Controller not ready for motion (state={state['state']}, error={state.get('error_code', 0)})"
            )
        if not state["safety"].get("safe", False):
            raise CalibrationError(
                f"Runtime robot pose violated safety guard: {state['safety'].get('message', 'Tool safety check failed')}"
            )

    def _format_start_pose_error(
        self,
        cur: dict[str, Any],
        reason: str,
        ws_min: np.ndarray,
        ws_max: np.ndarray,
    ) -> str:
        flange_pose = cur.get("flange_mm_deg") or pose_to_mm_deg(cur["X_base_flange"])
        tcp_pose = cur.get("tcp_mm_deg") or pose_to_mm_deg(
            tcp_from_flange(cur["X_base_flange"], self.tcp_offset_mm_rad)
        )
        flange_xyz = [round(float(v), 1) for v in flange_pose[:3]]
        tcp_xyz = [round(float(v), 1) for v in tcp_pose[:3]]
        offset_xyz = [round(float(v), 1) for v in self.tcp_offset_mm_rad[:3]]
        has_tcp_offset = bool(np.any(np.abs(self.tcp_offset_mm_rad) > 1e-6))

        pose_detail = f"current flange XYZ={flange_xyz} mm"
        if has_tcp_offset:
            pose_detail += f", TCP XYZ={tcp_xyz} mm, controller TCP offset XYZ={offset_xyz} mm"

        ws_min_mm = np.round(ws_min * 1000.0, 0).astype(int).tolist()
        ws_max_mm = np.round(ws_max * 1000.0, 0).astype(int).tolist()
        min_flange_z_mm = max(float(self.tool_cfg.get("min_flange_z_m", 0.300)), float(ws_min[2])) * 1000.0
        flange_z_mm = float(flange_pose[2])

        if flange_z_mm < min_flange_z_mm - 1e-3:
            lift_mm = min_flange_z_mm - flange_z_mm
            z_hint = f"flange Z >= {min_flange_z_mm:.1f} mm; lift +Z by >= {lift_mm:.1f} mm"
            if has_tcp_offset:
                z_hint += f", TCP Z >= {float(tcp_pose[2]) + lift_mm:.1f} mm at current orientation"
        else:
            z_hint = f"flange Z >= {min_flange_z_mm:.1f} mm"

        return (
            f"Current robot start pose violates safety guard before motion: {reason} ({pose_detail}). "
            f"Manually jog/raise the robot into the safe workspace {ws_min_mm} ~ {ws_max_mm} mm ({z_hint}) "
            "before starting calibration."
        )

    def assert_start_pose_safe(self, cur: dict[str, Any] | None = None) -> dict[str, Any]:
        """Verify that the robot's current pose and controller state allow starting calibration motion."""
        if not self.connected or self.arm is None:
            raise CalibrationError("Robot is not connected")
        ws = self.robot_cfg["workspace"]
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)
        state = self.read_state() if cur is None else cur
        if state.get("error_code", 0):
            raise CalibrationError(
                f"Controller hardware error (error={state['error_code']}, "
                f"warn={state.get('warn_code', 0)}, state={state['state']})"
            )
        if state.get("is_moving", False) or state["state"] == 1:
            raise CalibrationError("Robot is currently in motion")
        if self.moving_enabled and state["state"] not in (0, 2):
            raise CalibrationError(
                f"Controller not ready for motion (state={state['state']}, error={state.get('error_code', 0)})"
            )
        if not state["safety"].get("safe", False):
            reason = state["safety"].get("message", "Tool safety check failed")
            raise CalibrationError(self._format_start_pose_error(state, reason, ws_min, ws_max))
        try:
            check_tool_safety(state["X_base_flange"], self.tool_cfg, ws_min, ws_max)
        except RejectedSample as exc:
            raise CalibrationError(self._format_start_pose_error(state, str(exc), ws_min, ws_max)) from exc
        return state

    def plan_flange_pose(self, target_base_flange: np.ndarray) -> list[np.ndarray]:
        """Validate IK and FK along each joint segment without enabling motion."""
        if not self.connected or self.arm is None:
            raise CalibrationError("Robot is not connected")
        target_flange = valid_transform(target_base_flange)
        ws = self.robot_cfg["workspace"]
        motion = self.robot_cfg.get("motion", {})
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)

        cur = self.assert_start_pose_safe()
        try:
            check_tool_safety(target_flange, self.tool_cfg, ws_min, ws_max)
        except RejectedSample as exc:
            raise RejectedSample(f"Target waypoint rejected: {exc}") from exc

        q_cur = np.asarray(cur["joints_rad"], dtype=float)
        margin_rad = np.deg2rad(float(motion.get("joint_margin_deg", 8.0)))
        max_step_deg = float(motion.get("joint_step_deg", 8.0))

        steps = interpolate_poses(
            cur["X_base_flange"],
            target_flange,
            float(motion.get("translation_step_m", 0.015)),
            float(motion.get("rotation_step_deg", 5.0)),
        )
        joint_path: list[np.ndarray] = []
        q_ref = q_cur.copy()
        for step_flange in steps:
            try:
                check_tool_safety(step_flange, self.tool_cfg, ws_min, ws_max)
            except RejectedSample as exc:
                raise RejectedSample(f"Cartesian path interpolation violated safety guard: {exc}") from exc
            step_tcp = tcp_from_flange(step_flange, self.tcp_offset_mm_rad)
            code, raw_q = self.arm.get_inverse_kinematics(
                pose_to_mm_rad(step_tcp),
                input_is_radian=True,
                return_is_radian=True,
                limited=False,
                ref_angles=q_ref.tolist(),
            )
            if code != 0 or raw_q is None:
                raise RejectedSample(f"Target pose unreachable by inverse kinematics (code={code})")
            q_next = nearest_continuous_joints(np.asarray(raw_q), q_ref, self.limits_rad, margin_rad)
            if np.rad2deg(np.max(np.abs(q_next - q_ref))) > max_step_deg:
                raise RejectedSample("IK joint step exceeds threshold or approaches singularity; motion aborted")
            subdivisions = max(
                1,
                int(np.ceil(np.rad2deg(np.max(np.abs(q_next - q_ref))) / float(motion.get("path_sample_deg", 1.0)))),
            )
            for fraction in np.linspace(0.0, 1.0, subdivisions + 1)[1:]:
                joints = q_ref + fraction * (q_next - q_ref)
                tcp_pose = self._unwrap(
                    self.arm.get_forward_kinematics(joints.tolist(), input_is_radian=True, return_is_radian=True),
                    "evaluate path forward kinematics",
                )
                flange_pose = flange_from_tcp(pose_from_mm_rad(tcp_pose), self.tcp_offset_mm_rad)
                try:
                    check_tool_safety(flange_pose, self.tool_cfg, ws_min, ws_max)
                except RejectedSample as exc:
                    raise RejectedSample(f"Joint-space path interpolation violated safety guard: {exc}") from exc
            joint_path.append(q_next)
            q_ref = q_next
        return joint_path

    def move_to_flange_pose(self, target_base_flange: np.ndarray) -> dict[str, Any]:
        """Move only after validating the complete path; stop on any runtime fault."""
        joint_path = self.plan_flange_pose(target_base_flange)
        motion = self.robot_cfg.get("motion", {})
        ws = self.robot_cfg["workspace"]
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)

        if not self.moving_enabled:
            self.enable_motion()

        speed_rad = np.deg2rad(float(motion.get("speed_deg_s", 10.0)))
        acc_rad = np.deg2rad(float(motion.get("acceleration_deg_s2", 25.0)))
        timeout_s = float(motion.get("move_timeout_s", 20.0))

        try:
            for q_wp in joint_path:
                self._assert_motion_state(self.read_state(), allow_moving=False)
                code = self.arm.set_servo_angle(
                    angle=q_wp.tolist(),
                    speed=speed_rad,
                    mvacc=acc_rad,
                    is_radian=True,
                    wait=False,
                )
                if code != 0:
                    raise CalibrationError(f"Failed to submit joint motion command (code={code})")
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    st = self.read_state()
                    self._assert_motion_state(st, allow_moving=True)
                    try:
                        check_tool_safety(st["X_base_flange"], self.tool_cfg, ws_min, ws_max)
                    except RejectedSample as exc:
                        raise CalibrationError(f"Runtime robot pose violated safety guard: {exc}") from exc
                    if not st["is_moving"] and np.max(np.abs(np.asarray(st["joints_rad"]) - q_wp)) < np.deg2rad(0.2):
                        break
                    time.sleep(0.04)
                else:
                    raise CalibrationError("Robot motion timed out")
            time.sleep(float(motion.get("settle_s", 0.5)))
            final = self.read_state()
            self._assert_motion_state(final, allow_moving=False)
            position_mm, rotation_deg = pose_error(final["X_base_flange"], target_base_flange)
            if position_mm > float(motion.get("settle_position_mm", 0.3)) or rotation_deg > float(
                motion.get("settle_rotation_deg", 0.1)
            ):
                raise CalibrationError(f"Settled pose error exceeded tolerance ({position_mm:.2f} mm, {rotation_deg:.3f} deg)")
            return final
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.moving_enabled = False
        if self.arm is not None:
            try:
                self.arm.set_state(4)
            except Exception:
                pass

    def close(self) -> None:
        if self.arm is not None:
            try:
                self.arm.disconnect()
            except Exception:
                pass
        self.connected = False
        self.moving_enabled = False


class TcpCalibrationRobot:
    """TCP-space calibration motion controller with continuous IK path verification and settle gating."""

    def __init__(
        self,
        robot_config: dict[str, Any],
        config: dict[str, Any],
        event: Callable[..., None],
        arm: Any = None,
    ) -> None:
        if arm is None:
            ip = str(robot_config.get("ip", "")).strip()
            if not ip:
                raise CalibrationError(
                    "Robot IP is not configured; specify via --robot-ip, XARM_IP env var, or YAML config"
                )
            try:
                from xarm.wrapper import XArmAPI
            except ImportError as exc:
                raise CalibrationError("xarm-python-sdk is not installed; cannot connect to robot") from exc
            arm = XArmAPI(ip, is_radian=True, do_not_open=True)
        self.arm = arm
        self.config = config
        self.event = event
        self.moving_enabled = False
        self.cancel_check = lambda: None
        self.expected_tcp_offset: np.ndarray | None = None
        self.dof = int(robot_config.get("dof", 6))
        self.low = np.asarray(config["workspace_min_m"], dtype=float)
        self.high = np.asarray(config["workspace_max_m"], dtype=float)
        robot_low = np.asarray(robot_config["workspace_min"], dtype=float) / 1000.0
        robot_high = np.asarray(robot_config["workspace_max"], dtype=float) / 1000.0
        if (
            self.low.shape != (3,)
            or self.high.shape != (3,)
            or not np.isfinite([self.low, self.high]).all()
            or np.any(self.low >= self.high)
            or np.any(self.low < robot_low)
            or np.any(self.high > robot_high)
        ):
            raise CalibrationError("Calibration motion bounds must lie inside the robot workspace")

    @staticmethod
    def _value(reply: tuple[int, Any], name: str) -> Any:
        code, value = reply
        if code != 0 or value is None:
            raise CalibrationError(f"Failed to {name} (code={code})")
        return value

    @staticmethod
    def _code(code: int, name: str) -> None:
        if code != 0:
            raise CalibrationError(f"Failed to {name} (code={code})")

    def start(self) -> dict[str, Any]:
        if not self.arm.connected:
            self.arm.connect()
        if not self.arm.connected:
            raise CalibrationError("Failed to connect to robot")
        time.sleep(0.3)
        self.dof = int(self.arm.axis)
        if tuple(self.arm.version_number) < (2, 7, 103):
            raise CalibrationError("Automatic calibration requires controller firmware >= 2.7.103 for reference-joint IK")
        try:
            from xarm.core.config.x_config import XCONF

            self.limits = np.asarray(XCONF.Robot.JOINT_LIMITS[self.dof][self.arm.device_type], dtype=float)
        except ImportError:
            self.limits = np.column_stack(
                [np.full(self.dof, -np.deg2rad(350.0)), np.full(self.dof, np.deg2rad(350.0))]
            )
        except (KeyError, AttributeError) as exc:
            raise CalibrationError("SDK does not define joint limits for this robot model") from exc

        self.expected_tcp_offset = np.asarray(self.arm.tcp_offset, dtype=float)
        if self.expected_tcp_offset.shape != (6,) or not np.isfinite(self.expected_tcp_offset).all():
            raise CalibrationError("Failed to read controller TCP tool offset")
        if not np.allclose(self.arm.world_offset, np.zeros(6), atol=1e-6):
            raise CalibrationError("Controller world_offset is non-zero; readings cannot be treated directly as base frame")

        error, warn = self._value(self.arm.get_err_warn_code(), "read robot error/warning codes")
        state = self._value(self.arm.get_state(), "read robot state")
        if error:
            raise CalibrationError(f"Robot controller has error: error={error}, warn={warn}, state={state}")
        if state == 1:
            raise CalibrationError("Robot is currently moving; cannot start read-only inspection")
        if warn:
            self.event("Robot controller warning", warn=warn, state=state)

        sample = self.read()
        self.event("Robot read-only state", **sample, tcp_offset_controller=self.expected_tcp_offset.tolist())
        try:
            self.check_workspace(sample["X_base_tcp"])
        except RejectedSample as exc:
            self.event("Start pose outside calibration workspace; will move in during sampling", reason=str(exc))
        self.check_joints(sample["joints_rad"])
        fk = pose_from_mm_rad(
            self._value(
                self.arm.get_forward_kinematics(sample["joints_rad"], input_is_radian=True, return_is_radian=True),
                "read forward kinematics",
            )
        )
        mm, deg = pose_error(fk, sample["X_base_tcp"])
        if mm > 2.0 or deg > 1.0:
            raise CalibrationError("Controller FK disagrees with reported TCP pose; stopping to avoid double tool compensation")
        return {
            **sample,
            "tcp_offset_controller": self.expected_tcp_offset.tolist(),
            "firmware": self.arm.version,
            "joint_limits_rad": self.limits.tolist(),
            "self_collision_params": self.arm.self_collision_params,
        }

    def read(self) -> dict[str, Any]:
        self.cancel_check()
        if not self.arm.connected:
            raise CalibrationError("Robot connection lost")
        error, warn = self._value(self.arm.get_err_warn_code(), "read robot error/warning codes")
        state = self._value(self.arm.get_state(), "read robot state")
        if error:
            raise CalibrationError(f"Robot controller fault: error={error}, warn={warn}, state={state}")
        if self.moving_enabled and state not in (0, 1, 2):
            raise CalibrationError(f"Robot not ready during motion: state={state}, error={error}, warn={warn}")
        if self.expected_tcp_offset is not None and not np.allclose(
            self.arm.tcp_offset, self.expected_tcp_offset, atol=1e-6
        ):
            raise CalibrationError("Controller TCP offset changed during calibration")
        if not np.allclose(self.arm.world_offset, np.zeros(6), atol=1e-6):
            raise CalibrationError("Controller world_offset changed during calibration")
        before = time.monotonic()
        tcp = self._value(self.arm.get_position(is_radian=True), "read actual TCP pose")
        q = np.asarray(self._value(self.arm.get_servo_angle(is_radian=True, is_real=True), "read actual joints"))[
            : self.dof
        ]
        if q.shape != (self.dof,) or not np.isfinite(q).all():
            raise CalibrationError("Invalid actual joint angle reading")
        return {
            "X_base_tcp": pose_from_mm_rad(tcp),
            "tcp_mm_rpy_rad": tcp,
            "joints_rad": q.tolist(),
            "read_start_mono": before,
            "read_end_mono": time.monotonic(),
            "state": int(state),
            "error_code": int(error),
            "warn_code": int(warn),
        }

    def check_workspace(self, pose: np.ndarray) -> None:
        p = valid_transform(pose)[:3, 3]
        if np.any(p < self.low - 1e-6) or np.any(p > self.high + 1e-6):
            raise RejectedSample(f"TCP outside calibration workspace: {np.round(p * 1000, 1).tolist()} mm")

    def check_joints(self, q: Any) -> None:
        q_arr = np.asarray(q, dtype=float)
        margin = np.deg2rad(float(self.config["joint_margin_deg"]))
        if np.any(q_arr < self.limits[:, 0] + margin) or np.any(q_arr > self.limits[:, 1] - margin):
            raise RejectedSample("Actual or planned joint angle is too close to joint limits")

    def plan(self, target: np.ndarray) -> tuple[dict[str, Any], list[np.ndarray]]:
        state = self.read()
        self.check_workspace(target)
        q = np.asarray(state["joints_rad"], dtype=float)
        self.check_joints(q)
        path: list[np.ndarray] = []
        margin_rad = np.deg2rad(float(self.config["joint_margin_deg"]))
        for pose in interpolate_poses(
            state["X_base_tcp"],
            target,
            float(self.config["translation_step_m"]),
            float(self.config["rotation_step_deg"]),
        ):
            self.cancel_check()
            code, raw = self.arm.get_inverse_kinematics(
                pose_to_mm_rad(pose),
                input_is_radian=True,
                return_is_radian=True,
                limited=False,
                ref_angles=q.tolist(),
            )
            if code != 0 or raw is None:
                raise RejectedSample(f"Candidate pose unreachable by IK (code={code})")
            nxt = nearest_continuous_joints(np.asarray(raw)[: self.dof], q, self.limits, margin_rad)
            if np.rad2deg(np.abs(nxt - q)).max() > float(self.config["joint_step_deg"]):
                raise RejectedSample("IK joint step jumped or approached singularity; skipping candidate")
            count = max(1, int(np.ceil(np.rad2deg(np.abs(nxt - q)).max() / float(self.config["path_sample_deg"]))))
            fk = pose
            for test in np.linspace(q, nxt, count + 1):
                self.check_joints(test)
                if self._value(self.arm.is_joint_limit(test.tolist(), is_radian=True), "check joint limit"):
                    raise RejectedSample("Controller joint limit check rejected path")
                fk = pose_from_mm_rad(
                    self._value(
                        self.arm.get_forward_kinematics(test.tolist(), input_is_radian=True, return_is_radian=True),
                        "evaluate path forward kinematics",
                    )
                )
                self.check_workspace(fk)
                if self._value(self.arm.is_tcp_limit(pose_to_mm_rad(fk), is_radian=True), "check TCP limit"):
                    raise RejectedSample("Controller TCP limit check rejected path")
            mm, deg = pose_error(fk, pose)
            if mm > 1.0 or deg > 0.5:
                raise RejectedSample("Candidate IK solution disagrees with TCP forward kinematics")
            path.append(nxt)
            q = nxt
        return state, path

    def enable(self) -> None:
        pre = self.read()
        if pre["state"] == 1:
            raise CalibrationError("Robot is currently moving; cannot enable motion")
        if pre.get("warn_code"):
            self.event("Controller warning active before enable", warn=pre["warn_code"])
            if hasattr(self.arm, "clean_warn"):
                self.arm.clean_warn()

        self._code(self.arm.motion_enable(enable=True), "motion_enable")
        self._code(self.arm.set_mode(0), "set_mode(0)")
        self._code(self.arm.set_self_collision_detection(True), "set_self_collision_detection")
        self._code(self.arm.set_state(0), "set_state(0)")

        time.sleep(0.2)
        error, warn = self._value(self.arm.get_err_warn_code(), "read error/warning codes after set_state(0)")
        state = self._value(self.arm.get_state(), "read state after set_state(0)")
        if error or state not in (0, 2):
            raise CalibrationError(f"Robot not ready after set_state(0): state={state}, error={error}, warn={warn}")
        self.moving_enabled = True

    def move(self, target: np.ndarray, health: Callable[[], None]) -> float:
        initial, path = self.plan(target)
        if not self.moving_enabled:
            self.enable()
        args = {
            "is_radian": True,
            "speed": np.deg2rad(float(self.config["speed_deg_s"])),
            "mvacc": np.deg2rad(float(self.config["acceleration_deg_s2"])),
            "wait": False,
            "radius": -1,
        }
        code = 0
        for _ in (0, 1):
            for i, q in enumerate(path):
                code = self.arm.set_servo_angle(angle=q.tolist(), only_check_type=1 if i == 0 else 3, **args)
                if code != 0 or self.arm.only_check_result != 0:
                    break
            else:
                break
        else:
            raise RejectedSample(f"Controller path check failed: code={code}, result={self.arm.only_check_result}")

        actual = self.read()
        mm, deg = pose_error(initial["X_base_tcp"], actual["X_base_tcp"])
        if (
            mm > 0.5
            or deg > 0.2
            or np.max(np.abs(np.asarray(initial["joints_rad"]) - actual["joints_rad"])) > np.deg2rad(0.2)
        ):
            raise CalibrationError("Robot was moved externally during path planning")
        self.event("Executing verified path", waypoints=len(path), target=target)
        try:
            for q in path:
                health()
                self._code(self.arm.set_servo_angle(angle=q.tolist(), only_check_type=0, **args), "submit joint target")
                deadline = time.monotonic() + float(self.config["move_timeout_s"])
                while True:
                    current = self.read()
                    self.check_workspace(current["X_base_tcp"])
                    self.check_joints(current["joints_rad"])
                    if current["state"] != 1 and np.max(np.abs(np.asarray(current["joints_rad"]) - q)) < np.deg2rad(0.15):
                        break
                    if time.monotonic() >= deadline:
                        raise CalibrationError("Robot motion timed out")
                    health()
                    time.sleep(0.05)
            return self.settle(health)
        except BaseException as exc:
            self.stop()
            if isinstance(exc, RejectedSample):
                raise CalibrationError(f"Safety limit violated during motion: {exc}") from exc
            raise

    def settle(self, health: Callable[[], None]) -> float:
        deadline = time.monotonic() + float(self.config["move_timeout_s"])
        anchor, since = self.read(), time.monotonic()
        while time.monotonic() < deadline:
            health()
            current = self.read()
            mm, deg = pose_error(anchor["X_base_tcp"], current["X_base_tcp"])
            dq = np.rad2deg(np.abs(np.asarray(anchor["joints_rad"]) - current["joints_rad"])).max()
            if (
                current["state"] == 1
                or mm > float(self.config["settle_position_mm"])
                or deg > float(self.config["settle_rotation_deg"])
                or dq > 0.1
            ):
                anchor, since = current, time.monotonic()
            elif time.monotonic() - since >= float(self.config["settle_s"]):
                return time.monotonic()
            time.sleep(0.05)
        raise CalibrationError("Robot failed to settle")

    def stop(self) -> None:
        was_enabled = self.moving_enabled
        self.moving_enabled = False
        if was_enabled:
            try:
                self._code(self.arm.set_state(4), "stop robot")
            except Exception as exc:
                self.event("Stop command failed", reason=str(exc))

    def close(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
