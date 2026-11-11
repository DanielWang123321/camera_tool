"""Unified robot controller adapters for flange-space and TCP-space calibration workflows."""
from __future__ import annotations

import time
from typing import Any, Callable

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
        self.gripper_cfg = self.tool_cfg
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
            raise CalibrationError(f"{action}失败 (错误码 {code})")
        return val

    def connect(self) -> dict[str, Any]:
        if not self.ip:
            raise CalibrationError("未配置机械臂 IP，请通过 --robot-ip、环境变量 XARM_IP 或配置文件指定")
        try:
            from xarm.core.config.x_config import XCONF
            from xarm.wrapper import XArmAPI
        except ImportError as exc:
            raise CalibrationError("未安装 xarm-python-sdk，无法连接机械臂") from exc

        self.arm = XArmAPI(self.ip, is_radian=True, do_not_open=True)
        self.arm.connect()
        if not self.arm.connected:
            raise CalibrationError(f"无法连接机械臂 ({self.ip})")
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
            raise CalibrationError("无法读取控制器 TCP 工具偏移")
        self.tcp_offset_mm_rad = raw_offset.copy()

        err, warn = self._unwrap(self.arm.get_err_warn_code(), "读取错误状态")
        state = self._unwrap(self.arm.get_state(), "读取运行状态")
        if err != 0:
            raise CalibrationError(f"机械臂存在硬件错误码: {err} (警告 {warn}, 状态 {state})")

        self.connected = True
        info = self.read_state()
        self._log(
            f"已连接机械臂 ({self.ip}, {self.dof}轴)，当前法兰高度 {info['safety']['flange_z_mm']:.1f} mm，"
            f"末端距桌面 {info['safety']['table_clearance_mm']:.1f} mm"
        )
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
            raise CalibrationError("机械臂未连接")
        t_start = time.monotonic()
        tcp_mm_rad = self._unwrap(self.arm.get_position(is_radian=True), "读取 TCP 位姿")
        joints = self._unwrap(self.arm.get_servo_angle(is_radian=True, is_real=True), "读取实际关节角")
        state = self._unwrap(self.arm.get_state(), "读取机械臂状态")
        err, warn = self._unwrap(self.arm.get_err_warn_code(), "读取控制器错误状态")
        t_end = time.monotonic()

        x_base_tcp = pose_from_mm_rad(tcp_mm_rad)
        x_base_flange = flange_from_tcp(x_base_tcp, self.tcp_offset_mm_rad)
        try:
            safety = check_tool_safety(x_base_flange, self.tool_cfg)
            safety["safe"] = True
            safety["message"] = "安全"
        except RejectedSample as exc:
            flange_z = float(x_base_flange[2, 3] * 1000.0)
            tip_z = flange_z - float(self.tool_cfg.get("length_m", 0.185)) * 1000.0
            table_z = float(self.tool_cfg.get("table_z_m", 0.0)) * 1000.0
            safety = {
                "safe": False,
                "message": str(exc),
                "flange_z_mm": flange_z,
                "tool_tip_z_mm": tip_z,
                "gripper_tip_z_mm": tip_z,
                "table_clearance_mm": tip_z - table_z,
            }
        if err or int(state) not in (0, 1, 2):
            safety["safe"] = False
            safety["message"] = f"控制器状态 {state}，错误码 {err}，警告码 {warn}"

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
            raise CalibrationError("机械臂未连接")
        self._assert_motion_state(self.read_state(), allow_moving=False)
        for action, command in (
            ("启用自碰撞检测", lambda: self.arm.set_self_collision_detection(True)),
            ("使能机械臂", lambda: self.arm.motion_enable(enable=True)),
            ("设置位置控制模式", lambda: self.arm.set_mode(0)),
        ):
            code = command()
            if code != 0:
                raise CalibrationError(f"{action}失败 (返回码 {code})")
        # Never clear errors or reset a paused/stopped controller automatically.
        self._assert_motion_state(self.read_state(), allow_moving=False)
        code = self.arm.set_state(0)
        if code != 0:
            raise CalibrationError(f"准备运动失败 (返回码 {code})")
        self.moving_enabled = True
        time.sleep(0.2)

    @staticmethod
    def _assert_motion_state(state: dict[str, Any], allow_moving: bool) -> None:
        allowed = (0, 1, 2) if allow_moving else (0, 2)
        if state.get("error_code", 0) or state["state"] not in allowed:
            raise CalibrationError(f"控制器禁止运动 (状态 {state['state']}，错误码 {state.get('error_code', 0)})")
        if not state["safety"].get("safe", False):
            raise RejectedSample(state["safety"].get("message", "末端安全检查未通过"))

    def plan_flange_pose(self, target_base_flange: np.ndarray) -> list[np.ndarray]:
        """Validate IK and FK along each joint segment without enabling motion."""
        if not self.connected or self.arm is None:
            raise CalibrationError("机械臂未连接")
        target_flange = valid_transform(target_base_flange)
        ws = self.robot_cfg["workspace"]
        motion = self.robot_cfg.get("motion", {})
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)

        check_tool_safety(target_flange, self.tool_cfg, ws_min, ws_max)

        cur = self.read_state()
        self._assert_motion_state(cur, allow_moving=False)
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
            check_tool_safety(step_flange, self.tool_cfg, ws_min, ws_max)
            step_tcp = tcp_from_flange(step_flange, self.tcp_offset_mm_rad)
            code, raw_q = self.arm.get_inverse_kinematics(
                pose_to_mm_rad(step_tcp),
                input_is_radian=True,
                return_is_radian=True,
                limited=False,
                ref_angles=q_ref.tolist(),
            )
            if code != 0 or raw_q is None:
                raise RejectedSample(f"目标姿态逆运动学不可达 (返回码 {code})")
            q_next = nearest_continuous_joints(np.asarray(raw_q), q_ref, self.limits_rad, margin_rad)
            if np.rad2deg(np.max(np.abs(q_next - q_ref))) > max_step_deg:
                raise RejectedSample("逆解关节角发生跳变或接近奇异构型，已中止运动")
            subdivisions = max(
                1,
                int(np.ceil(np.rad2deg(np.max(np.abs(q_next - q_ref))) / float(motion.get("path_sample_deg", 1.0)))),
            )
            for fraction in np.linspace(0.0, 1.0, subdivisions + 1)[1:]:
                joints = q_ref + fraction * (q_next - q_ref)
                tcp_pose = self._unwrap(
                    self.arm.get_forward_kinematics(joints.tolist(), input_is_radian=True, return_is_radian=True),
                    "检查关节路径正运动学",
                )
                flange_pose = flange_from_tcp(pose_from_mm_rad(tcp_pose), self.tcp_offset_mm_rad)
                check_tool_safety(flange_pose, self.tool_cfg, ws_min, ws_max)
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
                    raise CalibrationError(f"机械臂关节运动下发失败 (返回码 {code})")
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    st = self.read_state()
                    self._assert_motion_state(st, allow_moving=True)
                    check_tool_safety(st["X_base_flange"], self.tool_cfg, ws_min, ws_max)
                    if not st["is_moving"] and np.max(np.abs(np.asarray(st["joints_rad"]) - q_wp)) < np.deg2rad(0.2):
                        break
                    time.sleep(0.04)
                else:
                    raise CalibrationError("机械臂运动超时")
            time.sleep(float(motion.get("settle_s", 0.5)))
            final = self.read_state()
            self._assert_motion_state(final, allow_moving=False)
            position_mm, rotation_deg = pose_error(final["X_base_flange"], target_base_flange)
            if position_mm > float(motion.get("settle_position_mm", 0.3)) or rotation_deg > float(
                motion.get("settle_rotation_deg", 0.1)
            ):
                raise CalibrationError(f"到位偏差超限 ({position_mm:.2f} mm, {rotation_deg:.3f}°)")
            return final
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
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
                raise CalibrationError("未配置机械臂 IP，请通过 --robot-ip、环境变量 XARM_IP 或配置文件指定")
            try:
                from xarm.wrapper import XArmAPI
            except ImportError as exc:
                raise CalibrationError("未安装 xarm-python-sdk，无法连接机械臂") from exc
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
            raise CalibrationError("标定运动范围必须在现有机械臂工作区内")

    @staticmethod
    def _value(reply: tuple[int, Any], name: str) -> Any:
        code, value = reply
        if code != 0 or value is None:
            raise CalibrationError(f"{name}失败，返回码 {code}")
        return value

    @staticmethod
    def _code(code: int, name: str) -> None:
        if code != 0:
            raise CalibrationError(f"{name}失败，返回码 {code}")

    def start(self) -> dict[str, Any]:
        if not self.arm.connected:
            self.arm.connect()
        if not self.arm.connected:
            raise CalibrationError("机械臂连接失败")
        time.sleep(0.3)
        self.dof = int(self.arm.axis)
        if tuple(self.arm.version_number) < (2, 7, 103):
            raise CalibrationError("自动标定需要支持参考关节逆解的控制器固件（至少 2.7.103）")
        try:
            from xarm.core.config.x_config import XCONF

            self.limits = np.asarray(XCONF.Robot.JOINT_LIMITS[self.dof][self.arm.device_type], dtype=float)
        except ImportError:
            self.limits = np.column_stack(
                [np.full(self.dof, -np.deg2rad(350.0)), np.full(self.dof, np.deg2rad(350.0))]
            )
        except (KeyError, AttributeError) as exc:
            raise CalibrationError("SDK 没有当前机械臂型号的关节限位") from exc

        self.expected_tcp_offset = np.asarray(self.arm.tcp_offset, dtype=float)
        if self.expected_tcp_offset.shape != (6,) or not np.isfinite(self.expected_tcp_offset).all():
            raise CalibrationError("无法读取控制器 TCP 工具偏移")
        if not np.allclose(self.arm.world_offset, np.zeros(6), atol=1e-6):
            raise CalibrationError("控制器世界坐标偏移不为零，不能把读数直接当机械臂基座坐标")
        error, warn = self._value(self.arm.get_err_warn_code(), "读取机械臂错误状态")
        state = self._value(self.arm.get_state(), "读取机械臂状态")
        if error or warn:
            raise CalibrationError(f"机械臂有错误或警告：错误 {error}，警告 {warn}，状态 {state}")
        if state not in (0, 2):
            raise CalibrationError(f"机械臂状态 {state}，需要人工恢复；只读检查不发送使能或复位命令")

        sample = self.read()
        self.event("机械臂只读状态", **sample, tcp_offset_controller=self.expected_tcp_offset.tolist())
        try:
            self.check_workspace(sample["X_base_tcp"])
        except RejectedSample as exc:
            self.event("起点不在标定工作区，采样时会先进入工作区", reason=str(exc))
        self.check_joints(sample["joints_rad"])
        fk = pose_from_mm_rad(
            self._value(
                self.arm.get_forward_kinematics(sample["joints_rad"], input_is_radian=True, return_is_radian=True),
                "读取正解",
            )
        )
        mm, deg = pose_error(fk, sample["X_base_tcp"])
        if mm > 2.0 or deg > 1.0:
            raise CalibrationError("控制器正解与实际 TCP 不一致，停止以免工具坐标重复补偿")
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
            raise CalibrationError("机械臂连接中断")
        error, warn = self._value(self.arm.get_err_warn_code(), "读取机械臂错误状态")
        state = self._value(self.arm.get_state(), "读取机械臂状态")
        if error or warn or state in (3, 4, 5):
            raise CalibrationError(f"机械臂未就绪：错误 {error}，警告 {warn}，状态 {state}")
        if self.expected_tcp_offset is not None and not np.allclose(
            self.arm.tcp_offset, self.expected_tcp_offset, atol=1e-6
        ):
            raise CalibrationError("标定期间控制器 TCP 偏移改变")
        if not np.allclose(self.arm.world_offset, np.zeros(6), atol=1e-6):
            raise CalibrationError("标定期间控制器世界坐标偏移改变")
        before = time.monotonic()
        tcp = self._value(self.arm.get_position(is_radian=True), "读取实际 TCP")
        q = np.asarray(self._value(self.arm.get_servo_angle(is_radian=True, is_real=True), "读取实际关节"))[
            : self.dof
        ]
        if q.shape != (self.dof,) or not np.isfinite(q).all():
            raise CalibrationError("实际关节数据无效")
        return {
            "X_base_tcp": pose_from_mm_rad(tcp),
            "tcp_mm_rpy_rad": tcp,
            "joints_rad": q.tolist(),
            "read_start_mono": before,
            "read_end_mono": time.monotonic(),
            "state": state,
        }

    def check_workspace(self, pose: np.ndarray) -> None:
        p = valid_transform(pose)[:3, 3]
        if np.any(p < self.low - 1e-6) or np.any(p > self.high + 1e-6):
            raise RejectedSample(f"TCP 不在标定上方工作区：{np.round(p * 1000, 1).tolist()} mm")

    def check_joints(self, q: Any) -> None:
        q_arr = np.asarray(q, dtype=float)
        margin = np.deg2rad(float(self.config["joint_margin_deg"]))
        if np.any(q_arr < self.limits[:, 0] + margin) or np.any(q_arr > self.limits[:, 1] - margin):
            raise RejectedSample("实际或规划关节接近限位")

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
                raise RejectedSample(f"候选姿态不可达，逆解返回 {code}")
            nxt = nearest_continuous_joints(np.asarray(raw)[: self.dof], q, self.limits, margin_rad)
            if np.rad2deg(np.abs(nxt - q)).max() > float(self.config["joint_step_deg"]):
                raise RejectedSample("逆解跳变或接近奇异姿态，跳过候选")
            count = max(1, int(np.ceil(np.rad2deg(np.abs(nxt - q)).max() / float(self.config["path_sample_deg"]))))
            fk = pose
            for test in np.linspace(q, nxt, count + 1):
                self.check_joints(test)
                if self._value(self.arm.is_joint_limit(test.tolist(), is_radian=True), "检查关节限位"):
                    raise RejectedSample("控制器关节限位检查拒绝路径")
                fk = pose_from_mm_rad(
                    self._value(
                        self.arm.get_forward_kinematics(test.tolist(), input_is_radian=True, return_is_radian=True),
                        "检查路径正解",
                    )
                )
                self.check_workspace(fk)
                if self._value(self.arm.is_tcp_limit(pose_to_mm_rad(fk), is_radian=True), "检查 TCP 限位"):
                    raise RejectedSample("控制器 TCP 限位检查拒绝路径")
            mm, deg = pose_error(fk, pose)
            if mm > 1.0 or deg > 0.5:
                raise RejectedSample("候选逆解与 TCP 正解不一致")
            path.append(nxt)
            q = nxt
        return state, path

    def enable(self) -> None:
        self.read()
        self.moving_enabled = True
        self._code(self.arm.motion_enable(enable=True), "使能机械臂")
        self._code(self.arm.set_mode(0), "进入位置模式")
        self._code(self.arm.set_state(0), "设置运行状态")
        self._code(self.arm.set_self_collision_detection(True), "启用自碰撞检查")

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
            raise RejectedSample(f"控制器路径检查未通过：{code}/{self.arm.only_check_result}")

        actual = self.read()
        mm, deg = pose_error(initial["X_base_tcp"], actual["X_base_tcp"])
        if (
            mm > 0.5
            or deg > 0.2
            or np.max(np.abs(np.asarray(initial["joints_rad"]) - actual["joints_rad"])) > np.deg2rad(0.2)
        ):
            raise CalibrationError("规划期间机械臂被其他程序移动")
        self.event("执行已检查路径", waypoints=len(path), target=target)
        try:
            for q in path:
                health()
                self._code(self.arm.set_servo_angle(angle=q.tolist(), only_check_type=0, **args), "运动下发")
                deadline = time.monotonic() + float(self.config["move_timeout_s"])
                while True:
                    current = self.read()
                    self.check_workspace(current["X_base_tcp"])
                    self.check_joints(current["joints_rad"])
                    if current["state"] != 1 and np.max(np.abs(np.asarray(current["joints_rad"]) - q)) < np.deg2rad(0.15):
                        break
                    if time.monotonic() >= deadline:
                        raise CalibrationError("机械臂运动超时")
                    health()
                    time.sleep(0.05)
            return self.settle(health)
        except BaseException as exc:
            self.stop()
            if isinstance(exc, RejectedSample):
                raise CalibrationError(f"运动途中违反限制：{exc}") from exc
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
        raise CalibrationError("机械臂未停稳")

    def stop(self) -> None:
        if self.moving_enabled:
            try:
                self._code(self.arm.set_state(4), "停止机械臂")
            except Exception as exc:
                self.event("停止命令失败", reason=str(exc))

    def close(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()


CalibrationRobot = TcpCalibrationRobot
