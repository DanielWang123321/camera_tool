"""Waypoint loader, Look-At target re-centering, and kinematic profile registry."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from camera_tool.core import (
    CalibrationError,
    check_tool_safety,
    load_yaml,
    make_transform,
    pose_from_mm_deg,
    pose_to_mm_deg,
    valid_transform,
)

PROFILE_ALIASES: dict[str, str] = {
    "xarm6": "arm_6dof",
    "xarm5": "arm_5dof",
    "xarm7": "arm_7dof",
    "lite6": "arm_compact_6dof",
    "uf850": "arm_long_6dof",
}


class WaypointPlanner:
    """Manages preset flange waypoints and workspace kinematic profiles."""

    def __init__(
        self,
        default_cfg_path: str = "configs/eye_in_hand/default.yaml",
        profiles_cfg_path: str = "configs/eye_in_hand/waypoints/profiles.yaml",
    ) -> None:
        self.default_cfg = load_yaml(default_cfg_path)
        self.profiles_cfg = load_yaml(profiles_cfg_path)
        self.models: dict[str, dict[str, Any]] = self.profiles_cfg.get("profiles") or self.profiles_cfg["models"]
        robot_section = self.default_cfg.get("robot", {})
        default_key = str(robot_section.get("default_profile") or robot_section.get("default_model", "arm_6dof"))
        self.current_model = PROFILE_ALIASES.get(default_key, default_key)
        self.robot_cfg: dict[str, Any] = {}
        self.waypoints: list[dict[str, Any]] = []
        self.target_center_m = np.array([0.40, 0.0, 0.0], dtype=float)
        self.select_model(self.current_model)

    def _tool_cfg(self) -> dict[str, Any]:
        robot_section = self.default_cfg.get("robot", {})
        return robot_section.get("tool") or robot_section.get("gripper") or {}

    def available_models(self) -> list[dict[str, Any]]:
        return [{"key": k, **v} for k, v in self.models.items()]

    def select_model(self, model_key: str, preset_file: str | None = None) -> dict[str, Any]:
        resolved_key = PROFILE_ALIASES.get(model_key, model_key)
        if resolved_key not in self.models:
            raise CalibrationError(f"未知的机械臂构型配置: {model_key}")
        self.current_model = resolved_key
        profile = self.models[resolved_key]
        robot_section = self.default_cfg.get("robot", {})
        default_key = str(robot_section.get("default_profile") or robot_section.get("default_model", "arm_6dof"))
        configured_preset = preset_file
        if configured_preset is None and resolved_key == PROFILE_ALIASES.get(default_key, default_key):
            configured_preset = robot_section.get("preset_file")
        if profile.get("supported", False) or configured_preset:
            self.robot_cfg = load_yaml(configured_preset or profile["preset_file"])
            self.target_center_m = np.asarray(
                self.robot_cfg.get("workspace", {}).get("target_center_m", profile["default_board_center_m"]),
                dtype=float,
            )
            self.waypoints = self._parse_and_validate_waypoints(self.robot_cfg["eye_in_hand_waypoints"])
        else:
            self.target_center_m = np.asarray(profile["default_board_center_m"], dtype=float)
            self.robot_cfg = {
                "profile": resolved_key,
                "dof": profile["dof"],
                "device_type": profile.get("device_type", profile["dof"]),
                "workspace": {
                    "flange_min_m": profile["workspace_min_m"],
                    "flange_max_m": profile["workspace_max_m"],
                    "target_center_m": profile["default_board_center_m"],
                },
                "eye_in_hand_waypoints": [],
            }
            self.waypoints = []
        return profile

    def _parse_and_validate_waypoints(self, raw_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_cfg = self._tool_cfg()
        ws = self.robot_cfg["workspace"]
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)
        parsed: list[dict[str, Any]] = []
        for item in raw_list:
            pose_mat = pose_from_mm_deg(item["flange_pose"])
            safety = check_tool_safety(pose_mat, tool_cfg, ws_min, ws_max)
            parsed.append(
                {
                    "id": int(item["id"]),
                    "name": str(item["name"]),
                    "role": str(item.get("role", "train")),
                    "flange_pose_mm_deg": list(item["flange_pose"]),
                    "X_base_flange": pose_mat,
                    "safety": safety,
                }
            )
        return parsed

    def recenter_waypoints(
        self,
        new_board_center_m: np.ndarray | list[float],
        yaw_offset_deg: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Translate and rotate the preset waypoint constellation around a newly observed board center."""
        if not self.waypoints:
            raise CalibrationError(f"当前构型 {self.current_model} 尚无预设点位可供重定位")

        old_center = np.asarray(self.robot_cfg["workspace"]["target_center_m"], dtype=float)
        new_center = np.asarray(new_board_center_m, dtype=float).reshape(3)
        rot_z = Rotation.from_euler("z", yaw_offset_deg, degrees=True).as_matrix()

        tool_cfg = self._tool_cfg()
        ws = self.robot_cfg["workspace"]
        ws_min = np.asarray(ws["flange_min_m"], dtype=float)
        ws_max = np.asarray(ws["flange_max_m"], dtype=float)

        raw_list = self.robot_cfg["eye_in_hand_waypoints"]
        updated: list[dict[str, Any]] = []
        for item in raw_list:
            orig_mat = pose_from_mm_deg(item["flange_pose"])
            rel_pos = orig_mat[:3, 3] - old_center
            new_pos = new_center + rot_z @ rel_pos
            new_rot = rot_z @ orig_mat[:3, :3]
            new_mat = valid_transform(make_transform(new_rot, new_pos))
            new_mat[:2, 3] = np.clip(new_mat[:2, 3], ws_min[:2] + 0.01, ws_max[:2] - 0.01)
            safety = check_tool_safety(new_mat, tool_cfg, ws_min, ws_max)
            updated.append(
                {
                    "id": int(item["id"]),
                    "name": str(item["name"]),
                    "role": str(item.get("role", "train")),
                    "flange_pose_mm_deg": pose_to_mm_deg(new_mat),
                    "X_base_flange": new_mat,
                    "safety": safety,
                }
            )
        self.target_center_m = new_center
        self.waypoints = updated
        return updated
