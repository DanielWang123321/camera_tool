"""Synthetic RGB-D camera and robot simulator for offline preview, UI testing, and verification."""
from __future__ import annotations

import time
from typing import Any, Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_tool.core import (
    CameraModel,
    check_tool_safety,
    inv_transform,
    make_transform,
    marker_object_points,
    pose_from_mm_deg,
    pose_to_mm_deg,
    tcp_from_flange,
    transform_points,
    valid_transform,
)


class SimulatedRgbdCamera:
    """Synthetic RGB-D camera rendering an ArUco board and planar depth map from virtual robot poses."""

    def __init__(self, camera_cfg: dict[str, Any], marker_cfg: dict[str, Any] | None = None) -> None:
        self.cfg = camera_cfg
        self.marker_cfg = marker_cfg or {"dictionary": "DICT_ARUCO_ORIGINAL", "id": 398, "side_m": 0.15}
        self.width = int(camera_cfg.get("width", 1280))
        self.height = int(camera_cfg.get("height", 720))
        self.fps = int(camera_cfg.get("fps", 30))
        self.connected = False
        self._frame_number = 0
        self._pointcloud_rays: dict[int, np.ndarray] = {}

        nominal = np.asarray(
            camera_cfg.get("nominal_flange_cam_mm_deg", [45.0, 0.0, 62.0, 0.0, 0.0, 0.0]),
            dtype=float,
        )
        gt_vec = nominal + np.array([3.0, -12.0, 3.0, 0.8, 2.0, -1.5], dtype=float)
        self.gt_flange_cam = pose_from_mm_deg(gt_vec)

        board_center = np.asarray(camera_cfg.get("board_center_m", [0.400, 0.000, 0.000]), dtype=float)
        board_rot = Rotation.from_euler("xyz", [0.0, 0.0, 2.5], degrees=True).as_matrix()
        self.gt_base_board = make_transform(board_rot, board_center)

        fx = 675.0 * (self.width / 1280.0)
        fy = fx
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.metadata: dict[str, Any] = {
            "name": "Simulated RGB-D Camera",
            "serial": "SIM-RGBD-001",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "actual_stream": [self.width, self.height, self.fps],
            "depth_scale": 0.0001,
            "sdk_depth_unit_m": 0.0001,
            "depth_scale_correction": 1.0,
            "model": "brown_conrady",
            "K": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        self._marker_tex = self._build_marker_texture()

    def _build_marker_texture(self) -> np.ndarray:
        dict_name = str(self.marker_cfg.get("dictionary", "DICT_ARUCO_ORIGINAL"))
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        marker_id = int(self.marker_cfg.get("id", 398))
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, 400)
        canvas = np.full((500, 500), 248, dtype=np.uint8)
        canvas[50:450, 50:450] = marker_img
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    def start(self) -> dict[str, Any]:
        self.connected = True
        self._frame_number = 0
        self._pointcloud_rays.clear()
        return self.metadata

    def read_frame(
        self,
        x_base_flange: np.ndarray | None = None,
        compute_pointcloud: bool = False,
    ) -> dict[str, Any]:
        if x_base_flange is None:
            x_base_flange = make_transform(
                Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix(),
                [0.40, 0.0, 0.355],
            )
        self._frame_number += 1
        x_base_cam = valid_transform(x_base_flange) @ self.gt_flange_cam
        x_cam_board = inv_transform(x_base_cam) @ self.gt_base_board

        bg = np.full((self.height, self.width, 3), (62, 66, 70), dtype=np.uint8)
        side_m = float(self.marker_cfg.get("side_m", 0.15))
        outer_pts_3d = marker_object_points(side_m * 1.25)
        pts_cam = transform_points(x_cam_board, outer_pts_3d)

        k = np.asarray(self.metadata["K"], dtype=float)
        dist = np.asarray(self.metadata["distortion"], dtype=float)
        if np.all(pts_cam[:, 2] > 0.05):
            img_pts, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), k, dist)
            dst_pts = img_pts.reshape(4, 2).astype(np.float32)
            src_pts = np.array([[0, 0], [499, 0], [499, 499], [0, 499]], dtype=np.float32)
            h_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(
                self._marker_tex,
                h_mat,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(248, 248, 248),
            )
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
            bg[mask > 0] = warped[mask > 0]

        x_cam_base = inv_transform(x_base_cam)
        n_cam = x_cam_base[:3, :3] @ np.array([0.0, 0.0, 1.0])
        board_origin = self.gt_base_board[:3, 3]
        p0_cam = transform_points(x_cam_base, board_origin.reshape(1, 3))[0]
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        ys, xs = np.mgrid[0 : self.height : 1, 0 : self.width : 1]
        rx = (xs - cx) / fx
        ry = (ys - cy) / fy
        denom = rx * n_cam[0] + ry * n_cam[1] + n_cam[2]
        num = float(p0_cam @ n_cam)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_m = np.where(np.abs(denom) > 1e-4, num / denom, 0.0).astype(np.float32)
        max_clip = max(1.5, float(self.cfg.get("max_depth_m", 0.80)) * 1.25)
        depth_m = np.clip(depth_m, 0.0, max_clip)

        points_cam = None
        colors_rgb = None
        if compute_pointcloud:
            min_d = float(self.cfg.get("min_depth_m", 0.07))
            max_d = float(self.cfg.get("max_depth_m", 0.80))
            points_cam, colors_rgb = CameraModel(self.metadata).deproject_depth_pointcloud(
                depth_m,
                bg,
                step=12,
                min_depth_m=min_d,
                max_depth_m=max_d,
                ray_cache=self._pointcloud_rays,
            )

        return {
            "timestamp_mono": time.monotonic(),
            "frame_number": self._frame_number,
            "metadata": dict(self.metadata),
            "color_bgr": bg,
            "depth_m": depth_m,
            "points_cam": points_cam,
            "colors_rgb": colors_rgb,
        }

    def close(self) -> None:
        self.connected = False

    stop = close
    read = read_frame


class SimulatedRobotController:
    """Simulated robot controller for GUI demonstration, testing, and offline preview."""

    def __init__(
        self,
        robot_cfg: dict[str, Any],
        tool_cfg: dict[str, Any],
        event_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.robot_cfg = robot_cfg
        self.tool_cfg = tool_cfg
        self.event_cb = event_cb or (lambda msg: None)
        self.connected = False
        self.dof = int(robot_cfg.get("dof", 6))
        self.tcp_offset_mm_rad = np.zeros(6, dtype=float)
        waypoints = robot_cfg.get("eye_in_hand_waypoints") or []
        if waypoints and "flange_pose" in waypoints[0]:
            self.current_flange = pose_from_mm_deg(waypoints[0]["flange_pose"])
        else:
            min_z_mm = float(tool_cfg.get("min_flange_z_m", 0.300)) * 1000.0
            self.current_flange = pose_from_mm_deg([400.0, 0.0, max(355.0, min_z_mm + 35.0), 180.0, 0.0, 0.0])

    def connect(self) -> dict[str, Any]:
        self.connected = True
        st = self.read_state()
        self.event_cb("已启用仿真机械臂控制器 (末端防撞保护已激活)")
        return {
            "ip": "SIM-LOCAL",
            "dof": self.dof,
            "device_type": self.dof,
            "firmware": "sim-2.7.103",
            "tcp_offset_mm_rad": self.tcp_offset_mm_rad.tolist(),
            **st,
        }

    def read_state(self) -> dict[str, Any]:
        now = time.monotonic()
        x_base_tcp = tcp_from_flange(self.current_flange, self.tcp_offset_mm_rad)
        safety = check_tool_safety(self.current_flange, self.tool_cfg)
        safety["safe"] = True
        safety["message"] = "安全"
        return {
            "X_base_flange": self.current_flange.copy(),
            "X_base_tcp": x_base_tcp,
            "flange_mm_deg": pose_to_mm_deg(self.current_flange),
            "tcp_mm_deg": pose_to_mm_deg(x_base_tcp),
            "joints_rad": [0.0, -0.5, -0.8, 0.0, 1.3, 0.0][: self.dof]
            + [0.0] * max(0, self.dof - 6),
            "state": 2,
            "is_moving": False,
            "read_start_mono": now,
            "read_end_mono": now,
            "safety": safety,
        }

    def move_to_flange_pose(self, target_base_flange: np.ndarray) -> dict[str, Any]:
        ws = self.robot_cfg["workspace"]
        check_tool_safety(
            target_base_flange,
            self.tool_cfg,
            ws["flange_min_m"],
            ws["flange_max_m"],
        )
        self.current_flange = valid_transform(target_base_flange)
        return self.read_state()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.connected = False
