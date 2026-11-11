"""Unified pinhole / Brown-Conrady / RealSense distortion camera model and point cloud deprojection."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from camera_tool.core.geometry import make_transform, transform_points
from camera_tool.core.io import CalibrationError, RejectedSample

SUPPORTED_DISTORTION_MODELS = (
    "pinhole",
    "brown_conrady",
    "inverse_brown_conrady",
    "modified_brown_conrady",
)


def marker_object_points(side_m: float) -> np.ndarray:
    """Return 3D coordinates of the 4 planar square marker corners in the marker frame (Z=0)."""
    side = float(side_m)
    if not (0.01 <= side <= 1.0):
        raise CalibrationError(f"标定码边长超出合理范围: {side} m")
    half = side / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=float,
    )


marker_points = marker_object_points


def _normalize_distortion_model(raw: Any) -> str:
    text = str(raw if raw is not None else "brown_conrady").strip()
    if text.startswith("OBCameraDistortionModel."):
        text = text.split(".")[-1].lower()
    elif text.startswith("distortion."):
        text = text.split(".", 1)[1].lower()
    if text == "none":
        return "pinhole"
    if text == "brown_conrady_k6":
        return "brown_conrady"
    return text


class CameraModel:
    """Camera intrinsics & distortion model shared across calibration, replay, and diagnostics."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.k = np.asarray(metadata["K"], dtype=float)
        self.distortion = np.asarray(metadata.get("distortion", [0.0] * 5), dtype=float).ravel()
        self.width = int(metadata["width"])
        self.height = int(metadata["height"])

        normalized = _normalize_distortion_model(metadata.get("model", "brown_conrady"))
        if normalized not in SUPPORTED_DISTORTION_MODELS:
            raise CalibrationError(f"求解器不接受未经处理的相机模型：{metadata.get('model')}")
        if self.distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(self.distortion).all():
            raise CalibrationError("相机畸变参数无效")
        if self.distortion.size == 0:
            self.distortion = np.zeros(5, dtype=float)
        self.model = "pinhole" if np.allclose(self.distortion, 0.0) else normalized

        if (
            self.k.shape != (3, 3)
            or not np.isfinite(self.k).all()
            or self.k[0, 0] <= 0
            or self.k[1, 1] <= 0
            or not np.allclose(self.k[2], [0.0, 0.0, 1.0])
            or self.width <= 0
            or self.height <= 0
        ):
            raise CalibrationError("相机内参矩阵 K 或分辨率无效")

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_3d, dtype=float).reshape(-1, 3)
        if self.model in ("inverse_brown_conrady", "modified_brown_conrady"):
            xy = pts[:, :2] / pts[:, 2, None]
            radius2 = np.sum(xy**2, axis=1)
            k1, k2, p1, p2, k3 = self.distortion[:5]
            radial = 1.0 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
            scaled = xy * radial[:, None]
            x, y = scaled.T
            tangent = np.column_stack(
                [
                    2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x**2),
                    2.0 * p2 * x * y + p1 * (radius2 + 2.0 * y**2),
                ]
            )
            return (scaled + tangent) * [self.k[0, 0], self.k[1, 1]] + self.k[:2, 2]
        proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.k, self.distortion)
        return proj.reshape(-1, 2)

    def deproject_pixels(self, pixels: np.ndarray, depth_m: np.ndarray | float = 1.0) -> np.ndarray:
        """Convert distorted pixels and optical Z depth to 3D camera-space points."""
        px = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if len(px) == 0:
            return np.zeros((0, 3), dtype=float)
        if self.model == "pinhole":
            fx, fy = self.k[0, 0], self.k[1, 1]
            cx, cy = self.k[0, 2], self.k[1, 2]
            xy = np.column_stack([(px[:, 0] - cx) / fx, (px[:, 1] - cy) / fy])
        else:
            xy = cv2.undistortPoints(px.reshape(-1, 1, 2), self.k, self.distortion).reshape(-1, 2)
            if self.model in ("inverse_brown_conrady", "modified_brown_conrady"):
                focal = np.array([self.k[0, 0], self.k[1, 1]])
                for _ in range(12):
                    rays = np.column_stack([xy, np.ones(len(xy))])
                    xy -= (self.project(rays) - px) / focal
        z = np.asarray(depth_m, dtype=float)
        z = np.full((len(px), 1), float(z)) if z.ndim == 0 else z.reshape(-1, 1)
        return np.column_stack([xy, np.ones(len(xy))]) * z

    def deproject_depth_pointcloud(
        self,
        depth_m: np.ndarray,
        color_bgr: np.ndarray,
        step: int = 6,
        min_depth_m: float = 0.07,
        max_depth_m: float = 0.80,
        ray_cache: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Subsample aligned metric depth + BGR into (points_cam_m, colors_rgb_01)."""
        h, w = depth_m.shape[:2]
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        z = depth_m[ys, xs].ravel()
        if ray_cache is not None and step in ray_cache:
            rays = ray_cache[step]
        else:
            pixels = np.column_stack([xs.ravel(), ys.ravel()])
            rays = self.deproject_pixels(pixels, 1.0)
            if ray_cache is not None:
                ray_cache[step] = rays
        valid = (z > float(min_depth_m)) & (z < float(max_depth_m)) & np.isfinite(z)
        pts = rays[valid] * z[valid, None]
        bgr = color_bgr[ys, xs].reshape(-1, 3)[valid]
        rgb = bgr[:, ::-1] / 255.0
        return pts, rgb

    def reprojection_residual(
        self,
        x_cam_board: np.ndarray,
        corners_2d: np.ndarray,
        side_m: float,
        min_z_m: float = 0.02,
    ) -> np.ndarray:
        from camera_tool.services.tasks import check_cancelled

        check_cancelled()
        pts_cam = transform_points(x_cam_board, marker_object_points(side_m))
        if np.any(pts_cam[:, 2] < min_z_m):
            penalty = np.maximum(min_z_m - pts_cam[:, 2], 0.0) * 1e4
            return np.full(8, 1e4) + np.repeat(penalty, 2)
        return (self.project(pts_cam) - np.asarray(corners_2d, dtype=float).reshape(4, 2)).ravel()

    residual = reprojection_residual

    def pnp_candidates(
        self,
        corners_2d: np.ndarray,
        side_m: float,
        min_z_m: float = 0.02,
    ) -> list[tuple[np.ndarray, float]]:
        """Solve planar square PnP via IPPE_SQUARE, returning sorted [(T_cam_board, rms_px), ...]."""
        obj_pts = marker_object_points(side_m)
        img_pts = np.asarray(corners_2d, dtype=float).reshape(4, 2)
        pnp_points, pnp_distortion = img_pts, self.distortion
        if self.model in ("inverse_brown_conrady", "modified_brown_conrady"):
            rays = self.deproject_pixels(img_pts, 1.0)
            pnp_points = rays[:, :2] * [self.k[0, 0], self.k[1, 1]] + self.k[:2, 2]
            pnp_distortion = np.zeros(5, dtype=float)
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            obj_pts,
            pnp_points,
            self.k,
            pnp_distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        candidates: list[tuple[np.ndarray, float]] = []
        if ok:
            for rvec, tvec in zip(rvecs, tvecs):
                rmat, _ = cv2.Rodrigues(rvec)
                pose = make_transform(rmat, tvec)
                if not np.isfinite(pose).all():
                    continue
                pts_cam = transform_points(pose, obj_pts)
                if np.min(pts_cam[:, 2]) <= min_z_m:
                    continue
                res = self.reprojection_residual(pose, img_pts, side_m, min_z_m=min_z_m)
                rms = float(np.sqrt(np.mean(res**2)))
                candidates.append((pose, rms))
        if not candidates:
            raise RejectedSample("ArUco 标定码 PnP 姿态求解失败")
        candidates.sort(key=lambda item: item[1])
        return candidates

    candidates = pnp_candidates
