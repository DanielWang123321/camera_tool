"""Unified ArUco marker detection, quality gates, IPPE disambiguation, and HUD overlay."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from camera_tool.core.camera_model import CameraModel
from camera_tool.core.geometry import pose_error, transform_points, valid_transform
from camera_tool.core.io import CalibrationError, RejectedSample


def choose_ippe_pose(
    candidates: list[tuple[np.ndarray, float]],
    predicted_cam_board: np.ndarray,
    rms_margin: float = 0.6,
) -> np.ndarray:
    """Select the IPPE planar pose candidate closest to a predicted SE(3) pose."""
    options = [item for item in candidates if item[1] <= candidates[0][1] + rms_margin]
    return min(
        options,
        key=lambda item: pose_error(item[0], predicted_cam_board)[0] / 10.0
        + pose_error(item[0], predicted_cam_board)[1] / 2.0,
    )[0]


class MarkerDetector:
    """ArUco marker detector with border, area, sharpness, and PnP RMS quality gates."""

    def __init__(self, marker_cfg: dict[str, Any], camera: CameraModel) -> None:
        self.cfg = marker_cfg
        self.config = marker_cfg
        self.camera = camera
        dict_name = str(marker_cfg.get("dictionary", "DICT_ARUCO_ORIGINAL"))
        if not hasattr(cv2.aruco, dict_name):
            raise CalibrationError(f"OpenCV 不存在 ArUco 字典: {dict_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

    def evaluate(self, image: np.ndarray) -> dict[str, Any]:
        """Evaluate an image frame without raising on gate failure; returns detailed status dict."""
        if image is None or image.shape[:2] != (self.camera.height, self.camera.width):
            return {
                "passed": False,
                "reason": "图像尺寸与相机内参不匹配",
                "detected": False,
                "area_px": 0.0,
                "sharpness": 0.0,
                "pnp_rms_px": 99.9,
                "corners": None,
                "pose_cam_board": None,
                "candidates": [],
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners_list, ids, _ = self.detector.detectMarkers(gray)
        target_id = int(self.cfg["id"])
        matches = [] if ids is None else np.flatnonzero(ids.ravel() == target_id)

        if len(matches) == 0:
            return {
                "passed": False,
                "reason": f"未检测到标定码 ID={target_id}",
                "detected": False,
                "area_px": 0.0,
                "sharpness": 0.0,
                "pnp_rms_px": 99.9,
                "corners": None,
                "pose_cam_board": None,
                "candidates": [],
            }
        if len(matches) > 1:
            return {
                "passed": False,
                "reason": f"检测到多个相同编号标定码 (ID={target_id})",
                "detected": False,
                "area_px": 0.0,
                "sharpness": 0.0,
                "pnp_rms_px": 99.9,
                "corners": None,
                "pose_cam_board": None,
                "candidates": [],
            }

        pts = np.asarray(corners_list[matches[0]][0], dtype=float)
        margin = 8.0
        near_border = bool(
            pts.min() < margin
            or np.any(pts[:, 0] > self.camera.width - margin - 1)
            or np.any(pts[:, 1] > self.camera.height - margin - 1)
        )
        area_px = float(abs(cv2.contourArea(pts.astype(np.float32))))

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        roi_pixels = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0]
        sharpness = float(roi_pixels.var()) if roi_pixels.size > 10 else 0.0
        if not np.isfinite(sharpness):
            sharpness = 0.0

        try:
            candidates = self.camera.pnp_candidates(pts, float(self.cfg["side_m"]))
            best_pose, pnp_rms_px = candidates[0]
        except RejectedSample as exc:
            return {
                "passed": False,
                "reason": str(exc),
                "detected": True,
                "area_px": area_px,
                "sharpness": sharpness,
                "pnp_rms_px": 99.9,
                "corners": pts.tolist(),
                "pose_cam_board": None,
                "candidates": [],
            }

        reasons: list[str] = []
        if near_border:
            reasons.append("标定码过于靠近画面边缘")
        if area_px < float(self.cfg["min_area_px"]):
            reasons.append(f"面积不足 ({area_px:.0f} < {self.cfg['min_area_px']} px²)")
        if sharpness < float(self.cfg["min_sharpness"]):
            reasons.append(f"清晰度不足 ({sharpness:.1f} < {self.cfg['min_sharpness']})")
        if pnp_rms_px > float(self.cfg["max_pnp_rms_px"]):
            reasons.append(f"PnP 误差超限 ({pnp_rms_px:.2f} > {self.cfg['max_pnp_rms_px']} px)")

        passed = len(reasons) == 0
        return {
            "passed": passed,
            "reason": "质检通过" if passed else "；".join(reasons),
            "detected": True,
            "area_px": area_px,
            "sharpness": sharpness,
            "pnp_rms_px": pnp_rms_px,
            "corners": pts.tolist(),
            "pose_cam_board": best_pose,
            "candidates": candidates,
        }

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """Strictly detect marker: raise CalibrationError on size mismatch or RejectedSample on gate failure."""
        if image is None or image.shape[:2] != (self.camera.height, self.camera.width):
            raise CalibrationError("检测图像尺寸与相机内参不一致")
        res = self.evaluate(image)
        if not res["passed"]:
            raise RejectedSample(res["reason"])
        return res

    detect_strict = detect

    def draw_overlay(self, image: np.ndarray, eval_result: dict[str, Any]) -> np.ndarray:
        """Render safe border, marker polygon, 3D axes, and distance HUD onto a BGR frame."""
        canvas = image.copy()
        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (8, 8), (w - 9, h - 9), (90, 90, 90), 1)

        if eval_result.get("corners") is not None:
            pts = np.asarray(eval_result["corners"], dtype=np.int32)
            color = (40, 220, 80) if eval_result["passed"] else (40, 165, 255)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                cv2.circle(canvas, tuple(pt), 5, (0, 255, 255) if idx == 0 else color, -1)
                cv2.putText(
                    canvas,
                    f"C{idx}",
                    (int(pt[0]) + 6, int(pt[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        pose = eval_result.get("pose_cam_board")
        if pose is not None:
            pose = valid_transform(pose)
            tvec = pose[:3, 3]
            axis_len = float(self.cfg["side_m"]) * 0.6
            axes_3d = transform_points(
                pose,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [axis_len, 0.0, 0.0],
                        [0.0, axis_len, 0.0],
                        [0.0, 0.0, axis_len],
                    ],
                    dtype=float,
                ),
            )
            if np.all(axes_3d[:, 2] > 0.02):
                uv = np.rint(self.camera.project(axes_3d)).astype(int)
                origin = (int(uv[0, 0]), int(uv[0, 1]))
                for end_pt, axis_color in zip(uv[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                    cv2.line(canvas, origin, (int(end_pt[0]), int(end_pt[1])), axis_color, 2, lineType=cv2.LINE_AA)
            dist_mm = float(np.linalg.norm(tvec) * 1000.0)
            cv2.putText(
                canvas,
                f"Dist: {dist_mm:.0f} mm",
                (18, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 240, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas
