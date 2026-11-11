"""Eye-to-Hand solver & verification gates (hardware-independent, supports offline replay)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from camera_tool.core import (
    CalibrationError,
    CameraModel,
    from_vector,
    inv,
    pose_error,
    pose_vector,
    rotation_coverage,
    transform,
    transform_points,
    valid_transform,
)


def split_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(samples) != 30:
        raise CalibrationError(f"需要 30 个有效姿态，当前为 {len(samples)}")
    train = [s for i, s in enumerate(samples) if i % 5 != 4]
    check = [s for i, s in enumerate(samples) if i % 5 == 4]
    return train, check


def _pose_candidates(samples: list[dict[str, Any]], camera: CameraModel, side: float) -> list[Any]:
    return [camera.candidates(s["corners"], side) for s in samples]


def _choose(candidates: list[tuple[np.ndarray, float]], predicted: np.ndarray) -> np.ndarray:
    options = [item for item in candidates if item[1] <= candidates[0][1] + 0.5]
    return min(
        options,
        key=lambda item: pose_error(item[0], predicted)[0] / 10.0 + pose_error(item[0], predicted)[1] / 2.0,
    )[0]


def _shah(tcp: list[np.ndarray], board_camera: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    base_tcp = [inv(t) for t in tcp]
    rb, tb, rc, tc = cv2.calibrateRobotWorldHandEye(
        [c[:3, :3] for c in board_camera],
        [c[:3, 3] for c in board_camera],
        [t[:3, :3] for t in base_tcp],
        [t[:3, 3] for t in base_tcp],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    return inv(valid_transform(transform(rc, tc))), inv(valid_transform(transform(rb, tb)))


def fit_hand_eye(
    samples: list[dict[str, Any]],
    camera: CameraModel,
    side: float,
    minimum_rotation_deg: float = 6.0,
) -> dict[str, Any]:
    coverage = rotation_coverage(samples, minimum_rotation_deg)
    tcp = [valid_transform(s["X_base_tcp"]) for s in samples]
    candidates = _pose_candidates(samples, camera, side)
    rng = np.random.default_rng(398)
    starts = [np.zeros(len(samples), dtype=int), np.ones(len(samples), dtype=int)]
    starts.extend(rng.integers(0, 2, len(samples)) for _ in range(10))
    seeds: list[np.ndarray] = []
    for assignment in starts:
        poses = [c[min(int(i), len(c) - 1)][0] for i, c in zip(assignment, candidates)]
        try:
            for _ in range(4):
                wrist, board = _shah(tcp, poses)
                poses = [_choose(c, inv(t @ wrist) @ board) for c, t in zip(candidates, tcp)]
            if np.linalg.norm(wrist[:3, 3]) > 0.6:
                continue
            seed = np.r_[pose_vector(wrist), pose_vector(board)]
            if not any(np.linalg.norm(seed - old) < 1e-5 for old in seeds):
                seeds.append(seed)
        except (cv2.error, ValueError, CalibrationError, np.linalg.LinAlgError):
            continue
    if not seeds:
        raise CalibrationError("Shah 初始化失败，检查转动、角点和单位")

    def residual(v: np.ndarray) -> np.ndarray:
        wrist, board = from_vector(v[:6]), from_vector(v[6:])
        return np.concatenate([camera.residual(inv(t @ wrist) @ board, s["corners"], side) for t, s in zip(tcp, samples)])

    seeds.sort(key=lambda v: float(np.mean(residual(v) ** 2)))
    solutions: list[tuple[float, np.ndarray, np.ndarray]] = []
    for seed in seeds[:4]:
        fit = least_squares(residual, seed, loss="soft_l1", f_scale=0.7, x_scale="jac", max_nfev=250)
        if fit.success and np.isfinite(fit.x).all():
            solutions.append((float(np.sqrt(np.mean(residual(fit.x) ** 2))), fit.x, fit.jac))
    if not solutions:
        raise CalibrationError("角点投影优化未收敛")
    solutions.sort(key=lambda item: item[0])
    rms, best, jac = solutions[0]
    wrist, board = from_vector(best[:6]), from_vector(best[6:])
    for other_rms, other, _ in solutions[1:]:
        distance, angle = pose_error(board, from_vector(other[6:]))
        if other_rms < rms + 0.1 and (distance > 5.0 or angle > 1.0):
            raise CalibrationError("平面码仍有不同的整体解，需要更丰富的转动")
    scaled = jac / np.maximum(np.linalg.norm(jac, axis=0), 1e-12)
    singular = np.linalg.svd(scaled, compute_uv=False)
    if singular[-1] < 1e-4 or np.linalg.norm(wrist[:3, 3]) > 0.6:
        raise CalibrationError("优化结果不稳定或腕部相机安装距离异常")
    return {
        "X_tcp_wrist": wrist,
        "X_base_board": board,
        "fit_rms_px": rms,
        "rotation_rms_deg": coverage,
        "jacobian_min_singular": float(singular[-1]),
    }


def check_samples(
    samples: list[dict[str, Any]],
    camera: CameraModel,
    side: float,
    solution: dict[str, Any],
    thresholds: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    wrist, board = solution["X_tcp_wrist"], solution["X_base_board"]
    errors, pixels = [], []
    for sample in samples:
        tcp = valid_transform(sample["X_base_tcp"])
        predicted = inv(tcp @ wrist) @ board
        observed = _choose(camera.candidates(sample["corners"], side), predicted)
        errors.append(pose_error(tcp @ wrist @ observed, board))
        pixels.append(float(np.sqrt(np.mean(camera.residual(predicted, sample["corners"], side) ** 2))))
    err_arr = np.asarray(errors)
    report = {
        "count": len(samples),
        "mean_mm": float(err_arr[:, 0].mean()),
        "max_mm": float(err_arr[:, 0].max()),
        "mean_deg": float(err_arr[:, 1].mean()),
        "max_deg": float(err_arr[:, 1].max()),
        "max_rms_px": max(pixels),
        "per_pose_mm_deg": err_arr.tolist(),
        "per_pose_rms_px": pixels,
    }
    gates = ("mean_mm", "max_mm", "mean_deg", "max_deg", "max_rms_px")
    failures = [f"{key}={report[key]:.3f}>{thresholds[key]}" for key in gates if report[key] > thresholds[key]]
    if failures:
        raise CalibrationError(f"{label}未通过，板可能移动或相机安装改变：" + "，".join(failures))
    return report


def check_board_geometry(solution: dict[str, Any], config: dict[str, Any]) -> None:
    board = valid_transform(solution["X_base_board"])
    tilt = np.rad2deg(np.arccos(np.clip(board[2, 2], -1.0, 1.0)))
    if tilt > config["board_max_tilt_deg"]:
        raise CalibrationError(f"标定板没有朝上平放（倾斜 {tilt:.1f}°）")
    if abs(board[2, 3] - config["table_z_m"]) > config["board_height_tolerance_m"]:
        raise CalibrationError("标定板高度不符合桌面范围，检查单位或坐标定义")


def solve_wrist(samples: list[dict[str, Any]], metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    train, check = split_samples(samples)
    camera = CameraModel(metadata)
    solution = fit_hand_eye(train, camera, config["marker"]["side_m"], config["checks"]["min_rotation_deg"])
    solution["train_ids"] = [s["id"] for s in train]
    solution["check_ids"] = [s["id"] for s in check]
    solution["train_check"] = check_samples(train, camera, config["marker"]["side_m"], solution, config["checks"], "拟合姿态检查")
    solution["independent_check"] = check_samples(
        check, camera, config["marker"]["side_m"], solution, config["checks"], "6 个独立姿态检查"
    )
    check_board_geometry(solution, config["checks"])
    return solution


def _marker_inliers(
    points_color: np.ndarray,
    camera: CameraModel,
    corners: Any,
    x_camera_board: np.ndarray,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_color, dtype=float)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05) & (points[:, 2] < 3.0)]
    if len(points) < 100:
        raise CalibrationError("固定相机有效原始深度不足")
    pixels = camera.project(points)
    corners_arr = np.asarray(corners, dtype=np.float32)
    expanded = (corners_arr.mean(axis=0) + (corners_arr - corners_arr.mean(axis=0)) * 1.35).astype(np.int32)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, expanded, 1)
    uv = np.rint(pixels).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    indices = np.flatnonzero(inside)
    selected = indices[mask[uv[indices, 1], uv[indices, 0]] != 0]
    points = points[selected]
    if len(points) < min_points:
        raise CalibrationError("标定码附近桌面有效深度不足")
    normal, origin = x_camera_board[:3, 2], x_camera_board[:3, 3]
    distance = np.abs((points - origin) @ normal)
    inliers = points[distance <= max(0.008, float(np.median(distance)) * 3.0)]
    if len(inliers) < min_points:
        raise CalibrationError("桌面深度平面不稳定")
    return points, inliers, distance, normal


def report_depth(
    points_color: np.ndarray,
    camera: CameraModel,
    corners: Any,
    x_camera_board: np.ndarray,
    x_base_board: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    points, inliers, distance, normal = _marker_inliers(
        points_color, camera, corners, x_camera_board, config["depth_min_points"]
    )
    median_mm, p95_mm = np.percentile(distance, [50, 95]) * 1000.0
    center = inliers.mean(axis=0)
    _, singular, vt = np.linalg.svd(inliers - center, full_matrices=False)
    if singular[1] / np.sqrt(len(inliers)) < 0.01:
        raise CalibrationError("桌面深度覆盖面积不足")
    normal_depth = vt[-1]
    angle = float(np.rad2deg(np.arccos(np.clip(abs(normal_depth @ normal), 0.0, 1.0))))
    points_base = transform_points(x_base_board @ inv(x_camera_board), inliers)
    table_error_mm = float(abs(np.median(points_base[:, 2]) - config["table_z_m"]) * 1000.0)
    result = {
        "points": int(len(points)),
        "median_mm": float(median_mm),
        "p95_mm": float(p95_mm),
        "normal_deg": angle,
        "table_height_error_mm": table_error_mm,
        "table_z_median_m": float(np.median(points_base[:, 2])),
        "inliers": inliers,
    }
    result["ok"] = not (
        median_mm > config["depth_median_mm"]
        or p95_mm > config["depth_p95_mm"]
        or angle > config["depth_normal_deg"]
        or table_error_mm > config["depth_table_height_mm"]
    )
    return result


def depth_check(
    points_color: np.ndarray,
    camera: CameraModel,
    corners: Any,
    x_camera_board: np.ndarray,
    x_base_board: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    result = report_depth(points_color, camera, corners, x_camera_board, x_base_board, config)
    if not result["ok"]:
        public = {k: result[k] for k in result if k != "inliers"}
        raise CalibrationError(f"桌面深度核对未通过：{public}")
    return {k: result[k] for k in result if k != "inliers"}


def fit_depth_scale(inliers: np.ndarray, x_base_camera: np.ndarray, table_z_m: float = 0.0) -> float:
    points = np.asarray(inliers, dtype=float)
    if len(points) < 20:
        return 1.0
    rotated_z = (points @ x_base_camera[:3, :3].T)[:, 2]
    med = float(np.median(rotated_z))
    if abs(med) < 1e-4:
        return 1.0
    scale = (float(table_z_m) - float(x_base_camera[2, 3])) / med
    if not np.isfinite(scale) or scale <= 0.5 or scale >= 1.5:
        return 1.0
    return float(scale)


def _public_depth(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {k: report[k] for k in report if k != "inliers"}


def solve_fixed_camera(
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    wrist_solution: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    fixed_cfg = config.get("fixed_camera") or config.get("realsense") or {}
    required_frames = int(config.get("fixed_frames") or config.get("realsense_frames") or fixed_cfg.get("frames", 12))
    if len(samples) < required_frames:
        raise CalibrationError("固定相机有效帧数不足")
    camera = CameraModel(metadata)
    side = config["marker"]["side_m"]
    corners = np.stack([s["corners"] for s in samples])
    drift = np.linalg.norm(corners - corners[0], axis=2).max()
    if drift > config["checks"]["static_corner_drift_px"]:
        raise CalibrationError("固定相机多帧期间板或相机移动")
    seeds = camera.candidates(corners.mean(axis=0), side)
    solutions = []
    for pose, _ in seeds:
        def residual(v: np.ndarray) -> np.ndarray:
            return np.concatenate([camera.residual(from_vector(v), s["corners"], side) for s in samples])

        fit = least_squares(residual, pose_vector(pose), loss="soft_l1", f_scale=0.7, max_nfev=150)
        rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
        if not fit.success or rms > config["checks"]["max_rms_px"]:
            continue
        x_camera_board = from_vector(fit.x)
        reports, warnings = [], []
        for sample in samples:
            try:
                reports.append(
                    report_depth(
                        sample["points_color"],
                        camera,
                        sample["corners"],
                        x_camera_board,
                        wrist_solution["X_base_board"],
                        config["checks"],
                    )
                )
            except CalibrationError as exc:
                warnings.append(str(exc))
                reports.append(None)
        solutions.append((rms, x_camera_board, reports, warnings))
    if not solutions:
        raise CalibrationError("固定相机彩色求解未通过（角点投影未收敛）")

    def score(item: Any) -> tuple[float, float]:
        rms, _, reports, _ = item
        heights = [r["table_height_error_mm"] for r in reports if r]
        return (rms, float(np.mean(heights)) if heights else 1e6)

    solutions.sort(key=score)
    rms, x_camera_board, reports, warnings = solutions[0]
    x_base_camera = wrist_solution["X_base_board"] @ inv(x_camera_board)
    inliers = np.concatenate([r["inliers"] for r in reports if r]) if any(reports) else np.zeros((0, 3))
    depth_scale = fit_depth_scale(inliers, x_base_camera, config["checks"]["table_z_m"])
    public = [_public_depth(r) for r in reports]
    depth_ok = bool(public) and all(r is not None and r["ok"] for r in public)
    scaled_height = None
    if len(inliers):
        scaled = transform_points(x_base_camera, inliers * depth_scale)
        scaled_height = float(abs(np.median(scaled[:, 2]) - config["checks"]["table_z_m"]) * 1000.0)
    return {
        "X_base_camera": x_base_camera,
        "X_camera_board": x_camera_board,
        "fixed_rms_px": rms,
        "realsense_rms_px": rms,
        "depth_checks": public,
        "depth_ok": depth_ok,
        "depth_scale": depth_scale,
        "depth_warnings": warnings,
        "depth_scale_table_height_error_mm": scaled_height,
        "camera_frame": "color_optical_frame",
        "depth_aligned_to": "color",
    }


solve_realsense = solve_fixed_camera
