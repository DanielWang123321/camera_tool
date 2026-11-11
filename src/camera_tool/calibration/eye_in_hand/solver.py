"""Eye-in-Hand Calibration Solver: OpenCV 7-method benchmark + IPPE disambiguation + Bundle Adjustment."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from camera_tool.core import (
    CalibrationError,
    CameraModel,
    average_se3,
    inv_transform,
    make_transform,
    pose_error,
    pose_from_mm_rad,
    pose_to_m_rad,
    pose_to_mm_deg,
    pose_to_vector6,
    rotation_coverage,
    transform_points,
    valid_transform,
    vector6_to_pose,
)

OPENCV_HAND_EYE_METHODS: dict[str, tuple[str, int]] = {
    "TSAI": ("calibrateHandEye", cv2.CALIB_HAND_EYE_TSAI),
    "PARK": ("calibrateHandEye", cv2.CALIB_HAND_EYE_PARK),
    "HORAUD": ("calibrateHandEye", cv2.CALIB_HAND_EYE_HORAUD),
    "ANDREFF": ("calibrateHandEye", cv2.CALIB_HAND_EYE_ANDREFF),
    "DANIILIDIS": ("calibrateHandEye", cv2.CALIB_HAND_EYE_DANIILIDIS),
    "SHAH": ("calibrateRobotWorldHandEye", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
    "LI": ("calibrateRobotWorldHandEye", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI),
}


def _choose_candidate(
    candidates: list[tuple[np.ndarray, float]],
    predicted_cam_board: np.ndarray,
) -> np.ndarray:
    options = [item for item in candidates if item[1] <= candidates[0][1] + 0.6]
    return min(
        options,
        key=lambda item: pose_error(item[0], predicted_cam_board)[0] / 10.0
        + pose_error(item[0], predicted_cam_board)[1] / 2.0,
    )[0]


def _solve_opencv_single(
    flange_poses: list[np.ndarray],
    cam_board_poses: list[np.ndarray],
    method_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one OpenCV hand-eye solver and return (X_flange_cam, X_base_board)."""
    family, flag = OPENCV_HAND_EYE_METHODS[method_name]
    if family == "calibrateHandEye":
        r_g2b = [p[:3, :3] for p in flange_poses]
        t_g2b = [p[:3, 3] for p in flange_poses]
        r_t2c = [c[:3, :3] for c in cam_board_poses]
        t_t2c = [c[:3, 3] for c in cam_board_poses]
        r_c2g, t_c2g = cv2.calibrateHandEye(r_g2b, t_g2b, r_t2c, t_t2c, method=flag)
        x_flange_cam = valid_transform(make_transform(r_c2g, t_c2g))
        board_estimates = [f @ x_flange_cam @ c for f, c in zip(flange_poses, cam_board_poses)]
        x_base_board = average_se3(board_estimates)
        return x_flange_cam, x_base_board

    base_flange = [inv_transform(f) for f in flange_poses]
    rb, tb, rc, tc = cv2.calibrateRobotWorldHandEye(
        [c[:3, :3] for c in cam_board_poses],
        [c[:3, 3] for c in cam_board_poses],
        [b[:3, :3] for b in base_flange],
        [b[:3, 3] for b in base_flange],
        method=flag,
    )
    x_flange_cam = inv_transform(valid_transform(make_transform(rc, tc)))
    x_base_board = inv_transform(valid_transform(make_transform(rb, tb)))
    return x_flange_cam, x_base_board


def evaluate_solution(
    samples: list[dict[str, Any]],
    camera: CameraModel,
    side_m: float,
    x_flange_cam: np.ndarray,
    x_base_board: np.ndarray,
) -> dict[str, Any]:
    """Compute per-sample and aggregate 3D pose & 2D reprojection metrics for a candidate solution."""
    x_flange_cam = valid_transform(x_flange_cam)
    x_base_board = valid_transform(x_base_board)
    pose_errors: list[tuple[float, float]] = []
    reproj_rms_list: list[float] = []
    sample_ids: list[Any] = []

    for s in samples:
        flange = valid_transform(s["X_base_flange"])
        corners = np.asarray(s["corners"], dtype=float)
        predicted_cam_board = inv_transform(flange @ x_flange_cam) @ x_base_board
        candidates = camera.pnp_candidates(corners, side_m)
        observed_cam_board = _choose_candidate(candidates, predicted_cam_board)
        err_mm, err_deg = pose_error(flange @ x_flange_cam @ observed_cam_board, x_base_board)
        res = camera.reprojection_residual(predicted_cam_board, corners, side_m)
        rms_px = float(np.sqrt(np.mean(res**2)))
        pose_errors.append((err_mm, err_deg))
        reproj_rms_list.append(rms_px)
        sample_ids.append(s.get("id", len(sample_ids) + 1))

    err_arr = np.asarray(pose_errors, dtype=float)
    return {
        "count": len(samples),
        "sample_ids": sample_ids,
        "mean_mm": float(err_arr[:, 0].mean()),
        "max_mm": float(err_arr[:, 0].max()),
        "std_mm": float(err_arr[:, 0].std()),
        "mean_deg": float(err_arr[:, 1].mean()),
        "max_deg": float(err_arr[:, 1].max()),
        "rms_px": float(np.sqrt(np.mean(np.square(reproj_rms_list)))),
        "max_rms_px": float(max(reproj_rms_list)),
        "per_sample_mm": err_arr[:, 0].tolist(),
        "per_sample_deg": err_arr[:, 1].tolist(),
        "per_sample_rms_px": reproj_rms_list,
    }


def solve_hand_eye_all(
    samples: list[dict[str, Any]],
    camera: CameraModel,
    side_m: float,
    minimum_rotation_deg: float = 6.0,
    tcp_offset_mm_rad: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run all 7 OpenCV algorithms plus IPPE-disambiguated Bundle Adjustment on active samples."""
    active = [s for s in samples if s.get("enabled", True)]
    if len(active) < 5:
        raise CalibrationError(f"启用样本数不足，至少需要 5 组有效姿态（当前 {len(active)} 组）")

    flange_poses = [valid_transform(s["X_base_flange"]) for s in active]
    coverage_deg = rotation_coverage(flange_poses, minimum_rotation_deg)
    all_candidates = [camera.pnp_candidates(s["corners"], side_m) for s in active]

    rng = np.random.default_rng(398)
    starts = [np.zeros(len(active), dtype=int), np.ones(len(active), dtype=int)]
    starts.extend(rng.integers(0, 2, len(active)) for _ in range(10))

    seeds: list[np.ndarray] = []
    disambiguated_poses: list[np.ndarray] | None = None

    for assignment in starts:
        poses = [c[min(int(idx), len(c) - 1)][0] for idx, c in zip(assignment, all_candidates)]
        try:
            for _ in range(4):
                flange_cam, base_board = _solve_opencv_single(flange_poses, poses, "SHAH")
                poses = [
                    _choose_candidate(c, inv_transform(f @ flange_cam) @ base_board)
                    for c, f in zip(all_candidates, flange_poses)
                ]
            if np.linalg.norm(flange_cam[:3, 3]) > 0.6:
                continue
            seed = np.r_[pose_to_vector6(flange_cam), pose_to_vector6(base_board)]
            if not any(np.linalg.norm(seed - old) < 1e-5 for old in seeds):
                seeds.append(seed)
                if disambiguated_poses is None:
                    disambiguated_poses = poses
        except (cv2.error, ValueError, CalibrationError, np.linalg.LinAlgError):
            continue

    if disambiguated_poses is None:
        disambiguated_poses = [c[0][0] for c in all_candidates]

    method_results: dict[str, dict[str, Any]] = {}
    for name in OPENCV_HAND_EYE_METHODS:
        try:
            x_fc, x_bb = _solve_opencv_single(flange_poses, disambiguated_poses, name)
            if np.linalg.norm(x_fc[:3, 3]) > 0.8:
                continue
            metrics = evaluate_solution(active, camera, side_m, x_fc, x_bb)
            method_results[name] = {
                "method": name,
                "X_flange_cam": x_fc,
                "X_base_board": x_bb,
                "flange_cam_mm_deg": pose_to_mm_deg(x_fc),
                "flange_cam_m_rad": pose_to_m_rad(x_fc),
                "base_board_mm_deg": pose_to_mm_deg(x_bb),
                "metrics": metrics,
            }
            seed = np.r_[pose_to_vector6(x_fc), pose_to_vector6(x_bb)]
            if not any(np.linalg.norm(seed - old) < 1e-5 for old in seeds):
                seeds.append(seed)
        except (cv2.error, ValueError, CalibrationError, np.linalg.LinAlgError):
            continue

    if not seeds:
        raise CalibrationError("所有手眼标定初值算法均未收敛，请检查角点质量与多轴旋转激励")

    def residual_fn(vec: np.ndarray) -> np.ndarray:
        from camera_tool.services.tasks import check_cancelled

        check_cancelled()
        x_fc = vector6_to_pose(vec[:6])
        x_bb = vector6_to_pose(vec[6:])
        return np.concatenate(
            [
                camera.reprojection_residual(inv_transform(f @ x_fc) @ x_bb, s["corners"], side_m)
                for f, s in zip(flange_poses, active)
            ]
        )

    seeds.sort(key=lambda v: float(np.mean(residual_fn(v) ** 2)))
    opt_solutions: list[tuple[float, np.ndarray, np.ndarray]] = []
    for seed in seeds[:4]:
        fit = least_squares(
            residual_fn,
            seed,
            loss="soft_l1",
            f_scale=0.7,
            x_scale="jac",
            max_nfev=300,
        )
        if fit.success and np.isfinite(fit.x).all():
            rms = float(np.sqrt(np.mean(residual_fn(fit.x) ** 2)))
            opt_solutions.append((rms, fit.x, fit.jac))

    if not opt_solutions:
        raise CalibrationError("2D 角点非线性重投影优化未收敛")

    opt_solutions.sort(key=lambda item: item[0])
    _, best_vec, best_jac = opt_solutions[0]
    best_fc = vector6_to_pose(best_vec[:6])
    best_bb = vector6_to_pose(best_vec[6:])

    scaled_jac = best_jac / np.maximum(np.linalg.norm(best_jac, axis=0), 1e-12)
    singular_vals = np.linalg.svd(scaled_jac, compute_uv=False)
    jac_min_sv = float(singular_vals[-1])

    ba_metrics = evaluate_solution(active, camera, side_m, best_fc, best_bb)
    method_results["BUNDLE_ADJUSTMENT"] = {
        "method": "BUNDLE_ADJUSTMENT",
        "X_flange_cam": best_fc,
        "X_base_board": best_bb,
        "flange_cam_mm_deg": pose_to_mm_deg(best_fc),
        "flange_cam_m_rad": pose_to_m_rad(best_fc),
        "base_board_mm_deg": pose_to_mm_deg(best_bb),
        "metrics": ba_metrics,
        "jacobian_min_singular": jac_min_sv,
    }

    if tcp_offset_mm_rad is not None:
        t_flange_tcp = pose_from_mm_rad(tcp_offset_mm_rad)
    else:
        t_flange_tcp = np.eye(4, dtype=float)
    best_tcp_cam = inv_transform(t_flange_tcp) @ best_fc

    return {
        "recommended_method": "BUNDLE_ADJUSTMENT",
        "X_flange_cam": best_fc,
        "X_tcp_cam": best_tcp_cam,
        "X_base_board": best_bb,
        "flange_cam_mm_deg": pose_to_mm_deg(best_fc),
        "flange_cam_m_rad": pose_to_m_rad(best_fc),
        "tcp_cam_mm_deg": pose_to_mm_deg(best_tcp_cam),
        "tcp_cam_m_rad": pose_to_m_rad(best_tcp_cam),
        "base_board_mm_deg": pose_to_mm_deg(best_bb),
        "rotation_coverage_deg": coverage_deg,
        "jacobian_min_singular": jac_min_sv,
        "metrics": ba_metrics,
        "methods": method_results,
    }


def evaluate_board_geometry(
    x_base_board: np.ndarray,
    table_z_m: float = 0.0,
    board_thickness_m: float = 0.0,
    max_tilt_deg: float = 12.0,
    height_tolerance_m: float = 0.04,
) -> dict[str, Any]:
    """Verify solved ArUco board top-surface height (table_z + board_thickness) and horizontal tilt."""
    board = valid_transform(x_base_board)
    tilt_deg = float(np.rad2deg(np.arccos(np.clip(abs(board[2, 2]), 0.0, 1.0))))
    expected_board_z_m = float(table_z_m) + float(board_thickness_m)
    board_z_m = float(board[2, 3])
    height_err_m = abs(board_z_m - expected_board_z_m)
    estimated_table_z_m = board_z_m - float(board_thickness_m)
    ok = bool(tilt_deg <= float(max_tilt_deg) and height_err_m <= float(height_tolerance_m))
    return {
        "ok": ok,
        "board_z_mm": board_z_m * 1000.0,
        "expected_board_z_mm": expected_board_z_m * 1000.0,
        "board_height_error_mm": height_err_m * 1000.0,
        "estimated_table_z_mm": estimated_table_z_m * 1000.0,
        "configured_table_z_mm": float(table_z_m) * 1000.0,
        "board_thickness_mm": float(board_thickness_m) * 1000.0,
        "tilt_deg": tilt_deg,
    }


def evaluate_depth_consistency(
    points_cam: np.ndarray,
    x_base_flange: np.ndarray,
    x_flange_cam: np.ndarray,
    x_base_board: np.ndarray,
    table_z_m: float = 0.0,
    board_thickness_m: float = 0.0,
    board_side_m: float = 0.15,
    min_depth_m: float = 0.07,
    max_depth_m: float = 1.20,
) -> dict[str, Any]:
    """Transform 3D points to robot base frame and verify tabletop plane height & normal consistency."""
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > float(min_depth_m)) & (pts[:, 2] < float(max_depth_m))
    pts = pts[valid]
    if len(pts) < 50:
        return {"ok": False, "reason": "有效深度点不足", "points_count": len(pts)}

    x_base_cam = valid_transform(x_base_flange) @ valid_transform(x_flange_cam)
    board_pose = valid_transform(x_base_board)
    pts_base = transform_points(x_base_cam, pts)

    board_center = board_pose[:3, 3]
    horiz_dist = np.linalg.norm(pts_base[:, :2] - board_center[:2], axis=1)
    near_board = pts_base[horiz_dist < 0.22]
    if len(near_board) < 30:
        near_board = pts_base

    board_normal = board_pose[:3, 2]
    leveled = near_board
    if abs(float(board_thickness_m)) > 1e-6:
        pts_board = transform_points(inv_transform(board_pose), near_board)
        half_outer = 0.5 * float(board_side_m) * 1.25
        on_board = (
            (np.abs(pts_board[:, 0]) <= half_outer)
            & (np.abs(pts_board[:, 1]) <= half_outer)
            & (np.abs(pts_board[:, 2]) <= 0.025)
        )
        if np.any(on_board):
            up_normal = board_normal if board_normal[2] >= 0.0 else -board_normal
            leveled = near_board.copy()
            leveled[on_board] -= float(board_thickness_m) * up_normal

    rng = np.random.default_rng(398)
    best_mask = np.zeros(len(leveled), dtype=bool)
    threshold_m = 0.004
    for _ in range(128):
        triangle = leveled[rng.choice(len(leveled), 3, replace=False)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        mask = np.abs((leveled - triangle[0]) @ (normal / norm)) <= threshold_m
        if mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask.sum() < max(30, len(leveled) * 0.5):
        return {"ok": False, "reason": "深度点中未找到稳定的主平面", "points_count": len(leveled)}
    for _ in range(2):
        plane = leveled[best_mask]
        center = plane.mean(axis=0)
        _, _, vt = np.linalg.svd(plane - center, full_matrices=False)
        mask = np.abs((leveled - center) @ vt[-1]) <= threshold_m
        if mask.sum() < max(30, len(leveled) * 0.5):
            break
        best_mask = mask
    plane = leveled[best_mask]
    z_median_m = float(np.median(plane[:, 2]))
    z_err_mm = abs(z_median_m - float(table_z_m)) * 1000.0
    center = plane.mean(axis=0)
    _, _, vt = np.linalg.svd(plane - center, full_matrices=False)
    normal = vt[-1]
    normal_deg = float(np.rad2deg(np.arccos(np.clip(abs(normal @ board_normal), 0.0, 1.0))))

    return {
        "ok": bool(z_err_mm <= 15.0 and normal_deg <= 5.0),
        "points_count": int(len(leveled)),
        "plane_inliers": int(best_mask.sum()),
        "plane_inlier_ratio": float(best_mask.mean()),
        "table_z_median_mm": z_median_m * 1000.0,
        "configured_table_z_mm": float(table_z_m) * 1000.0,
        "board_thickness_mm": float(board_thickness_m) * 1000.0,
        "table_height_error_mm": float(z_err_mm),
        "normal_angle_deg": normal_deg,
    }
