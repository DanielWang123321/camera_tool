"""Tests for Eye-in-Hand 7 OpenCV solvers, IPPE disambiguation, Bundle Adjustment, and Depth check."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from camera_tool.calibration.eye_in_hand.solver import evaluate_depth_consistency, solve_hand_eye_all
from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner
from camera_tool.core import (
    CameraModel,
    inv_transform,
    make_transform,
    marker_object_points,
    pose_error,
    pose_from_mm_rad,
    transform_points,
)
from camera_tool.paths import config_path


def test_eye_in_hand_solver_accuracy_with_noise() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    cam_meta = {
        "width": 1280,
        "height": 720,
        "model": "brown_conrady",
        "K": [[845.0, 0.0, 640.0], [0.0, 845.0, 360.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    camera = CameraModel(cam_meta)
    side_m = float(planner.default_cfg["marker"]["side_m"])

    gt_flange_cam = make_transform(
        Rotation.from_euler("xyz", [1.2, -1.8, 2.5], degrees=True).as_matrix(),
        [0.052, -0.015, 0.068],
    )
    gt_base_board = make_transform(
        Rotation.from_euler("xyz", [0.5, -0.4, 3.0], degrees=True).as_matrix(),
        [0.400, 0.005, 0.000],
    )

    rng = np.random.default_rng(42)
    obj_pts = marker_object_points(side_m)
    samples = []
    for wp in planner.waypoints:
        flange = wp["X_base_flange"]
        x_cam_board = inv_transform(flange @ gt_flange_cam) @ gt_base_board
        pts_3d = transform_points(x_cam_board, obj_pts)
        corners_2d = camera.project(pts_3d) + rng.normal(0.0, 0.15, size=(4, 2))
        samples.append(
            {
                "id": wp["id"],
                "name": wp["name"],
                "enabled": True,
                "X_base_flange": flange,
                "corners": corners_2d.tolist(),
            }
        )

    tcp_offset = [0.0, 0.0, 175.0, 0.0, 0.0, 0.0]
    solution = solve_hand_eye_all(
        samples,
        camera,
        side_m=side_m,
        minimum_rotation_deg=6.0,
        tcp_offset_mm_rad=tcp_offset,
    )

    assert "BUNDLE_ADJUSTMENT" in solution["methods"]
    assert len(solution["methods"]) >= 6

    err_mm, err_deg = pose_error(solution["X_flange_cam"], gt_flange_cam)
    board_mm, board_deg = pose_error(solution["X_base_board"], gt_base_board)
    assert err_mm < 0.6, f"Translation error too large: {err_mm:.3f} mm"
    assert err_deg < 0.08, f"Rotation error too large: {err_deg:.4f} deg"
    assert board_mm < 0.6
    assert board_deg < 0.08

    # Verify X_tcp_cam accounts for tcp_offset
    expected_tcp_cam = inv_transform(pose_from_mm_rad(tcp_offset)) @ solution["X_flange_cam"]
    tcp_mm, tcp_deg = pose_error(solution["X_tcp_cam"], expected_tcp_cam)
    assert tcp_mm < 1e-6 and tcp_deg < 1e-6


def test_depth_consistency_verification() -> None:
    flange = make_transform(
        Rotation.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True).as_matrix(),
        [0.40, 0.0, 0.35],
    )
    flange_cam = make_transform(np.eye(3), [0.05, 0.0, 0.06])
    base_board = make_transform(np.eye(3), [0.40, 0.0, 0.0])

    # Generate synthetic table points at Z_base = 0 transformed into camera frame
    xs, ys = np.meshgrid(np.linspace(0.32, 0.48, 15), np.linspace(-0.08, 0.08, 15))
    pts_base = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    pts_cam = transform_points(inv_transform(flange @ flange_cam), pts_base)

    report = evaluate_depth_consistency(pts_cam, flange, flange_cam, base_board, table_z_m=0.0)
    assert report["ok"] is True
    assert report["table_height_error_mm"] < 1.0
    assert report["normal_angle_deg"] < 1.0



def test_depth_plane_rejects_gripper_foreground_outliers():
    flange = pose_from_mm_rad([400,0,350,np.pi,0,0])
    fc = make_transform(np.eye(3), [0.05,0,0.06])
    board = make_transform(np.eye(3), [0.4,0,0])
    rng = np.random.default_rng(398)
    xs, ys = np.meshgrid(np.linspace(0.32,0.48,20),np.linspace(-0.08,0.08,20))
    table = np.column_stack([xs.ravel(),ys.ravel(),rng.normal(0,0.0003,xs.size)])
    gripper = rng.uniform([0.40,-0.04,0.10],[0.48,0.04,0.18],size=(100,3))
    points = transform_points(inv_transform(flange@fc),np.vstack([table,gripper]))
    report = evaluate_depth_consistency(points,flange,fc,board)
    assert report['ok']
    assert report['plane_inliers'] >= 390
    assert report['normal_angle_deg'] < 1.0
    assert report['table_height_error_mm'] < 1.0


def test_robust_depth_plane_still_detects_wrong_rotation():
    flange = pose_from_mm_rad([400,0,350,np.pi,0,0])
    fc = make_transform(np.eye(3), [0.05,0,0.06])
    board = make_transform(np.eye(3), [0.4,0,0])
    xs, ys = np.meshgrid(np.linspace(0.32,0.48,20),np.linspace(-0.08,0.08,20))
    table = np.column_stack([xs.ravel(),ys.ravel(),np.tan(np.deg2rad(15))*(xs.ravel()-0.4)])
    points = transform_points(inv_transform(flange@fc),table)
    report = evaluate_depth_consistency(points,flange,fc,board)
    assert not report['ok']
    assert report['normal_angle_deg'] > 14


def test_robust_depth_plane_still_detects_wrong_height():
    flange = pose_from_mm_rad([400,0,350,np.pi,0,0])
    fc = make_transform(np.eye(3), [0.05,0,0.06])
    board = make_transform(np.eye(3), [0.4,0,0])
    xs, ys = np.meshgrid(np.linspace(0.32,0.48,20),np.linspace(-0.08,0.08,20))
    table = np.column_stack([xs.ravel(),ys.ravel(),np.full(xs.size,0.04)])
    points = transform_points(inv_transform(flange@fc),table)
    report = evaluate_depth_consistency(points,flange,fc,board)
    assert not report['ok']
    assert report['table_height_error_mm'] > 39
