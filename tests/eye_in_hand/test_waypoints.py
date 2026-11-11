"""Tests for Eye-in-Hand preset waypoints, Flange/TCP decoupling, tool collision safety, and profile registry."""
from __future__ import annotations

import numpy as np
import pytest

from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner
from camera_tool.core import (
    CameraModel,
    MarkerDetector,
    RejectedSample,
    check_tool_safety,
    flange_from_tcp,
    pose_error,
    pose_from_mm_deg,
    rotation_coverage,
    tcp_from_flange,
)
from camera_tool.devices.sim import SimulatedRgbdCamera
from camera_tool.paths import config_path


def test_robot_profiles_registry_and_legacy_aliases() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    profiles = {m["key"]: m for m in planner.available_models()}
    assert set(profiles.keys()) == {"arm_6dof", "arm_5dof", "arm_7dof", "arm_compact_6dof", "arm_long_6dof"}
    assert profiles["arm_6dof"]["dof"] == 6 and profiles["arm_6dof"]["supported"] is True
    assert profiles["arm_5dof"]["dof"] == 5
    assert profiles["arm_7dof"]["dof"] == 7
    assert profiles["arm_compact_6dof"]["reach_mm"] == 440
    assert profiles["arm_long_6dof"]["reach_mm"] == 850

    # Legacy alias maps cleanly to arm_6dof
    planner.select_model("xarm6")
    assert planner.current_model == "arm_6dof"


def test_flange_tcp_decoupling_invariance() -> None:
    flange = pose_from_mm_deg([400.0, -50.0, 350.0, 172.0, -8.0, 15.0])
    tcp_offset = [0.0, 0.0, 172.0, 0.0, 0.0, 0.2]
    tcp_pose = tcp_from_flange(flange, tcp_offset)
    recovered_flange = flange_from_tcp(tcp_pose, tcp_offset)
    err_mm, err_deg = pose_error(flange, recovered_flange)
    assert err_mm < 1e-6
    assert err_deg < 1e-6


def test_tool_collision_protection_blocks_low_z() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    tool_cfg = planner.default_cfg["robot"]["tool"]

    unsafe_pose = pose_from_mm_deg([400.0, 0.0, 220.0, 180.0, 0.0, 0.0])
    with pytest.raises(RejectedSample):
        check_tool_safety(unsafe_pose, tool_cfg)


def test_close_range_20_preset_waypoints_safety_and_fov() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    planner.select_model("arm_6dof")
    assert len(planner.waypoints) == 20

    flange_poses = [wp["X_base_flange"] for wp in planner.waypoints]
    cov_deg = rotation_coverage(flange_poses, minimum_deg=6.0)
    assert cov_deg[0] >= 12.0
    assert cov_deg[1] >= 6.0

    sim_cam = SimulatedRgbdCamera(planner.default_cfg["camera"], planner.default_cfg["marker"])
    cam_meta = sim_cam.start()
    cam_model = CameraModel(cam_meta)
    detector = MarkerDetector(planner.default_cfg["marker"], cam_model)

    for wp in planner.waypoints:
        safety = wp["safety"]
        assert safety["flange_z_mm"] >= 325.0, f"Waypoint {wp['name']} flange too low"
        assert safety["table_clearance_mm"] >= 120.0, f"Waypoint {wp['name']} tool clearance too low"

        frame = sim_cam.read_frame(x_base_flange=wp["X_base_flange"])
        eval_res = detector.evaluate(frame["color_bgr"])
        assert eval_res["passed"], f"Waypoint #{wp['id']} ({wp['name']}) failed gate: {eval_res['reason']}"
        assert eval_res["area_px"] >= 1200.0
        assert eval_res["sharpness"] >= 30.0
        assert eval_res["pnp_rms_px"] <= 1.5


def test_waypoint_recentering() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    orig_first = planner.waypoints[0]["X_base_flange"][:2, 3].copy()
    recentered = planner.recenter_waypoints([0.42, 0.03, 0.0], yaw_offset_deg=0.0)
    assert len(recentered) == 20
    new_first = recentered[0]["X_base_flange"][:2, 3]
    assert np.allclose(new_first - orig_first, [0.02, 0.03], atol=1e-4)
