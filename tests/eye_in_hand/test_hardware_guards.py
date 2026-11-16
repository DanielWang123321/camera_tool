"""Hardware fault and xArm state machine regressions using SDK stubs, without contacting devices."""
from types import SimpleNamespace

import numpy as np
import pytest

from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner
from camera_tool.core import CalibrationError, RejectedSample, pose_from_mm_deg
from camera_tool.devices.robot import XArmController
from camera_tool.paths import config_path


@pytest.fixture
def controller():
    planner = WaypointPlanner(str(config_path("eye_in_hand/default.yaml")))
    robot = XArmController("192.0.2.10", planner.robot_cfg, planner.default_cfg["robot"]["tool"])
    robot.connected = True
    return robot


def state(code=2, error=0, warn=0):
    return {
        "state": code,
        "error_code": error,
        "warn_code": warn,
        "safety": {"safe": True, "message": "OK"},
        "joints_rad": [0.0] * 6,
        "is_moving": code == 1,
        "X_base_flange": pose_from_mm_deg([360, 10, 355, 180, 0, 0]),
    }


def test_enable_never_clears_hardware_error(controller, monkeypatch):
    calls = []
    controller.arm = SimpleNamespace(
        clean_error=lambda: calls.append("clean_error") or 0,
        motion_enable=lambda **kw: calls.append(("motion_enable", kw)) or 0,
        set_mode=lambda mode: calls.append(("set_mode", mode)) or 0,
        set_self_collision_detection=lambda on: calls.append(("self_collision", on)) or 0,
        set_state=lambda st: calls.append(("set_state", st)) or 0,
    )
    monkeypatch.setattr(controller, "read_state", lambda: state(code=4, error=1))
    with pytest.raises(CalibrationError, match="hardware error"):
        controller.enable_motion()
    assert calls == []
    assert not controller.moving_enabled


@pytest.mark.parametrize("pre_state,post_state", [(3, 2), (4, 2), (5, 2), (4, 0), (5, 0)])
def test_enable_transitions_pre_enable_states_and_queries_post_set_state(controller, monkeypatch, pre_state, post_state):
    calls = []
    current = {"state": pre_state}

    def set_state(val):
        calls.append(("set_state", val))
        if val == 0:
            current["state"] = post_state
        return 0

    controller.arm = SimpleNamespace(
        clean_error=lambda: calls.append(("clean_error", None)) or 0,
        clean_warn=lambda: calls.append(("clean_warn", None)) or 0,
        motion_enable=lambda **kw: calls.append(("motion_enable", kw.get("enable"))) or 0,
        set_mode=lambda mode: calls.append(("set_mode", mode)) or 0,
        set_self_collision_detection=lambda on: calls.append(("self_collision", on)) or 0,
        set_state=set_state,
    )
    monkeypatch.setattr(controller, "read_state", lambda: state(code=current["state"], error=0))
    controller.enable_motion()
    assert controller.moving_enabled is True
    assert calls == [
        ("motion_enable", True),
        ("set_mode", 0),
        ("self_collision", True),
        ("set_state", 0),
    ]


@pytest.mark.parametrize("stuck_state", [3, 4, 5])
def test_enable_fails_if_state_stays_unready_after_set_state(controller, monkeypatch, stuck_state):
    calls = []
    controller.arm = SimpleNamespace(
        clean_error=lambda: calls.append(("clean_error", None)) or 0,
        motion_enable=lambda **kw: calls.append(("motion_enable", kw.get("enable"))) or 0,
        set_mode=lambda mode: calls.append(("set_mode", mode)) or 0,
        set_self_collision_detection=lambda on: calls.append(("self_collision", on)) or 0,
        set_state=lambda val: calls.append(("set_state", val)) or 0,
    )
    monkeypatch.setattr(controller, "read_state", lambda: state(code=stuck_state, error=0))
    with pytest.raises(CalibrationError, match="not ready for motion"):
        controller.enable_motion()
    assert calls == [
        ("motion_enable", True),
        ("set_mode", 0),
        ("self_collision", True),
        ("set_state", 0),
    ]
    assert not controller.moving_enabled


@pytest.mark.parametrize("pre_state", [4, 5])
def test_plan_allows_pre_enable_state_and_stop_resets_moving_enabled(controller, monkeypatch, pre_state):
    stops = []
    controller.arm = SimpleNamespace(
        get_inverse_kinematics=lambda *a, **kw: (0, [0.01] * 6),
        get_forward_kinematics=lambda *a, **kw: (0, [360, 10, 355, np.pi, 0, 0]),
        set_state=lambda code: stops.append(code) or 0,
    )
    monkeypatch.setattr(controller, "read_state", lambda: state(code=pre_state, error=0))
    path = controller.plan_flange_pose(state()["X_base_flange"])
    assert len(path) >= 1
    assert not controller.moving_enabled

    controller.moving_enabled = True
    controller.stop()
    assert stops == [4]
    assert not controller.moving_enabled


def test_motion_fault_stops_without_submitting_another_joint_target(controller, monkeypatch):
    commands, stops = [], []
    controller.arm = SimpleNamespace(
        set_servo_angle=lambda **kw: commands.append(kw) or 0,
        set_state=lambda code: stops.append(code) or 0,
    )
    controller.moving_enabled = True
    readings = iter([state(), state(4)])
    monkeypatch.setattr(controller, "plan_flange_pose", lambda _: [np.zeros(6), np.ones(6) * 0.01])
    monkeypatch.setattr(controller, "read_state", lambda: next(readings))
    with pytest.raises(CalibrationError):
        controller.move_to_flange_pose(state()["X_base_flange"])
    assert len(commands) == 1
    assert stops == [4]
    assert not controller.moving_enabled


def test_unsafe_joint_interpolation_is_rejected_before_motion(controller, monkeypatch):
    controller.arm = SimpleNamespace(
        get_inverse_kinematics=lambda *a, **kw: (0, [0.01] * 6),
        get_forward_kinematics=lambda *a, **kw: (0, [360, 10, 220, np.pi, 0, 0]),
    )
    monkeypatch.setattr(controller, "read_state", state)
    with pytest.raises(RejectedSample):
        controller.plan_flange_pose(state()["X_base_flange"])
    assert not controller.moving_enabled


def test_failed_connection_can_still_release_sdk_socket(controller):
    disconnected = []
    controller.arm = SimpleNamespace(disconnect=lambda: disconnected.append(True))
    controller.connected = False
    controller.close()
    assert disconnected == [True]


def test_unsafe_start_pose_raises_clear_calibration_error_with_tcp_offset(monkeypatch):
    planner = WaypointPlanner(str(config_path("eye_in_hand/medium_range.yaml")))
    robot = XArmController("192.0.2.10", planner.robot_cfg, planner.default_cfg["robot"]["tool"])
    robot.connected = True
    robot.arm = SimpleNamespace()
    robot.tcp_offset_mm_rad = np.array([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])

    low_flange = pose_from_mm_deg([300.1, -0.1, 322.1, -180.0, 0.0, 0.0])
    low_state = {
        "state": 4,
        "error_code": 0,
        "warn_code": 0,
        "safety": {
            "safe": False,
            "message": "Flange Z height too low (322.1 mm < 410.0 mm); collision guard triggered",
        },
        "joints_rad": [0.0] * 6,
        "is_moving": False,
        "X_base_flange": low_flange,
        "flange_mm_deg": [300.1, -0.1, 322.1, -180.0, 0.0, 0.0],
        "tcp_mm_deg": [300.1, -0.1, 150.1, -180.0, 0.0, 0.0],
    }
    monkeypatch.setattr(robot, "read_state", lambda: low_state)

    with pytest.raises(CalibrationError) as exc:
        robot.plan_flange_pose(planner.waypoints[0]["X_base_flange"])

    assert not isinstance(exc.value, RejectedSample)
    msg = str(exc.value)
    assert "Current robot start pose violates safety guard before motion" in msg
    assert "current flange XYZ=[300.1, -0.1, 322.1] mm" in msg
    assert "TCP XYZ=[300.1, -0.1, 150.1] mm" in msg
    assert "controller TCP offset XYZ=[0.0, 0.0, 172.0] mm" in msg
    assert "lift +Z by >= 87.9 mm" in msg


def test_start_pose_outside_workspace_raises_calibration_error(monkeypatch):
    planner = WaypointPlanner(str(config_path("eye_in_hand/medium_range.yaml")))
    robot = XArmController("192.0.2.10", planner.robot_cfg, planner.default_cfg["robot"]["tool"])
    robot.connected = True
    robot.arm = SimpleNamespace()

    outside_flange = pose_from_mm_deg([150.0, 0.0, 450.0, -180.0, 0.0, 0.0])
    outside_state = {
        "state": 4,
        "error_code": 0,
        "warn_code": 0,
        "safety": {"safe": True, "message": "OK"},
        "joints_rad": [0.0] * 6,
        "is_moving": False,
        "X_base_flange": outside_flange,
        "flange_mm_deg": [150.0, 0.0, 450.0, -180.0, 0.0, 0.0],
        "tcp_mm_deg": [150.0, 0.0, 450.0, -180.0, 0.0, 0.0],
    }
    monkeypatch.setattr(robot, "read_state", lambda: outside_state)

    with pytest.raises(CalibrationError) as exc:
        robot.plan_flange_pose(planner.waypoints[0]["X_base_flange"])

    assert not isinstance(exc.value, RejectedSample)
    assert "Current robot start pose violates safety guard before motion" in str(exc.value)


def test_automatic_aborts_immediately_on_unsafe_start_pose_without_skipping_waypoints(tmp_path, monkeypatch):
    from camera_tool.services.eye_in_hand import EyeInHand
    from camera_tool.services.tasks import TaskContext

    service = EyeInHand(preset="medium_range", sim=True, output=tmp_path)
    robot = XArmController("192.0.2.10", service.planner.robot_cfg, service.planner.default_cfg["robot"]["tool"])
    robot.connected = True
    robot.arm = SimpleNamespace()
    robot.tcp_offset_mm_rad = np.array([0.0, 0.0, 172.0, 0.0, 0.0, 0.0])

    low_flange = pose_from_mm_deg([300.1, -0.1, 322.1, -180.0, 0.0, 0.0])
    low_state = {
        "state": 4,
        "error_code": 0,
        "warn_code": 0,
        "safety": {
            "safe": False,
            "message": "Flange Z height too low (322.1 mm < 410.0 mm); collision guard triggered",
        },
        "joints_rad": [0.0] * 6,
        "is_moving": False,
        "X_base_flange": low_flange,
        "flange_mm_deg": [300.1, -0.1, 322.1, -180.0, 0.0, 0.0],
        "tcp_mm_deg": [300.1, -0.1, 150.1, -180.0, 0.0, 0.0],
    }
    monkeypatch.setattr(robot, "read_state", lambda: low_state)
    service.robot = robot

    events: list[tuple[str, object]] = []
    context = TaskContext(lambda kind, value: events.append((kind, value)))

    with pytest.raises(CalibrationError, match="Current robot start pose violates safety guard before motion"):
        service._automatic(context)

    statuses = [str(val) for kind, val in events if kind == "status"]
    assert not any("Skipped" in msg for msg in statuses)
    assert not any("Moving to waypoint" in msg for msg in statuses)
