"""Hardware fault regressions using SDK stubs, without contacting devices."""
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
    robot = XArmController('192.0.2.10', planner.robot_cfg, planner.default_cfg['robot']['tool'])
    robot.connected = True
    return robot


def state(code=2, error=0):
    return {'state': code, 'error_code': error, 'safety': {'safe': True}, 'joints_rad': [0.0] * 6,
            'is_moving': code == 1, 'X_base_flange': pose_from_mm_deg([360, 10, 355, 180, 0, 0])}


@pytest.mark.parametrize('code,error', [(3, 0), (4, 0), (5, 0), (2, 1)])
def test_enable_never_resets_stopped_or_faulted_controller(controller, monkeypatch, code, error):
    calls = []
    controller.arm = SimpleNamespace(clean_error=lambda: calls.append('clear'), set_state=lambda _: calls.append('reset'))
    monkeypatch.setattr(controller, 'read_state', lambda: state(code, error))
    with pytest.raises(CalibrationError):
        controller.enable_motion()
    assert calls == []
    assert not controller.moving_enabled


def test_motion_fault_stops_without_submitting_another_joint_target(controller, monkeypatch):
    commands, stops = [], []
    controller.arm = SimpleNamespace(set_servo_angle=lambda **kw: commands.append(kw) or 0, set_state=lambda code: stops.append(code) or 0)
    controller.moving_enabled = True
    readings = iter([state(), state(4)])
    monkeypatch.setattr(controller, 'plan_flange_pose', lambda _: [np.zeros(6), np.ones(6) * 0.01])
    monkeypatch.setattr(controller, 'read_state', lambda: next(readings))
    with pytest.raises(CalibrationError):
        controller.move_to_flange_pose(state()['X_base_flange'])
    assert len(commands) == 1
    assert stops == [4]


def test_unsafe_joint_interpolation_is_rejected_before_motion(controller, monkeypatch):
    controller.arm = SimpleNamespace(
        get_inverse_kinematics=lambda *a, **kw: (0, [0.01] * 6),
        get_forward_kinematics=lambda *a, **kw: (0, [360, 10, 220, np.pi, 0, 0]),
    )
    monkeypatch.setattr(controller, 'read_state', state)
    with pytest.raises(RejectedSample):
        controller.plan_flange_pose(state()['X_base_flange'])
    assert not controller.moving_enabled


def test_failed_connection_can_still_release_sdk_socket(controller):
    disconnected = []
    controller.arm = SimpleNamespace(disconnect=lambda: disconnected.append(True))
    controller.connected = False
    controller.close()
    assert disconnected == [True]
