"""Cancellation, shared hardware ownership, USB accounting and offline boundaries."""
from __future__ import annotations

import csv
import importlib
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from camera_tool.core import CalibrationError
from camera_tool.services.tasks import Cancelled, DeviceManager, TaskContext, task_scope
from camera_tool.services.usb import UsbStatistics, usb_test


class NativeExitDriver:
    def __init__(self, config): pass
    def start(self): os._exit(9)
    def stop(self): pass


class BlockingDriver:
    def __init__(self, config): pass
    def start(self): return {"serial": "fake"}
    def read(self): time.sleep(60)
    def stop(self): pass


def test_native_sdk_exit_does_not_kill_the_host_and_releases_process():
    from camera_tool.devices.process_camera import ProcessCamera
    camera = ProcessCamera(f"{__name__}:NativeExitDriver", {})
    with pytest.raises(RuntimeError, match="SDK process exited"):
        camera.start()
    assert camera.process is None


def test_blocking_sdk_read_can_be_stopped_without_waiting_for_the_driver():
    from camera_tool.devices.process_camera import ProcessCamera
    camera = ProcessCamera(f"{__name__}:BlockingDriver", {})
    camera.start()
    started = time.monotonic()
    camera.stop()
    assert time.monotonic() - started < 2
    assert camera.process is None


def test_device_leases_reject_wildcards_and_release_after_failure(tmp_path):
    manager = DeviceManager(tmp_path)
    with manager.lease(["realsense:A"], "first"):
        with pytest.raises(RuntimeError):
            with manager.lease(["realsense:*"], "second"):
                pass
        with pytest.raises(RuntimeError):
            with manager.lease(["realsense:A"], "first"):
                pass
        with manager.lease(["realsense:B"], "third"):
            pass
    with manager.lease(["realsense:*", "robot:192.0.2.1"], "next"):
        pass
    assert manager._held == []


def test_separate_process_cannot_acquire_a_leased_device(tmp_path):
    manager = DeviceManager(tmp_path)
    code = "from camera_tool.services.tasks import DeviceManager; import sys; m=DeviceManager(sys.argv[1]);\nwith m.lease(['realsense:A'], 'child'): pass"
    with manager.lease(["realsense:*"], "desktop"):
        process = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True,
                                 env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        assert process.returncode != 0
        assert "another" in process.stderr


def test_frame_mailbox_is_bounded_and_cancellation_reaches_solver():
    context = TaskContext()
    for number in range(1000):
        context.frame("preview", {"frame_number": number})
    assert context.take_frames() == {"preview": {"frame_number": 999}}
    assert context.take_frames() == {}
    context.cancelled.set()
    from camera_tool.core import CameraModel
    model = CameraModel({"width": 640, "height": 480, "model": "pinhole", "K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]], "distortion": [0.] * 5})
    with task_scope(context), pytest.raises(Cancelled):
        model.residual(np.eye(4), np.zeros((4, 2)), .15)


def test_usb_statistics_include_missing_frames_resets_trailing_stalls_and_initial_link():
    stats = UsbStatistics(min_fps=29, max_gap=.1)
    assert not stats.snapshot(0, final=True)["stable"]
    stats.link("2.x", 480)
    stats.record(1., 10)
    stats.record(1. + 1/30, 11)
    stats.record(1. + 2/30, 14)
    stats.record(1. + 3/30, 1)
    report = stats.snapshot(2., "3.x", 5000, final=True)
    assert report["missing_frame_numbers"] == 2
    assert report["counter_resets"] == 1
    assert report["min_usb_speed_mbps"] == 480
    assert report["max_gap_s"] == pytest.approx(.9)
    assert report["low_fps"] and not report["stable"]


def test_usb_measures_every_capture_and_exports_final_partial_window(tmp_path):
    class Camera:
        count = 0
        closed = False
        def start(self): return {"serial": "fake"}
        def read(self):
            time.sleep(.003)
            self.count += 1
            return {"timestamp_mono": time.monotonic(), "frame_number": self.count}
        def stop(self): self.closed = True
    camera = Camera()
    events = []
    context = TaskContext(lambda kind, data: events.append((kind, data)))
    path = tmp_path / "stats.csv"
    payload = usb_test(context, {"backend": "realsense", "serial": "TEST-USB"}, duration=.08, interval=.03,
                       csv_path=path, camera_factory=lambda *args: camera, link_reader=lambda: ("3.x", 5000))
    assert camera.closed
    assert payload["summary"]["frame_count"] == camera.count
    assert payload["summary"]["stable"]
    assert sum(row["frame_count"] for row in payload["rows"]) == camera.count
    assert len(list(csv.DictReader(path.open()))) == len(payload["rows"])


def test_usb_failure_stops_partial_camera_and_releases_lease(tmp_path):
    class Camera:
        stopped = False
        def start(self): raise RuntimeError("USB start failed")
        def stop(self): self.stopped = True
    camera = Camera()
    with pytest.raises(RuntimeError, match="start failed"):
        usb_test(TaskContext(), {"serial": "TEST-USB"}, camera_factory=lambda *args: camera)
    assert camera.stopped
    from camera_tool.services.tasks import DEVICES
    with DEVICES.lease(["realsense:TEST-USB"], "after failure"):
        pass


def test_usb_stop_still_returns_statistics_and_csv(tmp_path):
    context = TaskContext()
    class Camera:
        number = 0
        def start(self): return {"serial": "fake"}
        def read(self):
            self.number += 1
            if self.number == 4:
                context.cancelled.set()
            return {"timestamp_mono": 10 + self.number / 30, "frame_number": self.number}
        def stop(self): pass
    payload = usb_test(context, {"serial": "TEST-STOP"}, csv_path=tmp_path / "stopped.csv",
                       camera_factory=lambda *args: Camera(), link_reader=lambda: ("3.x", 5000))
    assert payload["summary"]["stopped"]
    assert payload["summary"]["frame_count"] == 4
    assert (tmp_path / "stopped.csv").exists()


def test_preview_start_failure_always_closes_camera(monkeypatch):
    module = importlib.import_module("camera_tool.services.preview")
    class Camera:
        closed = False
        def start(self): raise RuntimeError("Camera start failed")
        def stop(self): self.closed = True
    camera = Camera()
    monkeypatch.setattr(module, "create_camera", lambda *args: camera)
    with pytest.raises(RuntimeError):
        module.preview(TaskContext(), "realsense", {"serial": "TEST-PREVIEW"})
    assert camera.closed


def test_eye_in_hand_check_sends_no_motion_or_result(tmp_path, monkeypatch):
    from camera_tool.services.eye_in_hand import EyeInHand
    backend = EyeInHand(sim=True, output=tmp_path)
    calls = []
    monkeypatch.setattr(backend, "move_next", lambda: calls.append("move"))
    result = backend.run(TaskContext(), check=True, result=tmp_path / "forbidden.yaml")
    assert result["motion_commands_sent"] is False
    assert calls == []
    assert not backend.camera.connected and not backend.robot.connected
    assert not (tmp_path / "forbidden.yaml").exists()


def test_eye_in_hand_check_accepts_pre_enable_state4(tmp_path, monkeypatch):
    from camera_tool.devices.robot import XArmController
    from camera_tool.services.eye_in_hand import EyeInHand

    calls = []
    fake_arm = SimpleNamespace(
        connected=True,
        axis=6,
        device_type=6,
        version="2.7.103",
        tcp_offset=[0.0] * 6,
        connect=lambda: None,
        disconnect=lambda: None,
        get_err_warn_code=lambda: (0, (0, 0)),
        get_state=lambda: (0, 4),
        get_position=lambda is_radian=True: (0, [400.0, 0.0, 355.0, np.pi, 0.0, 0.0]),
        get_servo_angle=lambda is_radian=True, is_real=True: (0, [0.0, -0.5, -0.8, 0.0, 1.3, 0.0]),
        motion_enable=lambda **kw: calls.append(("motion_enable", kw)) or 0,
        set_mode=lambda mode: calls.append(("set_mode", mode)) or 0,
        set_state=lambda st: calls.append(("set_state", st)) or 0,
    )
    monkeypatch.setitem(sys.modules, "xarm", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "xarm.core",
        SimpleNamespace(config=SimpleNamespace(x_config=SimpleNamespace(XCONF=SimpleNamespace(Robot=SimpleNamespace(JOINT_LIMITS={6: {6: [[-6.0, 6.0]] * 6}}))))),
    )
    monkeypatch.setitem(sys.modules, "xarm.core.config", sys.modules["xarm.core"].config)
    monkeypatch.setitem(sys.modules, "xarm.core.config.x_config", sys.modules["xarm.core"].config.x_config)
    monkeypatch.setitem(sys.modules, "xarm.wrapper", SimpleNamespace(XArmAPI=lambda *a, **kw: fake_arm))

    backend = EyeInHand(sim=True, output=tmp_path)
    orig_open = backend.open

    def open_with_xarm(context=None, persist=True):
        orig_open(context, persist=persist)
        ctrl = XArmController("192.0.2.10", backend.planner.robot_cfg, backend.planner.default_cfg["robot"]["tool"])
        ctrl.connect()
        backend.robot = ctrl

    monkeypatch.setattr(backend, "open", open_with_xarm)
    result = backend.run(TaskContext(), check=True, result=tmp_path / "forbidden.yaml")
    assert result["motion_commands_sent"] is False
    assert result["robot"]["state"] == 4
    assert result["robot"]["error_code"] == 0
    assert result["robot"]["ready_for_motion"] is True
    assert calls == []


@pytest.mark.parametrize("state", [3, 4, 5])
def test_eye_to_hand_start_allows_pre_enable_states_without_sending_commands(state):
    from camera_tool.calibration.eye_to_hand.workflow import load_settings
    from camera_tool.devices.robot import TcpCalibrationRobot
    from camera_tool.paths import config_path

    cfg = load_settings(config_path("eye_to_hand.yaml"))
    calls = []
    pose = [400.0, 0.0, 250.0, np.pi, 0.0, 0.0]
    arm = SimpleNamespace(
        connected=True,
        axis=6,
        device_type=6,
        version_number=(2, 7, 103),
        version="2.7.103",
        self_collision_params=[1],
        tcp_offset=[0.0] * 6,
        world_offset=[0.0] * 6,
        get_err_warn_code=lambda: (0, (0, 0)),
        get_state=lambda: (0, state),
        get_position=lambda is_radian=True: (0, list(pose)),
        get_servo_angle=lambda is_radian=True, is_real=True: (0, [0.0, -0.5, -0.8, 0.0, 1.3, 0.0]),
        get_forward_kinematics=lambda q, **kw: (0, list(pose)),
        set_state=lambda value: calls.append(("set_state", value)) or 0,
        motion_enable=lambda **kw: calls.append(("motion_enable", kw)) or 0,
    )
    robot = TcpCalibrationRobot(cfg["robot"], cfg["motion"], lambda *args, **kw: None, arm=arm)
    info = robot.start()
    assert info["state"] == state
    assert calls == []
    assert not robot.moving_enabled


def test_eye_to_hand_start_rejects_hardware_error_without_reset():
    from camera_tool.calibration.eye_to_hand.workflow import load_settings
    from camera_tool.devices.robot import TcpCalibrationRobot
    from camera_tool.paths import config_path

    cfg = load_settings(config_path("eye_to_hand.yaml"))
    calls = []
    arm = SimpleNamespace(
        connected=True,
        axis=6,
        device_type=6,
        version_number=(2, 7, 103),
        tcp_offset=[0.0] * 6,
        world_offset=[0.0] * 6,
        get_err_warn_code=lambda: (0, (22, 0)),
        get_state=lambda: (0, 4),
        clean_error=lambda: calls.append("clean_error") or 0,
        set_state=lambda value: calls.append(value),
        motion_enable=lambda **kw: calls.append(kw),
    )
    robot = TcpCalibrationRobot(cfg["robot"], cfg["motion"], lambda *args, **kw: None, arm=arm)
    with pytest.raises(CalibrationError, match="error=22"):
        robot.start()
    assert calls == []


@pytest.mark.parametrize("post_state,should_succeed", [(2, True), (0, True), (4, False), (5, False)])
def test_eye_to_hand_enable_queries_state_after_set_state(post_state, should_succeed):
    from camera_tool.calibration.eye_to_hand.workflow import load_settings
    from camera_tool.devices.robot import TcpCalibrationRobot
    from camera_tool.paths import config_path

    cfg = load_settings(config_path("eye_to_hand.yaml"))
    calls = []
    current = {"state": 4}
    pose = [400.0, 0.0, 250.0, np.pi, 0.0, 0.0]

    def set_state(value):
        calls.append(("set_state", value))
        if value == 0:
            current["state"] = post_state
        return 0

    arm = SimpleNamespace(
        connected=True,
        axis=6,
        device_type=6,
        version_number=(2, 7, 103),
        tcp_offset=[0.0] * 6,
        world_offset=[0.0] * 6,
        get_err_warn_code=lambda: (0, (0, 0)),
        get_state=lambda: (0, current["state"]),
        get_position=lambda is_radian=True: (0, list(pose)),
        get_servo_angle=lambda is_radian=True, is_real=True: (0, [0.0] * 6),
        motion_enable=lambda **kw: calls.append(("motion_enable", kw.get("enable"))) or 0,
        set_mode=lambda mode: calls.append(("set_mode", mode)) or 0,
        set_self_collision_detection=lambda on: calls.append(("self_collision", on)) or 0,
        set_state=set_state,
    )
    robot = TcpCalibrationRobot(cfg["robot"], cfg["motion"], lambda *args, **kw: None, arm=arm)
    if should_succeed:
        robot.enable()
        assert robot.moving_enabled is True
    else:
        with pytest.raises(CalibrationError, match="not ready after set_state"):
            robot.enable()
        assert robot.moving_enabled is False
    assert calls == [
        ("motion_enable", True),
        ("set_mode", 0),
        ("self_collision", True),
        ("set_state", 0),
    ]


def test_eye_to_hand_replay_cannot_publish_even_when_result_says_passed(tmp_path):
    from camera_tool.calibration.eye_to_hand.workflow import publish_camera
    from camera_tool.core import Session
    session = Session(tmp_path, "replay")
    with pytest.raises(RuntimeError, match="live hardware run"):
        publish_camera(tmp_path / "forbidden.yaml", {"passed": True}, session)
    assert not (tmp_path / "forbidden.yaml").exists()


def test_offline_cli_help_and_replay_do_not_import_hardware_sdks(tmp_path):
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'pyrealsense2', 'pyorbbecsdk', 'xarm', 'PySide6'}:
        raise AssertionError('offline code imported ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from camera_tool.cli import main
assert main(['replay', '/nonexistent/session']) == 1
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr
