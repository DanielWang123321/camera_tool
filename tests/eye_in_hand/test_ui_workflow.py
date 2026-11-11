"""Desktop workflows use real services with synthetic hardware and Qt events."""
from __future__ import annotations
import threading
import time
import numpy as np
import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from camera_tool.calibration.eye_in_hand.session import CalibrationSession
from camera_tool.ui.main_window import MainWindow


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        # Yield the GIL as well as processing Qt events for Python workers.
        time.sleep(.005)
    QApplication.processEvents()
    assert predicate(), "GUI task did not finish"


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = MainWindow(sim_mode=True, initial_page=1)
    widget.resize(1280, 720)
    widget.show()
    widget.start_calibration()
    wait_until(lambda: widget.payload and widget.payload.get("session"))
    yield widget
    widget.close()
    wait_until(lambda: not widget.workers)
    widget.deleteLater()
    app.processEvents()


def command(window, name, value=None):
    window._command(name, value)
    wait_until(lambda: not window._command_busy)


def samples(window, count=8):
    for _ in range(count):
        command(window, "next")
        command(window, "capture")


def solve(window):
    command(window, "solve")
    assert window.payload["solution"] is not None
    return window.payload["solution"]


def save_replay(window, base_dir):
    session = CalibrationSession(base_dir)
    folder = session.start_new_session(window.backend.session.camera_metadata, window.backend.session.robot_metadata)
    for sample in window.backend.session.samples:
        session.add_sample(sample["color_bgr"], sample, sample, waypoint_name=sample["name"])
    session._save_samples_index()
    return folder


def test_five_pages_and_camera_3d_navigation(window):
    assert window.pages.count() == 5
    for index, button in enumerate(window.nav_buttons):
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        assert window.pages.currentIndex() == index
    window._navigate(1)
    QTest.mouseClick(window.btn_view, Qt.MouseButton.LeftButton)
    assert window.capture_views.currentWidget() is window.scene3d_panel
    QTest.mouseClick(window.btn_view, Qt.MouseButton.LeftButton)
    assert window.capture_views.currentWidget() is window.video_panel


def test_automatic_capture_solves_and_keeps_desktop_responsive(window):
    ticks = []
    timer = QTimer(window)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    command(window, "auto")
    timer.stop()
    assert len(window.payload["samples"]) == 20
    assert window.payload["solution"]["metrics"]["rms_px"] < 1
    assert window.pages.currentIndex() == 2
    assert window.result_panel.btn_export.isEnabled()
    assert len(ticks) >= 10


def test_sample_changes_invalidate_result_views_and_export(window):
    samples(window)
    solve(window)
    window.result_panel.tbl_samples.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    wait_until(lambda: not window._command_busy)
    assert window.payload["solution"] is None
    assert window.result_panel.lbl_count.text() == "7"
    assert window.result_panel.lbl_rms.text() == "— px"
    assert window.result_panel.tbl_methods.rowCount() == 0
    assert not window.result_panel.btn_export.isEnabled()
    solve(window)
    command(window, "capture")
    assert window.payload["solution"] is None
    command(window, "clear")
    assert window.result_panel.tbl_samples.rowCount() == 0
    assert not window.result_panel.btn_solve.isEnabled()


def test_sample_preview_is_unchanged_by_stream_updates(window):
    samples(window, 2)
    window._navigate(2)
    window.select_sample(0)
    image = window.result_panel.sample_preview.image_bgr.copy()
    command(window, "next")
    window._consume_frames()
    assert np.array_equal(window.result_panel.sample_preview.image_bgr, image)


def test_space_capture_respects_text_editing_and_current_page(window):
    window._navigate(1)
    window.cal_config.setFocus()
    window._capture_shortcut()
    assert not window.backend.session.samples
    window.cal_config.clearFocus()
    window.setFocus()
    window._capture_shortcut()
    wait_until(lambda: not window._command_busy)
    assert len(window.backend.session.samples) == 1
    window._navigate(0)
    window._capture_shortcut()
    assert len(window.backend.session.samples) == 1


@pytest.mark.parametrize("count", [0, 2, 8])
def test_replay_has_no_hardware_and_inspects_partial_sessions(window, tmp_path, count):
    samples(window, count)
    folder = save_replay(window, tmp_path)
    camera, robot = window.backend.camera, window.backend.robot
    window.stop_task("calibration")
    wait_until(lambda: "calibration" not in window.workers)
    assert not camera.connected and not robot.connected
    window.load_replay_directory(folder)
    wait_until(lambda: "replay" not in window.workers)
    assert window.payload["replay"]
    assert len(window.payload["samples"]) == count
    assert window.pages.currentIndex() == 2
    assert window.result_panel.btn_solve.isEnabled() == (count >= 5)
    assert not window.btn_capture.isEnabled()


def test_replay_export_does_not_write_into_original_directory(window, tmp_path):
    samples(window)
    folder = save_replay(window, tmp_path / "old")
    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    window.stop_task("calibration")
    wait_until(lambda: not window.workers)
    window.load_replay_directory(folder)
    wait_until(lambda: not window.workers)
    window.payload["session"].export_result(window.payload["solution"], tmp_path / "new.yaml")
    after = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert before == after
    assert (tmp_path / "new.yaml").is_file()
    with pytest.raises(RuntimeError, match="只读"):
        window.payload["session"].export_result(window.payload["solution"], folder / "result.yaml")


def test_stream_failure_stops_and_releases_devices(window, monkeypatch):
    camera, robot = window.backend.camera, window.backend.robot
    def fail(**kwargs):
        raise RuntimeError("测试断流")
    monkeypatch.setattr(camera, "read", fail)
    wait_until(lambda: not window.workers)
    assert not camera.connected and not robot.connected
    assert "测试断流" in window.status_bar.currentMessage()
    assert not window.btn_capture.isEnabled()


def test_failed_motion_stops_without_capturing(window, monkeypatch):
    def fail(pose):
        raise RuntimeError("测试机械臂异常")
    robot = window.backend.robot
    monkeypatch.setattr(robot, "move_to_flange_pose", fail)
    window._command("auto")
    wait_until(lambda: not window.workers)
    assert not window.backend.session.samples
    assert not robot.connected
    assert "测试机械臂异常" in window.status_bar.currentMessage()


def test_close_waits_for_motion_cleanup(window, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    robot = window.backend.robot
    original = robot.move_to_flange_pose
    def blocked(pose):
        entered.set()
        assert release.wait(5)
        return original(pose)
    monkeypatch.setattr(robot, "move_to_flange_pose", blocked)
    window._command("next")
    wait_until(entered.is_set)
    try:
        window.close()
        assert window.isVisible() and window._closing
        assert robot.connected
    finally:
        release.set()
    wait_until(lambda: not window.workers)
    assert not window.isVisible() and not robot.connected


@pytest.mark.parametrize("size", [(1280, 720), (1366, 768), (1920, 1080)])
def test_controls_fit_supported_screens(window, size):
    window.resize(*size)
    for index in range(5):
        window._navigate(index)
        QApplication.processEvents()
        assert window.size().width() == size[0]
        assert window.size().height() == size[1]
        assert window.pages.width() <= window.width()


def test_eye_to_hand_hides_unsupported_controls(window):
    window.stop_task("calibration")
    wait_until(lambda: not window.workers)
    window.combo_mode.setCurrentIndex(1)
    window._show_payload({"mode": "eye-to-hand", "samples": [], "solution": None, "replay": True})
    assert window.capture_views.currentIndex() == 2
    assert not window.chk_sim.isEnabled()
    assert window.result_panel.tbl_samples.isColumnHidden(0)
    assert not window.result_panel.tabs.isTabVisible(window.result_panel.tabs.indexOf(window.result_panel.tbl_methods))
    assert not window.result_panel.btn_export.isEnabled()
