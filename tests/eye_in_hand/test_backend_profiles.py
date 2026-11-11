"""Tests for RGB-D camera backends, medium_range preset, board/table geometry checks, and offline TCP offset replay."""
from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from camera_tool.calibration.eye_in_hand.session import CalibrationSession
from camera_tool.calibration.eye_in_hand.solver import evaluate_board_geometry, evaluate_depth_consistency
from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner
from camera_tool.cli import main as cli_main
from camera_tool.core import (
    CameraModel,
    MarkerDetector,
    inv_transform,
    make_transform,
    marker_object_points,
    pose_error,
    pose_from_mm_deg,
    pose_from_mm_rad,
    rotation_coverage,
    transform_points,
)
from camera_tool.devices.realsense import RealSenseCamera
from camera_tool.diagnostics.usb_link import find_realsense_sysfs
from camera_tool.paths import config_path
from camera_tool.services.eye_in_hand import EyeInHand, detect_usb_camera_backend, solve_session


def _euler_xyz_to_matrix(euler6: list[float]) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = Rotation.from_euler("xyz", euler6[3:], degrees=False).as_matrix()
    mat[:3, 3] = np.asarray(euler6[:3], dtype=float)
    return mat


def test_usb_camera_backend_auto_detection(tmp_path) -> None:
    assert detect_usb_camera_backend(tmp_path) == "realsense"

    orbbec_dir = tmp_path / "2-1"
    orbbec_dir.mkdir()
    (orbbec_dir / "idVendor").write_text("2bc5\n", encoding="utf-8")
    (orbbec_dir / "idProduct").write_text("0840\n", encoding="utf-8")
    assert detect_usb_camera_backend(tmp_path) == "orbbec"

    rs_dir = tmp_path / "2-2"
    rs_dir.mkdir()
    (rs_dir / "idVendor").write_text("8086\n", encoding="utf-8")
    (rs_dir / "idProduct").write_text("0b3a\n", encoding="utf-8")
    assert detect_usb_camera_backend(tmp_path) == "realsense"


def test_realsense_camera_starts_color_and_depth_without_imu(monkeypatch) -> None:
    class Video:
        def __init__(self, kind, width, height):
            self.kind, self.w, self.h = kind, width, height

        def stream_type(self):
            return self.kind

        def format(self):
            return "bgr8" if self.kind == "color" else "z16"

        def is_video_stream_profile(self):
            return True

        def as_video_stream_profile(self):
            return self

        def width(self):
            return self.w

        def height(self):
            return self.h

        def fps(self):
            return 30

        def get_intrinsics(self):
            return SimpleNamespace(
                width=self.w,
                height=self.h,
                fx=911.69,
                fy=911.90,
                ppx=636.01,
                ppy=356.53,
                coeffs=[0.012, -0.021, 0.0, 0.0, 0.0],
                model="distortion.inverse_brown_conrady",
            )

        def get_extrinsics_to(self, _other):
            return SimpleNamespace(rotation=np.eye(3).ravel().tolist(), translation=[0.015, 0.0, 0.0])

    streams = [Video("color", 1280, 720), Video("depth", 1280, 720)]
    sensor = SimpleNamespace(get_stream_profiles=lambda: streams, get_depth_scale=lambda: 0.001)
    device = SimpleNamespace(
        supports=lambda _k: True,
        get_info=lambda key: {"serial": "MOCK_OPTICAL_001", "name": "RealSense RGB-D"}[key],
        query_sensors=lambda: [sensor],
        first_depth_sensor=lambda: sensor,
    )
    enabled_streams = []

    class Config:
        def enable_device(self, _serial):
            pass

        def enable_stream(self, *args):
            enabled_streams.append(args)

        def can_resolve(self, _wrapper):
            return True

    profile = SimpleNamespace(get_device=lambda: device, get_stream=lambda k: streams[0 if k == "color" else 1])
    pipeline = SimpleNamespace(
        start=lambda _cfg: profile,
        stop=lambda: None,
        wait_for_frames=lambda _timeout: None,
    )
    module = SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: [device]),
        pipeline=lambda: pipeline,
        pipeline_wrapper=lambda obj: obj,
        config=Config,
        align=lambda _: None,
        stream=SimpleNamespace(color="color", depth="depth", accel="accel", gyro="gyro"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        camera_info=SimpleNamespace(serial_number="serial", name="name"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", module)

    cam = RealSenseCamera({"width": 1280, "height": 720, "fps": 30, "align_depth_to_color": True})
    meta = cam.start()
    try:
        assert meta["name"] == "RealSense RGB-D"
        assert meta["serial"] == "MOCK_OPTICAL_001"
        assert meta["model"] == "inverse_brown_conrady"
        assert meta["camera_frame"] == "color_optical_frame"
        assert meta["depth_aligned_to_color"] is True
        assert enabled_streams == [("color", 1280, 720, "bgr8", 30), ("depth", 1280, 720, "z16", 30)]
    finally:
        cam.close()


def test_usb_link_resolves_via_physical_port_and_asic_serial(tmp_path) -> None:
    dev = tmp_path / "2-2"
    dev.mkdir()
    (dev / "idVendor").write_text("8086\n", encoding="utf-8")
    (dev / "idProduct").write_text("0b3a\n", encoding="utf-8")
    (dev / "serial").write_text("MOCK_ASIC_001\n", encoding="utf-8")
    (dev / "speed").write_text("5000\n", encoding="utf-8")
    (dev / "version").write_text(" 3.20\n", encoding="utf-8")

    found_by_port = find_realsense_sysfs(
        serial="MOCK_OPTICAL_001",
        physical_port="/sys/devices/pci0000:00/0000:00:14.0/usb2/2-2/2-2:1.0/video4linux/video0",
        sysfs_root=tmp_path,
    )
    assert found_by_port is not None
    assert found_by_port.path == "2-2"
    assert found_by_port.speed_mbps == 5000
    assert found_by_port.is_usb3()

    found_by_asic = find_realsense_sysfs(
        serial="MOCK_OPTICAL_001",
        asic_serial="MOCK_ASIC_001",
        sysfs_root=tmp_path,
    )
    assert found_by_asic is not None
    assert found_by_asic.path == "2-2"
    assert found_by_asic.speed_mbps == 5000


def test_medium_range_preset_waypoints_safety_fov_and_rotation_coverage() -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/medium_range.yaml")))
    assert len(planner.waypoints) == 20
    assert np.allclose(planner.target_center_m, [0.41, 0.0, 0.0])

    flange_poses = [wp["X_base_flange"] for wp in planner.waypoints]
    cov_deg = rotation_coverage(flange_poses, minimum_deg=6.0)
    assert cov_deg[0] >= 12.0
    assert cov_deg[1] >= 6.0

    cam_meta = {
        "width": 1280,
        "height": 720,
        "model": "inverse_brown_conrady",
        "K": [[911.69, 0.0, 636.01], [0.0, 911.90, 356.53], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam_model = CameraModel(cam_meta)
    obj_pts = marker_object_points(0.15)
    board_pose = make_transform(np.eye(3), [0.41, 0.0, 0.0])

    mounts = [
        pose_from_mm_deg([67.0, -31.0, 21.6, 0.0, 0.0, 90.0]),
        pose_from_mm_deg([64.4, 31.1, 22.8, 0.0, 0.0, -88.0]),
        pose_from_mm_deg([50.0, 0.0, 30.0, 0.0, 0.0, 0.0]),
    ]
    for mount in mounts:
        for wp in planner.waypoints:
            assert wp["safety"]["flange_z_mm"] >= 410.0
            assert wp["safety"]["table_clearance_mm"] >= 340.0
            x_cam_board = inv_transform(wp["X_base_flange"] @ mount) @ board_pose
            pts_cam = transform_points(x_cam_board, obj_pts)
            assert np.all(pts_cam[:, 2] >= 0.30) and np.all(pts_cam[:, 2] <= 0.65), wp["name"]
            uv = cam_model.project(pts_cam)
            assert np.all(uv[:, 0] >= 25.0) and np.all(uv[:, 0] <= 1255.0), wp["name"]
            assert np.all(uv[:, 1] >= 25.0) and np.all(uv[:, 1] <= 695.0), wp["name"]


def test_table_offset_and_board_thickness_geometry_and_depth_leveling() -> None:
    flange = pose_from_mm_deg([410.0, 0.0, 445.0, 180.0, 0.0, 0.0])
    flange_cam = pose_from_mm_deg([67.0, -31.0, 21.6, 0.0, 0.0, 90.0])
    base_board = make_transform(np.eye(3), [0.41, 0.0, -0.007])

    board_report = evaluate_board_geometry(base_board, table_z_m=-0.008, board_thickness_m=0.001)
    assert board_report["ok"] is True
    assert np.isclose(board_report["expected_board_z_mm"], -7.0)
    assert np.isclose(board_report["estimated_table_z_mm"], -8.0)
    assert board_report["board_height_error_mm"] < 1e-6

    rng = np.random.default_rng(123)
    xs, ys = np.meshgrid(np.linspace(0.28, 0.54, 24), np.linspace(-0.13, 0.13, 24))
    on_board = (np.abs(xs - 0.41) <= 0.08) & (np.abs(ys) <= 0.08)
    zs = np.where(on_board, -0.007, -0.008) + rng.normal(0.0, 0.0002, size=xs.shape)
    pts_base = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    pts_cam = transform_points(inv_transform(flange @ flange_cam), pts_base)

    depth_report = evaluate_depth_consistency(
        pts_cam,
        flange,
        flange_cam,
        base_board,
        table_z_m=-0.008,
        board_thickness_m=0.001,
        board_side_m=0.15,
        min_depth_m=0.20,
        max_depth_m=1.20,
    )
    assert depth_report["ok"] is True
    assert depth_report["table_height_error_mm"] < 0.3
    assert depth_report["normal_angle_deg"] < 0.5


def test_export_and_offline_replay_tcp_offset_override(tmp_path) -> None:
    planner = WaypointPlanner(str(config_path("eye_in_hand/medium_range.yaml")))
    cfg = copy.deepcopy(planner.default_cfg)
    cfg["checks"]["table_z_m"] = -0.008
    cfg["checks"]["board_thickness_m"] = 0.001

    cam_meta = {
        "name": "RealSense RGB-D",
        "serial": "MOCK_OPTICAL_001",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "model": "inverse_brown_conrady",
        "camera_frame": "color_optical_frame",
        "depth_aligned_to_color": True,
        "sdk_depth_unit_m": 0.001,
        "K": [[911.69, 0.0, 636.01], [0.0, 911.90, 356.53], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam_model = CameraModel(cam_meta)
    detector = MarkerDetector(cfg["marker"], cam_model)

    gt_flange_cam = pose_from_mm_deg([66.5, -30.8, 22.1, 0.8, -1.1, 89.4])
    gt_base_board = make_transform(
        Rotation.from_euler("xyz", [0.3, -0.2, 1.5], degrees=True).as_matrix(),
        [0.412, -0.004, -0.007],
    )

    custom_cfg_path = tmp_path / "custom_medium.yaml"
    custom_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    session = CalibrationSession(tmp_path / "runs")
    session_dir = session.start_new_session(
        cam_meta,
        {"profile": "arm_6dof", "tcp_offset_mm_rad": [0.0] * 6, "simulated": False, "preset": "medium_range"},
        persist=True,
    )
    assert session_dir is not None

    obj_pts = marker_object_points(0.15)
    dummy_img = np.full((720, 1280, 3), 200, dtype=np.uint8)
    for wp in planner.waypoints:
        flange = wp["X_base_flange"]
        x_cam_board = inv_transform(flange @ gt_flange_cam) @ gt_base_board
        uv = cam_model.project(transform_points(x_cam_board, obj_pts))
        eval_res = {
            "detected": True,
            "passed": True,
            "corners": uv.astype(np.float32),
            "area_px": 9500.0,
            "sharpness": 120.0,
            "pnp_rms_px": 0.05,
            "pose_cam_board": x_cam_board,
        }
        overlay = detector.draw_overlay(dummy_img, eval_res)
        assert overlay.shape == dummy_img.shape
        state = {"X_base_flange": flange, "X_base_tcp": flange, "safety": wp["safety"]}
        session.add_sample(dummy_img, eval_res, state, waypoint_name=wp["name"])

    solution = solve_session(session, cfg)
    assert solution["board_check"]["ok"] is True
    err_mm, err_deg = pose_error(solution["X_flange_cam"], gt_flange_cam)
    assert err_mm < 0.2 and err_deg < 0.05

    no_tool_yaml = tmp_path / "eye_in_hand_no_tool.yaml"
    session.export_result(solution, no_tool_yaml, cfg=cfg)
    exported = yaml.safe_load(no_tool_yaml.read_text(encoding="utf-8"))

    ext = exported["extrinsics"]
    uf_block = ext["ufactory_vision"]
    assert uf_block["camera_frame"] == "color_optical_frame"
    assert uf_block["depth_aligned_to_color"] is True
    assert uf_block["EULER_COLOR_TO_DEPTH_OPT"] == [0.0] * 6
    assert uf_block["table_z_base_m"] == -0.008
    assert uf_block["board_thickness_m"] == 0.001

    x_tcp_cam_no_tool = np.asarray(ext["X_tcp_cam"], dtype=float)
    assert np.allclose(x_tcp_cam_no_tool, np.asarray(ext["X_flange_cam"], dtype=float), atol=1e-12)
    reconstructed = _euler_xyz_to_matrix(uf_block["EULER_EEF_TO_COLOR_OPT"])
    assert np.max(np.abs(reconstructed - x_tcp_cam_no_tool)) <= 1e-9

    with_tool_yaml = tmp_path / "eye_in_hand_with_tool.yaml"
    rc = cli_main(
        [
            "replay",
            str(session_dir),
            "--config",
            str(custom_cfg_path),
            "--tcp-offset",
            "0,0,172,0,0,0.2",
            "--result",
            str(with_tool_yaml),
        ]
    )
    assert rc == 0
    exported_g = yaml.safe_load(with_tool_yaml.read_text(encoding="utf-8"))
    ext_g = exported_g["extrinsics"]
    assert ext_g["tcp_offset_mm_rad"] == [0.0, 0.0, 172.0, 0.0, 0.0, 0.2]
    assert np.allclose(ext_g["X_flange_cam"], ext["X_flange_cam"], atol=1e-8)

    expected_tcp_cam = inv_transform(pose_from_mm_rad([0.0, 0.0, 172.0, 0.0, 0.0, 0.2])) @ np.asarray(
        ext_g["X_flange_cam"], dtype=float
    )
    assert np.allclose(ext_g["X_tcp_cam"], expected_tcp_cam, atol=1e-8)
    reconstructed_g = _euler_xyz_to_matrix(ext_g["ufactory_vision"]["EULER_EEF_TO_COLOR_OPT"])
    assert np.max(np.abs(reconstructed_g - np.asarray(ext_g["X_tcp_cam"], dtype=float))) <= 1e-9


def test_robot_ip_override_priority(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("XARM_IP", raising=False)
    eih_default = EyeInHand(preset="medium_range", sim=False, output=tmp_path)
    assert eih_default.cfg["robot"]["ip"] == ""

    monkeypatch.setenv("XARM_IP", "192.0.2.10")
    eih_env = EyeInHand(preset="medium_range", sim=False, output=tmp_path)
    assert eih_env.cfg["robot"]["ip"] == "192.0.2.10"

    eih_cli = EyeInHand(preset="medium_range", sim=False, output=tmp_path, ip="192.0.2.20")
    assert eih_cli.cfg["robot"]["ip"] == "192.0.2.20"


def test_gui_backend_preset_selectors_and_robot_ip_input(monkeypatch) -> None:
    from PySide6.QtWidgets import QApplication
    import camera_tool.ui.main_window as mw_mod

    monkeypatch.delenv("XARM_IP", raising=False)
    monkeypatch.setattr(mw_mod, "detect_usb_camera_backend", lambda: "realsense")
    app = QApplication.instance() or QApplication([])
    window = mw_mod.MainWindow(sim_mode=False, initial_page=1)
    try:
        assert window.combo_eih_backend.currentData() == "realsense"
        assert "default.yaml" in window.cal_config.text()
        assert window.cal_robot_ip.text() == ""

        window.combo_eih_preset.setCurrentIndex(window.combo_eih_preset.findData("medium_range"))
        assert "medium_range.yaml" in window.cal_config.text()
        assert np.isclose(window.video_panel.max_depth_m, 1.20)
        assert np.isclose(window.scene3d_panel.tool_length_m, 0.060)

        window.cal_robot_ip.setText("192.0.2.99")
        window._on_ip_edited("192.0.2.99")
        window.combo_eih_preset.setCurrentIndex(window.combo_eih_preset.findData("orbbec"))
        assert "orbbec.yaml" in window.cal_config.text()
        assert window.cal_robot_ip.text() == "192.0.2.99"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
