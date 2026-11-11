"""离线验收：几何恢复、独立检查、深度报告与内参加载。"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_tool.calibration.eye_to_hand.solver import (check_samples, depth_check, fit_hand_eye, report_depth, solve_realsense,
                        solve_wrist, split_samples)
from camera_tool.calibration.eye_to_hand.wrist_model import load_wrist_intrinsics
from camera_tool.calibration.eye_to_hand.workflow import capture_consistency, color_metadata, depth_points, load_settings
from camera_tool.core import CalibrationError, CameraModel, MarkerDetector, RejectedSample, inv, marker_points, pose_error, transform, transform_points
from camera_tool.paths import config_path


def settings():
    return load_settings(config_path("eye_to_hand.yaml"))


def camera_metadata(width=960, height=960, focal=700.):
    return {"width": width, "height": height, "K": [[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]],
            "model": "pinhole", "distortion": [0.] * 5, "serial": "SYNTHETIC_ONLY"}


def synthetic(noise=.06, distorted=False):
    config = settings()
    meta = camera_metadata()
    if distorted:
        meta.update(model="brown_conrady", distortion=[-.12, .02, .001, -.002, 0])
    camera = CameraModel(meta)
    wrist = transform(Rotation.from_euler("xyz", [3, -6, 10], degrees=True).as_matrix(), [.04, -.02, -.10])
    board = transform(Rotation.from_euler("z", 20, degrees=True).as_matrix(), [.45, .04, 0.])
    rng = np.random.default_rng(20)
    samples = []
    for i in range(30):
        angles = [180+rng.uniform(-24, 24), rng.uniform(-24, 24), rng.uniform(-15, 15)]
        r = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        camera_pose = transform(r, board[:3, 3] - r @ np.r_[rng.uniform(-.05, .05, 2), rng.uniform(.35, .5)])
        corners = camera.project(transform_points(inv(camera_pose) @ board, marker_points(.15)))
        corners += rng.normal(0, noise, (4, 2))
        samples.append({"id": i, "X_base_tcp": camera_pose @ inv(wrist), "corners": corners})
    return config, camera, wrist, board, samples


def render_marker(camera, corners, seed=0):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    marker = cv2.aruco.generateImageMarker(dictionary, 398, 600)
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [599, 0], [599, 599], [0, 599]]), np.float32(corners))
    gray = cv2.warpPerspective(marker, h, (camera.width, camera.height), borderValue=255)
    noise = np.random.default_rng(seed).integers(-2, 3, gray.shape, dtype=np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def static_scene(board):
    meta = camera_metadata(640, 480, 520.)
    camera = CameraModel(meta)
    r = Rotation.from_euler("xyz", [156, 12, 8], degrees=True).as_matrix()
    base_camera = transform(r, board[:3, 3] - r @ [0, 0, .7])
    camera_board = inv(base_camera) @ board
    color_depth = transform(Rotation.from_euler("y", .5, degrees=True).as_matrix(), [.025, 0, .002])
    di = {"width": 640, "height": 480, "fx": 520., "fy": 520., "ppx": 320., "ppy": 240.,
          "coeffs": [0.] * 5, "model": "distortion.none"}
    meta["native"] = {"serial": "SYNTHETIC_RS", "color": di, "native_depth": di,
                      "X_color_depth": color_depth.tolist(), "depth_scale": .001}
    yy, xx = np.mgrid[:480, :640]
    rays = np.stack([(xx - 320) / 520, (yy - 240) / 520, np.ones_like(xx)], axis=-1)
    depth_board = inv(base_camera @ color_depth) @ board
    n, o = depth_board[:3, 2], depth_board[:3, 3]
    depth = ((o @ n) / (rays @ n))
    raw_depth = np.rint(depth * 1000).astype(np.uint16)
    geometry = {"pixels": np.stack([xx[::3, ::3], yy[::3, ::3]], axis=-1).reshape(-1, 2),
                "rays": rays[::3, ::3].reshape(-1, 3)}
    corners = camera.project(transform_points(camera_board, marker_points(.15)))
    points = depth_points(raw_depth, geometry, meta)
    samples = [{"corners": corners.copy(), "points_color": points} for _ in range(15)]
    return meta, base_camera, camera_board, raw_depth, geometry, samples


class GeometryTests(unittest.TestCase):
    def test_noisy_recovery_and_holdout(self):
        config, camera, wrist, board, samples = synthetic()
        result = solve_wrist(samples, camera.metadata, config)
        self.assertLess(pose_error(result["X_tcp_wrist"], wrist)[0], .5)
        self.assertLess(pose_error(result["X_base_board"], board)[0], .5)
        self.assertLess(pose_error(result["X_base_board"], board)[1], .1)
        self.assertEqual(result["check_ids"], [4, 9, 14, 19, 24, 29])
        self.assertEqual(len(result["train_ids"]), 24)
        self.assertLess(result["independent_check"]["mean_mm"], 3)

    def test_distortion_is_used(self):
        config, camera, _, board, samples = synthetic(distorted=True)
        result = solve_wrist(samples, camera.metadata, config)
        self.assertLess(pose_error(result["X_base_board"], board)[0], .5)

    def test_planar_candidate_order_does_not_change_solution(self):
        config, camera, _, board, samples = synthetic()
        original = camera.candidates
        camera.candidates = lambda corners, side: original(corners, side)[::-1]
        result = fit_hand_eye(split_samples(samples)[0], camera, .15)
        self.assertLess(pose_error(result["X_base_board"], board)[0], .5)

    def test_translation_only_and_one_axis_fail(self):
        config, camera, _, _, samples = synthetic()
        for one_axis in (False, True):
            bad = copy.deepcopy(samples)
            for i, sample in enumerate(bad):
                sample["X_base_tcp"][:3, :3] = Rotation.from_euler("z", i if one_axis else 0, degrees=True).as_matrix()
            with self.assertRaisesRegex(CalibrationError, "转动不足"):
                solve_wrist(bad, camera.metadata, config)

    def test_board_movement_in_independent_samples_fails(self):
        config, camera, _, _, samples = synthetic()
        for sample in split_samples(samples)[1]:
            sample["X_base_tcp"][0, 3] += .025
        with self.assertRaisesRegex(CalibrationError, "独立姿态检查"):
            solve_wrist(samples, camera.metadata, config)

    def test_wrong_units_fail(self):
        config, camera, _, _, samples = synthetic()
        for sample in samples:
            sample["X_base_tcp"][:3, 3] *= 1000
        with self.assertRaises(CalibrationError):
            solve_wrist(samples, camera.metadata, config)

    def test_realsense_composition_and_native_depth(self):
        config, _, wrist, board, _ = synthetic()
        meta, expected, _, _, _, samples = static_scene(board)
        solution = solve_realsense(samples, meta, {"X_base_board": board, "X_tcp_wrist": wrist}, config)
        self.assertLess(pose_error(expected, solution["X_base_camera"])[0], .01)
        self.assertTrue(solution["depth_ok"])
        self.assertLess(solution["depth_checks"][0]["median_mm"], .6)
        self.assertAlmostEqual(solution["depth_scale"], 1.0, places=2)

    def test_wrong_depth_warns_but_geometry_publishes(self):
        config, _, wrist, board, _ = synthetic()
        meta, expected, camera_board, _, _, samples = static_scene(board)
        with self.assertRaisesRegex(CalibrationError, "深度核对"):
            depth_check(samples[0]["points_color"] + [0, 0, .04], CameraModel(meta), samples[0]["corners"],
                        camera_board, board, config["checks"])
        report = report_depth(samples[0]["points_color"] + [0, 0, .04], CameraModel(meta), samples[0]["corners"],
                              camera_board, board, config["checks"])
        self.assertFalse(report["ok"])
        bad = copy.deepcopy(samples)
        for sample in bad:
            sample["points_color"] = np.asarray(sample["points_color"]) + [0, 0, .04]
        solution = solve_realsense(bad, meta, {"X_base_board": board, "X_tcp_wrist": wrist}, config)
        self.assertLess(pose_error(expected, solution["X_base_camera"])[0], .01)
        self.assertFalse(solution["depth_ok"])
        self.assertTrue(0.5 < solution["depth_scale"] < 1.5)
        self.assertNotAlmostEqual(solution["depth_scale"], 1.0, places=2)

    def test_static_board_movement_fails(self):
        config, _, _, board, _ = synthetic()
        meta, _, _, _, _, samples = static_scene(board)
        samples[-1]["corners"] = np.asarray(samples[-1]["corners"]) + 5
        with self.assertRaisesRegex(CalibrationError, "期间板或相机移动"):
            solve_realsense(samples, meta, {"X_base_board": board}, config)


class FrameTests(unittest.TestCase):
    def test_detection_occlusion_and_blur(self):
        detector = MarkerDetector(settings()["marker"], CameraModel(camera_metadata()))
        pose = transform(Rotation.from_euler("xyz", [170, 8, 0], degrees=True).as_matrix(), [0, 0, .3])
        corners = detector.camera.project(transform_points(pose, marker_points(.15)))
        image = render_marker(detector.camera, corners)
        self.assertEqual(len(detector.detect(image)["corners"]), 4)
        image[:, 400:] = 255
        with self.assertRaises(RejectedSample):
            detector.detect(image)
        image = render_marker(detector.camera, corners)
        with self.assertRaises(RejectedSample):
            detector.detect(cv2.GaussianBlur(image, (81, 81), 25))

    def test_raw_seucm_cannot_be_used_as_pinhole(self):
        with self.assertRaisesRegex(CalibrationError, "未经处理"):
            CameraModel({**camera_metadata(), "model": "SEUCM"})

    def test_unknown_realsense_distortion_fails(self):
        _, _, _, board, _ = synthetic()
        meta, *_ = static_scene(board)
        native = meta["native"]
        native["color"].update(model="distortion.inverse_brown_conrady", coeffs=[.1, 0, 0, 0, 0])
        with self.assertRaises(CalibrationError):
            color_metadata(native)

    def test_capture_bracketing_and_motion(self):
        state = {"X_base_tcp": np.eye(4), "joints_rad": [0] * 6, "state": 0,
                 "read_start_mono": 1., "read_end_mono": 2.}
        after = {**state, "read_start_mono": 4., "read_end_mono": 5.}
        capture_consistency(state, after, 3., settings()["motion"])
        moving = {**after, "X_base_tcp": transform(translation=[0, 0, .01])}
        with self.assertRaises(RejectedSample):
            capture_consistency(state, moving, 3., settings()["motion"])
        with self.assertRaises(CalibrationError):
            capture_consistency(state, after, 6., settings()["motion"])

    def test_wrist_intrinsics_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            npz_path = tmp_dir / "wrist_MOCK_WRIST_001.npz"
            json_path = tmp_dir / "wrist_MOCK_WRIST_001.json"
            grid_y, grid_x = np.meshgrid(np.arange(960, dtype=np.float32), np.arange(960, dtype=np.float32), indexing="ij")
            np.savez_compressed(npz_path, mapping=np.stack([grid_x, grid_y], axis=-1))
            json_path.write_text(
                json.dumps(
                    {
                        "serial": "MOCK_WRIST_001",
                        "model": "pinhole",
                        "width": 960,
                        "height": 960,
                        "K": [[500.0, 0.0, 480.0], [0.0, 500.0, 480.0], [0.0, 0.0, 1.0]],
                        "distortion": [0.0] * 5,
                        "mapping_file": npz_path.name,
                    }
                ),
                encoding="utf-8",
            )
            metadata, mapping = load_wrist_intrinsics(json_path)
            self.assertEqual(metadata["serial"], "MOCK_WRIST_001")
            self.assertEqual(metadata["model"], "pinhole")
            self.assertEqual(mapping.shape, (960, 960, 2))
            CameraModel(metadata)

    def test_independent_check_helper(self):
        config, camera, _, _, samples = synthetic()
        result = solve_wrist(samples, camera.metadata, config)
        train, check = split_samples(samples)
        report = check_samples(check, camera, .15, result, config["checks"], "6 个独立姿态检查")
        self.assertEqual(report["count"], 6)


if __name__ == "__main__":
    unittest.main()
