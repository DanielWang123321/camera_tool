"""三脚架监控：已知微动应换成约 3 mm 桌面位移；运动物体不应报相机在动。"""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from camera_tool.core import CameraModel, inv, pose_error, transform, transform_points, valid_transform
from camera_tool.diagnostics.scene_lock import (ConfirmGate, classify_status, estimate_camera_motion, filter_dynamic,
                            intersect_table, load_baseline, lock_scene, observe_frame, save_baseline,
                            static_mask, table_apparent_shift, watch_settings)


def synthetic_camera():
    return CameraModel({
        "width": 1280, "height": 720, "model": "pinhole", "distortion": [0.0] * 5,
        "K": [[910.867431640625, 0.0, 647.6005859375],
              [0.0, 909.6834106445312, 359.2757263183594],
              [0.0, 0.0, 1.0]],
    })


def published_extrinsics():
    raw = yaml.safe_load((ROOT / "tests/fixtures/eye_to_hand.yaml").read_text(encoding="utf-8"))
    return (valid_transform(raw["X_base_camera"]),
            np.asarray(raw["bbox_min"], dtype=float),
            np.asarray(raw["bbox_max"], dtype=float))


def static_points(x_base_camera, bbox_min, bbox_max, camera, n_extra=40, seed=3):
    rng = np.random.default_rng(seed)
    pts = []
    xs = np.linspace(-0.08, 0.82, 14)
    ys = np.linspace(-0.50, 0.70, 14)
    lo, hi = bbox_min, bbox_max
    for x in xs:
        for y in ys:
            if lo[0] - 0.08 <= x <= hi[0] + 0.08 and lo[1] - 0.08 <= y <= hi[1] + 0.08:
                if lo[0] < x < hi[0] and lo[1] < y < hi[1]:
                    continue
            pts.append([x, y, 0.0])
    for _ in range(n_extra):
        pts.append([rng.uniform(-0.1, 0.9), rng.uniform(-0.55, 0.75), rng.uniform(-0.05, 0.35)])
    pts = np.asarray(pts, dtype=float)
    cam = transform_points(inv(x_base_camera), pts)
    px = camera.project(cam)
    keep = ((cam[:, 2] > 0.25)
            & (px[:, 0] > 8) & (px[:, 0] < camera.width - 8)
            & (px[:, 1] > 8) & (px[:, 1] < camera.height - 8))
    return cam[keep], px[keep]


def synthetic_rgbd(camera, x_base_camera, seed=0):
    h, w = camera.height, camera.width
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((h, w))
    board = np.where(((xx // 16) + (yy // 16)) % 2, 210, 45).astype(np.int16)
    color = np.clip(np.dstack([board, board, board]) + rng.integers(-18, 19, (h, w, 3)), 0, 255).astype(np.uint8)
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.column_stack([uu.ravel(), vv.ravel()])
    hit = intersect_table(pixels, camera, x_base_camera, 0.0).reshape(h, w, 3)
    p_cam = np.full((h, w, 3), np.nan)
    valid = np.isfinite(hit).all(axis=2)
    p_cam[valid] = transform_points(inv(x_base_camera), hit[valid])
    depth = np.full((h, w), 1.25, np.float32)
    use = valid & (p_cam[:, :, 2] > 0.2) & (p_cam[:, :, 2] < 2.8)
    depth[use] = p_cam[:, :, 2][use].astype(np.float32)
    return color, depth


class SceneLockTests(unittest.TestCase):
    def setUp(self):
        self.camera = synthetic_camera()
        self.x_bc, self.bbox_min, self.bbox_max = published_extrinsics()
        self.cfg = watch_settings({})

    def test_pitch_0_2deg_moves_table_about_3mm(self):
        old_cam, _ = static_points(self.x_bc, self.bbox_min, self.bbox_max, self.camera)
        x_gt = transform(Rotation.from_euler("x", 0.2, degrees=True).as_matrix())
        px_now = self.camera.project(transform_points(x_gt, old_cam))
        motion = estimate_camera_motion(old_cam, px_now, self.camera, self.cfg)
        self.assertIsNotNone(motion)
        mm, deg = pose_error(motion["X_now_from_old"], x_gt)
        self.assertLess(mm, 1.0)
        self.assertLess(deg, 0.05)
        table = table_apparent_shift(
            motion["X_now_from_old"], self.camera, self.x_bc,
            self.bbox_min, self.bbox_max, 0.0, self.cfg["grid_n"])
        truth = table_apparent_shift(
            x_gt, self.camera, self.x_bc, self.bbox_min, self.bbox_max, 0.0, self.cfg["grid_n"])
        self.assertIsNotNone(table)
        self.assertIsNotNone(truth)
        self.assertGreater(truth["p95_mm"], 1.5)
        self.assertLess(truth["p95_mm"], 8.0)
        self.assertLess(abs(table["p95_mm"] - truth["p95_mm"]), 0.4)
        self.assertEqual(classify_status(table["p95_mm"], motion["n_inliers"], self.cfg), "FAIL")

    def test_moving_object_outliers_do_not_report_camera_motion(self):
        old_cam, px = static_points(self.x_bc, self.bbox_min, self.bbox_max, self.camera, n_extra=80)
        self.assertGreater(len(px), 80)
        px_now = px.copy()
        px_now[:25] += np.array([30.0, -12.0])
        motion = estimate_camera_motion(old_cam, px_now, self.camera, self.cfg)
        self.assertIsNotNone(motion)
        mm, deg = pose_error(motion["X_now_from_old"], np.eye(4))
        self.assertLess(mm, 0.5)
        self.assertLess(deg, 0.05)
        table = table_apparent_shift(
            motion["X_now_from_old"], self.camera, self.x_bc,
            self.bbox_min, self.bbox_max, 0.0, self.cfg["grid_n"])
        self.assertLess(table["p95_mm"], 0.5)
        self.assertEqual(classify_status(table["p95_mm"], motion["n_inliers"], self.cfg), "OK")

    def test_static_mask_keeps_outside_workspace(self):
        color, depth = synthetic_rgbd(self.camera, self.x_bc)
        mask = static_mask(depth, self.camera, self.x_bc, self.bbox_min, self.bbox_max, self.cfg, 0.0)
        self.assertEqual(mask.shape, depth.shape)
        self.assertGreater(int((mask > 0).sum()), 5000)
        self.assertLess(int((mask > 0).sum()), mask.size * 0.95)

    def test_filter_dynamic_drops_workspace_and_depth_jump(self):
        old_cam, px = static_points(self.x_bc, self.bbox_min, self.bbox_max, self.camera)
        h, w = self.camera.height, self.camera.width
        depth = np.full((h, w), 1.0, np.float32)
        from camera_tool.diagnostics.scene_lock import Baseline
        base = Baseline(
            color=np.zeros((h, w, 3), np.uint8), gray=np.zeros((h, w), np.uint8), depth_m=depth,
            pixels=px.astype(np.float32), points_cam=old_cam,
            orb_pixels=np.zeros((0, 2), np.float32), orb_points_cam=np.zeros((0, 3)),
            orb_des=np.zeros((0, 32), np.uint8), camera=self.camera, x_base_camera=self.x_bc,
            bbox_min=self.bbox_min, bbox_max=self.bbox_max, table_z_m=0.0, depth_scale=1.0,
            plane=None, serial="t", extrinsics_hash="x", static_mask=np.zeros((h, w), np.uint8),
        )
        u = np.clip(np.round(px[:, 0]).astype(int), 0, w - 1)
        v = np.clip(np.round(px[:, 1]).astype(int), 0, h - 1)
        depth[v, u] = old_cam[:, 2].astype(np.float32)
        keep = filter_dynamic(base, px, old_cam, depth, self.cfg)
        self.assertGreater(int(keep.sum()), 20)
        jumped = depth.copy()
        jumped[v[0], u[0]] = old_cam[0, 2] + 0.2
        keep2 = filter_dynamic(base, px[:1], old_cam[:1], jumped, self.cfg)
        self.assertFalse(bool(keep2[0]))
        mid = np.array([[(self.bbox_min[0] + self.bbox_max[0]) / 2,
                         (self.bbox_min[1] + self.bbox_max[1]) / 2, 0.0]])
        mid_cam = transform_points(inv(self.x_bc), mid)
        mid_px = self.camera.project(mid_cam)
        uu, vv = int(round(mid_px[0, 0])), int(round(mid_px[0, 1]))
        if 0 <= uu < w and 0 <= vv < h:
            depth[vv, uu] = float(mid_cam[0, 2])
            keep3 = filter_dynamic(base, mid_px, mid_cam, depth, self.cfg)
            self.assertFalse(bool(keep3[0]))

    def test_confirm_gate_needs_consecutive_frames(self):
        gate = ConfirmGate(3)
        self.assertEqual(gate.update("WARN"), "OK")
        self.assertEqual(gate.update("WARN"), "OK")
        self.assertEqual(gate.update("WARN"), "WARN")
        self.assertEqual(gate.update("OK"), "OK")
        self.assertEqual(gate.update("FAIL"), "OK")
        self.assertEqual(gate.update("TRACK_LOST"), "TRACK_LOST")

    def test_lock_and_identical_frame_is_ok(self):
        color, depth = synthetic_rgbd(self.camera, self.x_bc)
        baseline = lock_scene(
            color, depth, self.camera, self.x_bc,
            bbox_min=self.bbox_min, bbox_max=self.bbox_max, table_z_m=0.0,
            depth_scale=1.008, serial="MOCK_FIXED_001", cfg=self.cfg)
        self.assertGreaterEqual(len(baseline.pixels), self.cfg["min_lock_features"])
        tmp = ROOT / "tests" / "_tmp_scene_lock"
        if tmp.exists():
            shutil.rmtree(tmp)
        try:
            save_baseline(baseline, tmp)
            loaded = load_baseline(tmp, x_base_camera=self.x_bc, serial="MOCK_FIXED_001", camera=self.camera)
            obs = observe_frame(loaded, color, depth, self.cfg)
            self.assertNotEqual(obs.status, "TRACK_LOST")
            self.assertIsNotNone(obs.table_p95_mm)
            self.assertLess(obs.table_p95_mm, 1.0)
            self.assertLess(obs.cam_mm, 1.0)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
