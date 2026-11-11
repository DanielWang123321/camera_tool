"""Markerless tripod scene lock: estimate camera micro-drift from static background features."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera_tool.core import (
    CalibrationError,
    CameraModel,
    inv,
    pose_error,
    transform,
    transform_points,
    valid_transform,
    write_json,
)
from camera_tool.paths import DATA_ROOT, project_path

DEFAULT_WATCH = {
    "baseline_dir": "result/scene_lock",
    "lock_frames": 8,
    "max_features": 400,
    "min_lock_features": 80,
    "min_inliers": 30,
    "warn_table_mm": 1.5,
    "fail_table_mm": 3.0,
    "confirm_frames": 15,
    "workspace_margin_m": 0.05,
    "table_rim_m": 0.04,
    "max_flow_fb_px": 1.5,
    "ransac_reproj_px": 2.5,
    "depth_jump_m": 0.06,
    "min_depth_m": 0.15,
    "max_depth_m": 3.0,
    "grid_n": 8,
    "orb_fallback_tracks": 40,
    "table_height_tolerance_m": 0.03,
    "plane_fit_points": 4000,
    "plane_inlier_m": 0.012,
}

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def watch_settings(config: dict | None) -> dict:
    merged = dict(DEFAULT_WATCH)
    if config:
        merged.update(config)
    return merged


def camera_from_rs_color(ci: dict) -> CameraModel:
    coeffs = np.asarray(ci["coeffs"], dtype=float)
    model_name = str(ci.get("model") or "")
    if np.all(coeffs == 0) or model_name in ("distortion.none", "none", "distortion.inverse_brown_conrady"):
        model = "pinhole"
    elif model_name in ("distortion.brown_conrady", "brown_conrady"):
        model = "brown_conrady"
    else:
        raise CalibrationError(f"彩色畸变模型尚不支持：{model_name}")
    return CameraModel(
        {
            "width": int(ci["width"]),
            "height": int(ci["height"]),
            "model": model,
            "distortion": coeffs.tolist(),
            "K": [
                [float(ci["fx"]), 0.0, float(ci["ppx"])],
                [0.0, float(ci["fy"]), float(ci["ppy"])],
                [0.0, 0.0, 1.0],
            ],
        }
    )


def camera_from_meta(meta: dict) -> CameraModel:
    return CameraModel(
        {
            "width": meta["width"],
            "height": meta["height"],
            "model": meta["model"],
            "distortion": meta["distortion"],
            "K": meta["K"],
        }
    )


def hash_transform(x: Any) -> str:
    return hashlib.sha256(np.asarray(valid_transform(x), dtype=np.float64).tobytes()).hexdigest()[:16]


def resolve_baseline_dir(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = DATA_ROOT / path
    return project_path(path)


def gray_image(color: np.ndarray) -> np.ndarray:
    if color.ndim == 2:
        return color
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


def unproject_pixels(pixels: np.ndarray, z: np.ndarray, camera: CameraModel) -> np.ndarray:
    return camera.deproject_pixels(pixels, z)


def sample_depth(depth_m: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape[:2]
    u = np.rint(np.asarray(pixels)[:, 0]).astype(int)
    v = np.rint(np.asarray(pixels)[:, 1]).astype(int)
    z = np.full(len(u), np.nan, dtype=np.float64)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z[ok] = depth_m[v[ok], u[ok]]
    z[z <= 0] = np.nan
    return z


def table_grid(bbox_min: Any, bbox_max: Any, table_z_m: float, n: int = 8) -> np.ndarray:
    lo, hi = np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, table_z_m)])


def inside_aabb(points: Any, lo: Any, hi: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    lo_arr, hi_arr = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return np.all((pts >= lo_arr) & (pts <= hi_arr), axis=1)


def static_mask(
    depth_m: np.ndarray,
    camera: CameraModel,
    x_base_camera: np.ndarray,
    bbox_min: Any,
    bbox_max: Any,
    cfg: dict,
    table_z_m: float,
) -> np.ndarray:
    h, w = depth_m.shape[:2]
    z = np.asarray(depth_m, dtype=np.float32)
    valid = (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"]) & np.isfinite(z)
    ys, xs = np.indices((h, w))
    pts = unproject_pixels(np.column_stack([xs.ravel(), ys.ravel()]), z.ravel(), camera).reshape(h, w, 3)
    pts[~valid] = np.nan
    finite = np.isfinite(pts).all(axis=2)
    base = np.full((h, w, 3), np.nan)
    base[finite] = transform_points(x_base_camera, pts[finite])
    margin = float(cfg["workspace_margin_m"])
    lo = np.asarray(bbox_min, dtype=float) - margin
    hi = np.asarray(bbox_max, dtype=float) + margin
    lo[2] = min(float(bbox_min[2]), table_z_m) - 0.05
    hi[2] = float(bbox_max[2]) + 0.05
    in_dyn = finite & inside_aabb(base.reshape(-1, 3), lo, hi).reshape(h, w)
    orig_lo, orig_hi = np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)
    xy_in = finite & (base[:, :, 0] >= orig_lo[0]) & (base[:, :, 0] <= orig_hi[0])
    xy_in &= (base[:, :, 1] >= orig_lo[1]) & (base[:, :, 1] <= orig_hi[1])
    on_table = np.abs(base[:, :, 2] - table_z_m) <= cfg["table_height_tolerance_m"]
    dx = np.minimum(base[:, :, 0] - orig_lo[0], orig_hi[0] - base[:, :, 0])
    dy = np.minimum(base[:, :, 1] - orig_lo[1], orig_hi[1] - base[:, :, 1])
    rim = float(cfg["table_rim_m"])
    table_rim = xy_in & on_table & ((dx <= rim) | (dy <= rim))
    static = finite & ((~in_dyn) | table_rim)
    mask = np.zeros((h, w), np.uint8)
    mask[static] = 255
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    return mask


def fit_plane_ransac(points: np.ndarray, cfg: dict, rng: Any = None) -> dict[str, Any] | None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 20:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    thresh = float(cfg["plane_inlier_m"])
    best = None
    for _ in range(80):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        d = float(normal @ p0)
        n_in = int(np.count_nonzero(np.abs(pts @ normal - d) < thresh))
        if best is None or n_in > best[0]:
            best = (n_in, normal, d)
    if best is None or best[0] < 20:
        return None
    _, normal, d = best
    inliers = pts[np.abs(pts @ normal - d) < thresh]
    center = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    if np.linalg.norm(normal) < 1e-9:
        return None
    normal = normal / np.linalg.norm(normal)
    if normal[2] > 0:
        normal = -normal
    d = float(normal @ center)
    return {"normal": normal, "d": d, "n_inliers": int(len(inliers))}


def table_points_camera(
    depth_m: np.ndarray,
    camera: CameraModel,
    x_base_camera: np.ndarray,
    bbox_min: Any,
    bbox_max: Any,
    table_z_m: float,
    cfg: dict,
) -> np.ndarray:
    h, w = depth_m.shape[:2]
    step = max(1, int(np.sqrt((h * w) / max(int(cfg["plane_fit_points"]), 200))))
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    pixels = np.column_stack([xs.ravel(), ys.ravel()])
    z = sample_depth(depth_m, pixels)
    ok = np.isfinite(z) & (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"])
    if not np.any(ok):
        return np.zeros((0, 3))
    pts = unproject_pixels(pixels[ok], z[ok], camera)
    base = transform_points(x_base_camera, pts)
    lo, hi = np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)
    keep = (
        (base[:, 0] >= lo[0])
        & (base[:, 0] <= hi[0])
        & (base[:, 1] >= lo[1])
        & (base[:, 1] <= hi[1])
        & (np.abs(base[:, 2] - table_z_m) <= cfg["table_height_tolerance_m"])
    )
    return pts[keep]


def plane_delta(baseline_plane: dict | None, current_plane: dict | None) -> tuple[float | None, float | None]:
    if not baseline_plane or not current_plane:
        return None, None
    n0 = np.asarray(baseline_plane["normal"], dtype=float)
    n1 = np.asarray(current_plane["normal"], dtype=float)
    cos = float(np.clip(abs(n0 @ n1), 0.0, 1.0))
    angle = float(np.rad2deg(np.arccos(cos)))
    dist0 = abs(float(baseline_plane["d"]))
    dist1 = abs(float(current_plane["d"]))
    return angle, float((dist1 - dist0) * 1000.0)


@dataclass
class Baseline:
    color: np.ndarray
    gray: np.ndarray
    depth_m: np.ndarray
    pixels: np.ndarray
    points_cam: np.ndarray
    orb_pixels: np.ndarray
    orb_points_cam: np.ndarray
    orb_des: np.ndarray
    camera: CameraModel
    x_base_camera: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    table_z_m: float
    depth_scale: float
    plane: dict | None
    serial: str
    extrinsics_hash: str
    static_mask: np.ndarray


def _features_with_depth(
    pixels: np.ndarray, depth_m: np.ndarray, camera: CameraModel, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    z = sample_depth(depth_m, pixels)
    ok = np.isfinite(z) & (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"])
    if not np.any(ok):
        return np.zeros((0, 2), np.float32), np.zeros((0, 3), np.float64)
    return pixels[ok].astype(np.float32), unproject_pixels(pixels[ok], z[ok], camera)


def lock_scene(
    color: np.ndarray,
    depth_m: np.ndarray,
    camera: CameraModel,
    x_base_camera: np.ndarray,
    *,
    bbox_min: Any,
    bbox_max: Any,
    table_z_m: float,
    depth_scale: float,
    serial: str,
    cfg: dict | None = None,
) -> Baseline:
    cfg = watch_settings(cfg)
    if color.shape[:2] != (camera.height, camera.width):
        raise CalibrationError("锁基线图像尺寸与内参不一致")
    if depth_m.shape[:2] != (camera.height, camera.width):
        raise CalibrationError("锁基线深度尺寸与内参不一致")
    x_base_camera = valid_transform(x_base_camera)
    mask = static_mask(depth_m, camera, x_base_camera, bbox_min, bbox_max, cfg, table_z_m)
    gray = gray_image(color)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=int(cfg["max_features"]), qualityLevel=0.01, minDistance=10, blockSize=7, mask=mask
    )
    if corners is None:
        raise CalibrationError("静物角点不足，无法锁基线（画面里工作区外要有桌沿/地面/机座）")
    pixels, points = _features_with_depth(corners.reshape(-1, 2).astype(np.float32), depth_m, camera, cfg)
    if len(pixels) < int(cfg["min_lock_features"]):
        raise CalibrationError(
            f"有效静物三维点只有 {len(pixels)} 个，需要至少 {cfg['min_lock_features']} 个才能锁基线"
        )
    orb = cv2.ORB_create(nfeatures=800)
    kps, des = orb.detectAndCompute(gray, mask)
    orb_px = np.zeros((0, 2), np.float32)
    orb_pts = np.zeros((0, 3), np.float64)
    if kps and des is not None and len(kps):
        raw_px = np.array([kp.pt for kp in kps], dtype=np.float32)
        z = sample_depth(depth_m, raw_px)
        keep = np.isfinite(z) & (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"])
        orb_px, orb_pts = _features_with_depth(raw_px[keep], depth_m, camera, cfg)
        des = des[keep]
    if des is None or len(des) != len(orb_px):
        des = np.zeros((0, 32), np.uint8)
        orb_px = np.zeros((0, 2), np.float32)
        orb_pts = np.zeros((0, 3), np.float64)
    plane = fit_plane_ransac(
        table_points_camera(depth_m, camera, x_base_camera, bbox_min, bbox_max, table_z_m, cfg), cfg
    )
    return Baseline(
        color=np.ascontiguousarray(color),
        gray=gray,
        depth_m=np.asarray(depth_m, dtype=np.float32),
        pixels=pixels,
        points_cam=points,
        orb_pixels=orb_px,
        orb_points_cam=orb_pts,
        orb_des=np.asarray(des),
        camera=camera,
        x_base_camera=x_base_camera,
        bbox_min=np.asarray(bbox_min, dtype=float),
        bbox_max=np.asarray(bbox_max, dtype=float),
        table_z_m=float(table_z_m),
        depth_scale=float(depth_scale),
        plane=plane,
        serial=str(serial),
        extrinsics_hash=hash_transform(x_base_camera),
        static_mask=mask,
    )


def save_baseline(baseline: Baseline, directory: str | Path) -> Path:
    directory = resolve_baseline_dir(directory)
    directory.mkdir(parents=True, exist_ok=True)
    color_path = project_path(directory / "baseline_color.png")
    if not cv2.imwrite(str(color_path), baseline.color):
        raise CalibrationError(f"无法写入 {color_path}")
    np.save(project_path(directory / "baseline_depth.npy"), baseline.depth_m)
    plane = None
    if baseline.plane is not None:
        plane = {
            "normal": baseline.plane["normal"].tolist(),
            "d": float(baseline.plane["d"]),
            "n_inliers": int(baseline.plane["n_inliers"]),
        }
    np.savez_compressed(
        project_path(directory / "features.npz"),
        pixels=baseline.pixels,
        points_cam=baseline.points_cam,
        orb_pixels=baseline.orb_pixels,
        orb_points_cam=baseline.orb_points_cam,
        orb_des=baseline.orb_des,
        static_mask=baseline.static_mask,
        plane_normal=np.zeros(3) if plane is None else np.asarray(plane["normal"]),
        plane_d=np.array(0.0 if plane is None else plane["d"]),
        has_plane=np.array(plane is not None),
    )
    write_json(
        directory / "meta.json",
        {
            "schema": 1,
            "locked_at": datetime.now().astimezone().isoformat(),
            "serial": baseline.serial,
            "width": baseline.camera.width,
            "height": baseline.camera.height,
            "model": baseline.camera.metadata["model"],
            "K": baseline.camera.k.tolist(),
            "distortion": baseline.camera.distortion.tolist(),
            "X_base_camera": baseline.x_base_camera.tolist(),
            "extrinsics_hash": baseline.extrinsics_hash,
            "bbox_min": baseline.bbox_min.tolist(),
            "bbox_max": baseline.bbox_max.tolist(),
            "table_z_m": baseline.table_z_m,
            "depth_scale": baseline.depth_scale,
            "n_features": int(len(baseline.pixels)),
            "n_orb": int(len(baseline.orb_pixels)),
            "plane": plane,
        },
    )
    return directory


def load_baseline(
    directory: str | Path,
    x_base_camera: Any = None,
    serial: str | None = None,
    camera: CameraModel | None = None,
) -> Baseline:
    directory = resolve_baseline_dir(directory)
    meta_path = directory / "meta.json"
    color_path = directory / "baseline_color.png"
    depth_path = directory / "baseline_depth.npy"
    feat_path = directory / "features.npz"
    for path in (meta_path, color_path, depth_path, feat_path):
        if not path.exists():
            raise CalibrationError(f"没有场景基线 {path.name}，请先运行 camera-tool tripod lock")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = camera_from_meta(meta)
    if camera is not None:
        if (
            camera.width != stored.width
            or camera.height != stored.height
            or not np.allclose(camera.k, stored.k, atol=0.5)
            or not np.allclose(camera.distortion, stored.distortion, atol=1e-6)
        ):
            raise CalibrationError("当前相机内参与锁基线时不一致，请重新执行 tripod lock")
    if serial is not None and str(meta.get("serial")) != str(serial):
        raise CalibrationError(f"基线序列号 {meta.get('serial')} 与当前相机 {serial} 不一致")
    if x_base_camera is not None and hash_transform(x_base_camera) != meta.get("extrinsics_hash"):
        raise CalibrationError("标定结果外参已更新，请重新执行 tripod lock")
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise CalibrationError("无法读取 baseline_color.png")
    depth_m = np.load(depth_path)
    feat = np.load(feat_path)
    plane = None
    if bool(np.asarray(feat["has_plane"]).item()) if "has_plane" in feat else False:
        plane = {
            "normal": np.asarray(feat["plane_normal"], dtype=float),
            "d": float(np.asarray(feat["plane_d"]).reshape(-1)[0]),
            "n_inliers": int((meta.get("plane") or {}).get("n_inliers") or 0),
        }
    x_cam = valid_transform(meta["X_base_camera"] if x_base_camera is None else x_base_camera)
    mask = np.asarray(feat["static_mask"]) if "static_mask" in feat else np.zeros(color.shape[:2], np.uint8)
    return Baseline(
        color=color,
        gray=gray_image(color),
        depth_m=np.asarray(depth_m, dtype=np.float32),
        pixels=np.asarray(feat["pixels"], dtype=np.float32),
        points_cam=np.asarray(feat["points_cam"], dtype=float),
        orb_pixels=np.asarray(feat["orb_pixels"], dtype=np.float32),
        orb_points_cam=np.asarray(feat["orb_points_cam"], dtype=float),
        orb_des=np.asarray(feat["orb_des"]),
        camera=stored,
        x_base_camera=x_cam,
        bbox_min=np.asarray(meta["bbox_min"], dtype=float),
        bbox_max=np.asarray(meta["bbox_max"], dtype=float),
        table_z_m=float(meta["table_z_m"]),
        depth_scale=float(meta["depth_scale"]),
        plane=plane,
        serial=str(meta["serial"]),
        extrinsics_hash=str(meta["extrinsics_hash"]),
        static_mask=mask,
    )


def lk_track(
    prev_gray: np.ndarray, gray: np.ndarray, pixels: np.ndarray, cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    if len(pixels) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 2), np.float32)
    p0 = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **LK_PARAMS)
    if p1 is None or p0r is None:
        return np.zeros(0, dtype=int), np.zeros((0, 2), np.float32)
    fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < float(cfg["max_flow_fb_px"]))
    idx = np.flatnonzero(good)
    return idx, p1.reshape(-1, 2)[idx].astype(np.float32)


def orb_correspondences(baseline: Baseline, gray: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    del cfg
    if len(baseline.orb_des) < 8:
        return np.zeros((0, 3)), np.zeros((0, 2), np.float32)
    orb = cv2.ORB_create(nfeatures=800)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or kps is None or len(kps) < 8:
        return np.zeros((0, 3)), np.zeros((0, 2), np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(baseline.orb_des, des, k=2)
    pixels, points = [], []
    for pair in raw:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            points.append(baseline.orb_points_cam[a.queryIdx])
            pixels.append(kps[a.trainIdx].pt)
    if not pixels:
        return np.zeros((0, 3)), np.zeros((0, 2), np.float32)
    return np.asarray(points, dtype=float), np.asarray(pixels, dtype=np.float32)


def filter_dynamic(
    baseline: Baseline,
    pixels_now: np.ndarray,
    points_cam: np.ndarray,
    depth_m: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    px = np.asarray(pixels_now, dtype=float).reshape(-1, 2)
    pts_cam = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    if len(px) == 0:
        return np.zeros(0, dtype=bool)
    z_now = sample_depth(depth_m, px)
    jump = np.isfinite(z_now) & (np.abs(z_now - pts_cam[:, 2]) > float(cfg["depth_jump_m"]))
    in_ws = np.zeros(len(px), dtype=bool)
    ok_z = np.isfinite(z_now) & (z_now >= cfg["min_depth_m"])
    if np.any(ok_z):
        pts = unproject_pixels(px[ok_z], z_now[ok_z], baseline.camera)
        base = transform_points(baseline.x_base_camera, pts)
        margin = float(cfg["workspace_margin_m"])
        lo = baseline.bbox_min.copy()
        hi = baseline.bbox_max.copy()
        lo[:2] -= margin
        hi[:2] += margin
        in_ws[ok_z] = inside_aabb(base, lo, hi)
    return ~(jump | in_ws)


def estimate_camera_motion(
    points_old_cam: np.ndarray,
    pixels_now: np.ndarray,
    camera: CameraModel,
    cfg: dict | None = None,
) -> dict[str, Any] | None:
    cfg = watch_settings(cfg)
    obj = np.asarray(points_old_cam, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(pixels_now, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 6 or len(obj) != len(img):
        return None
    dist = camera.distortion if camera.model == "brown_conrady" else np.zeros(5, dtype=float)
    if camera.model not in ("pinhole", "brown_conrady"):
        rays = camera.deproject_pixels(img, np.ones(len(img)))
        img_pinhole = (camera.k[:2, :2] @ rays[:, :2].T).T + camera.k[:2, 2]
        img_solve = img_pinhole
        dist = np.zeros(5, dtype=float)
    else:
        img_solve = img
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj.astype(np.float32),
        img_solve.astype(np.float32),
        camera.k,
        dist,
        iterationsCount=200,
        reprojectionError=float(cfg["ransac_reproj_px"]),
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None:
        return None
    inliers = np.asarray(inliers).ravel()
    if len(inliers) < int(cfg["min_inliers"]):
        return None
    ok2, rvec, tvec = cv2.solvePnP(
        obj[inliers].astype(np.float32),
        img_solve[inliers].astype(np.float32),
        camera.k,
        dist,
        rvec,
        tvec,
        True,
        cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok2:
        return None
    rotation = cv2.Rodrigues(rvec)[0]
    x = transform(rotation, tvec)
    projected = camera.project(transform_points(x, obj[inliers]))
    rms = float(np.sqrt(np.mean(np.sum((projected - img[inliers]) ** 2, axis=1))))
    return {"X_now_from_old": x, "inliers": inliers, "n_inliers": int(len(inliers)), "rms_px": rms}


def intersect_table(
    pixels: np.ndarray, camera: CameraModel, x_base_camera: np.ndarray, table_z_m: float
) -> np.ndarray:
    px = np.asarray(pixels, dtype=float).reshape(-1, 2)
    dirs = camera.deproject_pixels(px, np.ones(len(px), dtype=float))
    rotation, t = x_base_camera[:3, :3], x_base_camera[:3, 3]
    dir_base = dirs @ rotation.T
    denom = dir_base[:, 2]
    out = np.full((len(px), 3), np.nan)
    ok = np.abs(denom) > 1e-8
    lam = np.zeros(len(px))
    lam[ok] = (table_z_m - t[2]) / denom[ok]
    ok &= lam > 1e-6
    out[ok] = t + lam[ok, None] * dir_base[ok]
    return out


def table_apparent_shift(
    x_now_from_old: np.ndarray,
    camera: CameraModel,
    x_base_camera: np.ndarray,
    bbox_min: Any,
    bbox_max: Any,
    table_z_m: float,
    grid_n: int = 8,
) -> dict[str, Any] | None:
    x_now_from_old = valid_transform(x_now_from_old)
    grid = table_grid(bbox_min, bbox_max, table_z_m, grid_n)
    old_cam = transform_points(inv(x_base_camera), grid)
    now_cam = transform_points(x_now_from_old, old_cam)
    front = (old_cam[:, 2] > 0.05) & (now_cam[:, 2] > 0.05)
    if not np.any(front):
        return None
    px_old = camera.project(old_cam)
    px_now = camera.project(now_cam)
    in_img = (
        (px_now[:, 0] >= 0)
        & (px_now[:, 0] < camera.width)
        & (px_now[:, 1] >= 0)
        & (px_now[:, 1] < camera.height)
        & (px_old[:, 0] >= 0)
        & (px_old[:, 0] < camera.width)
        & (px_old[:, 1] >= 0)
        & (px_old[:, 1] < camera.height)
    )
    keep = front & in_img
    if not np.any(keep):
        keep = front
    est = intersect_table(px_now[keep], camera, x_base_camera, table_z_m)
    true = grid[keep]
    valid = np.isfinite(est).all(axis=1)
    if not np.any(valid):
        return None
    err_mm = np.linalg.norm(est[valid, :2] - true[valid, :2], axis=1) * 1000.0
    pix = np.linalg.norm(px_now[keep][valid] - px_old[keep][valid], axis=1)
    return {
        "grid": true[valid],
        "px_old": px_old[keep][valid],
        "px_now": px_now[keep][valid],
        "err_mm": err_mm,
        "pixel_shift": pix,
        "median_mm": float(np.median(err_mm)),
        "p95_mm": float(np.percentile(err_mm, 95)),
        "max_mm": float(np.max(err_mm)),
        "pixel_p95": float(np.percentile(pix, 95)),
    }


def workspace_hull(
    camera: CameraModel, x_base_camera: np.ndarray, bbox_min: Any, bbox_max: Any
) -> np.ndarray | None:
    lo, hi = np.asarray(bbox_min, dtype=float), np.asarray(bbox_max, dtype=float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    cam = transform_points(inv(x_base_camera), corners)
    front = cam[:, 2] > 0.05
    if np.count_nonzero(front) < 3:
        return None
    pixels = np.round(camera.project(cam[front])).astype(np.int32)
    hull = cv2.convexHull(pixels.reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def classify_status(p95_mm: float | None, n_inliers: int, cfg: dict) -> str:
    cfg = watch_settings(cfg)
    if p95_mm is None or n_inliers < int(cfg["min_inliers"]):
        return "TRACK_LOST"
    if p95_mm >= float(cfg["fail_table_mm"]):
        return "FAIL"
    if p95_mm >= float(cfg["warn_table_mm"]):
        return "WARN"
    return "OK"


class ConfirmGate:
    def __init__(self, confirm_frames: int) -> None:
        self.n = max(int(confirm_frames), 1)
        self.streak = 0

    def update(self, raw: str) -> str:
        if raw in ("OK", "TRACK_LOST"):
            self.streak = 0
            return raw
        self.streak += 1
        if self.streak >= self.n:
            return raw
        return "OK"


@dataclass
class Observation:
    status: str
    reason: str
    x_now_from_old: np.ndarray | None
    cam_mm: float | None
    cam_deg: float | None
    table_median_mm: float | None
    table_p95_mm: float | None
    table_max_mm: float | None
    pixel_p95: float | None
    n_tracks: int
    n_inliers: int
    rms_px: float | None
    inlier_now: np.ndarray
    rejected_now: np.ndarray
    grid_px_now: np.ndarray
    grid_px_old: np.ndarray
    grid_err_mm: np.ndarray
    plane_angle_deg: float | None
    plane_dist_delta_mm: float | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "cam_mm": self.cam_mm,
            "cam_deg": self.cam_deg,
            "table_median_mm": self.table_median_mm,
            "table_p95_mm": self.table_p95_mm,
            "table_max_mm": self.table_max_mm,
            "pixel_p95": self.pixel_p95,
            "n_tracks": self.n_tracks,
            "n_inliers": self.n_inliers,
            "rms_px": self.rms_px,
            "plane_angle_deg": self.plane_angle_deg,
            "plane_dist_delta_mm": self.plane_dist_delta_mm,
        }


def _empty_observation(status: str, reason: str, n_tracks: int = 0) -> Observation:
    empty = np.zeros((0, 2), np.float32)
    return Observation(
        status=status,
        reason=reason,
        x_now_from_old=None,
        cam_mm=None,
        cam_deg=None,
        table_median_mm=None,
        table_p95_mm=None,
        table_max_mm=None,
        pixel_p95=None,
        n_tracks=n_tracks,
        n_inliers=0,
        rms_px=None,
        inlier_now=empty,
        rejected_now=empty,
        grid_px_now=empty,
        grid_px_old=empty,
        grid_err_mm=np.zeros(0),
        plane_angle_deg=None,
        plane_dist_delta_mm=None,
    )


def observe_frame(
    baseline: Baseline, color: np.ndarray, depth_m: np.ndarray, cfg: dict | None = None
) -> Observation:
    cfg = watch_settings(cfg)
    if color.shape[:2] != (baseline.camera.height, baseline.camera.width):
        raise CalibrationError("当前图像尺寸与基线不一致")
    gray = gray_image(color)
    idx, px_lk = lk_track(baseline.gray, gray, baseline.pixels, cfg)
    points = baseline.points_cam[idx] if len(idx) else np.zeros((0, 3))
    used_orb = False
    if len(px_lk):
        keep = filter_dynamic(baseline, px_lk, points, depth_m, cfg)
        points, px_now = points[keep], px_lk[keep]
        rejected = px_lk[~keep]
    else:
        points, px_now = np.zeros((0, 3)), np.zeros((0, 2), np.float32)
        rejected = np.zeros((0, 2), np.float32)
    n_tracks = len(px_now)
    if n_tracks < int(cfg["orb_fallback_tracks"]):
        orb_pts, orb_px = orb_correspondences(baseline, gray, cfg)
        if len(orb_px):
            keep = filter_dynamic(baseline, orb_px, orb_pts, depth_m, cfg)
            orb_pts, orb_px = orb_pts[keep], orb_px[keep]
            if len(orb_px) > n_tracks:
                points, px_now = orb_pts, orb_px
                n_tracks = len(px_now)
                used_orb = True
                rejected = np.zeros((0, 2), np.float32)
    motion = estimate_camera_motion(points, px_now, baseline.camera, cfg)
    if motion is None:
        return _empty_observation("TRACK_LOST", "静物内点不足，无法判断三脚架是否移动", n_tracks)
    table = table_apparent_shift(
        motion["X_now_from_old"],
        baseline.camera,
        baseline.x_base_camera,
        baseline.bbox_min,
        baseline.bbox_max,
        baseline.table_z_m,
        int(cfg["grid_n"]),
    )
    if table is None:
        return _empty_observation("TRACK_LOST", "桌面网格无法投影", n_tracks)
    cam_mm, cam_deg = pose_error(motion["X_now_from_old"], np.eye(4))
    current_plane = fit_plane_ransac(
        table_points_camera(
            depth_m,
            baseline.camera,
            baseline.x_base_camera,
            baseline.bbox_min,
            baseline.bbox_max,
            baseline.table_z_m,
            cfg,
        ),
        cfg,
    )
    angle, dist_mm = plane_delta(baseline.plane, current_plane)
    inlier_now = px_now[motion["inliers"]]
    rej = px_now[np.setdiff1d(np.arange(len(px_now)), motion["inliers"])]
    if len(rejected):
        rej = np.vstack([rej, rejected]) if len(rej) else rejected
    status = classify_status(table["p95_mm"], motion["n_inliers"], cfg)
    reason = "ORB 兜底" if used_orb else ""
    return Observation(
        status=status,
        reason=reason,
        x_now_from_old=motion["X_now_from_old"],
        cam_mm=cam_mm,
        cam_deg=cam_deg,
        table_median_mm=table["median_mm"],
        table_p95_mm=table["p95_mm"],
        table_max_mm=table["max_mm"],
        pixel_p95=table["pixel_p95"],
        n_tracks=n_tracks,
        n_inliers=motion["n_inliers"],
        rms_px=motion["rms_px"],
        inlier_now=inlier_now.astype(np.float32),
        rejected_now=np.asarray(rej, dtype=np.float32).reshape(-1, 2),
        grid_px_now=table["px_now"].astype(np.float32),
        grid_px_old=table["px_old"].astype(np.float32),
        grid_err_mm=table["err_mm"],
        plane_angle_deg=angle,
        plane_dist_delta_mm=dist_mm,
    )
