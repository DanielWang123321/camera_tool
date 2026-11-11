"""Monitor whether a fixed tripod RGB-D camera has drifted relative to its locked scene baseline."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from camera_tool.calibration.eye_to_hand.cameras import BufferedRgbdCamera
from camera_tool.core import CalibrationError, CameraModel, json_default, valid_transform
from camera_tool.diagnostics.scene_lock import (
    ConfirmGate,
    Observation,
    camera_from_meta,
    camera_from_rs_color,
    load_baseline,
    lock_scene,
    observe_frame,
    resolve_baseline_dir,
    save_baseline,
    watch_settings,
    workspace_hull,
)

__all__ = [
    "ConfirmGate",
    "Observation",
    "append_log",
    "camera_from_camera_driver",
    "draw_observation",
    "grab_corrected",
    "load_baseline",
    "load_result",
    "load_yaml",
    "lock_from_camera",
    "observe_frame",
    "resolve_baseline_dir",
    "watch_settings",
]


def median_depth_stack(depths: list[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(depths, axis=0), axis=0).astype(np.float32)


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CalibrationError(f"{path} 不是有效配置")
    return raw


def load_result(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    if raw.get("X_base_camera") is None:
        raise CalibrationError(f"{path} 没有 X_base_camera，请先完成标定")
    return raw


def camera_from_camera_driver(cam: BufferedRgbdCamera) -> CameraModel:
    meta = cam.calibration_metadata
    if "color" in meta:
        return camera_from_rs_color(meta["color"])
    return camera_from_meta(meta)


def mm_color(mm: float, warn: float, fail: float) -> tuple[int, int, int]:
    if mm >= fail:
        return (0, 0, 255)
    if mm >= warn:
        return (0, 200, 255)
    return (0, 220, 0)


def status_color(status: str) -> tuple[int, int, int]:
    return {
        "OK": (0, 220, 0),
        "WARN": (0, 200, 255),
        "FAIL": (0, 0, 255),
        "TRACK_LOST": (180, 180, 180),
    }.get(status, (255, 255, 255))


def draw_dashed_hull(image: np.ndarray, hull: np.ndarray | None, color: tuple[int, int, int], gap: int = 8) -> None:
    if hull is None or len(hull) < 2:
        return
    pts = np.vstack([hull.reshape(-1, 2), hull.reshape(-1, 2)[:1]])
    for i in range(len(pts) - 1):
        a = pts[i].astype(float)
        b = pts[i + 1].astype(float)
        dist = float(np.linalg.norm(b - a))
        if dist < 2:
            continue
        n = max(int(dist / gap), 1)
        for j in range(n):
            if j % 2:
                continue
            t0, t1 = j / n, min((j + 1) / n, 1.0)
            p0 = tuple(np.round(a * (1 - t0) + b * t0).astype(int))
            p1 = tuple(np.round(a * (1 - t1) + b * t1).astype(int))
            cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)


def put_banner(image: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> None:
    y = 28
    for line in lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        y += 26


def draw_observation(
    image: np.ndarray, baseline: Any, obs: Observation, cfg: dict[str, Any], confirmed: str
) -> np.ndarray:
    out = image.copy()
    warn, fail = float(cfg["warn_table_mm"]), float(cfg["fail_table_mm"])
    hull = workspace_hull(baseline.camera, baseline.x_base_camera, baseline.bbox_min, baseline.bbox_max)
    draw_dashed_hull(out, hull, (0, 255, 255))
    for pt in np.asarray(obs.rejected_now).reshape(-1, 2):
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 220), -1, cv2.LINE_AA)
    for pt in np.asarray(obs.inlier_now).reshape(-1, 2):
        cv2.circle(out, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 220, 0), -1, cv2.LINE_AA)
    for p_now, p_old, err in zip(obs.grid_px_now, obs.grid_px_old, obs.grid_err_mm):
        a = tuple(np.round(p_now).astype(int))
        delta = p_now - p_old
        b = tuple(np.round(p_now + delta).astype(int))
        color = mm_color(float(err), warn, fail)
        cv2.arrowedLine(out, a, b, color, 1, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(out, a, 2, color, -1, cv2.LINE_AA)
    p95 = obs.table_p95_mm
    cam = f"{obs.cam_mm:.1f}mm/{obs.cam_deg:.2f}deg" if obs.cam_mm is not None else "--"
    table = f"{p95:.1f}mm" if p95 is not None else "--"
    line = f"table p95={table}  cam {cam}  inliers={obs.n_inliers}  {confirmed}"
    extra = []
    if obs.plane_angle_deg is not None and obs.plane_dist_delta_mm is not None:
        extra.append(f"plane {obs.plane_angle_deg:.2f}deg  dz={obs.plane_dist_delta_mm:.1f}mm")
    if obs.reason:
        extra.append(obs.reason)
    put_banner(out, [line, *extra], status_color(confirmed))
    return out


def grab_corrected(
    cam: BufferedRgbdCamera,
    depth_scale: float,
    wait_s: float = 2.0,
    after_frame_number: int | None = None,
) -> dict[str, Any]:
    frame = cam.get_frame(wait_s=wait_s, max_age_s=0.5, after_frame_number=after_frame_number)
    if frame is None:
        raise CalibrationError("固定相机断流")
    return {
        **frame,
        "depth_m": np.asarray(frame["depth_m"], dtype=np.float32) * float(depth_scale),
    }


def lock_from_camera(
    cam: BufferedRgbdCamera,
    result: dict[str, Any],
    cfg: dict[str, Any],
    directory: Path,
    context: Any = None,
) -> Path:
    camera = camera_from_camera_driver(cam)
    x_base_camera = valid_transform(result["X_base_camera"])
    bbox_min = result.get("bbox_min", [0.20, -0.22, -0.03])
    bbox_max = result.get("bbox_max", [0.61, 0.45, 0.40])
    table_z = float(result.get("table_z_m", 0.0))
    depth_scale = float(result.get("depth_scale") or 1.0)
    colors, depths = [], []
    last_fn = None
    for _ in range(int(cfg["lock_frames"])):
        if context:
            context.check()
        frame = grab_corrected(cam, depth_scale, after_frame_number=last_fn)
        last_fn = frame["frame_number"]
        colors.append(frame["color"])
        depths.append(frame["depth_m"])
        if context:
            context.frame("tripod", {"color_bgr": frame["color"]})
            context.emit("status", f"锁定基线 {len(colors)}/{cfg['lock_frames']}")
    color = colors[-1]
    depth_m = median_depth_stack(depths)
    baseline = lock_scene(
        color,
        depth_m,
        camera,
        x_base_camera,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        table_z_m=table_z,
        depth_scale=depth_scale,
        serial=str(cam.calibration_metadata.get("serial") or cam.serial),
        cfg=cfg,
    )
    path = save_baseline(baseline, directory)
    print(f"已锁基线：{path}  静物点 {len(baseline.pixels)}  ORB {len(baseline.orb_pixels)}", flush=True)
    return path


def append_log(path: Path, obs: Observation, confirmed: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": datetime.now().astimezone().isoformat(), **obs.public_dict(), "confirmed": confirmed}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default, allow_nan=False) + "\n")
