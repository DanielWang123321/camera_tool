"""Four-corner tabletop marker calibration: read-only TCP + fixed RGB-D camera."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from camera_tool.calibration.eye_to_hand.cameras import BufferedRgbdCamera
from camera_tool.core import CalibrationError, CameraModel, MarkerDetector, marker_points, matrix_list
from camera_tool.diagnostics.tripod import camera_from_camera_driver
from camera_tool.paths import DATA_ROOT, config_path

LABELS = ("左下", "右下", "右上", "左上")
CYCLES = (
    ("左上", "右上", "右下", "左下"),
    ("右上", "右下", "左下", "左上"),
    ("右下", "左下", "左上", "右上"),
    ("左下", "左上", "右上", "右下"),
)


def load_corners(path: Path) -> dict[str, np.ndarray]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get("corners_mm") or {}
    out = {}
    for name in LABELS:
        if name not in block:
            raise SystemExit(f"{path} 缺少 {name}")
        out[name] = np.asarray(block[name][:3], dtype=np.float64)
    return out


def print_geometry(tcp_mm: dict[str, np.ndarray], side_mm: float) -> None:
    pts = np.stack([tcp_mm[k] for k in LABELS], axis=0)
    print("边长（相对印刷边）：")
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))
    names = ("左下-右下", "右下-右上", "右上-左上", "左上-左下")
    for (i, j), name in zip(pairs, names):
        d = float(np.linalg.norm(pts[i] - pts[j]))
        print(f"  {name}: {d:.1f} mm  ({d - side_mm:+.1f})")
    print(f"中心: {np.round(pts.mean(axis=0), 1)} mm")


def orthonormalize(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    u, _, vt = np.linalg.svd(out[:3, :3])
    out[:3, :3] = u @ vt
    if np.linalg.det(out[:3, :3]) < 0:
        out[:3, :2] *= -1
    return out


def T_base_marker(tcp_mm: dict[str, np.ndarray], labels: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    p = np.stack([tcp_mm[k] / 1000.0 for k in labels], axis=0)
    x = p[1] - p[0]
    y = p[0] - p[3]
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    z = z / np.linalg.norm(z)
    if z[2] < 0:
        z = -z
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    t = np.eye(4)
    t[:3, :3] = np.stack([x, y, z], axis=1)
    t[:3, 3] = p.mean(axis=0)
    return t, p


def score_pose(x: np.ndarray, marker_c: np.ndarray) -> tuple[bool, float]:
    cam = x[:3, 3]
    look = x[:3, 2]
    to_m = marker_c - cam
    n = np.linalg.norm(to_m)
    if n < 1e-6:
        return False, 0.0
    align = float(np.dot(look, to_m / n))
    return bool(cam[2] > 0.15 and align > 0.5), align


def write_camera_yaml(
    path: Path,
    x: np.ndarray,
    mean_e: float,
    max_e: float,
    settings: dict[str, Any],
    serial: str,
) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raw = {}
    fixed_cfg = settings.get("fixed_camera") or settings["realsense"]
    marker = settings["marker"]
    raw["X_base_camera"] = matrix_list(x)
    raw.update(
        serial=serial,
        width=fixed_cfg["width"],
        height=fixed_cfg["height"],
        fps=fixed_cfg["fps"],
        num_points=raw.get("num_points", 1024),
        bbox_min=raw.get("bbox_min", [0.20, -0.22, -0.03]),
        bbox_max=raw.get("bbox_max", [0.61, 0.45, 0.40]),
        marker_length_m=marker["side_m"],
        marker_id=marker["id"],
        aruco_dict=marker["dictionary"],
    )
    raw["calibration"] = {
        "method": "four_corners",
        "mean_mm": mean_e,
        "max_mm": max_e,
        "camera_frame": "color_optical_frame",
        "depth_aligned_to": "color",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 桌面码四角 TCP 写入 X_base_camera。\n")
        yaml.safe_dump(raw, handle, allow_unicode=True, sort_keys=False)


def _read_tcp_m(arm: Any) -> np.ndarray | None:
    code, tcp = arm.get_position_aa(is_radian=True)
    if code != 0 or tcp is None:
        return None
    return np.asarray(tcp[:3], dtype=np.float64) / 1000.0


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path("eye_to_hand.yaml")))
    parser.add_argument("--corners", default=str(config_path("corners.yaml")))
    parser.add_argument("--result", default=str(DATA_ROOT / "result" / "eye_to_hand.yaml"))
    parser.add_argument("--robot-ip", default=None, help="机械臂控制器 IP")
    parser.add_argument("--hypothesis", type=int, default=-1, help="强制使用 0～3 号朝向，默认自动选")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写结果 YAML")
    args = parser.parse_args(argv)

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fixed_cfg = settings.get("fixed_camera") or settings["realsense"]
    tcp_mm = load_corners(Path(args.corners))
    side_mm = float(settings["marker"]["side_m"]) * 1000.0
    print_geometry(tcp_mm, side_mm)

    cam = BufferedRgbdCamera(fixed_cfg)
    try:
        cam.start()
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1

    arm = None
    try:
        robot_ip = str(args.robot_ip or os.environ.get("XARM_IP") or settings.get("robot", {}).get("ip") or "").strip()
        if robot_ip:
            try:
                from xarm.wrapper import XArmAPI

                arm = XArmAPI(robot_ip)
                time.sleep(0.2)
                if not arm.connected:
                    print("机械臂未连上，朝向需要 --hypothesis 0～3。")
                    arm = None
            except Exception as exc:
                print(f"机械臂未连上（{exc}），朝向需要 --hypothesis 0～3。")
                arm = None

        camera: CameraModel = camera_from_camera_driver(cam)
        detector = MarkerDetector(settings["marker"], camera)
        hit = None
        frame = None
        for _ in range(15):
            frame = cam.get_frame(max_age_s=2.0, wait_s=1.0)
            if frame is None:
                continue
            try:
                detection = detector.detect(frame["color"])
            except Exception:
                continue
            poses = [p for p, _ in camera.candidates(detection["corners"], settings["marker"]["side_m"])]
            hit = {
                "img_pts": np.asarray(detection["corners"]),
                "obj": marker_points(settings["marker"]["side_m"]),
                "poses": poses,
            }
            break
        if hit is None:
            if frame is None:
                print("相机没出帧。")
                return 1
            out = DATA_ROOT / "last_nomarker.png"
            cv2.imwrite(str(out), frame["color"])
            print(f"相机没看到码 {settings['marker']['id']}。末端工具让开后再跑。画面已存 {out}")
            return 1

        t_cm = hit["poses"][0]
        obj = hit["obj"]
        img_pts = hit["img_pts"]
        fx, fy, cx, cy = camera.k[0, 0], camera.k[1, 1], camera.k[0, 2], camera.k[1, 2]
        depth = frame["depth_m"]
        grip = _read_tcp_m(arm) if arm is not None else None
        marker_c = np.mean([tcp_mm[n] / 1000.0 for n in LABELS], axis=0)

        rows = []
        for i, labels in enumerate(CYCLES):
            t_bm, p_tcp = T_base_marker(tcp_mm, labels)
            x = orthonormalize(t_bm @ np.linalg.inv(t_cm))
            cam_pts = (t_cm @ np.hstack([obj, np.ones((4, 1))]).T).T[:, :3]
            pred = (x[:3, :3] @ cam_pts.T).T + x[:3, 3]
            err = np.linalg.norm(pred - p_tcp, axis=1) * 1000.0
            ok, align = score_pose(x, marker_c)
            depth_err = None
            pix = None
            if grip is not None and depth is not None:
                t_cb = np.linalg.inv(x)
                u, v = camera.project(grip.reshape(1, 3) @ t_cb[:3, :3].T + t_cb[:3, 3]).reshape(2)
                ui, vi = int(round(u)), int(round(v))
                pix = (ui, vi)
                if 0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]:
                    z_d = float(depth[vi, ui])
                    if z_d > 0.05:
                        p_cam = np.array([(u - cx) * z_d / fx, (v - cy) * z_d / fy, z_d, 1.0])
                        p_base = x @ p_cam
                        depth_err = float(np.linalg.norm(p_base[:3] - grip) * 1000.0)
            rows.append(
                dict(i=i, labels=labels, X=x, err=err, ok=ok, align=align, cam_xyz=x[:3, 3], depth_err=depth_err, pix=pix)
            )
            extra = f"  TCP投影深度差 {depth_err:.0f} mm" if depth_err is not None else ""
            print(
                f"朝向 {i} {labels}: 相机 {np.round(x[:3, 3] * 1000, 0)} mm  "
                f"残差 {err.mean():.2f}/{err.max():.2f} mm  {'可用' if ok else '丢弃'}{extra}"
            )

        if 0 <= args.hypothesis <= 3:
            best = rows[args.hypothesis]
            print(f"使用 --hypothesis {args.hypothesis}")
        else:
            usable = [r for r in rows if r["ok"]]
            if not usable:
                print("没有相机在桌上方且朝向码的解。")
                return 1
            with_depth = [r for r in usable if r["depth_err"] is not None]
            if with_depth:
                best = min(with_depth, key=lambda r: r["depth_err"])
            else:
                best = min(usable, key=lambda r: float(r["err"].max()))
                if grip is None:
                    print("未读到 TCP，按残差最小选取。请用 --hypothesis 确认朝向。")

        mean_e = float(best["err"].mean())
        max_e = float(best["err"].max())
        print(f"选用朝向 {best['i']} {best['labels']}")
        print(f"三维残差平均 {mean_e:.2f} mm，最大 {max_e:.2f} mm")
        print("相机位置 mm", np.round(best["cam_xyz"] * 1000, 1))
        print("X_base_camera =\n", np.array_str(best["X"], precision=6, suppress_small=True))

        vis = frame["color"].copy()
        for j, p in enumerate(np.round(img_pts).astype(int)):
            cv2.circle(vis, tuple(p), 7, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{j}:{best['labels'][j]}",
                (int(p[0] + 6), int(p[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        overlay = DATA_ROOT / "last_overlay.png"
        cv2.imwrite(str(overlay), vis)
        print("画面已存", overlay)
        if args.dry_run:
            print("dry-run，未写入结果 YAML")
            return 0
        write_camera_yaml(Path(args.result), best["X"], mean_e, max_e, settings, cam.serial)
        print(f"已写入 {args.result}")
        return 0
    finally:
        cam.stop()
        if arm is not None:
            arm.disconnect()


def main(argv: list[str] | None = None) -> int:
    from camera_tool.services.tasks import DEVICES, resource

    with DEVICES.lease([resource("realsense"), resource("orbbec"), resource("robot")], "维护工具"):
        return _run(argv)
