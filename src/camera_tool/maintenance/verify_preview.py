"""Verify published Eye-to-Hand extrinsics on live RGB-D stream (marker overlay + tabletop height)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from camera_tool.calibration.eye_to_hand.cameras import BufferedRgbdCamera
from camera_tool.core import CalibrationError, CameraModel, MarkerDetector, inv, marker_points, transform_points, valid_transform
from camera_tool.diagnostics.tripod import camera_from_camera_driver
from camera_tool.paths import DATA_ROOT, config_path


def load_result(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("X_base_camera") is None:
        raise CalibrationError(f"{path} 没有 X_base_camera，请先完成标定")
    return raw


def table_height_mm(
    depth_m: np.ndarray,
    camera: CameraModel,
    x_base_camera: np.ndarray,
    depth_scale: float,
    bbox: tuple[Any, Any],
) -> float | None:
    ys, xs = np.where(depth_m > 0.05)
    if len(xs) < 200:
        return None
    step = max(1, len(xs) // 4000)
    xs, ys = xs[::step], ys[::step]
    z = depth_m[ys, xs] * depth_scale
    pts = camera.deproject_pixels(np.column_stack([xs, ys]), z)
    base = transform_points(x_base_camera, pts)
    lo, hi = np.asarray(bbox[0]), np.asarray(bbox[1])
    inside = np.all((base >= lo) & (base <= hi), axis=1)
    picked = base[inside]
    if len(picked) < 50:
        picked = base
    return float(np.median(picked[:, 2]) * 1000.0)


def verify(context: Any, config: str | Path, result_path: str | Path, once: bool = False) -> dict[str, Any]:
    from camera_tool.paths import output_path
    from camera_tool.services.tasks import DEVICES, resource

    settings = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    fixed_cfg = settings.get("fixed_camera") or settings["realsense"]
    backend = str(fixed_cfg.get("backend", "realsense")).strip().lower()
    result = load_result(output_path(result_path))
    x_base_camera = valid_transform(result["X_base_camera"])
    x_base_board = valid_transform(result["X_base_board"]) if result.get("X_base_board") is not None else None
    correction = float(result.get("depth_scale") or 1.0)
    cam = BufferedRgbdCamera(fixed_cfg)
    with DEVICES.lease([resource(backend, fixed_cfg.get("serial"))], "外参核对"):
        try:
            context.check()
            cam.start()
            camera = camera_from_camera_driver(cam)
            detector = MarkerDetector(settings["marker"], camera)
            predicted = (
                camera.project(
                    transform_points(inv(x_base_camera) @ x_base_board, marker_points(settings["marker"]["side_m"]))
                )
                if x_base_board is not None
                else None
            )
            bbox = (result.get("bbox_min", [0.20, -0.22, -0.03]), result.get("bbox_max", [0.61, 0.45, 0.40]))
            number = None
            while True:
                context.check()
                frame = cam.get_frame(wait_s=2.0, after_frame_number=number)
                if frame is None:
                    raise CalibrationError("固定相机断流")
                number = frame["frame_number"]
                image = frame["color"].copy()
                height = table_height_mm(frame["depth_m"], camera, x_base_camera, correction, bbox)
                rms = None
                try:
                    detection = detector.detect(image)
                    cv2.polylines(image, [np.round(detection["corners"]).astype(int)], True, (0, 255, 0), 2)
                    if predicted is not None:
                        rms = float(np.sqrt(np.mean((predicted - detection["corners"]) ** 2)))
                except CalibrationError:
                    pass
                if predicted is not None:
                    cv2.polylines(image, [np.round(predicted).astype(int)], True, (0, 255, 255), 2)
                metrics = {"table_height_mm": height, "corner_rms_px": rms, "depth_scale_correction": correction}
                context.frame("preview", {"color_bgr": image, "depth_m": frame["depth_m"]})
                context.emit("status", str(metrics))
                if once:
                    return metrics
        finally:
            cam.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path("eye_to_hand.yaml")))
    parser.add_argument("--result", default=str(DATA_ROOT / "result" / "eye_to_hand.yaml"))
    parser.add_argument("--once", action="store_true", help="无界面读取一帧后退出")
    args = parser.parse_args(argv)
    if args.once:
        import json
        from camera_tool.services.tasks import TaskContext

        print(json.dumps(verify(TaskContext(), args.config, args.result, once=True)))
        return 0
    from camera_tool.ui.qt_runtime import prepare_qt

    prepare_qt()
    from PySide6.QtWidgets import QApplication
    from camera_tool.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(["camera-tool"])
    window = MainWindow()
    window.setWindowTitle("camera_tool · 已发布外参核对")
    window.run_task("preview", lambda context: verify(context, args.config, args.result))
    window.show()
    return app.exec()
