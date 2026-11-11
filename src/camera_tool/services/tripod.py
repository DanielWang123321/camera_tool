"""Tripod baseline locking and drift monitoring shared by desktop UI and CLI."""
from __future__ import annotations

from pathlib import Path

from camera_tool.calibration.eye_to_hand.cameras import BufferedRgbdCamera
from camera_tool.core import load_yaml, valid_transform
from camera_tool.diagnostics.scene_lock import (
    ConfirmGate,
    camera_from_rs_color,
    load_baseline,
    observe_frame,
    resolve_baseline_dir,
    watch_settings,
)
from camera_tool.diagnostics.tripod import append_log, draw_observation, grab_corrected, load_result, lock_from_camera
from camera_tool.paths import config_path, output_path
from camera_tool.services.tasks import DEVICES, resource


def tripod(context, action, config=None, result="result/eye_to_hand.yaml", baseline=None, once=False, log=None):
    settings = load_yaml(Path(config or config_path("eye_to_hand.yaml")))
    calibration = load_result(output_path(result))
    cfg = watch_settings(settings.get("tripod_watch"))
    calibration.setdefault("table_z_m", settings.get("checks", {}).get("table_z_m", 0))
    directory = resolve_baseline_dir(baseline or cfg["baseline_dir"])
    fixed_cfg = settings.get("fixed_camera") or settings.get("realsense") or {}
    backend = str(fixed_cfg.get("backend", "realsense")).strip().lower() or "realsense"
    camera = BufferedRgbdCamera(fixed_cfg)
    with DEVICES.lease([resource(backend, fixed_cfg.get("serial"))], "三脚架检测"):
        try:
            context.check()
            camera.start()
            if action == "lock":
                path = lock_from_camera(camera, calibration, cfg, directory, context=context)
                return {"baseline": str(path)}
            model = camera_from_rs_color(camera.calibration_metadata["color"])
            locked = load_baseline(
                directory,
                x_base_camera=valid_transform(calibration["X_base_camera"]),
                serial=str(camera.calibration_metadata["serial"]),
                camera=model,
            )
            gate = ConfirmGate(1 if once else cfg["confirm_frames"])
            last_number = None
            while True:
                context.check()
                frame = grab_corrected(camera, locked.depth_scale, after_frame_number=last_number)
                last_number = frame["frame_number"]
                obs = observe_frame(locked, frame["color"], frame["depth_m"], cfg)
                confirmed = gate.update(obs.status)
                payload = {
                    **obs.public_dict(),
                    "confirmed": confirmed,
                    "warn_mm": cfg["warn_table_mm"],
                    "fail_mm": cfg["fail_table_mm"],
                }
                context.frame(
                    "tripod",
                    {
                        "color_bgr": draw_observation(frame["color"], locked, obs, cfg, confirmed),
                        "observation": payload,
                    },
                )
                context.emit("tripod", payload)
                if once:
                    if log:
                        append_log(output_path(log), obs, confirmed)
                    return payload
                if log:
                    append_log(output_path(log), obs, confirmed)
        finally:
            camera.stop()
