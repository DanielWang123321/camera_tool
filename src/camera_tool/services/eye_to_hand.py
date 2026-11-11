"""Eye-to-Hand calibration service; only a complete live run can publish YAML."""
from __future__ import annotations

import os
from pathlib import Path

from camera_tool.calibration.eye_to_hand.workflow import LiveCalibration, load_settings, publish_camera
from camera_tool.core import Session, write_json
from camera_tool.paths import config_path
from camera_tool.services.tasks import DEVICES, resource


def calibrate(
    context,
    config: str | Path | None = None,
    output: str | Path = "runs/eye_to_hand",
    result: str | Path = "result/eye_to_hand.yaml",
    check: bool = False,
    publish: bool = False,
    ip: str | None = None,
    serial: str | None = None,
):
    cfg = load_settings(config or config_path("eye_to_hand.yaml"))
    resolved_ip = str(ip or os.environ.get("XARM_IP", "")).strip()
    if resolved_ip:
        cfg["robot"]["ip"] = resolved_ip
    fixed_cfg = cfg.get("fixed_camera") or cfg.get("realsense") or {}
    wrist_cfg = cfg.get("wrist_camera") or cfg.get("wrist") or {}
    if serial is not None and str(serial).strip():
        fixed_cfg["serial"] = str(serial).strip()
    fixed_backend = str(fixed_cfg.get("backend", "realsense")).strip().lower() or "realsense"
    wrist_backend = str(wrist_cfg.get("backend", "uvc")).strip().lower() or "uvc"
    resources = [
        resource(fixed_backend, fixed_cfg.get("serial")),
        resource(wrist_backend, wrist_cfg.get("serial")),
        resource("robot", cfg["robot"].get("ip")),
    ]
    session = live = None
    failed = True

    def progress(event, manifest):
        fixed_samples = manifest.get("fixed_samples") or manifest.get("realsense") or []
        context.emit(
            "progress",
            {
                "stage": event["event"],
                "fields": event,
                "samples": [dict(s) for s in manifest["samples"]],
                "fixed_count": len(fixed_samples),
            },
        )

    with DEVICES.lease(resources, "眼在手外标定"):
        try:
            context.check()
            session = Session(output, "check" if check else "live", on_event=progress)
            session.manifest["config"] = cfg
            session.save()
            live = LiveCalibration(cfg, session, context)
            live.start()
            solution = live.check_only() if check else live.run()
            context.check()
            write_json(session.path / "result.json", solution)
            session.finish("checked" if check else "passed", result_file="result.json")
            if publish and not check:
                publish_camera(result, solution, session)
                solution["published"] = True
            failed = False
            return {
                "mode": "eye-to-hand",
                "solution": solution,
                "session": session,
                "check": check,
                "samples": session.manifest["samples"],
                "directory": str(session.path),
            }
        except BaseException as exc:
            if session:
                session.finish("failed", failure=str(exc) or type(exc).__name__)
                context.emit(
                    "snapshot",
                    {
                        "mode": "eye-to-hand",
                        "samples": session.manifest["samples"],
                        "solution": None,
                        "directory": str(session.path),
                    },
                )
            raise
        finally:
            if live:
                live.close(failed)
