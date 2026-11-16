"""Unified CLI and GUI entry point; headless commands never import or initialize Qt."""
from __future__ import annotations

import argparse
import json
import signal
import sys


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="camera-tool",
        description="Camera preview, hand-eye calibration, offline replay, and diagnostics tool",
    )
    root.add_argument("--sim", action="store_true", help="Run in offline simulation mode")
    sub = root.add_subparsers(dest="command")

    preview = sub.add_parser("preview", help="Open live camera preview or list available devices")
    preview.add_argument(
        "--backend",
        choices=("realsense", "orbbec", "uvc", "sim"),
        default="realsense",
        help="Camera backend (default: realsense)",
    )
    preview.add_argument("--serial", default="", help="Device serial number (leave empty to auto-select single device)")
    preview.add_argument("--width", type=int)
    preview.add_argument("--height", type=int)
    preview.add_argument("--fps", type=int)
    preview.add_argument("--no-depth", action="store_true", help="Enable color stream only")
    preview.add_argument("--max-depth", type=float, help="Max depth range for colormap visualization (meters)")
    preview.add_argument("--list", action="store_true", help="Discover backends and connected devices as JSON")
    preview.add_argument("--headless", action="store_true", help="Grab one headless frame and verify stream status")

    cal = sub.add_parser("calibrate", help="Run hand-eye calibration (live mode commands robot motion)")
    cal.add_argument("--mode", choices=("eye-in-hand", "eye-to-hand"), required=True)
    cal.add_argument(
        "--backend",
        choices=("auto", "realsense", "orbbec", "sim"),
        default="auto",
        help="Camera backend (default: auto)",
    )
    cal.add_argument(
        "--preset",
        choices=("auto", "default", "medium_range", "orbbec"),
        default="auto",
        help="Eye-in-Hand preset config template (default: auto)",
    )
    cal.add_argument("--profile", help="Robot kinematic profile (e.g., arm_6dof, arm_7dof, arm_compact_6dof)")
    cal.add_argument("--serial", default="", help="Camera serial number (leave empty to auto-select single device)")
    cal.add_argument("--robot-ip", help="Override robot IP (falls back to XARM_IP env var and YAML config)")
    cal.add_argument("--config", help="Path to custom YAML configuration file")
    cal.add_argument("--check", action="store_true", help="Read-only check; never enable, move robot, or publish YAML")
    cal.add_argument("--auto", action="store_true", help="Headless automatic capture across waypoints and solve")
    cal.add_argument("--sim", action="store_true", help="Offline simulation mode (eye-in-hand only)")
    cal.add_argument("--output", help="Calibration session output directory")
    cal.add_argument("--result", help="Exported calibration result YAML path")
    cal.add_argument("--gui", action="store_true", help="Open calibration page in desktop GUI")

    replay = sub.add_parser("replay", help="Auto-detect recorded session and recompute offline (no hardware SDK import)")
    replay.add_argument("directory", help="Path to recorded session directory")
    replay.add_argument("--config", help="Optional Eye-in-Hand config YAML for replay verification")
    replay.add_argument("--result", help="Eye-in-Hand output YAML path (eye-to-hand replay cannot publish)")
    replay.add_argument(
        "--tcp-offset",
        help="Override tool TCP offset x,y,z,r,p,y (mm, rad), e.g. 0,0,172,0,0,0",
    )
    replay.add_argument("--output", default="runs/eye_to_hand/replay", help="Eye-to-Hand offline replay output directory")
    replay.add_argument("--gui", action="store_true", help="Open replay results in desktop GUI")

    tripod = sub.add_parser("tripod", help="Fixed camera baseline lock and tripod drift monitoring")
    tripod.add_argument("action", choices=("lock", "watch"))
    tripod.add_argument("--config", help="Configuration YAML file path")
    tripod.add_argument("--result", default="result/eye_to_hand.yaml", help="Published Eye-to-Hand result YAML")
    tripod.add_argument("--baseline", help="Baseline save/load directory")
    tripod.add_argument("--once", action="store_true", help="Headless single-frame check; returns exit code 2 on FAIL")
    tripod.add_argument("--headless", action="store_true", help="Headless continuous monitoring stream")
    tripod.add_argument("--log", help="Monitoring log file path (JSONL)")

    usb = sub.add_parser("usb-test", help="Headless USB link speed and frame-rate stability test")
    usb.add_argument("--backend", choices=("realsense", "orbbec"), default="realsense", help="RGB-D camera backend")
    usb.add_argument("--serial", default="", help="Device serial number (empty auto-selects)")
    usb.add_argument("--width", type=int, default=1280)
    usb.add_argument("--height", type=int, default=720)
    usb.add_argument("--fps", type=int, default=30)
    usb.add_argument("--duration", type=float, default=3600)
    usb.add_argument("--interval", type=float, default=5)
    usb.add_argument("--min-fps", type=float, default=29.0)
    usb.add_argument("--max-gap", type=float, default=0.1)
    usb.add_argument("--csv", help="Export window statistics CSV path")
    usb.add_argument("--quiet", action="store_true", help="Print only final summary JSON")

    wrist = sub.add_parser("wrist-intrinsics", help="Export wrist camera undistortion map for Eye-to-Hand calibration")
    wrist.add_argument("--config", default=None, help="Eye-to-Hand configuration YAML path")
    wrist.add_argument("--serial", default=None, help="Wrist camera serial number (optional when a single device is connected)")
    wrist.add_argument("--from-run", default=None, help="Copy intrinsics and rectification map from a recorded session directory")
    wrist.add_argument("--output", default=None, help="Output JSON path (default: intrinsics/wrist_<serial>.json)")
    return root


def _json(value) -> str:
    from camera_tool.core import json_default

    return json.dumps(value, ensure_ascii=False, default=json_default)


def _parse_tcp_offset(raw: str | None) -> list[float] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if len(parts) != 6:
        raise ValueError("--tcp-offset must be 6 comma-separated values x,y,z,r,p,y (mm, rad), e.g. 0,0,172,0,0,0")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--tcp-offset contains non-numeric values") from exc


def launch_gui(args=None) -> int:
    from camera_tool.ui.qt_runtime import prepare_qt

    prepare_qt()
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from camera_tool.ui.main_window import MainWindow

    args = args or argparse.Namespace(command=None, sim=False)
    command = args.command
    page = {"preview": 0, "calibrate": 1, "replay": 2, "tripod": 3}.get(command, 0)
    sim_flag = bool(getattr(args, "sim", False) or getattr(args, "backend", "") == "sim")
    app = QApplication.instance() or QApplication(["camera-tool"])
    window = MainWindow(
        sim_mode=sim_flag if command != "preview" else False,
        initial_page=page,
        initial_mode=getattr(args, "mode", "eye-in-hand"),
        replay_dir=args.directory if command == "replay" else None,
    )
    if command == "preview":
        index = window.combo_camera.findData(args.backend)
        if index >= 0:
            window.combo_camera.setCurrentIndex(index)
        window.combo_device.setEditText(args.serial)
        from camera_tool.devices.registry import PROFILES

        w, h, fps = PROFILES[args.backend][0]
        window.combo_profile.setEditText(f"{args.width or w}x{args.height or h}@{args.fps or fps}")
        window.chk_depth.setChecked(not args.no_depth and args.backend != "uvc")
        if args.max_depth is not None:
            window.max_depth.setValue(args.max_depth)
        QTimer.singleShot(0, window.start_preview)
    elif command == "calibrate":
        backend = getattr(args, "backend", "auto")
        if backend not in ("auto", "sim") and hasattr(window, "combo_eih_backend"):
            idx = window.combo_eih_backend.findData(backend)
            if idx >= 0:
                window.combo_eih_backend.setCurrentIndex(idx)
        preset = getattr(args, "preset", "auto")
        if preset != "auto" and hasattr(window, "combo_eih_preset"):
            idx = window.combo_eih_preset.findData(preset)
            if idx >= 0:
                window.combo_eih_preset.setCurrentIndex(idx)
        if args.config:
            window.cal_config.setText(args.config)
        if getattr(args, "robot_ip", None) and hasattr(window, "cal_robot_ip"):
            window.cal_robot_ip.setText(args.robot_ip)
            window._on_ip_edited(args.robot_ip)
        if args.check:
            QTimer.singleShot(0, lambda: window.start_calibration(check=True))
    elif command == "tripod":
        if args.config:
            window.tripod_config.setText(args.config)
        window.tripod_result.setText(args.result)
        if args.baseline:
            window.tripod_baseline.setText(args.baseline)
        QTimer.singleShot(0, lambda: window.start_tripod(args.action))
    window.show()
    return app.exec()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    command = args.command
    if command is None or (command in ("calibrate", "replay") and args.gui):
        return launch_gui(args)
    if command == "preview":
        if args.list:
            from camera_tool.devices.registry import discover

            print(_json(discover()))
            return 0
        if not args.headless:
            return launch_gui(args)
    if command == "tripod" and args.action == "watch" and not args.once and not args.headless:
        return launch_gui(args)

    from camera_tool.services.tasks import Cancelled, TaskContext, task_scope

    def emit(kind, value):
        if kind in ("usb", "tripod", "usb_summary") and not getattr(args, "quiet", False):
            print(_json({kind: value}), flush=True)
        elif kind == "status":
            print(str(value), file=sys.stderr, flush=True)

    context = TaskContext(emit)
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: context.cancelled.set())
    try:
        with task_scope(context):
            if command == "preview":
                from camera_tool.devices.registry import (
                    PROFILES,
                    create_camera,
                    device_resource,
                )
                from camera_tool.services.tasks import DEVICES

                w, h, fps = PROFILES[args.backend][0]
                cfg = {
                    "serial": args.serial,
                    "width": args.width or w,
                    "height": args.height or h,
                    "fps": args.fps or fps,
                    "enable_depth": not args.no_depth and args.backend != "uvc",
                }
                cam = create_camera(args.backend, cfg)
                with DEVICES.lease([device_resource(args.backend, args.serial)], "Headless preview check"):
                    try:
                        meta = cam.start()
                        frame = cam.read()
                    finally:
                        cam.stop()
                print(
                    _json(
                        {
                            "backend": args.backend,
                            "metadata": meta,
                            "color_shape": list(frame["color_bgr"].shape),
                            "has_depth": frame.get("depth_m") is not None,
                        }
                    )
                )
                return 0
            if command == "calibrate":
                sim = bool(args.sim or args.backend == "sim")
                if sim and args.mode != "eye-in-hand":
                    raise ValueError("Offline simulation is only supported for eye-in-hand mode")
                output = args.output or f"runs/{args.mode.replace('-', '_')}"
                result = args.result or f"result/{args.mode.replace('-', '_')}.yaml"
                if args.mode == "eye-in-hand":
                    from camera_tool.services.eye_in_hand import EyeInHand

                    backend = None if args.backend in ("auto", "sim") else args.backend
                    preset = None if args.preset == "auto" else args.preset
                    payload = EyeInHand(
                        config=args.config,
                        sim=sim,
                        output=output,
                        profile=args.profile,
                        ip=args.robot_ip,
                        backend=backend,
                        preset=preset,
                        serial=args.serial,
                    ).run(context, check=args.check, automatic=not args.check, result=result)
                else:
                    from camera_tool.services.eye_to_hand import calibrate

                    payload = calibrate(
                        context,
                        config=args.config,
                        output=output,
                        result=result,
                        check=args.check,
                        publish=True,
                        ip=args.robot_ip,
                        serial=args.serial,
                    )
                keys = (
                    "mode",
                    "directory",
                    "output",
                    "check",
                    "motion_commands_sent",
                    "camera",
                    "robot",
                    "evaluation",
                )
                print(_json({k: payload[k] for k in keys if k in payload}))
                return 0
            if command == "replay":
                from camera_tool.services.replay import detect_session, load_session

                mode, _ = detect_session(args.directory)
                tcp_offset = _parse_tcp_offset(args.tcp_offset)
                if tcp_offset is not None and mode != "eye-in-hand":
                    raise ValueError("--tcp-offset is only supported for eye-in-hand replay")
                if args.result and mode == "eye-to-hand":
                    raise ValueError("Eye-to-hand replay cannot publish extrinsics; inspect result.json in the output directory")
                payload = load_session(
                    context,
                    args.directory,
                    config=args.config,
                    strict=True,
                    output=args.output,
                    tcp_offset_mm_rad=tcp_offset,
                )
                if args.result:
                    payload["session"].export_result(
                        payload["solution"],
                        args.result,
                        tcp_offset_override_mm_rad=tcp_offset,
                        cfg=payload.get("cfg"),
                    )
                print(_json({"mode": mode, "source": payload["directory"], "solution": payload["solution"]}))
                return 0
            if command == "usb-test":
                from camera_tool.services.usb import usb_test

                payload = usb_test(
                    context,
                    {
                        "backend": args.backend,
                        "serial": args.serial,
                        "width": args.width,
                        "height": args.height,
                        "fps": args.fps,
                    },
                    args.duration,
                    args.interval,
                    args.min_fps,
                    args.max_gap,
                    args.csv,
                )
                print(_json(payload["summary"]))
                return 130 if payload["summary"]["stopped"] else 0 if payload["summary"]["stable"] else 1
            if command == "tripod":
                from camera_tool.services.tripod import tripod

                payload = tripod(context, args.action, args.config, args.result, args.baseline, args.once, args.log)
                print(_json(payload))
                return 2 if payload.get("confirmed") == "FAIL" else 0
            if command == "wrist-intrinsics":
                from camera_tool.calibration.eye_to_hand.wrist_model import (
                    export_wrist_intrinsics,
                )

                dest = export_wrist_intrinsics(
                    config_file=args.config,
                    serial=args.serial,
                    from_run=args.from_run,
                    output=args.output,
                )
                print(_json({"output": str(dest)}))
                return 0
    except (Cancelled, KeyboardInterrupt):
        print("Task stopped; devices released", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
