"""Unified CLI and GUI entry point; headless commands never import or initialize Qt."""
from __future__ import annotations

import argparse
import json
import signal
import sys


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="camera-tool",
        description="相机预览、手眼标定、离线回放与设备诊断工具",
    )
    root.add_argument("--sim", action="store_true", help="离线仿真模式")
    sub = root.add_subparsers(dest="command")

    preview = sub.add_parser("preview", help="打开相机实时预览或枚举设备")
    preview.add_argument(
        "--backend",
        choices=("realsense", "orbbec", "uvc", "sim"),
        default="realsense",
        help="相机后端（默认 realsense）",
    )
    preview.add_argument("--serial", default="", help="设备序列号（留空自动选择单台设备）")
    preview.add_argument("--width", type=int)
    preview.add_argument("--height", type=int)
    preview.add_argument("--fps", type=int)
    preview.add_argument("--no-depth", action="store_true", help="仅开启彩色流")
    preview.add_argument("--max-depth", type=float, help="深度伪彩色显示上限（米）")
    preview.add_argument("--list", action="store_true", help="仅枚举可用后端与在线设备")
    preview.add_argument("--headless", action="store_true", help="无界面抓取单帧校验流状态")

    cal = sub.add_parser("calibrate", help="执行手眼标定；正式模式会驱动机械臂运动")
    cal.add_argument("--mode", choices=("eye-in-hand", "eye-to-hand"), required=True)
    cal.add_argument(
        "--backend",
        choices=("auto", "realsense", "orbbec", "sim"),
        default="auto",
        help="相机后端（默认 auto 自动检测）",
    )
    cal.add_argument(
        "--preset",
        choices=("auto", "default", "medium_range", "orbbec"),
        default="auto",
        help="眼在手上配置预设模板（默认 auto）",
    )
    cal.add_argument("--profile", help="机械臂构型配置（如 arm_6dof, arm_7dof, arm_compact_6dof）")
    cal.add_argument("--serial", default="", help="指定相机序列号（留空则单设备自动选择）")
    cal.add_argument("--robot-ip", help="覆盖机械臂 IP（未指定时依次读取 XARM_IP 环境变量与 YAML 配置）")
    cal.add_argument("--config", help="自定义 YAML 配置文件路径")
    cal.add_argument("--check", action="store_true", help="只读检查，不使能、移动机械臂或发布外参")
    cal.add_argument("--auto", action="store_true", help="无界面自动采集全部点位并求解")
    cal.add_argument("--sim", action="store_true", help="离线仿真模式（仅支持眼在手上）")
    cal.add_argument("--output", help="标定会话保存目录")
    cal.add_argument("--result", help="导出结果 YAML 路径")
    cal.add_argument("--gui", action="store_true", help="打开桌面端标定页面")

    replay = sub.add_parser("replay", help="自动识别历史会话并离线复算（不导入硬件 SDK）")
    replay.add_argument("directory", help="历史会话目录路径")
    replay.add_argument("--config", help="眼在手上回放检查配置 YAML")
    replay.add_argument("--result", help="眼在手上导出 YAML 路径；眼在手外回放不允许发布外参")
    replay.add_argument(
        "--tcp-offset",
        help="离线覆盖末端 TCP 偏移 x,y,z,r,p,y（单位 mm,rad，如 0,0,172,0,0,0）",
    )
    replay.add_argument("--output", default="runs/eye_to_hand/replay", help="眼在手外离线复算输出目录")
    replay.add_argument("--gui", action="store_true", help="在桌面端打开回放结果")

    tripod = sub.add_parser("tripod", help="固定相机基线锁定与三脚架位移监测")
    tripod.add_argument("action", choices=("lock", "watch"))
    tripod.add_argument("--config", help="配置文件路径")
    tripod.add_argument("--result", default="result/eye_to_hand.yaml", help="眼在手外标定结果 YAML")
    tripod.add_argument("--baseline", help="基线保存/读取目录")
    tripod.add_argument("--once", action="store_true", help="无界面检查单帧；FAIL 返回退出码 2")
    tripod.add_argument("--headless", action="store_true", help="无界面持续打印监测数据")
    tripod.add_argument("--log", help="监测日志文件路径")

    usb = sub.add_parser("usb-test", help="无界面 USB 链路与帧率稳定性测试")
    usb.add_argument("--backend", choices=("realsense", "orbbec"), default="realsense", help="RGB-D 相机后端")
    usb.add_argument("--serial", default="", help="设备序列号（留空自动选择）")
    usb.add_argument("--width", type=int, default=1280)
    usb.add_argument("--height", type=int, default=720)
    usb.add_argument("--fps", type=int, default=30)
    usb.add_argument("--duration", type=float, default=3600)
    usb.add_argument("--interval", type=float, default=5)
    usb.add_argument("--min-fps", type=float, default=29.0)
    usb.add_argument("--max-gap", type=float, default=0.1)
    usb.add_argument("--csv", help="导出窗口统计 CSV 路径")
    usb.add_argument("--quiet", action="store_true", help="仅在结束时输出汇总 JSON")

    maintenance = sub.add_parser("maintenance", help="腕部内参转换、角点标定与取流核对")
    maintenance.add_argument("action", choices=("wrist-intrinsics", "corners", "verify"))
    maintenance.add_argument("arguments", nargs=argparse.REMAINDER)
    return root


def _json(value) -> str:
    from camera_tool.core import json_default

    return json.dumps(value, ensure_ascii=False, default=json_default)


def _parse_tcp_offset(raw: str | None) -> list[float] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip() != ""]
    if len(parts) != 6:
        raise ValueError("--tcp-offset 须为逗号分隔的 6 个数值 x,y,z,r,p,y（单位 mm,rad），例如 0,0,172,0,0,0")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--tcp-offset 包含无法解析的数值") from exc


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
                from camera_tool.devices.registry import PROFILES, create_camera, device_resource
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
                with DEVICES.lease([device_resource(args.backend, args.serial)], "无界面预览检查"):
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
                    raise ValueError("离线仿真仅支持眼在手上 (eye-in-hand)")
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
                    raise ValueError("--tcp-offset 仅适用于眼在手上回放复算")
                if args.result and mode == "eye-to-hand":
                    raise ValueError("眼在手外回放不允许发布外参；可查看输出目录的 result.json")
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
            if command == "maintenance":
                if args.action == "verify":
                    from camera_tool.maintenance.verify_preview import main as maintenance
                elif args.action == "corners":
                    from camera_tool.maintenance.calibrate_corners import main as maintenance
                else:
                    from camera_tool.maintenance.dump_wrist_model import main as maintenance
                return maintenance(args.arguments)
    except (Cancelled, KeyboardInterrupt):
        print("任务已停止，设备已释放", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
