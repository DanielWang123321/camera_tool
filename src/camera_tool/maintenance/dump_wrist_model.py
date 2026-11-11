"""Export wrist camera undistortion maps from native SDK or an existing calibration run."""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from camera_tool.core import CalibrationError, write_json
from camera_tool.paths import DATA_ROOT, RESOURCE_ROOT, config_path, project_path


def _worker(connection: Any, library: str, serial: str, camera_index: int, workdir: str) -> None:
    os.chdir(workdir)
    log = open("wrist_sdk.log", "ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    lib = None
    try:
        lib = ct.CDLL(library)
        lib.calib_error.restype = ct.c_char_p
        lib.calib_info.restype = ct.c_char_p
        lib.calib_open.argtypes = [ct.c_char_p, ct.c_int]
        lib.calib_model.argtypes = [ct.c_int, ct.c_int]
        lib.calib_map.argtypes = [ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p]
        code = lib.calib_open(serial.encode(), camera_index)
        overview = lib.calib_info()
        if overview:
            write_json(Path(workdir) / "wrist_device.json", json.loads(overview.decode()))
        if code != 0:
            raise RuntimeError(lib.calib_error().decode())
        connection.send((True, "opened"))
        while True:
            command, args = connection.recv()
            if command == "close":
                break
            if command == "model":
                raw_w, raw_h, width, height, k = args
                if lib.calib_model(raw_w, raw_h) != 0:
                    raise RuntimeError(lib.calib_error().decode())
                metadata = json.loads(lib.calib_info().decode())
                k_arr = np.asarray(k, dtype=np.float64)
                mapping = np.empty((height, width, 2), dtype=np.float32)
                valid = lib.calib_map(width, height, k_arr.ctypes.data, mapping.ctypes.data)
                if valid < width * height * 0.7:
                    raise RuntimeError("设备相机模型的投影/反投影检查失败，或可用视野不足 70%")
                metadata["valid_rectification_ratio"] = valid / (width * height)
                connection.send((True, (metadata, mapping)))
            else:
                raise RuntimeError("未知请求")
    except BaseException as exc:
        try:
            connection.send((False, str(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if lib is not None:
            lib.calib_close()
        connection.close()


def dump_from_device(config: dict[str, Any], dest_json: Path | None, serial_override: str | None = None) -> None:
    wrist_cfg = config.get("wrist_camera") or config["wrist"]
    serial = str(serial_override or wrist_cfg.get("serial") or "").strip()
    camera_index = int(wrist_cfg["camera_index"])
    raw_w, raw_h = wrist_cfg["capture_size"]
    rect_w, rect_h = wrist_cfg["rectified_size"]
    fov = float(wrist_cfg["rectified_fov_deg"])
    workdir = project_path(DATA_ROOT / "intrinsics" / "dump_work")
    workdir.mkdir(parents=True, exist_ok=True)
    build = workdir / "build"
    build.mkdir(exist_ok=True)
    source = RESOURCE_ROOT / "native" / "fastumi_bridge.cpp"
    library = build / "fastumi_bridge.so"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        "-I/usr/include/xvsdk",
        str(source),
        "-o",
        str(library),
        "-lxvsdk",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=build,
            env={**os.environ, "TMPDIR": str(build)},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CalibrationError(f"腕部相机参数接口编译失败，需要本机 g++ 和 xvsdk：{exc}") from exc
    (build / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise CalibrationError(f"腕部相机参数接口编译失败，见 {build / 'compile.log'}")

    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_worker,
        args=(child, str(library), serial, camera_index, str(workdir)),
        daemon=True,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(15):
            raise CalibrationError("腕部相机 SDK 超时，详见 dump_work/wrist_sdk.log")
        ok, value = parent.recv()
        if not ok:
            raise CalibrationError(f"腕部相机 SDK：{value}")
        focal = rect_w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        k = [focal, focal, (rect_w - 1) / 2.0, (rect_h - 1) / 2.0]
        parent.send(("model", (raw_w, raw_h, rect_w, rect_h, k)))
        if not parent.poll(15):
            raise CalibrationError("读取腕部相机模型超时")
        ok, value = parent.recv()
        if not ok:
            raise CalibrationError(f"腕部相机 SDK：{value}")
        metadata, mapping = value
        actual_serial = str(metadata["serial"])
        if serial and actual_serial != serial:
            raise CalibrationError("腕部相机内参与取流设备的序列号不一致")
        target_json = dest_json or (DATA_ROOT / "intrinsics" / f"wrist_{actual_serial}.json")
        target_json.parent.mkdir(parents=True, exist_ok=True)
        mapping_name = f"wrist_{actual_serial}.npz"
        dest_npz = target_json.parent / mapping_name
        metadata.update(
            width=rect_w,
            height=rect_h,
            model="pinhole",
            distortion=[0.0] * 5,
            K=[[focal, 0.0, k[2]], [0.0, focal, k[3]], [0.0, 0.0, 1.0]],
            rectification="SDK CameraModel.project; roundtrip < 0.05px",
            mapping_file=mapping_name,
            image_size_verified=False,
        )
        np.savez_compressed(dest_npz, mapping=mapping)
        write_json(target_json, metadata)
        print(f"已写入 {target_json} 和 {dest_npz}")
    finally:
        try:
            parent.send(("close", None))
        except (BrokenPipeError, OSError):
            pass
        process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(2)
        parent.close()


def copy_from_run(run_dir: Path, dest_json: Path | None, serial: str) -> None:
    run_dir = Path(run_dir).expanduser().resolve()
    src_json = run_dir / "wrist_camera.json"
    src_npz = run_dir / "wrist_rectification.npz"
    if not src_json.is_file() or not src_npz.is_file():
        raise CalibrationError(f"{run_dir} 缺少 wrist_camera.json 或 wrist_rectification.npz")
    metadata = json.loads(src_json.read_text(encoding="utf-8"))
    actual_serial = str(metadata.get("serial") or "").strip()
    if serial and actual_serial != serial:
        raise CalibrationError(f"运行记录序列号 {actual_serial} 与指定序列号 {serial} 不一致")
    if not actual_serial:
        raise CalibrationError("运行记录缺少腕部相机序列号")
    target_json = dest_json or (DATA_ROOT / "intrinsics" / f"wrist_{actual_serial}.json")
    target_json.parent.mkdir(parents=True, exist_ok=True)
    mapping_name = f"wrist_{actual_serial}.npz"
    dest_npz = target_json.parent / mapping_name
    shutil.copy2(src_npz, dest_npz)
    keep = {k: v for k, v in metadata.items() if k not in ("video_device", "backend", "frame_clock")}
    keep["mapping_file"] = mapping_name
    write_json(target_json, keep)
    print(f"已从 {run_dir} 拷贝到 {target_json} 和 {dest_npz}")


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(config_path("eye_to_hand.yaml")))
    parser.add_argument("--serial", default=None, help="腕部相机序列号（单设备时可省略自动发现）")
    parser.add_argument("--from-run", default=None, help="从已有标定运行目录拷贝映射")
    parser.add_argument("--output", default=None, help="输出 json 路径，默认 intrinsics/wrist_<serial>.json")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    wrist_cfg = config.get("wrist_camera") or config["wrist"]
    serial = str(args.serial or wrist_cfg.get("serial") or "").strip()
    dest = Path(args.output) if args.output else (DATA_ROOT / "intrinsics" / f"wrist_{serial}.json" if serial else None)
    try:
        if args.from_run:
            copy_from_run(Path(args.from_run), dest, serial)
        else:
            dump_from_device(config, dest, serial_override=serial)
    except CalibrationError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    from camera_tool.services.tasks import DEVICES, resource

    with DEVICES.lease([resource("uvc")], "维护工具"):
        return _run(argv)
