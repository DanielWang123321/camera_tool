"""Wrist camera undistortion loading and intrinsics export helpers."""
from __future__ import annotations

import ctypes as ct
import json
import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from camera_tool.core import CalibrationError, load_yaml, write_json
from camera_tool.paths import (
    DATA_ROOT,
    RESOURCE_ROOT,
    config_path,
    output_path,
    resource_path,
)


def load_wrist_intrinsics(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    resolved = resource_path(path)
    if not resolved.is_file():
        raise CalibrationError(
            f"Wrist intrinsics file not found: {resolved} (run 'camera-tool wrist-intrinsics' first)"
        )
    metadata = json.loads(resolved.read_text(encoding="utf-8"))
    mapping_name = metadata.get("mapping_file")
    if not mapping_name:
        raise CalibrationError(f"{resolved} is missing mapping_file")
    mapping_path = (resolved.parent / mapping_name).resolve()
    if not mapping_path.is_relative_to(resolved.parent.resolve()):
        raise CalibrationError("Undistortion mapping path escapes intrinsics directory")
    if not mapping_path.is_file():
        raise CalibrationError(f"Wrist undistortion mapping file not found: {mapping_path}")
    with np.load(mapping_path, allow_pickle=False) as data:
        if "mapping" not in data:
            raise CalibrationError("Undistortion NPZ file is missing 'mapping' array")
        mapping = np.asarray(data["mapping"], dtype=np.float32)
    width, height = int(metadata["width"]), int(metadata["height"])
    if mapping.shape != (height, width, 2) or not np.isfinite(mapping).all():
        raise CalibrationError("Invalid wrist undistortion mapping shape or values")
    if metadata.get("model") not in ("pinhole", "brown_conrady"):
        raise CalibrationError("Wrist intrinsics must use a rectified pinhole or brown_conrady model")
    metadata = dict(metadata)
    metadata["intrinsics_json"] = str(resolved)
    metadata["mapping_path"] = str(mapping_path)
    return metadata, mapping


def _worker(connection: Any, library: str, serial: str, camera_index: int, workdir: str) -> None:
    os.chdir(workdir)
    lib = None
    try:
        with open("wrist_sdk.log", "ab", buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
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
                    raise RuntimeError("Device camera model projection/raytrace check failed or valid FOV < 70%")
                metadata["valid_rectification_ratio"] = valid / (width * height)
                connection.send((True, (metadata, mapping)))
            else:
                raise RuntimeError("Unknown worker command")
    except BaseException as exc:
        try:
            connection.send((False, str(exc)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if lib is not None:
            lib.calib_close()
        connection.close()


def dump_from_device(
    config: dict[str, Any],
    dest_json: Path | None,
    serial_override: str | None = None,
) -> Path:
    wrist_cfg = config.get("wrist_camera") or config["wrist"]
    serial = str(serial_override or wrist_cfg.get("serial") or "").strip()
    camera_index = int(wrist_cfg["camera_index"])
    raw_w, raw_h = wrist_cfg["capture_size"]
    rect_w, rect_h = wrist_cfg["rectified_size"]
    fov = float(wrist_cfg["rectified_fov_deg"])
    workdir = output_path(DATA_ROOT / "intrinsics" / "dump_work")
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
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=build,
            env={**os.environ, "TMPDIR": str(build)},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CalibrationError(f"Failed to compile wrist camera bridge (requires g++ and xvsdk): {exc}") from exc
    (build / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise CalibrationError(f"Failed to compile wrist camera bridge; see {build / 'compile.log'}")

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
            raise CalibrationError("Wrist camera SDK timed out; check dump_work/wrist_sdk.log")
        ok, value = parent.recv()
        if not ok:
            raise CalibrationError(f"Wrist camera SDK error: {value}")
        focal = rect_w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        k = [focal, focal, (rect_w - 1) / 2.0, (rect_h - 1) / 2.0]
        parent.send(("model", (raw_w, raw_h, rect_w, rect_h, k)))
        if not parent.poll(15):
            raise CalibrationError("Timed out reading wrist camera model")
        ok, value = parent.recv()
        if not ok:
            raise CalibrationError(f"Wrist camera SDK error: {value}")
        metadata, mapping = value
        actual_serial = str(metadata["serial"])
        if serial and actual_serial != serial:
            raise CalibrationError("Wrist camera intrinsics serial does not match live device serial")
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
        return target_json
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


def copy_from_run(run_dir: Path, dest_json: Path | None, serial: str) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    src_json = run_dir / "wrist_camera.json"
    src_npz = run_dir / "wrist_rectification.npz"
    if not src_json.is_file() or not src_npz.is_file():
        raise CalibrationError(f"{run_dir} is missing wrist_camera.json or wrist_rectification.npz")
    metadata = json.loads(src_json.read_text(encoding="utf-8"))
    actual_serial = str(metadata.get("serial") or "").strip()
    if serial and actual_serial != serial:
        raise CalibrationError(f"Run serial {actual_serial} does not match requested serial {serial}")
    if not actual_serial:
        raise CalibrationError("Run metadata is missing wrist camera serial")
    target_json = dest_json or (DATA_ROOT / "intrinsics" / f"wrist_{actual_serial}.json")
    target_json.parent.mkdir(parents=True, exist_ok=True)
    mapping_name = f"wrist_{actual_serial}.npz"
    dest_npz = target_json.parent / mapping_name
    shutil.copy2(src_npz, dest_npz)
    keep = {k: v for k, v in metadata.items() if k not in ("video_device", "backend", "frame_clock")}
    keep["mapping_file"] = mapping_name
    write_json(target_json, keep)
    return target_json


def export_wrist_intrinsics(
    config_file: str | Path | None = None,
    serial: str | None = None,
    from_run: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    cfg = load_yaml(config_path(config_file or "eye_to_hand.yaml"))
    wrist_cfg = cfg.get("wrist_camera") or cfg["wrist"]
    resolved_serial = str(serial or wrist_cfg.get("serial") or "").strip()
    dest = (
        output_path(output)
        if output
        else (DATA_ROOT / "intrinsics" / f"wrist_{resolved_serial}.json" if resolved_serial else None)
    )
    if from_run:
        return copy_from_run(Path(from_run), dest, resolved_serial)
    from camera_tool.services.tasks import DEVICES, resource

    with DEVICES.lease([resource("uvc")], "Wrist intrinsics export"):
        return dump_from_device(cfg, dest, serial_override=resolved_serial)
