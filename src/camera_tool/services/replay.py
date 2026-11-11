"""Detect session layouts and recompute offline without importing hardware SDKs."""
from __future__ import annotations

import json
from pathlib import Path

from camera_tool.paths import output_path


def detect_session(directory):
    path = Path(directory).expanduser().resolve()
    if path.is_file():
        path = path.parent
    if (path / "session_meta.json").is_file():
        return "eye-in-hand", path
    if (path / "session.json").is_file():
        raw = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if raw.get("schema") == 1 and "config" in raw:
            return "eye-to-hand", path
    raise ValueError("无法识别会话；需要 session_meta.json + samples.json 或 schema=1 的 session.json")


def load_session(
    context,
    directory,
    config=None,
    solve=True,
    strict=False,
    output="runs/eye_to_hand/replay",
    tcp_offset_mm_rad=None,
):
    mode, path = detect_session(directory)
    context.check()
    if mode == "eye-in-hand":
        from camera_tool.calibration.eye_in_hand.session import CalibrationSession
        from camera_tool.core import load_yaml
        from camera_tool.services.eye_in_hand import config_for_camera_metadata, solve_session

        session = CalibrationSession.load_replay(path)
        cfg = load_yaml(config_for_camera_metadata(session.camera_metadata, config, session.robot_metadata))
        payload = {
            "mode": mode,
            "session": session,
            "samples": session.samples,
            "solution": None,
            "cfg": cfg,
            "tcp_offset_mm_rad": tcp_offset_mm_rad,
            "metadata": session.camera_metadata,
            "directory": str(path),
            "replay": True,
        }
        if solve and sum(s.get("enabled", True) for s in session.samples) >= 5:
            try:
                payload["solution"] = solve_session(
                    session, cfg, tcp_offset_override_mm_rad=tcp_offset_mm_rad
                )
            except Exception as exc:
                if strict:
                    raise
                payload["error"] = str(exc)
        elif solve and strict:
            raise ValueError("回放有效样本不足 5 组")
    else:
        from camera_tool.calibration.eye_to_hand.workflow import replay
        from camera_tool.core import Session, write_json

        manifest = json.loads((path / "session.json").read_text(encoding="utf-8"))
        if output_path(output).is_relative_to(path):
            raise ValueError("回放源目录保持只读，请将复算记录写入其他目录")
        session = Session(output, "replay")
        payload = {
            "mode": mode,
            "session": session,
            "samples": manifest["samples"],
            "solution": None,
            "directory": str(path),
            "replay": True,
        }
        if solve:
            try:
                payload["solution"] = replay(path, session)
                context.check()
                write_json(session.path / "result.json", payload["solution"])
                session.finish("passed", result_file="result.json")
            except Exception as exc:
                session.finish("failed", failure=str(exc))
                if strict:
                    raise
                payload["error"] = str(exc)
    context.check()
    return payload
