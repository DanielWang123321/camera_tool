"""Eye-in-Hand calibration session recording, replay loader, and YAML/JSON result exporter."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from camera_tool.core import (
    CalibrationError,
    inv_transform,
    json_default,
    pose_from_mm_rad,
    pose_to_m_rad,
    pose_to_mm_deg,
    valid_transform,
)
from camera_tool.paths import project_path


class CalibrationSession:
    """Manages captured calibration samples, disk persistence, and result export."""

    def __init__(self, base_dir: str | Path = "runs/eye_in_hand") -> None:
        self.base_dir = project_path(base_dir)
        self.session_dir: Path | None = None
        self.samples: list[dict[str, Any]] = []
        self.camera_metadata: dict[str, Any] | None = None
        self.robot_metadata: dict[str, Any] | None = None
        self.last_solution: dict[str, Any] | None = None
        self.read_only = False

    def start_new_session(
        self,
        camera_metadata: dict[str, Any],
        robot_metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> Path | None:
        self.samples.clear()
        self.last_solution = None
        self.camera_metadata = camera_metadata
        self.robot_metadata = robot_metadata or {}
        if persist:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.session_dir = self.base_dir / stamp
            (self.session_dir / "images").mkdir(parents=True, exist_ok=True)
            self._write_metadata()
            self._save_samples_index()
        else:
            self.session_dir = None
        return self.session_dir

    def _write_metadata(self) -> None:
        if self.session_dir is None or self.read_only:
            return
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "camera": self.camera_metadata,
            "robot": self.robot_metadata,
        }
        (self.session_dir / "session_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=json_default),
            encoding="utf-8",
        )

    def add_sample(
        self,
        color_bgr: np.ndarray,
        eval_result: dict[str, Any],
        robot_state: dict[str, Any],
        depth_mm: np.ndarray | None = None,
        points_cam: np.ndarray | None = None,
        waypoint_name: str = "manual",
    ) -> dict[str, Any]:
        sample_id = len(self.samples) + 1
        x_base_flange = valid_transform(robot_state["X_base_flange"])
        x_base_tcp = valid_transform(robot_state.get("X_base_tcp", x_base_flange))

        img_rel = f"images/sample_{sample_id:03d}.png"
        if self.session_dir is not None and color_bgr is not None:
            cv2.imwrite(str(self.session_dir / img_rel), color_bgr)
            if depth_mm is not None:
                np.save(str(self.session_dir / f"images/depth_{sample_id:03d}.npy"), depth_mm)

        sample: dict[str, Any] = {
            "id": sample_id,
            "name": waypoint_name,
            "timestamp": time.time(),
            "enabled": True,
            "image_path": img_rel,
            "color_bgr": color_bgr.copy() if color_bgr is not None else None,
            "points_cam": points_cam.copy() if points_cam is not None else None,
            "corners": eval_result["corners"],
            "area_px": float(eval_result["area_px"]),
            "sharpness": float(eval_result["sharpness"]),
            "pnp_rms_px": float(eval_result["pnp_rms_px"]),
            "pose_cam_board": valid_transform(eval_result["pose_cam_board"]),
            "X_base_flange": x_base_flange,
            "X_base_tcp": x_base_tcp,
            "flange_mm_deg": pose_to_mm_deg(x_base_flange),
            "tcp_mm_deg": pose_to_mm_deg(x_base_tcp),
            "joints_rad": robot_state.get("joints_rad", [0.0] * 6),
            "safety": robot_state.get("safety", {}),
        }
        self.samples.append(sample)
        self._save_samples_index()
        return sample

    def _save_samples_index(self) -> None:
        if self.session_dir is None or self.read_only:
            return
        serializable = [
            {k: v for k, v in s.items() if k not in ("color_bgr", "points_cam")}
            for s in self.samples
        ]
        (self.session_dir / "samples.json").write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=json_default),
            encoding="utf-8",
        )

    @classmethod
    def load_replay(cls, run_dir: str | Path) -> CalibrationSession:
        folder = project_path(run_dir)
        meta_path = folder / "session_meta.json"
        samples_path = folder / "samples.json"
        if not meta_path.exists():
            raise CalibrationError(f"回放目录缺少 session_meta.json: {folder}")

        session = cls(base_dir=folder.parent)
        session.session_dir = folder
        session.read_only = True
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        session.camera_metadata = meta["camera"]
        session.robot_metadata = meta.get("robot", {})

        raw_samples = json.loads(samples_path.read_text(encoding="utf-8")) if samples_path.is_file() else []
        for item in raw_samples:
            img_file = (folder / item["image_path"]).resolve()
            if not img_file.is_relative_to(folder.resolve()):
                raise CalibrationError("回放图像路径超出会话目录")
            item["color_bgr"] = cv2.imread(str(img_file), cv2.IMREAD_COLOR) if img_file.exists() else None
            item["X_base_flange"] = valid_transform(np.asarray(item["X_base_flange"], dtype=float))
            item["X_base_tcp"] = valid_transform(
                np.asarray(item.get("X_base_tcp", item["X_base_flange"]), dtype=float)
            )
            item["pose_cam_board"] = valid_transform(np.asarray(item["pose_cam_board"], dtype=float))
            item["points_cam"] = None
            session.samples.append(item)
        return session

    def export_result(
        self,
        solution: dict[str, Any],
        output_path: str | Path = "result/eye_in_hand.yaml",
        tcp_offset_override_mm_rad: list[float] | np.ndarray | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> Path:
        out = project_path(output_path)
        if self.read_only and self.session_dir is not None and out.resolve().is_relative_to(self.session_dir.resolve()):
            raise CalibrationError("回放源目录保持只读，请将结果导出到其他目录")
        out.parent.mkdir(parents=True, exist_ok=True)
        self.last_solution = solution

        x_flange_cam = valid_transform(np.asarray(solution["X_flange_cam"], dtype=float))
        if tcp_offset_override_mm_rad is not None:
            tcp_offset = [float(x) for x in np.asarray(tcp_offset_override_mm_rad, dtype=float).reshape(6)]
            x_tcp_cam = inv_transform(pose_from_mm_rad(tcp_offset)) @ x_flange_cam
        else:
            tcp_offset = [float(x) for x in (self.robot_metadata or {}).get("tcp_offset_mm_rad", [0.0] * 6)]
            x_tcp_cam = valid_transform(np.asarray(solution["X_tcp_cam"], dtype=float))

        flange_cam_mm_deg = pose_to_mm_deg(x_flange_cam)
        tcp_cam_mm_deg = pose_to_mm_deg(x_tcp_cam)
        flange_cam_m_rad = [round(float(x), 10) for x in pose_to_m_rad(x_flange_cam)]
        tcp_cam_m_rad = [round(float(x), 10) for x in pose_to_m_rad(x_tcp_cam)]

        checks_cfg = (cfg or {}).get("checks", {})
        marker_cfg = (cfg or {}).get("marker", {})
        robot_section = (cfg or {}).get("robot", {})
        tool_cfg = robot_section.get("tool") or robot_section.get("gripper") or {}
        table_z_m = checks_cfg.get("table_z_m", tool_cfg.get("table_z_m"))
        board_thickness_m = checks_cfg.get("board_thickness_m", marker_cfg.get("thickness_m"))
        if table_z_m is None and "board_check" in solution:
            table_z_m = float(solution["board_check"].get("configured_table_z_mm", 0.0)) / 1000.0
        if board_thickness_m is None and "board_check" in solution:
            board_thickness_m = float(solution["board_check"].get("board_thickness_mm", 0.0)) / 1000.0

        uf_vision_block: dict[str, Any] = {
            "EULER_EEF_TO_COLOR_OPT": tcp_cam_m_rad,
            "EULER_FLANGE_TO_COLOR_OPT": flange_cam_m_rad,
            "EULER_COLOR_TO_DEPTH_OPT": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "camera_frame": "color_optical_frame",
            "depth_aligned_to_color": True,
        }
        if table_z_m is not None:
            uf_vision_block["table_z_base_m"] = round(float(table_z_m), 6)
        if board_thickness_m is not None:
            uf_vision_block["board_thickness_m"] = round(float(board_thickness_m), 6)

        robot_profile = (self.robot_metadata or {}).get("profile") or (self.robot_metadata or {}).get("model", "arm_6dof")
        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "calibration_mode": "eye_in_hand",
            "robot_profile": robot_profile,
            "camera": {
                "name": (self.camera_metadata or {}).get("name", ""),
                "model": (self.camera_metadata or {}).get("model", "brown_conrady"),
                "serial": (self.camera_metadata or {}).get("serial", ""),
                "width": (self.camera_metadata or {}).get("width", 1280),
                "height": (self.camera_metadata or {}).get("height", 720),
                "fps": (self.camera_metadata or {}).get("fps"),
                "camera_frame": (self.camera_metadata or {}).get("camera_frame", "color_optical_frame"),
                "depth_aligned_to_color": bool((self.camera_metadata or {}).get("depth_aligned_to_color", True)),
                "sdk_depth_unit_m": (self.camera_metadata or {}).get(
                    "sdk_depth_unit_m", (self.camera_metadata or {}).get("depth_scale")
                ),
                "depth_scale_correction": 1.0,
                "K": (self.camera_metadata or {}).get("K", []),
                "distortion": (self.camera_metadata or {}).get("distortion", []),
            },
            "extrinsics": {
                "recommended_method": solution["recommended_method"],
                "X_flange_cam": x_flange_cam.tolist(),
                "flange_cam_mm_deg": [round(float(x), 4) for x in flange_cam_mm_deg],
                "flange_cam_m_rad": flange_cam_m_rad,
                "X_tcp_cam": x_tcp_cam.tolist(),
                "tcp_cam_mm_deg": [round(float(x), 4) for x in tcp_cam_mm_deg],
                "tcp_cam_m_rad": tcp_cam_m_rad,
                "X_base_board": np.asarray(solution["X_base_board"], dtype=float).tolist(),
                "base_board_mm_deg": [round(float(x), 4) for x in solution["base_board_mm_deg"]],
                "tcp_offset_mm_rad": tcp_offset,
                "ufactory_vision": uf_vision_block,
            },
            "metrics": {
                "active_samples": solution["metrics"]["count"],
                "mean_mm": round(float(solution["metrics"]["mean_mm"]), 3),
                "max_mm": round(float(solution["metrics"]["max_mm"]), 3),
                "mean_deg": round(float(solution["metrics"]["mean_deg"]), 4),
                "max_deg": round(float(solution["metrics"]["max_deg"]), 4),
                "reprojection_rms_px": round(float(solution["metrics"]["rms_px"]), 3),
                "rotation_coverage_deg": [round(float(x), 2) for x in solution["rotation_coverage_deg"]],
                **({"board_check": solution["board_check"]} if "board_check" in solution else {}),
                **({"depth_check": solution["depth_check"]} if "depth_check" in solution else {}),
            },
            "methods_comparison": {
                name: {
                    "flange_cam_mm_deg": [round(float(x), 3) for x in m["flange_cam_mm_deg"]],
                    "mean_mm": round(float(m["metrics"]["mean_mm"]), 3),
                    "max_mm": round(float(m["metrics"]["max_mm"]), 3),
                    "mean_deg": round(float(m["metrics"]["mean_deg"]), 4),
                    "rms_px": round(float(m["metrics"]["rms_px"]), 3),
                }
                for name, m in solution["methods"].items()
            },
        }
        out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if self.session_dir is not None and not self.read_only:
            (self.session_dir / "result.yaml").write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        return out
