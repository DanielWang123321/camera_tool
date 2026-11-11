"""SE(3) rigid transformations, Flange/TCP conversion, kinematics helpers, and tool safety."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from camera_tool.core.io import CalibrationError, RejectedSample


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | list[float] | None = None) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    if rotation is not None:
        mat[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if translation is not None:
        mat[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return mat


transform = make_transform


def valid_transform(matrix: Any, atol: float = 1e-3) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise CalibrationError("4x4 刚体变换矩阵包含无效数值或维度错误")
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=atol) or abs(np.linalg.det(r) - 1.0) > atol:
        raise CalibrationError("旋转矩阵不满足正交约束 SO(3)")
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise CalibrationError("齐次变换矩阵底行必须为 [0, 0, 0, 1]")
    return m.copy()


def inv_transform(matrix: np.ndarray) -> np.ndarray:
    m = valid_transform(matrix)
    r = m[:3, :3]
    t = m[:3, 3]
    return make_transform(r.T, -r.T @ t)


inv = inv_transform


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    m = valid_transform(matrix)
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return pts @ m[:3, :3].T + m[:3, 3]


def pose_to_vector6(matrix: np.ndarray) -> np.ndarray:
    """Convert 4x4 SE(3) matrix to 6D vector [rx, ry, rz, tx, ty, tz] (axis-angle + meters)."""
    m = valid_transform(matrix)
    rotvec = Rotation.from_matrix(m[:3, :3]).as_rotvec()
    return np.r_[rotvec, m[:3, 3]]


pose_vector = pose_to_vector6


def vector6_to_pose(vec: np.ndarray | list[float]) -> np.ndarray:
    """Convert 6D vector [rx, ry, rz, tx, ty, tz] to 4x4 SE(3) matrix."""
    v = np.asarray(vec, dtype=float).reshape(6)
    return make_transform(Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])


from_vector = vector6_to_pose


def pose_from_mm_deg(pose_mm_deg: list[float] | np.ndarray) -> np.ndarray:
    """Convert [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg] (extrinsic 'xyz') to 4x4 meters."""
    p = np.asarray(pose_mm_deg, dtype=float)
    if p.size != 6 or not np.isfinite(p).all():
        raise CalibrationError("位姿数组包含非有限值或维度无效")
    p = p.reshape(6)
    rot = Rotation.from_euler("xyz", p[3:], degrees=True).as_matrix()
    return make_transform(rot, p[:3] / 1000.0)


def pose_to_mm_deg(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 meters SE(3) matrix to [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]."""
    m = valid_transform(matrix)
    rpy_deg = Rotation.from_matrix(m[:3, :3]).as_euler("xyz", degrees=True)
    return np.r_[m[:3, 3] * 1000.0, rpy_deg].tolist()


def pose_from_mm_rad(pose_mm_rad: list[float] | np.ndarray) -> np.ndarray:
    """Convert [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad] to 4x4 meters."""
    p = np.asarray(pose_mm_rad, dtype=float)
    if p.size != 6 or not np.isfinite(p).all():
        raise CalibrationError("控制器位姿数据无效")
    p = p.reshape(6)
    rot = Rotation.from_euler("xyz", p[3:], degrees=False).as_matrix()
    return make_transform(rot, p[:3] / 1000.0)


def pose_to_mm_rad(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 meters SE(3) matrix to [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]."""
    m = valid_transform(matrix)
    rpy_rad = Rotation.from_matrix(m[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[m[:3, 3] * 1000.0, rpy_rad].tolist()


def pose_to_m_rad(matrix: np.ndarray) -> list[float]:
    """Convert 4x4 meters SE(3) matrix to [x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]."""
    m = valid_transform(matrix)
    rpy_rad = Rotation.from_matrix(m[:3, :3]).as_euler("xyz", degrees=False)
    return np.r_[m[:3, 3], rpy_rad].tolist()


def pose_error(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    """Compute (translation_error_mm, rotation_error_deg) between two 4x4 SE(3) transforms."""
    a, b = valid_transform(pose_a), valid_transform(pose_b)
    trans_mm = float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000.0)
    rel_rot = a[:3, :3].T @ b[:3, :3]
    rot_deg = float(np.rad2deg(Rotation.from_matrix(rel_rot).magnitude()))
    return trans_mm, rot_deg


def flange_from_tcp(x_base_tcp: np.ndarray, tcp_offset_mm_rad: list[float] | np.ndarray) -> np.ndarray:
    """Recover Flange pose T_base_flange from controller TCP pose and tcp_offset."""
    t_flange_tcp = pose_from_mm_rad(tcp_offset_mm_rad)
    return valid_transform(x_base_tcp) @ inv_transform(t_flange_tcp)


def tcp_from_flange(x_base_flange: np.ndarray, tcp_offset_mm_rad: list[float] | np.ndarray) -> np.ndarray:
    """Compute controller TCP pose T_base_tcp from Flange pose and controller tcp_offset."""
    t_flange_tcp = pose_from_mm_rad(tcp_offset_mm_rad)
    return valid_transform(x_base_flange) @ t_flange_tcp


def tool_lowest_z_m(
    x_base_flange: np.ndarray,
    length_m: float = 0.185,
    radius_m: float = 0.065,
) -> float:
    """Compute lowest Z height (meters, base frame) of a cylindrical end-effector envelope."""
    m = valid_transform(x_base_flange)
    z_flange = float(m[2, 3])
    z_axis_z = float(np.clip(m[2, 2], -1.0, 1.0))
    sin_tilt = float(np.sqrt(max(0.0, 1.0 - z_axis_z * z_axis_z)))
    z_disk_center = z_flange + float(length_m) * z_axis_z
    return float(min(z_flange, z_disk_center - float(radius_m) * sin_tilt))


def check_tool_safety(
    x_base_flange: np.ndarray,
    tool_cfg: dict[str, Any],
    workspace_min_m: np.ndarray | list[float] | None = None,
    workspace_max_m: np.ndarray | list[float] | None = None,
) -> dict[str, float]:
    """Verify Flange height, cylindrical tool clearance above table, and workspace bounds."""
    m = valid_transform(x_base_flange)
    pos = m[:3, 3]
    if workspace_min_m is not None and workspace_max_m is not None:
        low = np.asarray(workspace_min_m, dtype=float)
        high = np.asarray(workspace_max_m, dtype=float)
        if np.any(pos < low - 1e-6) or np.any(pos > high + 1e-6):
            raise RejectedSample(
                f"法兰位置超出安全工作区: {np.round(pos * 1000, 1).tolist()} mm "
                f"(允许范围 {np.round(low * 1000, 0).tolist()} ~ {np.round(high * 1000, 0).tolist()} mm)"
            )

    length_m = float(tool_cfg.get("length_m", 0.185))
    radius_m = float(tool_cfg.get("radius_m", 0.065))
    min_flange_z_m = float(tool_cfg.get("min_flange_z_m", 0.300))
    min_tip_z_m = float(tool_cfg.get("min_tip_z_m", 0.080))
    table_z_m = float(tool_cfg.get("table_z_m", 0.000))

    flange_z = float(pos[2])
    tip_z = tool_lowest_z_m(m, length_m, radius_m)
    clearance_m = tip_z - table_z_m

    tool_name = str(tool_cfg.get("name", "末端工具"))
    if flange_z < min_flange_z_m - 1e-6:
        raise RejectedSample(
            f"法兰高度过低 ({flange_z * 1000:.1f} mm < {min_flange_z_m * 1000:.1f} mm)，触发防撞保护"
        )
    if clearance_m < min_tip_z_m - 1e-6:
        raise RejectedSample(
            f"{tool_name}最低点距桌面过近 ({clearance_m * 1000:.1f} mm < {min_tip_z_m * 1000:.1f} mm)，"
            "已拦截以防碰撞桌面"
        )
    return {
        "flange_z_mm": flange_z * 1000.0,
        "tool_tip_z_mm": tip_z * 1000.0,
        "table_clearance_mm": clearance_m * 1000.0,
    }


def interpolate_poses(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    translation_step_m: float = 0.015,
    rotation_step_deg: float = 5.0,
) -> list[np.ndarray]:
    """Interpolate SE(3) poses along straight line + Slerp with bounded step sizes."""
    a, b = valid_transform(pose_start), valid_transform(pose_end)
    dist_mm, angle_deg = pose_error(a, b)
    steps = max(
        1,
        int(np.ceil((dist_mm / 1000.0) / max(float(translation_step_m), 1e-4))),
        int(np.ceil(angle_deg / max(float(rotation_step_deg), 1e-2))),
    )
    times = np.linspace(0.0, 1.0, steps + 1)[1:]
    rots = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(times).as_matrix()
    return [make_transform(r, a[:3, 3] * (1.0 - t) + b[:3, 3] * t) for r, t in zip(rots, times)]


def average_se3(transforms: list[np.ndarray]) -> np.ndarray:
    """Compute mean SE(3) transform from a list of 4x4 matrices."""
    mats = [valid_transform(t) for t in transforms]
    mean_trans = np.mean([m[:3, 3] for m in mats], axis=0)
    mean_rot = Rotation.from_matrix([m[:3, :3] for m in mats]).mean().as_matrix()
    return make_transform(mean_rot, mean_trans)


def rotation_coverage(
    poses_or_samples: list[Any],
    minimum_deg: float = 6.0,
    pose_key: str | None = None,
) -> list[float]:
    """Compute 3-axis rotational excitation RMS in degrees via SVD of relative rotation vectors."""
    if len(poses_or_samples) < 4:
        raise CalibrationError(f"有效样本数量不足 ({len(poses_or_samples)} < 4)")
    mats: list[np.ndarray] = []
    for item in poses_or_samples:
        if isinstance(item, dict):
            raw = item[pose_key] if pose_key else item.get("X_base_flange", item.get("X_base_tcp"))
            mats.append(valid_transform(raw))
        else:
            mats.append(valid_transform(item))
    rotations = np.stack([m[:3, :3] for m in mats])
    rel_rotvecs = Rotation.from_matrix(rotations[0].T @ rotations).as_rotvec()
    singular = np.linalg.svd(rel_rotvecs - rel_rotvecs.mean(axis=0), compute_uv=False)
    rms_deg = np.rad2deg(singular / np.sqrt(len(mats)))
    if len(rms_deg) < 2 or rms_deg[1] < minimum_deg:
        raise CalibrationError(
            f"机械臂多轴转动不足或只有一个转轴（第二方向幅度 {rms_deg[1]:.1f}° < 阈值 {minimum_deg:g}°）"
        )
    return [float(x) for x in rms_deg]


def nearest_continuous_joints(
    joints: np.ndarray | list[float],
    reference: np.ndarray | list[float],
    limits: np.ndarray | list[list[float]],
    margin_rad: float,
) -> np.ndarray:
    """Select continuous IK joint solution within joint limits and margin."""
    ref = np.asarray(reference, dtype=float).ravel()
    dof = len(ref)
    q_raw = np.asarray(joints, dtype=float).ravel()[:dof]
    lim = np.asarray(limits, dtype=float)[:dof]
    if len(q_raw) != dof or not np.isfinite(q_raw).all():
        raise RejectedSample("逆运动学返回关节角包含非有限值或维度不匹配")
    out: list[float] = []
    for q, r, (low, high) in zip(q_raw, ref, lim):
        candidates = q + np.arange(-2, 3) * 2.0 * np.pi
        valid = candidates[(candidates >= low + margin_rad) & (candidates <= high - margin_rad)]
        if len(valid) == 0:
            raise RejectedSample("逆解关节角过于接近软限位")
        out.append(float(valid[np.argmin(np.abs(valid - r))]))
    return np.asarray(out, dtype=float)
