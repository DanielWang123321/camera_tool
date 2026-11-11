"""3D Coordinate Frame, Tool Envelope, Waypoints, and Point Cloud Visualization Panel."""
from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QVector3D
from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from camera_tool.core import marker_object_points, transform_points, valid_transform

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    gl = None
    HAS_OPENGL = False


class Scene3DPanel(QWidget):
    """Switchable 3D view of coordinate frames, waypoints, tool envelope, and point cloud."""

    def __init__(
        self,
        marker_side_m: float = 0.15,
        tool_length_m: float = 0.185,
        default_board_z_m: float = 0.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.marker_side_m = marker_side_m
        self.tool_length_m = tool_length_m
        self.default_board_z_m = default_board_z_m
        self.nominal_flange_cam = np.array(
            [[1, 0, 0, 0.045], [0, 1, 0, 0.0], [0, 0, 1, 0.062], [0, 0, 0, 1.0]], dtype=float
        )
        self._items: list[Any] = []
        self._build_ui()

    def configure(
        self,
        marker_side_m: float,
        tool_length_m: float = 0.185,
        default_board_z_m: float = 0.0,
        nominal_flange_cam: np.ndarray | None = None,
    ) -> None:
        self.marker_side_m = float(marker_side_m)
        self.tool_length_m = float(tool_length_m)
        self.default_board_z_m = float(default_board_z_m)
        if nominal_flange_cam is not None:
            self.nominal_flange_cam = valid_transform(nominal_flange_cam)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("基座坐标系")
        title.setObjectName("muted")
        title.setToolTip("坐标轴：X 红 / Y 绿 / Z 蓝；品红线：末端包络；黄框：标定板；青色点：预设点位。")
        self.chk_waypoints = QCheckBox("点位")
        self.chk_waypoints.setChecked(True)
        self.chk_pointcloud = QCheckBox("点云")
        self.chk_pointcloud.setChecked(True)
        self.btn_reset_cam = QPushButton("重置视角")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.chk_waypoints)
        header.addWidget(self.chk_pointcloud)
        header.addWidget(self.btn_reset_cam)
        root.addLayout(header)

        is_offscreen = QApplication.platformName() == "offscreen"
        if HAS_OPENGL and gl is not None and not is_offscreen:
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setBackgroundColor("#191d24")
            self.reset_camera_view()
            root.addWidget(self.gl_view, stretch=1)
            self.btn_reset_cam.clicked.connect(self.reset_camera_view)
            self._init_static_grid()
        else:
            self.gl_view = None
            fallback = QLabel("OpenGL 视图不可用")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(fallback, stretch=1)

    def reset_camera_view(self) -> None:
        if self.gl_view is None:
            return
        self.gl_view.opts["center"] = QVector3D(0.32, 0.0, 0.16)
        self.gl_view.setCameraPosition(distance=0.95, elevation=26, azimuth=-135)

    def _init_static_grid(self) -> None:
        if self.gl_view is None or gl is None:
            return
        grid = gl.GLGridItem()
        grid.setSize(x=1.0, y=1.0, z=1.0)
        grid.setSpacing(x=0.1, y=0.1, z=0.1)
        grid.translate(0.35, 0.0, 0.0)
        self.gl_view.addItem(grid)
        self._add_frame_axes(np.eye(4), axis_len=0.12, width=3.0, permanent=True)

    def _clear_dynamic_items(self) -> None:
        if self.gl_view is None:
            return
        for item in self._items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._items.clear()

    def _add_line(
        self,
        pts: np.ndarray,
        color: tuple[float, float, float, float],
        width: float = 2.0,
        mode: str = "lines",
        permanent: bool = False,
    ) -> None:
        if self.gl_view is None or gl is None:
            return
        item = gl.GLLinePlotItem(pos=pts, color=color, width=width, mode=mode, antialias=True)
        self.gl_view.addItem(item)
        if not permanent:
            self._items.append(item)

    def _add_frame_axes(
        self,
        transform: np.ndarray,
        axis_len: float = 0.07,
        width: float = 2.5,
        permanent: bool = False,
    ) -> None:
        t = valid_transform(transform)
        origin = t[:3, 3]
        x_end = origin + t[:3, 0] * axis_len
        y_end = origin + t[:3, 1] * axis_len
        z_end = origin + t[:3, 2] * axis_len
        self._add_line(np.vstack([origin, x_end]), (1.0, 0.25, 0.25, 1.0), width, permanent=permanent)
        self._add_line(np.vstack([origin, y_end]), (0.25, 0.95, 0.35, 1.0), width, permanent=permanent)
        self._add_line(np.vstack([origin, z_end]), (0.30, 0.65, 1.0, 1.0), width, permanent=permanent)

    def update_scene(
        self,
        x_base_flange: np.ndarray | None = None,
        x_flange_cam: np.ndarray | None = None,
        x_base_board: np.ndarray | None = None,
        waypoints: list[dict[str, Any]] | None = None,
        captured_samples: list[dict[str, Any]] | None = None,
        points_cam: np.ndarray | None = None,
        colors_rgb: np.ndarray | None = None,
    ) -> None:
        if self.gl_view is None or gl is None:
            return
        self._clear_dynamic_items()

        board_pose = valid_transform(x_base_board) if x_base_board is not None else np.array(
            [[1, 0, 0, 0.40], [0, 1, 0, 0.0], [0, 0, 1, self.default_board_z_m], [0, 0, 0, 1.0]], dtype=float
        )
        corners_local = marker_object_points(self.marker_side_m)
        corners_base = transform_points(board_pose, corners_local)
        loop_pts = np.vstack([corners_base, corners_base[0]])
        self._add_line(loop_pts, (0.95, 0.80, 0.20, 1.0), width=2.5, mode="line_strip")
        self._add_frame_axes(board_pose, axis_len=0.08, width=2.0)

        if self.chk_waypoints.isChecked() and waypoints:
            wp_pts = np.array([wp["X_base_flange"][:3, 3] for wp in waypoints], dtype=float)
            sp = gl.GLScatterPlotItem(pos=wp_pts, color=(0.35, 0.85, 0.95, 0.65), size=6, pxMode=True)
            self.gl_view.addItem(sp)
            self._items.append(sp)

        if captured_samples:
            cap_pts = np.array(
                [s["X_base_flange"][:3, 3] for s in captured_samples if s.get("enabled", True)],
                dtype=float,
            )
            if len(cap_pts) > 0:
                sp_cap = gl.GLScatterPlotItem(pos=cap_pts, color=(0.40, 0.95, 0.45, 0.95), size=9, pxMode=True)
                self.gl_view.addItem(sp_cap)
                self._items.append(sp_cap)

        if x_base_flange is not None:
            flange = valid_transform(x_base_flange)
            self._add_frame_axes(flange, axis_len=0.075, width=3.0)
            shoulder = np.array([0.0, 0.0, 0.267])
            self._add_line(
                np.vstack([[0.0, 0.0, 0.0], shoulder, flange[:3, 3]]),
                (0.75, 0.78, 0.85, 0.75),
                width=3.5,
                mode="line_strip",
            )
            tool_tip = transform_points(flange, np.array([[0.0, 0.0, self.tool_length_m]]))[0]
            self._add_line(
                np.vstack([flange[:3, 3], tool_tip]),
                (0.95, 0.35, 0.85, 0.95),
                width=3.0,
            )

            cam_extr = valid_transform(x_flange_cam) if x_flange_cam is not None else self.nominal_flange_cam
            x_base_cam = flange @ cam_extr
            self._add_frame_axes(x_base_cam, axis_len=0.06, width=2.0)

            if self.chk_pointcloud.isChecked() and points_cam is not None and len(points_cam) > 0:
                pts_base = transform_points(x_base_cam, points_cam)
                cols = (
                    np.column_stack([colors_rgb, np.full(len(pts_base), 0.85)])
                    if colors_rgb is not None and len(colors_rgb) == len(pts_base)
                    else (0.4, 0.8, 1.0, 0.75)
                )
                pc_item = gl.GLScatterPlotItem(pos=pts_base, color=cols, size=2.5, pxMode=True)
                self.gl_view.addItem(pc_item)
                self._items.append(pc_item)
