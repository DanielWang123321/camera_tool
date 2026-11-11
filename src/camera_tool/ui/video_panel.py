"""Camera preview and compact image quality indicators."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


def numpy_bgr_to_qpixmap(bgr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        max(1, target_w),
        max(1, target_h),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class ImagePreview(QLabel):
    """An aspect-preserving image that never sets the layout's preferred size."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.image_bgr: np.ndarray | None = None
        self.setObjectName("imagePreview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(240, 135)

    def set_image(self, bgr: np.ndarray) -> None:
        self.image_bgr = bgr
        self._render_image()

    def reset(self, text: str) -> None:
        self.image_bgr = None
        self.clear()
        self.setText(text)

    def _render_image(self) -> None:
        if self.image_bgr is not None:
            self.setPixmap(numpy_bgr_to_qpixmap(self.image_bgr, self.width() - 12, self.height() - 12))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._render_image()


def colorize_depth(depth_m: np.ndarray, max_depth_m: float = 0.80) -> np.ndarray:
    scaled = np.clip(depth_m / max(max_depth_m, 0.05), 0.0, 1.0)
    cmap = cv2.applyColorMap((scaled * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    cmap[depth_m <= 0.01] = (25, 25, 25)
    return cmap


class GateIndicator(QFrame):
    """Single-line metric with a textual pass/fail state and a threshold tooltip."""

    def __init__(self, title: str, unit: str, threshold: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.unit = unit
        self.setObjectName("qualityGate")
        self.setToolTip(threshold)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("muted")
        self.val_lbl = QLabel("—")
        layout.addWidget(self.title_lbl)
        layout.addStretch()
        layout.addWidget(self.val_lbl)

    def reset(self) -> None:
        self.val_lbl.setText("—")
        self.val_lbl.setStyleSheet("color: #939ba9;")

    def update_gate(self, value_text: str, passed: bool) -> None:
        state = "通过" if passed else "未通过"
        self.val_lbl.setText(f"{value_text} {self.unit} · {state}".replace("  ", " "))
        self.val_lbl.setStyleSheet(f"color: {'#8fc9a2' if passed else '#ec9898'};")


class VideoPanel(QWidget):
    """Live RGB/depth preview; recorded sample images live in CalibrationPanel."""

    def __init__(
        self,
        marker_cfg: dict[str, Any],
        tool_cfg: dict[str, Any],
        max_depth_m: float = 0.80,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.marker_cfg = marker_cfg
        self.tool_cfg = tool_cfg
        self.max_depth_m = float(max_depth_m)
        self._annotated_bgr: np.ndarray | None = None
        self._depth_m: np.ndarray | None = None
        self._build_ui()

    def _marker_summary(self) -> str:
        marker_id = self.marker_cfg.get("id", 398)
        side_mm = float(self.marker_cfg.get("side_m", 0.15)) * 1000.0
        return f"ArUco #{marker_id} · {side_mm:.0f} mm"

    def configure(
        self,
        marker_cfg: dict[str, Any],
        tool_cfg: dict[str, Any],
        max_depth_m: float = 0.80,
    ) -> None:
        self.marker_cfg = marker_cfg
        self.tool_cfg = tool_cfg
        self.max_depth_m = float(max_depth_m)
        self.gate_area.setToolTip(f"标定板面积 ≥ {self.marker_cfg['min_area_px']} px²")
        self.gate_sharp.setToolTip(f"Laplacian 方差 ≥ {self.marker_cfg['min_sharpness']}")
        self.gate_rms.setToolTip(f"单帧重投影 RMS ≤ {self.marker_cfg['max_pnp_rms_px']} px")
        self._render_frame()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        header = QHBoxLayout()
        self.status_badge = QLabel("等待相机")
        self.status_badge.setObjectName("muted")
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["RGB", "深度", "RGB + 深度"])
        self.view_mode_combo.setAccessibleName("相机显示模式")
        self.view_mode_combo.currentIndexChanged.connect(self._render_frame)
        header.addWidget(self.status_badge)
        header.addStretch()
        header.addWidget(self.view_mode_combo)
        root.addLayout(header)
        self.video_lbl = ImagePreview("未连接相机\n连接设备后显示实时画面")
        root.addWidget(self.video_lbl, stretch=1)

        gates = QHBoxLayout()
        gates.setSpacing(8)
        self.gate_area = GateIndicator("面积", "px²", f"标定板面积 ≥ {self.marker_cfg['min_area_px']} px²")
        self.gate_sharp = GateIndicator("清晰度", "", f"Laplacian 方差 ≥ {self.marker_cfg['min_sharpness']}")
        self.gate_rms = GateIndicator("PnP RMS", "px", f"单帧重投影 RMS ≤ {self.marker_cfg['max_pnp_rms_px']} px")
        for gate in (self.gate_area, self.gate_sharp, self.gate_rms):
            gates.addWidget(gate, stretch=1)
        root.addLayout(gates)
        self.lbl_reason = QLabel(self._marker_summary())
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setObjectName("muted")
        root.addWidget(self.lbl_reason)

    def reset(self, message: str = "未连接相机\n连接设备后显示实时画面") -> None:
        self._annotated_bgr = None
        self._depth_m = None
        self.video_lbl.reset(message)
        self.status_badge.setText("等待相机")
        self.status_badge.setStyleSheet("")
        self.lbl_reason.setText(self._marker_summary())
        for gate in (self.gate_area, self.gate_sharp, self.gate_rms):
            gate.reset()

    def show_error(self, reason: str) -> None:
        self.status_badge.setText("取流异常")
        self.status_badge.setStyleSheet("color: #ec9898;")
        self.lbl_reason.setText(reason)
        for gate in (self.gate_area, self.gate_sharp, self.gate_rms):
            gate.reset()

    def _render_frame(self) -> None:
        if self._annotated_bgr is None:
            return
        mode = self.view_mode_combo.currentIndex()
        display = self._annotated_bgr
        if mode == 1 and self._depth_m is not None:
            display = colorize_depth(self._depth_m, self.max_depth_m)
        elif mode == 2 and self._depth_m is not None:
            display = np.hstack([display, colorize_depth(self._depth_m, self.max_depth_m)])
        self.video_lbl.set_image(display)

    def update_frame(
        self,
        annotated_bgr: np.ndarray,
        depth_m: np.ndarray | None,
        eval_result: dict[str, Any],
        robot_safety: dict[str, Any] | None = None,
    ) -> None:
        self._annotated_bgr, self._depth_m = annotated_bgr, depth_m
        self._render_frame()
        detected = bool(eval_result.get("detected", False))
        area = float(eval_result.get("area_px", 0.0))
        sharp = float(eval_result.get("sharpness", 0.0))
        rms = float(eval_result.get("pnp_rms_px", float("inf")))
        self.gate_area.update_gate(
            f"{area:.0f}" if detected else "—", detected and area >= self.marker_cfg["min_area_px"]
        )
        self.gate_sharp.update_gate(
            f"{sharp:.1f}" if detected else "—", detected and sharp >= self.marker_cfg["min_sharpness"]
        )
        self.gate_rms.update_gate(
            f"{rms:.2f}" if detected and np.isfinite(rms) else "—",
            detected and rms <= self.marker_cfg["max_pnp_rms_px"],
        )
        safe = robot_safety is None or robot_safety.get("safe", False)
        passed = bool(eval_result.get("passed", False)) and safe
        self.status_badge.setText("质量通过 · 可采集" if passed else "暂不可采集")
        self.status_badge.setStyleSheet(f"color: {'#8fc9a2' if passed else '#ec9898'};")
        reason = str(eval_result.get("reason", ""))
        if not safe:
            reason = str(robot_safety.get("message", "安全状态异常"))
        marker_id = self.marker_cfg.get("id", 398)
        self.lbl_reason.setText(f"ArUco #{marker_id} · 质量通过" if passed else reason)
