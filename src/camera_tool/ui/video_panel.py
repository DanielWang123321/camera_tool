"""Camera preview and compact image quality indicators."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from camera_tool.ui.i18n import DEFAULT_LANG, normalize_lang, tr


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

    def __init__(
        self,
        title: str,
        unit: str,
        threshold: str,
        parent: QWidget | None = None,
        lang: str = DEFAULT_LANG,
    ) -> None:
        super().__init__(parent)
        self.unit = unit
        self.lang = normalize_lang(lang)
        self._last_value: str | None = None
        self._last_passed: bool | None = None
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

    def set_labels(self, title: str, threshold: str, lang: str) -> None:
        self.lang = normalize_lang(lang)
        self.title_lbl.setText(title)
        self.setToolTip(threshold)
        if self._last_value is not None and self._last_passed is not None:
            self.update_gate(self._last_value, self._last_passed)

    def reset(self) -> None:
        self._last_value = None
        self._last_passed = None
        self.val_lbl.setText("—")
        self.val_lbl.setStyleSheet("color: #939ba9;")

    def update_gate(self, value_text: str, passed: bool) -> None:
        self._last_value = value_text
        self._last_passed = passed
        state = tr("video.gate_pass" if passed else "video.gate_fail", self.lang)
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
        lang: str = DEFAULT_LANG,
    ) -> None:
        super().__init__(parent)
        self.marker_cfg = marker_cfg
        self.tool_cfg = tool_cfg
        self.max_depth_m = float(max_depth_m)
        self.lang = normalize_lang(lang)
        self._annotated_bgr: np.ndarray | None = None
        self._depth_m: np.ndarray | None = None
        self._last_eval: dict[str, Any] | None = None
        self._last_safety: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._build_ui()

    def _marker_summary(self) -> str:
        marker_id = self.marker_cfg.get("id", 398)
        side_mm = float(self.marker_cfg.get("side_m", 0.15)) * 1000.0
        return f"ArUco #{marker_id} · {side_mm:.0f} mm"

    def set_language(self, lang: str) -> None:
        self.lang = normalize_lang(lang)
        with QSignalBlocker(self.view_mode_combo):
            self.view_mode_combo.setItemText(0, tr("video.view_rgb", self.lang))
            self.view_mode_combo.setItemText(1, tr("video.view_depth", self.lang))
            self.view_mode_combo.setItemText(2, tr("video.view_both", self.lang))
        self.view_mode_combo.setAccessibleName(tr("video.view_acc", self.lang))
        self.gate_area.set_labels(
            tr("video.gate_area", self.lang),
            tr("video.tip_area", self.lang, val=self.marker_cfg["min_area_px"]),
            self.lang,
        )
        self.gate_sharp.set_labels(
            tr("video.gate_sharp", self.lang),
            tr("video.tip_sharp", self.lang, val=self.marker_cfg["min_sharpness"]),
            self.lang,
        )
        self.gate_rms.set_labels(
            tr("video.gate_rms", self.lang),
            tr("video.tip_rms", self.lang, val=self.marker_cfg["max_pnp_rms_px"]),
            self.lang,
        )
        if self._last_error is not None:
            self.show_error(self._last_error)
        elif self._last_eval is not None and self._annotated_bgr is not None:
            self._apply_evaluation(self._last_eval, self._last_safety)
        else:
            self.status_badge.setText(tr("video.waiting", self.lang))
            if self.video_lbl.image_bgr is None:
                self.video_lbl.setText(tr("video.disconnected", self.lang))
            self.lbl_reason.setText(self._marker_summary())

    def configure(
        self,
        marker_cfg: dict[str, Any],
        tool_cfg: dict[str, Any],
        max_depth_m: float = 0.80,
    ) -> None:
        self.marker_cfg = marker_cfg
        self.tool_cfg = tool_cfg
        self.max_depth_m = float(max_depth_m)
        self.gate_area.setToolTip(tr("video.tip_area", self.lang, val=self.marker_cfg["min_area_px"]))
        self.gate_sharp.setToolTip(tr("video.tip_sharp", self.lang, val=self.marker_cfg["min_sharpness"]))
        self.gate_rms.setToolTip(tr("video.tip_rms", self.lang, val=self.marker_cfg["max_pnp_rms_px"]))
        self._render_frame()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        header = QHBoxLayout()
        self.status_badge = QLabel(tr("video.waiting", self.lang))
        self.status_badge.setObjectName("muted")
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(
            [tr("video.view_rgb", self.lang), tr("video.view_depth", self.lang), tr("video.view_both", self.lang)]
        )
        self.view_mode_combo.setAccessibleName(tr("video.view_acc", self.lang))
        self.view_mode_combo.currentIndexChanged.connect(self._render_frame)
        header.addWidget(self.status_badge)
        header.addStretch()
        header.addWidget(self.view_mode_combo)
        root.addLayout(header)
        self.video_lbl = ImagePreview(tr("video.disconnected", self.lang))
        root.addWidget(self.video_lbl, stretch=1)

        gates = QHBoxLayout()
        gates.setSpacing(8)
        self.gate_area = GateIndicator(
            tr("video.gate_area", self.lang),
            "px²",
            tr("video.tip_area", self.lang, val=self.marker_cfg["min_area_px"]),
            lang=self.lang,
        )
        self.gate_sharp = GateIndicator(
            tr("video.gate_sharp", self.lang),
            "",
            tr("video.tip_sharp", self.lang, val=self.marker_cfg["min_sharpness"]),
            lang=self.lang,
        )
        self.gate_rms = GateIndicator(
            tr("video.gate_rms", self.lang),
            "px",
            tr("video.tip_rms", self.lang, val=self.marker_cfg["max_pnp_rms_px"]),
            lang=self.lang,
        )
        for gate in (self.gate_area, self.gate_sharp, self.gate_rms):
            gates.addWidget(gate, stretch=1)
        root.addLayout(gates)
        self.lbl_reason = QLabel(self._marker_summary())
        self.lbl_reason.setWordWrap(True)
        self.lbl_reason.setObjectName("muted")
        root.addWidget(self.lbl_reason)

    def reset(self, message: str | None = None) -> None:
        self._annotated_bgr = None
        self._depth_m = None
        self._last_eval = None
        self._last_safety = None
        self._last_error = None
        self.video_lbl.reset(message if message is not None else tr("video.disconnected", self.lang))
        self.status_badge.setText(tr("video.waiting", self.lang))
        self.status_badge.setStyleSheet("")
        self.lbl_reason.setText(self._marker_summary())
        for gate in (self.gate_area, self.gate_sharp, self.gate_rms):
            gate.reset()

    def show_error(self, reason: str) -> None:
        self._last_error = reason
        self.status_badge.setText(tr("video.stream_error", self.lang))
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

    def _apply_evaluation(
        self,
        eval_result: dict[str, Any],
        robot_safety: dict[str, Any] | None = None,
    ) -> None:
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
        self.status_badge.setText(tr("video.ready" if passed else "video.not_ready", self.lang))
        self.status_badge.setStyleSheet(f"color: {'#8fc9a2' if passed else '#ec9898'};")
        reason = str(eval_result.get("reason", ""))
        if not safe:
            reason = str(robot_safety.get("message", tr("video.safety_fault", self.lang)))
        marker_id = self.marker_cfg.get("id", 398)
        self.lbl_reason.setText(tr("video.reason_passed", self.lang, marker_id=marker_id) if passed else reason)

    def update_frame(
        self,
        annotated_bgr: np.ndarray,
        depth_m: np.ndarray | None,
        eval_result: dict[str, Any],
        robot_safety: dict[str, Any] | None = None,
    ) -> None:
        self._last_error = None
        self._last_eval = eval_result
        self._last_safety = robot_safety
        self._annotated_bgr, self._depth_m = annotated_bgr, depth_m
        self._render_frame()
        self._apply_evaluation(eval_result, robot_safety)
