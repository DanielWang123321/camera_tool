"""Calibration results, sample inspection, algorithm comparison and residuals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from camera_tool.ui.video_panel import ImagePreview


class CalibrationPanel(QWidget):
    """Result page with an independent recorded-sample preview."""

    sample_toggled = Signal(int, bool)
    sample_selected = Signal(int)
    solve_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating_table = False
        self._build_ui()
        self.mode = "eye-in-hand"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        editable = mode == "eye-in-hand"
        self.tbl_samples.setColumnHidden(0, not editable)
        self.tabs.setTabVisible(self.tabs.indexOf(self.tbl_methods), editable)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        bar = QHBoxLayout()
        self.lbl_result_state = QLabel("尚未求解")
        self.lbl_result_state.setObjectName("muted")
        self.btn_solve = QPushButton("重新求解")
        self.btn_solve.setEnabled(False)
        self.btn_export = QPushButton("导出 YAML")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        bar.addWidget(self.lbl_result_state)
        bar.addStretch()
        bar.addWidget(self.btn_solve)
        bar.addWidget(self.btn_export)
        root.addLayout(bar)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.lbl_count = self._add_metric(metrics, "有效样本", "0")
        self.lbl_rms = self._add_metric(metrics, "重投影 RMS", "— px")
        self.lbl_mean = self._add_metric(metrics, "平均位置残差", "— mm")
        root.addLayout(metrics)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.samples_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.samples_splitter.setChildrenCollapsible(False)
        self.tbl_samples = QTableWidget(0, 4)
        self.tbl_samples.setHorizontalHeaderLabels(["启用", "编号", "点位", "误差 (px)"])
        hdr = self.tbl_samples.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tbl_samples.horizontalHeaderItem(3).setToolTip("未求解时显示单帧 PnP RMS；求解后显示该样本的重投影 RMS。")
        self.tbl_samples.verticalHeader().hide()
        self.tbl_samples.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_samples.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.tbl_samples.setAlternatingRowColors(True)
        self.tbl_samples.itemChanged.connect(self._on_sample_item_changed)
        self.tbl_samples.currentCellChanged.connect(self._on_current_sample_changed)
        self.samples_splitter.addWidget(self.tbl_samples)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setMinimumWidth(340)
        detail_widget = QWidget()
        details = QVBoxLayout(detail_widget)
        details.setContentsMargins(16, 12, 8, 12)
        details.setSpacing(10)
        self.lbl_sample_title = QLabel("样本详情")
        self.lbl_sample_title.setObjectName("sectionTitle")
        self.sample_preview = ImagePreview("无样本图像")
        self.sample_preview.setMinimumHeight(180)
        self.lbl_sample_details = QLabel()
        self.lbl_sample_details.setWordWrap(True)
        self.lbl_sample_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.addWidget(self.lbl_sample_title)
        details.addWidget(self.sample_preview, stretch=1)
        details.addWidget(self.lbl_sample_details)
        self.details_scroll.setWidget(detail_widget)
        self.samples_splitter.addWidget(self.details_scroll)
        self.samples_splitter.setSizes([620, 500])
        self.details_scroll.hide()
        self.tabs.addTab(self.samples_splitter, "样本")

        self.tbl_methods = QTableWidget(0, 7)
        self.tbl_methods.setHorizontalHeaderLabels(
            ["算法", "平移 XYZ (mm)", "旋转 RPY (°)", "平均 (mm)", "最大 (mm)", "角度 (°)", "RMS (px)"]
        )
        self.tbl_methods.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_methods.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_methods.verticalHeader().hide()
        self.tbl_methods.setAlternatingRowColors(True)
        self.tabs.addTab(self.tbl_methods, "算法")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#191d24")
        self.plot_widget.setLabel("left", "误差", color="#939ba9")
        self.plot_widget.setLabel("bottom", "样本编号", color="#939ba9")
        self.plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.addLegend(offset=(12, 12))
        self.tabs.addTab(self.plot_widget, "残差")
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.tabs.addTab(self.txt_report, "报告")
        self.btn_solve.clicked.connect(self.solve_requested.emit)
        self.btn_export.clicked.connect(self.export_requested.emit)

    @staticmethod
    def _add_metric(layout: QHBoxLayout, title: str, value: str) -> QLabel:
        card = QFrame()
        card.setObjectName("metricCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 12, 16, 12)
        caption = QLabel(title)
        caption.setObjectName("muted")
        label = QLabel(value)
        label.setObjectName("metricValue")
        card_layout.addWidget(caption)
        card_layout.addWidget(label)
        layout.addWidget(card, stretch=1)
        return label

    def refresh_samples(self, samples: list[dict[str, Any]], per_sample_rms: dict[int, float] | None = None) -> None:
        selected_row = self.tbl_samples.currentRow()
        self._updating_table = True
        self.tbl_samples.setRowCount(len(samples))
        self.lbl_count.setText(str(sum(1 for s in samples if s.get("enabled", True))))
        for row, sample in enumerate(samples):
            sid = int(sample["id"])
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Checked if sample.get("enabled", True) else Qt.CheckState.Unchecked)
            self.tbl_samples.setItem(row, 0, chk)
            rms = per_sample_rms.get(sid) if per_sample_rms is not None else sample["pnp_rms_px"]
            for col, value in enumerate([str(sid), str(sample.get("name", "manual")), f"{rms:.2f}" if rms is not None else "—"], start=1):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                self.tbl_samples.setItem(row, col, item)
        if 0 <= selected_row < len(samples):
            self.tbl_samples.selectRow(selected_row)
        else:
            self.tbl_samples.clearSelection()
            self.tbl_samples.setCurrentCell(-1, -1)
            self.details_scroll.hide()
            self.sample_preview.reset("无样本图像")
            self.lbl_sample_details.clear()
        self._updating_table = False

    def invalidate_solution(self, samples: list[dict[str, Any]]) -> None:
        self.btn_export.setEnabled(False)
        self.lbl_result_state.setText("待重新求解" if samples else "尚未求解")
        self.lbl_result_state.setStyleSheet("color: #d6b577;" if samples else "")
        self.lbl_rms.setText("— px")
        self.lbl_mean.setText("— mm")
        self.tbl_methods.setRowCount(0)
        self.plot_widget.clear()
        self.plot_widget.getAxis("bottom").setTicks(None)
        self.txt_report.setPlainText("样本变化后需要重新求解。" if samples else "采集至少 5 组有效样本后开始标定，推荐 20 组。")
        self.refresh_samples(samples)

    def show_sample(self, sample: dict[str, Any], annotated_bgr: np.ndarray | None = None) -> None:
        self.details_scroll.show()
        self.lbl_sample_title.setText(f"样本 #{sample['id']} · {sample.get('name', 'manual')}")
        image = annotated_bgr if annotated_bgr is not None else sample.get("color_bgr")
        if image is not None:
            self.sample_preview.set_image(image)
        else:
            self.sample_preview.reset("该样本未保存图像")
        if self.mode == "eye-to-hand":
            import json
            from camera_tool.core import json_default
            self.lbl_sample_details.setText(json.dumps({k: sample[k] for k in ("role", "area_px", "sharpness", "pnp_rms_px", "X_base_tcp") if k in sample},
                                                      indent=2, ensure_ascii=False, default=json_default))
            return
        pose = sample["flange_mm_deg"]
        tcp = sample.get("tcp_mm_deg", pose)
        safety = sample.get("safety", {})
        clearance = safety.get("table_clearance_mm")
        lines = [
            f"法兰 XYZ: {pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f} mm",
            f"法兰 RPY: {pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f}°",
            f"TCP XYZ: {tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f} mm",
            f"TCP RPY: {tcp[3]:.1f}, {tcp[4]:.1f}, {tcp[5]:.1f}°",
            f"面积 {sample['area_px']:.0f} px² · 清晰度 {sample['sharpness']:.1f}",
            f"单帧 PnP RMS {sample['pnp_rms_px']:.3f} px",
            f"末端离桌 {clearance:.1f} mm · {safety.get('message', '未知')}" if clearance is not None else "安全状态未知",
        ]
        self.lbl_sample_details.setText("\n".join(lines))

    def _on_sample_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._updating_table and item.column() == 0:
            self.sample_toggled.emit(item.row(), item.checkState() == Qt.CheckState.Checked)

    def _on_current_sample_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if not self._updating_table and row >= 0:
            self.sample_selected.emit(row)

    def display_solution(self, solution: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        self.btn_export.setEnabled(True)
        self.lbl_result_state.setText("求解完成")
        self.lbl_result_state.setStyleSheet("color: #8fc9a2;")
        metrics = solution["metrics"]
        self.lbl_rms.setText(f"{metrics['rms_px']:.3f} px")
        self.lbl_mean.setText(f"{metrics['mean_mm']:.2f} mm")
        methods = solution.get("methods", {})
        names = [k for k in ["BUNDLE_ADJUSTMENT", *[k for k in methods if k != "BUNDLE_ADJUSTMENT"]] if k in methods]
        self.tbl_methods.setRowCount(len(names))
        for row, name in enumerate(names):
            method = methods[name]
            pose, met = method["flange_cam_mm_deg"], method["metrics"]
            values = [
                "重投影精化" if name == "BUNDLE_ADJUSTMENT" else name,
                ", ".join(f"{v:.2f}" for v in pose[:3]), ", ".join(f"{v:.2f}" for v in pose[3:]),
                f"{met['mean_mm']:.2f}", f"{met['max_mm']:.2f}", f"{met['mean_deg']:.3f}", f"{met['rms_px']:.3f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                if name == "BUNDLE_ADJUSTMENT":
                    item.setForeground(QColor("#8fc9a2"))
                self.tbl_methods.setItem(row, col, item)
        per_rms = dict(zip(metrics["sample_ids"], metrics["per_sample_rms_px"]))
        self.refresh_samples(samples, per_rms)
        self.plot_widget.clear()
        x_ids = np.arange(len(metrics["sample_ids"]), dtype=float)
        self.plot_widget.getAxis("bottom").setTicks([list(zip(x_ids, map(str, metrics["sample_ids"])))])
        self.plot_widget.addItem(pg.BarGraphItem(x=x_ids - 0.18, height=metrics["per_sample_mm"], width=0.34, brush="#6ca8f5", name="位置残差 (mm)"))
        self.plot_widget.addItem(pg.BarGraphItem(x=x_ids + 0.18, height=metrics["per_sample_rms_px"], width=0.34, brush="#d6b577", name="重投影 RMS (px)"))
        self._display_report(solution)

    def display_eye_to_hand(self, solution: dict[str, Any], samples: list[dict[str, Any]], replay: bool = False) -> None:
        import json
        from camera_tool.core import json_default
        self.set_mode("eye-to-hand")
        self.refresh_samples(samples)
        self.btn_export.setEnabled(not replay and solution.get("passed") is True)
        self.lbl_result_state.setText("离线复算通过 · 保留发布限制" if replay else "完整检查通过 · 可导出")
        report = solution.get("independent_check", {})
        self.lbl_mean.setText(f"{report.get('mean_mm', 0):.2f} mm")
        self.lbl_rms.setText(f"{report.get('max_rms_px', 0):.3f} px（最大）")
        self.plot_widget.clear()
        self.plot_widget.getAxis("bottom").setTicks(None)
        values = report.get("per_pose_mm_deg", [])
        if values:
            self.plot_widget.addItem(pg.BarGraphItem(x=np.arange(len(values)), height=[v[0] for v in values], width=.6, brush="#6ca8f5", name="独立姿态残差 (mm)"))
        self.txt_report.setPlainText(json.dumps(solution, indent=2, ensure_ascii=False, default=json_default))

    def _display_report(self, solution: dict[str, Any]) -> None:
        from camera_tool.core import pose_to_m_rad

        met, cov = solution["metrics"], solution["rotation_coverage_deg"]
        lines = [
            "手眼标定报告 · Eye-in-Hand",
            f"推荐算法: {solution['recommended_method']}",
            f"有效样本: {met['count']} 组 · 三轴旋转激励 RMS: {cov[0]:.2f}, {cov[1]:.2f}, {cov[2]:.2f}°",
            f"位置残差: 平均 {met['mean_mm']:.3f} mm · 最大 {met['max_mm']:.3f} mm",
            f"角度残差: 平均 {met['mean_deg']:.4f}° · 最大 {met['max_deg']:.4f}°",
            f"重投影 RMS: {met['rms_px']:.3f} px · 最大 {met['max_rms_px']:.3f} px",
        ]
        for title, key, pose_key in [
            ("法兰到相机", "X_flange_cam", "flange_cam_mm_deg"),
            ("TCP 到相机", "X_tcp_cam", "tcp_cam_mm_deg"),
            ("基座到标定板", "X_base_board", "base_board_mm_deg"),
        ]:
            pose = ", ".join(f"{float(value):.3f}" for value in solution[pose_key])
            lines.extend(["", f"{title} · {key}", f"XYZ (mm), RPY (°): {pose}", np.array2string(np.asarray(solution[key]), precision=6, suppress_small=True)])

        eef_m_rad = solution.get("tcp_cam_m_rad") or pose_to_m_rad(np.asarray(solution["X_tcp_cam"], dtype=float))
        flange_m_rad = solution.get("flange_cam_m_rad") or pose_to_m_rad(np.asarray(solution["X_flange_cam"], dtype=float))
        lines.extend([
            "",
            "ufactory_vision 外参 (color_optical_frame, 单位: m, rad):",
            f"EULER_EEF_TO_COLOR_OPT = [{', '.join(f'{float(v):.6f}' for v in eef_m_rad)}]",
            f"EULER_FLANGE_TO_COLOR_OPT = [{', '.join(f'{float(v):.6f}' for v in flange_m_rad)}]",
        ])

        bc = solution.get("board_check")
        if bc is not None:
            lines.extend([
                "",
                f"标定板平面校验: {'通过' if bc.get('ok') else '警告'}",
                f"求解板面 Z {bc['board_z_mm']:.2f} mm（期望 {bc['expected_board_z_mm']:.2f} mm，误差 {bc['board_height_error_mm']:.2f} mm）· 板倾角 {bc['tilt_deg']:.2f}°",
            ])

        dc = solution.get("depth_check")
        if dc is not None:
            lines.extend(["", f"深度一致性: {'通过' if dc.get('ok') else '警告'}"])
            if dc.get("reason"):
                lines.append(str(dc["reason"]))
            else:
                lines.append(f"桌面高度误差 {dc['table_height_error_mm']:.2f} mm · 法向偏差 {dc['normal_angle_deg']:.2f}°")
            if "plane_inliers" in dc:
                lines.append(f"平面内点 {dc['plane_inliers']} / {dc['points_count']}")
        self.txt_report.setPlainText("\n".join(lines))
