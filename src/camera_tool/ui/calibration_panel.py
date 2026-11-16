"""Calibration results, sample inspection, algorithm comparison and residuals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from camera_tool.ui.i18n import DEFAULT_LANG, normalize_lang, tr
from camera_tool.ui.video_panel import ImagePreview


class CalibrationPanel(QWidget):
    """Result page with an independent recorded-sample preview."""

    sample_toggled = Signal(int, bool)
    sample_selected = Signal(int)
    solve_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent: QWidget | None = None, lang: str = DEFAULT_LANG) -> None:
        super().__init__(parent)
        self.lang = normalize_lang(lang)
        self._updating_table = False
        self._last_samples: list[dict[str, Any]] = []
        self._last_solution: dict[str, Any] | None = None
        self._last_replay: bool = False
        self._selected_sample: dict[str, Any] | None = None
        self._selected_annotated: np.ndarray | None = None
        self._build_ui()
        self.mode = "eye-in-hand"

    def set_language(self, lang: str) -> None:
        self.lang = normalize_lang(lang)
        self.btn_solve.setText(tr("panel.btn_resolve", self.lang))
        self.btn_export.setText(tr("panel.btn_export", self.lang))
        self.lbl_count_title.setText(tr("panel.metric_count", self.lang))
        self.lbl_rms_title.setText(tr("panel.metric_rms", self.lang))
        self.lbl_mean_title.setText(tr("panel.metric_mean", self.lang))
        self.tbl_samples.setHorizontalHeaderLabels([
            tr("panel.hdr_use", self.lang),
            tr("panel.hdr_id", self.lang),
            tr("panel.hdr_waypoint", self.lang),
            tr("panel.hdr_err", self.lang),
        ])
        self.tbl_samples.horizontalHeaderItem(3).setToolTip(tr("panel.hdr_err_tip", self.lang))
        self.tbl_methods.setHorizontalHeaderLabels([
            tr("panel.method_hdr_name", self.lang),
            tr("panel.method_hdr_trans", self.lang),
            tr("panel.method_hdr_rot", self.lang),
            tr("panel.method_hdr_mean", self.lang),
            tr("panel.method_hdr_max", self.lang),
            tr("panel.method_hdr_angle", self.lang),
            tr("panel.method_hdr_rms", self.lang),
        ])
        self.tabs.setTabText(0, tr("panel.tab_samples", self.lang))
        self.tabs.setTabText(1, tr("panel.tab_methods", self.lang))
        self.tabs.setTabText(2, tr("panel.tab_residuals", self.lang))
        self.tabs.setTabText(3, tr("panel.tab_report", self.lang))
        self.plot_widget.setLabel("left", tr("panel.plot_err", self.lang), color="#939ba9")
        self.plot_widget.setLabel("bottom", tr("panel.plot_id", self.lang), color="#939ba9")

        if self._last_solution is not None:
            if self.mode == "eye-in-hand":
                self.display_solution(self._last_solution, self._last_samples)
            else:
                self.display_eye_to_hand(self._last_solution, self._last_samples, self._last_replay)
        else:
            self.invalidate_solution(self._last_samples)

        if self._selected_sample is not None and self.details_scroll.isVisible():
            self.show_sample(self._selected_sample, self._selected_annotated)
        else:
            self.lbl_sample_title.setText(tr("panel.sample_title", self.lang))
            if self.sample_preview.image_bgr is None:
                self.sample_preview.reset(tr("panel.no_sample_image", self.lang))

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
        self.lbl_result_state = QLabel(tr("panel.not_solved", self.lang))
        self.lbl_result_state.setObjectName("muted")
        self.btn_solve = QPushButton(tr("panel.btn_resolve", self.lang))
        self.btn_solve.setEnabled(False)
        self.btn_export = QPushButton(tr("panel.btn_export", self.lang))
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        bar.addWidget(self.lbl_result_state)
        bar.addStretch()
        bar.addWidget(self.btn_solve)
        bar.addWidget(self.btn_export)
        root.addLayout(bar)

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.lbl_count_title, self.lbl_count = self._add_metric(metrics, tr("panel.metric_count", self.lang), "0")
        self.lbl_rms_title, self.lbl_rms = self._add_metric(metrics, tr("panel.metric_rms", self.lang), "— px")
        self.lbl_mean_title, self.lbl_mean = self._add_metric(metrics, tr("panel.metric_mean", self.lang), "— mm")
        root.addLayout(metrics)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.samples_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.samples_splitter.setChildrenCollapsible(False)
        self.tbl_samples = QTableWidget(0, 4)
        self.tbl_samples.setHorizontalHeaderLabels([
            tr("panel.hdr_use", self.lang),
            tr("panel.hdr_id", self.lang),
            tr("panel.hdr_waypoint", self.lang),
            tr("panel.hdr_err", self.lang),
        ])
        hdr = self.tbl_samples.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.tbl_samples.horizontalHeaderItem(3).setToolTip(tr("panel.hdr_err_tip", self.lang))
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
        self.lbl_sample_title = QLabel(tr("panel.sample_title", self.lang))
        self.lbl_sample_title.setObjectName("sectionTitle")
        self.sample_preview = ImagePreview(tr("panel.no_sample_image", self.lang))
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
        self.tabs.addTab(self.samples_splitter, tr("panel.tab_samples", self.lang))

        self.tbl_methods = QTableWidget(0, 7)
        self.tbl_methods.setHorizontalHeaderLabels([
            tr("panel.method_hdr_name", self.lang),
            tr("panel.method_hdr_trans", self.lang),
            tr("panel.method_hdr_rot", self.lang),
            tr("panel.method_hdr_mean", self.lang),
            tr("panel.method_hdr_max", self.lang),
            tr("panel.method_hdr_angle", self.lang),
            tr("panel.method_hdr_rms", self.lang),
        ])
        self.tbl_methods.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_methods.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_methods.verticalHeader().hide()
        self.tbl_methods.setAlternatingRowColors(True)
        self.tabs.addTab(self.tbl_methods, tr("panel.tab_methods", self.lang))

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#191d24")
        self.plot_widget.setLabel("left", tr("panel.plot_err", self.lang), color="#939ba9")
        self.plot_widget.setLabel("bottom", tr("panel.plot_id", self.lang), color="#939ba9")
        self.plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.addLegend(offset=(12, 12))
        self.tabs.addTab(self.plot_widget, tr("panel.tab_residuals", self.lang))
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.tabs.addTab(self.txt_report, tr("panel.tab_report", self.lang))
        self.btn_solve.clicked.connect(self.solve_requested.emit)
        self.btn_export.clicked.connect(self.export_requested.emit)

    @staticmethod
    def _add_metric(layout: QHBoxLayout, title: str, value: str) -> tuple[QLabel, QLabel]:
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
        return caption, label

    def refresh_samples(self, samples: list[dict[str, Any]], per_sample_rms: dict[int, float] | None = None) -> None:
        self._last_samples = list(samples)
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
            self._selected_sample = None
            self._selected_annotated = None
            self.tbl_samples.clearSelection()
            self.tbl_samples.setCurrentCell(-1, -1)
            self.details_scroll.hide()
            self.sample_preview.reset(tr("panel.no_sample_image", self.lang))
            self.lbl_sample_details.clear()
        self._updating_table = False

    def invalidate_solution(self, samples: list[dict[str, Any]]) -> None:
        self._last_solution = None
        self._last_samples = list(samples)
        self.btn_export.setEnabled(False)
        self.lbl_result_state.setText(tr("panel.needs_resolve" if samples else "panel.not_solved", self.lang))
        self.lbl_result_state.setStyleSheet("color: #d6b577;" if samples else "")
        self.lbl_rms.setText("— px")
        self.lbl_mean.setText("— mm")
        self.tbl_methods.setRowCount(0)
        self.plot_widget.clear()
        self.plot_widget.getAxis("bottom").setTicks(None)
        self.txt_report.setPlainText(tr("panel.dirty_report" if samples else "panel.empty_report", self.lang))
        self.refresh_samples(samples)

    def show_sample(self, sample: dict[str, Any], annotated_bgr: np.ndarray | None = None) -> None:
        self._selected_sample = sample
        self._selected_annotated = annotated_bgr
        self.details_scroll.show()
        self.lbl_sample_title.setText(
            tr("panel.sample_title_fmt", self.lang, id=sample["id"], name=sample.get("name", "manual"))
        )
        image = annotated_bgr if annotated_bgr is not None else sample.get("color_bgr")
        if image is not None:
            self.sample_preview.set_image(image)
        else:
            self.sample_preview.reset(tr("panel.sample_missing_img", self.lang))
        if self.mode == "eye-to-hand":
            import json

            from camera_tool.core import json_default
            self.lbl_sample_details.setText(
                json.dumps(
                    {k: sample[k] for k in ("role", "area_px", "sharpness", "pnp_rms_px", "X_base_tcp") if k in sample},
                    indent=2,
                    ensure_ascii=False,
                    default=json_default,
                )
            )
            return
        pose = sample["flange_mm_deg"]
        tcp = sample.get("tcp_mm_deg", pose)
        safety = sample.get("safety", {})
        clearance = safety.get("table_clearance_mm")
        lines = [
            tr("panel.detail_flange_xyz", self.lang, x=pose[0], y=pose[1], z=pose[2]),
            tr("panel.detail_flange_rpy", self.lang, r=pose[3], p=pose[4], y=pose[5]),
            tr("panel.detail_tcp_xyz", self.lang, x=tcp[0], y=tcp[1], z=tcp[2]),
            tr("panel.detail_tcp_rpy", self.lang, r=tcp[3], p=tcp[4], y=tcp[5]),
            tr("panel.detail_quality", self.lang, area=sample["area_px"], sharp=sample["sharpness"]),
            tr("panel.detail_pnp", self.lang, rms=sample["pnp_rms_px"]),
            tr(
                "panel.detail_clearance",
                self.lang,
                clearance=clearance,
                msg=safety.get("message", tr("report.unknown", self.lang)),
            )
            if clearance is not None
            else tr("panel.detail_safety_unknown", self.lang),
        ]
        self.lbl_sample_details.setText("\n".join(lines))

    def _on_sample_item_changed(self, item: QTableWidgetItem) -> None:
        if not self._updating_table and item.column() == 0:
            self.sample_toggled.emit(item.row(), item.checkState() == Qt.CheckState.Checked)

    def _on_current_sample_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if not self._updating_table and row >= 0:
            self.sample_selected.emit(row)

    def display_solution(self, solution: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        self._last_solution = solution
        self._last_samples = list(samples)
        self._last_replay = False
        self.btn_export.setEnabled(True)
        self.lbl_result_state.setText(tr("panel.solved", self.lang))
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
                tr("panel.method_ba", self.lang) if name == "BUNDLE_ADJUSTMENT" else name,
                ", ".join(f"{v:.2f}" for v in pose[:3]),
                ", ".join(f"{v:.2f}" for v in pose[3:]),
                f"{met['mean_mm']:.2f}",
                f"{met['max_mm']:.2f}",
                f"{met['mean_deg']:.3f}",
                f"{met['rms_px']:.3f}",
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
        self.plot_widget.addItem(
            pg.BarGraphItem(x=x_ids - 0.18, height=metrics["per_sample_mm"], width=0.34, brush="#6ca8f5", name=tr("panel.bar_pos", self.lang))
        )
        self.plot_widget.addItem(
            pg.BarGraphItem(x=x_ids + 0.18, height=metrics["per_sample_rms_px"], width=0.34, brush="#d6b577", name=tr("panel.bar_rms", self.lang))
        )
        self._display_report(solution)

    def display_eye_to_hand(self, solution: dict[str, Any], samples: list[dict[str, Any]], replay: bool = False) -> None:
        import json

        from camera_tool.core import json_default
        self._last_solution = solution
        self._last_samples = list(samples)
        self._last_replay = replay
        self.set_mode("eye-to-hand")
        self.refresh_samples(samples)
        self.btn_export.setEnabled(not replay and solution.get("passed") is True)
        self.lbl_result_state.setText(tr("panel.eth_replay" if replay else "panel.eth_live", self.lang))
        report = solution.get("independent_check", {})
        self.lbl_mean.setText(f"{report.get('mean_mm', 0):.2f} mm")
        self.lbl_rms.setText(tr("panel.eth_max_rms", self.lang, val=report.get("max_rms_px", 0)))
        self.plot_widget.clear()
        self.plot_widget.getAxis("bottom").setTicks(None)
        values = report.get("per_pose_mm_deg", [])
        if values:
            self.plot_widget.addItem(
                pg.BarGraphItem(
                    x=np.arange(len(values)),
                    height=[v[0] for v in values],
                    width=0.6,
                    brush="#6ca8f5",
                    name=tr("panel.bar_holdout", self.lang),
                )
            )
        self.txt_report.setPlainText(json.dumps(solution, indent=2, ensure_ascii=False, default=json_default))

    def _display_report(self, solution: dict[str, Any]) -> None:
        from camera_tool.core import pose_to_m_rad

        met, cov = solution["metrics"], solution["rotation_coverage_deg"]
        lines = [
            tr("report.title", self.lang),
            tr("report.method", self.lang, method=solution["recommended_method"]),
            tr("report.samples", self.lang, count=met["count"], r0=cov[0], r1=cov[1], r2=cov[2]),
            tr("report.pos_err", self.lang, mean=met["mean_mm"], max=met["max_mm"]),
            tr("report.rot_err", self.lang, mean=met["mean_deg"], max=met["max_deg"]),
            tr("report.reproj", self.lang, rms=met["rms_px"], max_rms=met["max_rms_px"]),
        ]
        for title_key, key, pose_key in [
            ("report.flange_cam", "X_flange_cam", "flange_cam_mm_deg"),
            ("report.tcp_cam", "X_tcp_cam", "tcp_cam_mm_deg"),
            ("report.base_board", "X_base_board", "base_board_mm_deg"),
        ]:
            pose = ", ".join(f"{float(value):.3f}" for value in solution[pose_key])
            lines.extend([
                "",
                f"{tr(title_key, self.lang)} · {key}",
                tr("report.pose_line", self.lang, pose=pose),
                np.array2string(np.asarray(solution[key]), precision=6, suppress_small=True),
            ])

        eef_m_rad = solution.get("tcp_cam_m_rad") or pose_to_m_rad(np.asarray(solution["X_tcp_cam"], dtype=float))
        flange_m_rad = solution.get("flange_cam_m_rad") or pose_to_m_rad(np.asarray(solution["X_flange_cam"], dtype=float))
        lines.extend([
            "",
            tr("report.uf_header", self.lang),
            f"EULER_EEF_TO_COLOR_OPT = [{', '.join(f'{float(v):.6f}' for v in eef_m_rad)}]",
            f"EULER_FLANGE_TO_COLOR_OPT = [{', '.join(f'{float(v):.6f}' for v in flange_m_rad)}]",
        ])

        bc = solution.get("board_check")
        if bc is not None:
            state = tr("report.pass" if bc.get("ok") else "report.warn", self.lang)
            lines.extend([
                "",
                tr("report.board_check", self.lang, state=state),
                tr(
                    "report.board_vals",
                    self.lang,
                    z=bc["board_z_mm"],
                    exp=bc["expected_board_z_mm"],
                    err=bc["board_height_error_mm"],
                    tilt=bc["tilt_deg"],
                ),
            ])

        dc = solution.get("depth_check")
        if dc is not None:
            state = tr("report.pass" if dc.get("ok") else "report.warn", self.lang)
            lines.extend(["", tr("report.depth_check", self.lang, state=state)])
            if dc.get("reason"):
                lines.append(str(dc["reason"]))
            else:
                lines.append(
                    tr("report.depth_vals", self.lang, err=dc["table_height_error_mm"], ang=dc["normal_angle_deg"])
                )
            if "plane_inliers" in dc:
                lines.append(tr("report.depth_inliers", self.lang, inliers=dc["plane_inliers"], total=dc["points_count"]))
        self.txt_report.setPlainText("\n".join(lines))
