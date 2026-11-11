"""Unified five-page desktop UI. All SDK, motion, solve, and disk replay work runs asynchronously."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from camera_tool.core import load_yaml, pose_from_mm_deg
from camera_tool.devices.registry import BACKENDS, PROFILES, availability
from camera_tool.paths import config_path, output_path
from camera_tool.services.eye_in_hand import (
    EyeInHand,
    config_for_camera_metadata,
    detect_usb_camera_backend,
    resolve_preset_config,
    solve_session,
)
from camera_tool.services.replay import load_session
from camera_tool.ui.calibration_panel import CalibrationPanel
from camera_tool.ui.scene3d_panel import Scene3DPanel
from camera_tool.ui.tasks import TaskWorker
from camera_tool.ui.theme import STYLE
from camera_tool.ui.video_panel import ImagePreview, VideoPanel, colorize_depth


class MainWindow(QMainWindow):
    def __init__(self, sim_mode=False, replay_dir=None, initial_page=0, initial_mode="eye-in-hand"):
        super().__init__()
        self.setWindowTitle("camera_tool · 相机与手眼标定")
        self.setMinimumSize(1080, 640)
        self.resize(1440, 900)
        self.setStyleSheet(STYLE)
        self.workers = {}
        self.backend = None
        self.payload = None
        self.usb_rows = []
        self._closing = False
        self._command_busy = False
        self._last_preview = None
        self._last_metadata = None
        self._selected_mode = initial_mode
        self._build_ui(sim_mode, initial_mode)
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(80)
        self.frame_timer.timeout.connect(self._consume_frames)
        self.frame_timer.start()
        self.capture_shortcut = QShortcut(QKeySequence("Space"), self)
        self.capture_shortcut.activated.connect(self._capture_shortcut)
        self._navigate(initial_page)
        self._sync_controls()
        if replay_dir:
            QTimer.singleShot(0, lambda: self.load_replay_directory(replay_dir))

    def _build_ui(self, sim, mode):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(20, 14, 20, 8)
        top = QHBoxLayout()
        title = QLabel("camera_tool")
        title.setObjectName("appTitle")
        top.addWidget(title)
        self.navigation = QButtonGroup(self)
        self.nav_buttons = []
        for index, text in enumerate(("相机预览", "手眼标定", "结果与回放", "三脚架检测", "USB 稳定性测试")):
            button = QPushButton(text)
            button.setObjectName("nav")
            button.setCheckable(True)
            self.navigation.addButton(button, index)
            self.nav_buttons.append(button)
            top.addWidget(button)
        self.navigation.idClicked.connect(self._navigate)
        top.addStretch()
        root.addLayout(top)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self._build_preview()
        self._build_calibration(sim, mode)
        self._build_results()
        self._build_tripod()
        self._build_usb()
        self.status_bar = QStatusBar()
        self.status_bar.setSizeGripEnabled(False)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 · 选择功能后启动设备")

    @staticmethod
    def _button(text, layout, slot, primary=False):
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(slot)
        layout.addWidget(button)
        return button

    def _build_preview(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.combo_camera = QComboBox()
        available = availability()
        for backend, (name, sdk) in BACKENDS.items():
            self.combo_camera.addItem(name + ("" if available[backend] else f"（缺少 {sdk}）"), backend)
            self.combo_camera.model().item(self.combo_camera.count() - 1).setEnabled(available[backend])
        if any(available.values()):
            self.combo_camera.setCurrentIndex(next(i for i, b in enumerate(BACKENDS) if available[b]))
        self.combo_device = QComboBox()
        self.combo_device.setEditable(True)
        self.combo_device.setMinimumWidth(170)
        self.combo_device.setPlaceholderText("序列号；空值自动选择")
        self.combo_profile = QComboBox()
        self.combo_profile.setEditable(True)
        self.chk_depth = QCheckBox("开启深度")
        self.chk_depth.setChecked(True)
        self.combo_view = QComboBox()
        self.combo_view.addItems(["彩色", "深度"])
        self.max_depth = QDoubleSpinBox()
        self.max_depth.setRange(0.1, 20.0)
        self.max_depth.setValue(4.0)
        self.max_depth.setSuffix(" m")
        for widget in (
            self.combo_camera,
            self.combo_device,
            self.combo_profile,
            self.chk_depth,
            self.combo_view,
            self.max_depth,
        ):
            controls.addWidget(widget)
        self.btn_scan = self._button("发现设备", controls, self.scan_devices)
        self.btn_preview_start = self._button("启动", controls, self.start_preview, True)
        self.btn_preview_stop = self._button("停止", controls, lambda: self.stop_task("preview"))
        layout.addLayout(controls)
        self.preview_info = QLabel("彩色与深度在同一预览区切换。流规格按实际设备能力协商。")
        self.preview_info.setWordWrap(True)
        layout.addWidget(self.preview_info)
        info_bar = QHBoxLayout()
        self.chk_crosshair = QCheckBox("中心十字线")
        self.chk_crosshair.setChecked(True)
        self.chk_crosshair.toggled.connect(self._render_preview)
        self.preview_distance = QLabel("中心距离 — m")
        info_bar.addWidget(self.chk_crosshair)
        info_bar.addWidget(self.preview_distance)
        info_bar.addStretch()
        layout.addLayout(info_bar)
        self.preview_image = ImagePreview("启动相机后显示画面")
        layout.addWidget(self.preview_image, 1)
        self.pages.addWidget(page)
        self.combo_camera.currentIndexChanged.connect(self._camera_changed)
        self.combo_view.currentIndexChanged.connect(self._render_preview)
        self.max_depth.valueChanged.connect(self._render_preview)
        self._camera_changed()

    def _camera_changed(self):
        backend = self.combo_camera.currentData()
        self.combo_profile.clear()
        self.combo_profile.addItems([f"{w}x{h}@{fps}" for w, h, fps in PROFILES[backend]])
        has_depth = backend != "uvc"
        self.chk_depth.setEnabled(has_depth)
        self.chk_depth.setChecked(has_depth)
        self.combo_view.model().item(1).setEnabled(has_depth)
        self.combo_view.setCurrentIndex(0)
        self.max_depth.setValue({"realsense": 4.0, "orbbec": 1.0, "uvc": 1.0, "sim": 1.0}.get(backend, 4.0))
        self.combo_device.clear()
        self.combo_device.setEditText("")

    def scan_devices(self):
        from camera_tool.devices.registry import discover

        def completed(payload):
            backend = self.combo_camera.currentData()
            self.combo_device.clear()
            self.combo_device.addItems([d["serial"] for d in payload["devices"] if d["kind"] == backend])
            self.preview_info.setText("；".join(payload["errors"]) or f"发现 {len(payload['devices'])} 个设备")

        self.run_task("scan", lambda context: discover(), completed)

    def preview_config(self):
        try:
            shape, rate = self.combo_profile.currentText().split("@")
            width, height = map(int, shape.lower().split("x"))
            fps = int(rate)
            if min(width, height, fps) <= 0:
                raise ValueError()
        except ValueError as exc:
            raise ValueError("流规格格式应为 1280x720@30") from exc
        return {
            "serial": self.combo_device.currentText().strip(),
            "width": width,
            "height": height,
            "fps": fps,
            "enable_depth": self.chk_depth.isChecked(),
        }

    def start_preview(self):
        from camera_tool.services.preview import preview

        try:
            backend, config = self.combo_camera.currentData(), self.preview_config()
        except ValueError as exc:
            self._status(str(exc))
            return
        self.run_task("preview", lambda context: preview(context, backend, config))

    def _build_calibration(self, sim, mode):
        cfg = load_yaml(config_path("eye_in_hand/default.yaml"))
        tool_cfg = cfg.get("robot", {}).get("tool") or cfg.get("robot", {}).get("gripper") or {}
        self._ip_customized = False
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("眼在手上 (Eye-in-Hand)", "eye-in-hand")
        self.combo_mode.addItem("眼在手外 (Eye-to-Hand)", "eye-to-hand")
        self.combo_mode.setCurrentIndex(0 if mode == "eye-in-hand" else 1)

        self.combo_eih_backend = QComboBox()
        self.combo_eih_backend.addItem("自动识别后端", "auto")
        self.combo_eih_backend.addItem("RealSense RGB-D", "realsense")
        self.combo_eih_backend.addItem("Orbbec RGB-D", "orbbec")
        if not sim:
            detected = detect_usb_camera_backend()
            idx = self.combo_eih_backend.findData(detected)
            self.combo_eih_backend.setCurrentIndex(idx if idx >= 0 else 0)

        self.combo_eih_preset = QComboBox()
        self.combo_eih_preset.addItem("自动模板", "auto")
        self.combo_eih_preset.addItem("近距模板 (default)", "default")
        self.combo_eih_preset.addItem("中距模板 (medium_range)", "medium_range")
        self.combo_eih_preset.addItem("Orbbec 模板 (orbbec)", "orbbec")

        self.chk_sim = QCheckBox("离线仿真")
        self.chk_sim.setChecked(sim)
        self.lbl_robot_ip = QLabel("机械臂 IP")
        self.cal_robot_ip = QLineEdit()
        self.cal_robot_ip.setMaximumWidth(165)
        self.cal_robot_ip.setPlaceholderText("例如 192.168.x.x / XARM_IP")
        self.cal_robot_ip.textEdited.connect(self._on_ip_edited)

        bar.addWidget(self.combo_mode)
        bar.addWidget(self.combo_eih_backend)
        bar.addWidget(self.combo_eih_preset)
        bar.addWidget(self.chk_sim)
        bar.addWidget(self.lbl_robot_ip)
        bar.addWidget(self.cal_robot_ip)
        bar.addStretch(1)
        self.btn_connect = self._button("连接设备", bar, self.start_calibration, True)
        self.btn_check = self._button("只读检查", bar, lambda: self.start_calibration(check=True))
        self.btn_cal_stop = self._button("停止并释放设备", bar, lambda: self.stop_task("calibration"))
        layout.addLayout(bar)

        cfg_bar = QHBoxLayout()
        cfg_bar.addWidget(QLabel("配置文件"))
        self.cal_config = QLineEdit(
            str(config_path("eye_in_hand/default.yaml" if mode == "eye-in-hand" else "eye_to_hand.yaml"))
        )
        cfg_bar.addWidget(self.cal_config, 1)
        layout.addLayout(cfg_bar)

        self.cal_stage = QLabel("眼在手上：连接设备后可手动采集、走预设点位或自动采集并求解。")
        self.cal_stage.setWordWrap(True)
        layout.addWidget(self.cal_stage)

        self.capture_views = QStackedWidget()
        self.video_panel = VideoPanel(
            cfg["marker"], tool_cfg, float(cfg["camera"].get("max_depth_m", 0.80))
        )
        self.capture_views.addWidget(self.video_panel)
        self.scene3d_panel = Scene3DPanel(
            float(cfg["marker"]["side_m"]), float(tool_cfg.get("length_m", 0.185))
        )
        self.capture_views.addWidget(self.scene3d_panel)
        dual = QSplitter(Qt.Orientation.Horizontal)
        self.wrist_image = ImagePreview("腕部相机 · 等待画面")
        self.fixed_image = ImagePreview("固定相机 · 等待画面")
        dual.addWidget(self.wrist_image)
        dual.addWidget(self.fixed_image)
        self.capture_views.addWidget(dual)
        layout.addWidget(self.capture_views, 1)

        actions = QHBoxLayout()
        self.btn_view = self._button(
            "相机 / 3D", actions, lambda: self.capture_views.setCurrentIndex(1 - self.capture_views.currentIndex())
        )
        self.btn_capture = self._button("采集样本（空格）", actions, lambda: self._command("capture"))
        self.btn_next = self._button("下一点位", actions, lambda: self._command("next"))
        self.btn_recenter = self._button("点位重定位", actions, lambda: self._command("recenter"))
        self.btn_auto = self._button("自动采集并求解", actions, lambda: self._command("auto"), True)
        self.btn_solve_capture = self._button("求解", actions, self.solve)
        self.btn_clear = self._button("清空样本", actions, lambda: self._command("clear"))
        layout.addLayout(actions)
        self.safety_status = QLabel("设备未连接 · 安全状态未知")
        self.safety_status.setWordWrap(True)
        layout.addWidget(self.safety_status)
        self.pages.addWidget(page)

        self.combo_mode.currentIndexChanged.connect(self._cal_mode_changed)
        self.combo_eih_backend.currentIndexChanged.connect(self._cal_mode_changed)
        self.combo_eih_preset.currentIndexChanged.connect(self._cal_mode_changed)
        self.chk_sim.toggled.connect(self._cal_mode_changed)
        self._cal_mode_changed()

    def _on_ip_edited(self, _text):
        self._ip_customized = True

    def _apply_eih_visual_cfg(self, cfg):
        cam_cfg = cfg.get("camera", {})
        marker_cfg = cfg.get("marker", {})
        robot_section = cfg.get("robot", {})
        tool_cfg = robot_section.get("tool") or robot_section.get("gripper") or {}
        checks_cfg = cfg.get("checks", {})
        self.video_panel.configure(
            marker_cfg,
            tool_cfg,
            max_depth_m=float(cam_cfg.get("max_depth_m", 0.80)),
        )
        default_board_z = float(checks_cfg.get("table_z_m", tool_cfg.get("table_z_m", 0.0))) + float(
            checks_cfg.get("board_thickness_m", marker_cfg.get("thickness_m", 0.0))
        )
        nominal = pose_from_mm_deg(
            cam_cfg.get("nominal_flange_cam_mm_deg", [45.0, 0.0, 62.0, 0.0, 0.0, 0.0])
        )
        self.scene3d_panel.configure(
            marker_side_m=float(marker_cfg.get("side_m", 0.15)),
            tool_length_m=float(tool_cfg.get("length_m", 0.185)),
            default_board_z_m=default_board_z,
            nominal_flange_cam=nominal,
        )

    def _cal_mode_changed(self):
        mode = self.combo_mode.currentData()
        if mode != "eye-in-hand":
            target_cfg = config_path("eye_to_hand.yaml")
        else:
            preset_sel = self.combo_eih_preset.currentData()
            backend_sel = self.combo_eih_backend.currentData()
            if self.chk_sim.isChecked() and preset_sel == "auto":
                target_cfg = config_path("eye_in_hand/default.yaml")
            else:
                target_cfg = resolve_preset_config(preset=preset_sel, backend=backend_sel)
        self.cal_config.setText(str(target_cfg))
        try:
            loaded = load_yaml(target_cfg)
            if not self._ip_customized:
                env_ip = os.environ.get("XARM_IP", "").strip()
                self.cal_robot_ip.setText(env_ip or str(loaded.get("robot", {}).get("ip", "")))
            if mode == "eye-in-hand":
                self._apply_eih_visual_cfg(loaded)
        except Exception:
            pass
        is_eih = mode == "eye-in-hand"
        self.combo_eih_backend.setVisible(is_eih)
        self.combo_eih_preset.setVisible(is_eih)
        self.combo_eih_backend.setEnabled(is_eih and not self.chk_sim.isChecked())
        self.combo_eih_preset.setEnabled(is_eih)
        self.chk_sim.setEnabled(is_eih)
        self.capture_views.setCurrentIndex(0 if is_eih else 2)
        self.btn_view.setVisible(is_eih)
        self.btn_connect.setText("连接设备" if is_eih else "开始完整标定")

    def start_calibration(self, check=False):
        if "calibration" in self.workers:
            return
        mode = self.combo_mode.currentData()
        self._selected_mode = mode
        ip_override = self.cal_robot_ip.text().strip() or None
        self._show_payload({"mode": mode, "samples": [], "solution": None})
        if mode == "eye-in-hand":
            backend_sel = self.combo_eih_backend.currentData()
            preset_sel = self.combo_eih_preset.currentData()
            backend = None if self.chk_sim.isChecked() or backend_sel == "auto" else backend_sel
            preset = None if preset_sel == "auto" else preset_sel
            try:
                self.backend = EyeInHand(
                    config=self.cal_config.text(),
                    sim=self.chk_sim.isChecked(),
                    ip=ip_override,
                    backend=backend,
                    preset=preset,
                )
                self._apply_eih_visual_cfg(self.backend.cfg)
            except Exception as exc:
                self._status(f"配置无效：{exc}")
                return
            operation = lambda context: self.backend.run(context, check=check)
        else:
            from camera_tool.services.eye_to_hand import calibrate

            config = self.cal_config.text()
            operation = lambda context: calibrate(context, config=config, check=check, ip=ip_override)
        self.run_task("calibration", operation, self._calibration_done)

    def _calibration_done(self, payload):
        if payload and payload.get("motion_commands_sent") is False:
            self._status(f"只读检查完成 · {payload.get('evaluation', {}).get('reason', '未发送运动命令')}")
        elif payload and payload.get("check"):
            self._status("只读检查完成 · 未发送运动命令")
        elif payload:
            self._show_payload(payload)
            self._navigate(2)

    def _capture_shortcut(self):
        focused = QApplication.focusWidget()
        if self.pages.currentIndex() == 1 and not isinstance(
            focused, (QLineEdit, QComboBox, QAbstractSpinBox, QTextEdit, QPlainTextEdit)
        ):
            self._command("capture")

    def _command(self, name, value=None):
        if (
            self.backend
            and "calibration" in self.workers
            and not self._command_busy
            and self._selected_mode == "eye-in-hand"
        ):
            self._command_busy = True
            self._sync_controls()
            self.backend.submit(name, value)

    def _build_results(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        bar = QHBoxLayout()
        self.replay_path = QLineEdit()
        self.replay_path.setPlaceholderText("历史会话或新会话目录")
        bar.addWidget(self.replay_path, 1)
        self.btn_replay = self._button("打开会话…", bar, self.browse_replay)
        layout.addLayout(bar)
        self.result_panel = CalibrationPanel()
        self.result_panel.solve_requested.connect(self.solve)
        self.result_panel.export_requested.connect(self.export_result)
        self.result_panel.sample_toggled.connect(self.toggle_sample)
        self.result_panel.sample_selected.connect(self.select_sample)
        layout.addWidget(self.result_panel, 1)
        self.pages.addWidget(page)

    def browse_replay(self):
        folder = self.replay_path.text().strip() or QFileDialog.getExistingDirectory(
            self, "选择历史会话", str(output_path("runs"))
        )
        if folder:
            self.load_replay_directory(folder)

    def load_replay_directory(self, directory):
        if "calibration" in self.workers:
            self._status("请先停止并释放标定设备，再打开历史会话")
            return
        self.replay_path.setText(str(directory))
        self.run_task("replay", lambda context: load_session(context, directory), self._replay_done)

    def _replay_done(self, payload):
        self._show_payload(payload)
        self._navigate(2)

    def _show_payload(self, payload):
        self.payload = payload
        panel = self.result_panel
        mode = payload["mode"]
        panel.set_mode(mode)
        samples, solution = payload.get("samples", []), payload.get("solution")
        if mode == "eye-to-hand":
            samples = [
                {
                    **s,
                    "name": f"{s.get('role', 'sample')} · {s.get('attempt', s['id'])}",
                    "pnp_rms_px": s.get("pnp_rms_px", 0),
                }
                for s in samples
            ]
            payload["samples"] = samples
        if solution:
            if mode == "eye-in-hand":
                panel.display_solution(solution, samples)
            else:
                panel.display_eye_to_hand(solution, samples, payload.get("replay", False))
        else:
            panel.invalidate_solution(samples)
        if payload.get("error"):
            panel.txt_report.setPlainText(payload["error"])
            self._status(payload["error"])
        if mode == "eye-in-hand":
            self.cal_stage.setText(
                f"有效样本 {sum(s.get('enabled', True) for s in samples)} / {len(samples)} · 点位 {payload.get('index', -1) + 1}"
            )
        row = panel.tbl_samples.currentRow()
        if 0 <= row < len(samples):
            self.select_sample(row)
        self._sync_controls()

    def solve(self):
        if self.payload and (self.payload.get("replay") or "calibration" not in self.workers):
            if self.payload["mode"] == "eye-to-hand":
                self.load_replay_directory(self.payload["directory"])
            else:
                payload = self.payload

                def operation(context):
                    context.check()
                    cfg = payload.get("cfg") or load_yaml(
                        config_for_camera_metadata(
                            payload["session"].camera_metadata,
                            robot_metadata=payload["session"].robot_metadata,
                        )
                    )
                    solution = solve_session(
                        payload["session"],
                        cfg,
                        tcp_offset_override_mm_rad=payload.get("tcp_offset_mm_rad"),
                    )
                    context.check()
                    return {**payload, "cfg": cfg, "solution": solution, "error": None}

                payload["solution"] = None
                self.result_panel.invalidate_solution(payload["samples"])
                self.run_task("replay", operation, self._replay_done)
        else:
            self._command("solve")

    def toggle_sample(self, row, enabled):
        if not self.payload or self.payload["mode"] != "eye-in-hand":
            return
        if self.payload.get("replay"):
            if "replay" in self.workers:
                return
            self.payload["session"].samples[row]["enabled"] = enabled
            self.payload["session"].last_solution = None
            self.payload["solution"] = None
            self._show_payload(self.payload)
        else:
            self._command("toggle", (row, enabled))

    def select_sample(self, row):
        if not self.payload or not 0 <= row < len(self.payload["samples"]):
            return
        if self.result_panel.tbl_samples.currentRow() != row:
            self.result_panel.tbl_samples.selectRow(row)
            return
        sample = self.payload["samples"][row]
        if sample.get("color_bgr") is None and self.payload["mode"] == "eye-to-hand" and sample.get("image"):
            folder = Path(self.payload["directory"]).resolve()
            path = (folder / sample["image"]).resolve()
            if path.is_relative_to(folder):
                import cv2

                sample = {**sample, "color_bgr": cv2.imread(str(path))}
        self.result_panel.show_sample(sample)

    def export_result(self):
        if not self.payload or self.payload.get("solution") is None:
            return
        mode = self.payload["mode"]
        default = output_path("result/eye_in_hand.yaml" if mode == "eye-in-hand" else "result/eye_to_hand.yaml")
        filename, _ = QFileDialog.getSaveFileName(self, "导出标定结果", str(default), "YAML (*.yaml)")
        if not filename:
            return
        if mode == "eye-in-hand" and not self.payload.get("replay") and "calibration" in self.workers:
            self._command("export", filename)
            return
        payload = self.payload
        if mode == "eye-to-hand" and payload.get("replay"):
            self._status("眼在手外回放仅供离线复核，不能发布外参")
            return

        def operation(context):
            context.check()
            if mode == "eye-in-hand":
                return str(
                    payload["session"].export_result(
                        payload["solution"],
                        filename,
                        tcp_offset_override_mm_rad=payload.get("tcp_offset_mm_rad"),
                        cfg=payload.get("cfg"),
                    )
                )
            from camera_tool.calibration.eye_to_hand.workflow import publish_camera

            return publish_camera(filename, payload["solution"], payload["session"])

        self.run_task("export", operation, lambda path: self._status(f"已导出 {path}"))

    def _build_tripod(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.tripod_config = QLineEdit(str(config_path("eye_to_hand.yaml")))
        self.tripod_result = QLineEdit(str(output_path("result/eye_to_hand.yaml")))
        self.tripod_baseline = QLineEdit(str(output_path("result/scene_lock")))
        form.addRow("相机与阈值配置", self.tripod_config)
        form.addRow("眼在手外结果", self.tripod_result)
        form.addRow("基线目录", self.tripod_baseline)
        layout.addLayout(form)
        bar = QHBoxLayout()
        self.btn_lock = self._button("锁定基线", bar, lambda: self.start_tripod("lock"))
        self.btn_watch = self._button("开始监测", bar, lambda: self.start_tripod("watch"), True)
        self.btn_tripod_stop = self._button("停止", bar, lambda: self.stop_task("tripod"))
        layout.addLayout(bar)
        self.tripod_metric = QLabel("桌面 p95 位移 — mm · 跟踪状态 — · 阈值按配置读取")
        layout.addWidget(self.tripod_metric)
        self.tripod_image = ImagePreview("先在标定姿态锁定基线，再监测三脚架是否移动")
        layout.addWidget(self.tripod_image, 1)
        self.pages.addWidget(page)

    def start_tripod(self, action):
        from camera_tool.services.tripod import tripod

        args = (self.tripod_config.text(), self.tripod_result.text(), self.tripod_baseline.text())
        self.run_task(
            "tripod",
            lambda context: tripod(context, action, config=args[0], result=args[1], baseline=args[2]),
            lambda payload: self._status(
                f"基线已保存：{payload['baseline']}" if payload and "baseline" in payload else "监测结束"
            ),
        )

    def _build_usb(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.usb_kind = QComboBox()
        self.usb_kind.addItems(["realsense", "orbbec"])
        self.usb_serial = QLineEdit()
        self.usb_profile = QLineEdit("1280x720@30")
        self.usb_duration = QSpinBox()
        self.usb_duration.setRange(1, 86400)
        self.usb_duration.setValue(1800)
        self.usb_interval = QSpinBox()
        self.usb_interval.setRange(1, 600)
        self.usb_interval.setValue(5)
        self.usb_min_fps = QDoubleSpinBox()
        self.usb_min_fps.setRange(0.1, 200)
        self.usb_min_fps.setValue(29)
        self.usb_max_gap = QDoubleSpinBox()
        self.usb_max_gap.setDecimals(3)
        self.usb_max_gap.setRange(0.001, 10)
        self.usb_max_gap.setValue(0.1)
        for title, widget in (
            ("相机后端", self.usb_kind),
            ("序列号（空值自动选择）", self.usb_serial),
            ("彩色流规格", self.usb_profile),
            ("测试时长（秒）", self.usb_duration),
            ("统计间隔（秒）", self.usb_interval),
            ("最低 FPS", self.usb_min_fps),
            ("最大间隔（秒）", self.usb_max_gap),
        ):
            form.addRow(title, widget)
        layout.addLayout(form)
        bar = QHBoxLayout()
        self.btn_usb_start = self._button("开始测试", bar, self.start_usb, True)
        self.btn_usb_stop = self._button("停止", bar, lambda: self.stop_task("usb"))
        self.btn_usb_export = self._button("导出 CSV", bar, self.export_usb)
        layout.addLayout(bar)
        self.usb_metric = QLabel("FPS — · 丢帧 — · 最大间隔 — ms · USB — Mbps")
        layout.addWidget(self.usb_metric)
        self.usb_log = QPlainTextEdit()
        self.usb_log.setReadOnly(True)
        self.usb_log.setMaximumBlockCount(2000)
        layout.addWidget(self.usb_log, 1)
        self.pages.addWidget(page)

    def start_usb(self):
        from camera_tool.services.usb import usb_test

        try:
            shape, rate = self.usb_profile.text().split("@")
            width, height = map(int, shape.lower().split("x"))
            config = {
                "backend": self.usb_kind.currentText(),
                "serial": self.usb_serial.text().strip(),
                "width": width,
                "height": height,
                "fps": int(rate),
            }
        except ValueError:
            self._status("流规格格式应为 1280x720@30")
            return
        self.usb_rows = []
        self.usb_log.clear()
        duration = self.usb_duration.value()
        interval = self.usb_interval.value()
        min_fps = self.usb_min_fps.value()
        max_gap = self.usb_max_gap.value()

        def completed(payload):
            self.usb_rows = payload["rows"]
            self._usb_update(payload["summary"], final=True)

        self.run_task("usb", lambda context: usb_test(context, config, duration, interval, min_fps, max_gap), completed)

    def export_usb(self):
        from camera_tool.services.usb import export_csv

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出 USB 测试 CSV", str(output_path("usb_stability.csv")), "CSV (*.csv)"
        )
        if filename:
            rows = list(self.usb_rows)
            self.run_task(
                "export",
                lambda context: str(export_csv(rows, filename)),
                lambda path: self._status(f"已导出 {path}"),
            )

    def run_task(self, name, operation, completed=None):
        if name in self.workers:
            return
        worker = TaskWorker(operation, self)
        self.workers[name] = worker
        worker.update.connect(lambda kind, value: self._task_update(name, kind, value))
        worker.failed.connect(lambda message: self._status(f"{name}：{message}"))
        if completed:
            worker.completed.connect(completed)
        worker.finished.connect(lambda: self._task_finished(name, worker))
        self._sync_controls()
        worker.start()

    def stop_task(self, name):
        if name in self.workers:
            self.workers[name].stop()
            self._status("正在停止任务并释放设备…")

    def _task_finished(self, name, worker):
        if self.workers.get(name) is worker:
            del self.workers[name]
        if name == "calibration":
            self._command_busy = False
            self.safety_status.setText("标定设备已释放 · 当前安全状态未知")
        worker.deleteLater()
        self._sync_controls()
        if self._closing and not self.workers:
            self.close()

    def _task_update(self, name, kind, value):
        if kind == "status":
            self._status(str(value))
        elif kind == "metadata":
            if name == "preview":
                import json

                self._last_metadata = value
                self.preview_info.setText(
                    f"{value.get('name', '')} · SN {value.get('serial')} · 实际 {value['width']}×{value['height']}@{value.get('fps')} · SDK 深度单位 {value.get('sdk_depth_unit_m')} m"
                )
                self.preview_info.setToolTip(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        elif kind in ("snapshot", "connected"):
            self._show_payload(value)
        elif kind == "busy":
            self._command_busy = value
            if not value and self.payload and self.payload.get("solution"):
                self._navigate(2)
            self._sync_controls()
        elif kind == "progress":
            self.cal_stage.setText(
                f"{value['stage']} · 有效腕部样本 {len(value['samples'])}/30 · 固定相机 {value['fixed_count']} 帧"
            )
            fields = value.get("fields", {})
            if "report" in fields:
                self._status(str(fields["report"]))
        elif kind == "usb":
            self.usb_rows.append(value)
            self._usb_update(value)
        elif kind == "usb_summary":
            self._usb_update(value, final=True)
        elif kind == "tripod":
            self.tripod_metric.setText(
                f"桌面 p95 位移 {value.get('table_p95_mm')} mm · {value['confirmed']} · 内点 {value.get('n_inliers')} · 警告 {value['warn_mm']} / 失败 {value['fail_mm']} mm"
            )

    def _usb_update(self, row, final=False):
        text = (
            f"FPS {row['fps']:.2f} · 丢帧 {row['missing_frame_numbers']} · "
            f"最大间隔 {row['max_gap_s'] * 1000:.1f} ms · USB {row['usb_speed_mbps']} Mbps ({row['usb_descriptor']})"
        )
        self.usb_metric.setText(text)
        self.usb_log.appendPlainText(("汇总 · " if final else "") + text)
        if final:
            self.usb_log.appendPlainText(
                "已停止" if row.get("stopped") else "测试稳定" if row["stable"] else "存在不稳定或链路降级迹象"
            )

    def _consume_frames(self):
        for worker in list(self.workers.values()):
            for name, packet in worker.context.take_frames().items():
                if name == "preview":
                    self._last_preview = packet
                    self._render_preview()
                elif name == "calibration":
                    frame, evaluation, state = packet["frame"], packet["evaluation"], packet["state"]
                    self.video_panel.update_frame(packet["annotated"], frame.get("depth_m"), evaluation, state["safety"])
                    self.safety_status.setText(
                        f"末端离桌 {state['safety'].get('table_clearance_mm', 0):.1f} mm · {state['safety'].get('message', '未知')}"
                    )
                    if self.capture_views.currentIndex() == 1:
                        solution = (self.payload or {}).get("solution") or {}
                        self.scene3d_panel.update_scene(
                            x_base_flange=state["X_base_flange"],
                            x_flange_cam=solution.get("X_flange_cam"),
                            x_base_board=solution.get("X_base_board"),
                            waypoints=(self.payload or {}).get("waypoints", []),
                            captured_samples=(self.payload or {}).get("samples", []),
                            points_cam=frame.get("points_cam"),
                            colors_rgb=frame.get("colors_rgb"),
                        )
                elif name == "wrist":
                    self.wrist_image.set_image(packet["color"])
                elif name == "fixed":
                    self.fixed_image.set_image(packet["color"])
                elif name == "tripod":
                    self.tripod_image.set_image(packet["color_bgr"])

    def _render_preview(self):
        if self._last_preview is None:
            return
        frame = self._last_preview
        if self.combo_view.currentIndex() == 1:
            if frame.get("depth_m") is None:
                self.preview_image.reset("当前流没有深度数据")
                return
            image = colorize_depth(frame["depth_m"], self.max_depth.value())
        else:
            image = frame["color_bgr"]
        depth = frame.get("depth_m")
        if depth is not None:
            h, w = depth.shape[:2]
            distance = float(depth[h // 2, w // 2])
            self.preview_distance.setText(f"中心距离 {distance:.3f} m" if distance > 0 else "中心距离 — m")
        else:
            self.preview_distance.setText("当前流无深度")
        if self.chk_crosshair.isChecked():
            import cv2

            image = image.copy()
            h, w = image.shape[:2]
            cv2.drawMarker(image, (w // 2, h // 2), (80, 220, 80), cv2.MARKER_CROSS, 32, 1)
        self.preview_image.set_image(image)

    def _sync_controls(self):
        if not hasattr(self, "btn_usb_start"):
            return
        calibration = "calibration" in self.workers
        for widget in (
            self.combo_mode,
            self.combo_eih_backend,
            self.combo_eih_preset,
            self.chk_sim,
            self.cal_robot_ip,
            self.cal_config,
            self.btn_connect,
            self.btn_check,
        ):
            widget.setEnabled(not calibration)
        is_eih = self.combo_mode.currentData() == "eye-in-hand"
        self.chk_sim.setEnabled(not calibration and is_eih)
        self.combo_eih_backend.setEnabled(not calibration and is_eih and not self.chk_sim.isChecked())
        self.combo_eih_preset.setEnabled(not calibration and is_eih)
        self.btn_cal_stop.setEnabled(calibration)
        live = (
            calibration
            and self._selected_mode == "eye-in-hand"
            and self.payload is not None
            and self.payload.get("session") is not None
            and not self.payload.get("replay")
        )
        for widget in (self.btn_capture, self.btn_next, self.btn_auto, self.btn_recenter, self.btn_clear):
            widget.setEnabled(live and not self._command_busy)
        active = sum(s.get("enabled", True) for s in (self.payload or {}).get("samples", []))
        replay_busy = "replay" in self.workers or "export" in self.workers
        mode = (self.payload or {}).get("mode")
        can_solve = (
            self.payload is not None
            and not self._command_busy
            and not replay_busy
            and (live or self.payload.get("replay") or mode == "eye-in-hand")
            and (active >= 5 if mode == "eye-in-hand" else True)
        )
        self.btn_solve_capture.setEnabled(can_solve)
        self.result_panel.btn_solve.setEnabled(can_solve)
        self.result_panel.tbl_samples.setEnabled(not self._command_busy and not replay_busy)
        self.result_panel.btn_export.setEnabled(
            bool((self.payload or {}).get("solution"))
            and not self._command_busy
            and not replay_busy
            and not (mode == "eye-to-hand" and self.payload.get("replay"))
        )
        self.btn_replay.setEnabled(not calibration and not replay_busy)
        for name, start, stop in (
            ("preview", self.btn_preview_start, self.btn_preview_stop),
            ("usb", self.btn_usb_start, self.btn_usb_stop),
        ):
            start.setEnabled(name not in self.workers)
            stop.setEnabled(name in self.workers)
        for widget in (self.combo_camera, self.combo_device, self.combo_profile, self.chk_depth, self.btn_scan):
            widget.setEnabled("preview" not in self.workers and "scan" not in self.workers)
        self.chk_depth.setEnabled("preview" not in self.workers and self.combo_camera.currentData() != "uvc")
        for widget in (
            self.usb_kind,
            self.usb_serial,
            self.usb_profile,
            self.usb_duration,
            self.usb_interval,
            self.usb_min_fps,
            self.usb_max_gap,
        ):
            widget.setEnabled("usb" not in self.workers)
        for widget in (self.tripod_config, self.tripod_result, self.tripod_baseline):
            widget.setEnabled("tripod" not in self.workers)
        self.btn_lock.setEnabled("tripod" not in self.workers)
        self.btn_watch.setEnabled("tripod" not in self.workers)
        self.btn_tripod_stop.setEnabled("tripod" in self.workers)
        self.btn_usb_export.setEnabled(
            bool(self.usb_rows) and "usb" not in self.workers and "export" not in self.workers
        )

    def _navigate(self, index):
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)

    def _status(self, message):
        self.status_bar.showMessage(message)
        self.status_bar.setToolTip(message)

    def closeEvent(self, event):
        if self.workers:
            self._closing = True
            for worker in list(self.workers.values()):
                worker.stop()
            self._status("正在停止任务，释放设备后关闭…")
            event.ignore()
            return
        self.frame_timer.stop()
        super().closeEvent(event)
