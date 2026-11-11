"""Tests for ArUco #398 quality gates and headless PySide6 GUI end-to-end calibration."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from camera_tool.core import CameraModel, MarkerDetector, load_yaml
from camera_tool.paths import config_path
from camera_tool.ui.main_window import MainWindow


def test_aruco_398_detector_quality_gates() -> None:
    cfg = load_yaml(config_path("eye_in_hand/default.yaml"))
    cam_meta = {
        "width": 1280,
        "height": 720,
        "model": "pinhole",
        "K": [[845.0, 0.0, 640.0], [0.0, 845.0, 360.0], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    detector = MarkerDetector(cfg["marker"], CameraModel(cam_meta))

    # 1. Blank frame fails detection
    blank = np.full((720, 1280, 3), 120, dtype=np.uint8)
    res_blank = detector.evaluate(blank)
    assert res_blank["passed"] is False
    assert res_blank["detected"] is False

    # 2. Sharp valid marker #398 passes all gates
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    marker = cv2.aruco.generateImageMarker(dictionary, 398, 220)
    sharp_img = np.full((720, 1280, 3), 240, dtype=np.uint8)
    sharp_img[250:470, 530:750] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    res_sharp = detector.evaluate(sharp_img)
    assert res_sharp["passed"] is True
    assert res_sharp["area_px"] >= 1200.0
    assert res_sharp["sharpness"] >= 30.0
    assert res_sharp["pnp_rms_px"] <= 1.5

    # 3. Heavily blurred marker fails sharpness gate
    blurred_img = cv2.GaussianBlur(sharp_img, (35, 35), 12.0)
    res_blur = detector.evaluate(blurred_img)
    assert res_blur["passed"] is False
    assert "清晰度不足" in res_blur["reason"] or not res_blur["detected"]

