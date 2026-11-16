"""Tests for ArUco #398 quality gates."""
from __future__ import annotations

import cv2
import numpy as np

from camera_tool.core import CameraModel, MarkerDetector, load_yaml
from camera_tool.paths import config_path


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

    blank = np.full((720, 1280, 3), 120, dtype=np.uint8)
    res_blank = detector.evaluate(blank)
    assert res_blank["passed"] is False
    assert res_blank["detected"] is False

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    marker = cv2.aruco.generateImageMarker(dictionary, 398, 220)
    sharp_img = np.full((720, 1280, 3), 240, dtype=np.uint8)
    sharp_img[250:470, 530:750] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    res_sharp = detector.evaluate(sharp_img)
    assert res_sharp["passed"] is True
    assert res_sharp["area_px"] >= 1200.0
    assert res_sharp["sharpness"] >= 30.0
    assert res_sharp["pnp_rms_px"] <= 1.5

    blurred_img = cv2.GaussianBlur(sharp_img, (35, 35), 12.0)
    res_blur = detector.evaluate(blurred_img)
    assert res_blur["passed"] is False
    assert "Sharpness" in res_blur["reason"] or not res_blur["detected"]

