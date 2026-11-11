"""Initialize PySide6 after removing only OpenCV's injected Qt5 paths."""
import os


def prepare_qt():
    # cv2 injects these on import, so import it before cleaning the environment.
    import cv2  # noqa: F401
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH"):
        value = os.environ.get(key, "")
        if "/cv2/" in value.replace("\\", "/"):
            os.environ.pop(key, None)
    if os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
