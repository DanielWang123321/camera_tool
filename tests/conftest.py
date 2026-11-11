import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from camera_tool.ui.qt_runtime import prepare_qt

prepare_qt()
