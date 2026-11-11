"""Regression for the observed PySide6 6.12 / CPython 3.11 fatal refcount bug."""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtWidgets import QApplication, QLabel


@pytest.mark.skipif(sys.version_info >= (3, 12), reason='CPython 3.12 makes None immortal')
def test_qt_void_calls_do_not_consume_none_references():
    app = QApplication.instance() or QApplication([])
    label = QLabel('runtime probe')
    label.setEnabled(True)
    before = sys.getrefcount(None)
    for _ in range(500):
        label.setEnabled(True)
    after = sys.getrefcount(None)
    assert after >= before - 10, 'Qt binding consumes None references; install PySide6 < 6.12'
    label.deleteLater()
    app.processEvents()
