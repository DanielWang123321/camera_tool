"""Cooperative cancellation, bounded frame delivery and process-wide device leases."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import time


class Cancelled(RuntimeError):
    pass


_CURRENT_CONTEXT = ContextVar("camera_tool_task", default=None)


@contextmanager
def task_scope(context):
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


def check_cancelled():
    context = _CURRENT_CONTEXT.get()
    if context:
        context.check()


class TaskContext:
    def __init__(self, emit=None):
        self.cancelled = threading.Event()
        self.emit = emit or (lambda kind, data: None)
        self._frames = {}
        self._frames_lock = threading.Lock()

    def check(self):
        if self.cancelled.is_set():
            raise Cancelled("任务已停止")

    def sleep(self, seconds):
        self.cancelled.wait(seconds)
        self.check()

    def frame(self, name, data):
        # A slow or hidden desktop consumes only the latest frame, without
        # retaining an unbounded queue of full-resolution images in Qt signals.
        with self._frames_lock:
            self._frames[name] = data

    def take_frames(self):
        with self._frames_lock:
            frames, self._frames = self._frames, {}
        return frames


class DeviceManager:
    """Atomic leases shared by all pages and by separate CLI processes.

    A family wildcard conflicts with any serial in that family. Multi-device
    calibration reserves all its devices together before opening any SDK.
    """
    def __init__(self, directory=None):
        self.directory = Path(directory or Path(tempfile.gettempdir()) / f"camera-tool-{os.getuid()}")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._mutex = threading.RLock()
        self._held = []

    @contextmanager
    def lease(self, resources, owner):
        handles = []
        acquired = False
        keys = sorted(set(resources))
        try:
            with self._mutex:
                for key in keys:
                    family, serial = key.split(":", 1)
                    for held, held_owner in self._held:
                        hf, hs = held.split(":", 1)
                        if hf == family and (serial == "*" or hs == "*" or serial == hs):
                            raise RuntimeError(f"设备 {key} 正由 {held_owner} 使用")
                for key in keys:
                    family, serial = key.split(":", 1)
                    family_file = (self.directory / f"{family}.lock").open("a")
                    handles.append(family_file)
                    fcntl.flock(family_file, (fcntl.LOCK_EX if serial == "*" else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                    if serial != "*":
                        digest = hashlib.sha256(key.encode()).hexdigest()
                        serial_file = (self.directory / f"{digest}.lock").open("a")
                        handles.append(serial_file)
                        fcntl.flock(serial_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._held.extend((key, owner) for key in keys)
                acquired = True
            yield
        except BlockingIOError as exc:
            raise RuntimeError("所选设备正由另一个 camera-tool 任务使用") from exc
        finally:
            with self._mutex:
                # Only remove entries acquired by this invocation.
                for key in keys:
                    if acquired and (key, owner) in self._held:
                        self._held.remove((key, owner))
                for handle in reversed(handles):
                    handle.close()


DEVICES = DeviceManager()


def resource(family, serial=""):
    return f"{family}:{serial or '*'}"
