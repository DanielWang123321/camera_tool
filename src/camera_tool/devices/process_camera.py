"""Isolate SDKs that may exit natively or block inside a device read."""
from __future__ import annotations

import importlib
import multiprocessing as mp
import queue
import time


def _run_driver(specification, config, packets, stop, ready, cleaning_up):
    driver = None
    try:
        module, name = specification.split(":")
        driver = getattr(importlib.import_module(module), name)(config)
        packets.put(("metadata", driver.start()))
        if not ready.wait(10):
            return
        while not stop.is_set():
            frame = driver.read()
            try:
                packets.put_nowait(("frame", frame))
            except queue.Full:
                # Old UI frames can be discarded; acquisition and SDK errors
                # remain independent of the main process's rendering rate.
                try:
                    packets.get_nowait()
                    packets.put_nowait(("frame", frame))
                except (queue.Empty, queue.Full):
                    pass
    except BaseException as exc:
        try:
            while True:
                packets.get_nowait()
        except queue.Empty:
            pass
        try:
            packets.put(("error", str(exc)), timeout=.5)
        except queue.Full:
            pass
    finally:
        cleaning_up.set()
        if driver is not None:
            driver.stop()


class ProcessCamera:
    def __init__(self, specification, config):
        self.specification, self.config = specification, config
        self.process = None
        self.metadata = {}
        self.connected = False

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.packets.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(f"相机 SDK 进程退出（{self.process.exitcode}）；请检查设备、SDK 和系统权限")
        raise RuntimeError("相机 SDK 响应超时")

    def start(self):
        context = mp.get_context("spawn")
        self.packets, self.stop_event, self.ready, self.cleaning_up = context.Queue(2), context.Event(), context.Event(), context.Event()
        self.process = context.Process(target=_run_driver, args=(self.specification, self.config, self.packets, self.stop_event, self.ready, self.cleaning_up), daemon=True)
        self.process.start()
        try:
            kind, value = self._receive(10)
            if kind != "metadata":
                raise RuntimeError(value)
            self.metadata = value
            self.connected = True
            self.ready.set()
            return value
        except BaseException:
            self.stop()
            raise

    def read(self, fresh=False, **kwargs):
        if fresh and self.process is not None:
            try:
                while True:
                    kind, value = self.packets.get_nowait()
                    if kind != "frame":
                        raise RuntimeError(value)
            except queue.Empty:
                pass
        kind, value = self._receive(3)
        if kind != "frame":
            raise RuntimeError(value)
        self.metadata = value.get("metadata", self.metadata)
        return value

    def stop(self):
        if self.process is not None:
            self.stop_event.set()
            self.ready.set()
            self.process.join(.4)
            if self.process.is_alive() and self.cleaning_up.is_set():
                self.process.join(3.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1)
            self.packets.cancel_join_thread()
            self.packets.close()
            self.process = None
        self.connected = False

    close = stop
