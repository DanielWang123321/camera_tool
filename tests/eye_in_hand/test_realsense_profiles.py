"""Stream negotiation and resource cleanup regressions for RealSense backend."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from camera_tool.core import CalibrationError
from camera_tool.devices.realsense import RealSenseCamera


@pytest.fixture
def fake_rs(monkeypatch):
    class Video:
        def __init__(self, kind, width, height):
            self.kind, self.w, self.h = kind, width, height

        def stream_type(self): return self.kind
        def format(self): return 'bgr8' if self.kind == 'color' else 'z16'
        def is_video_stream_profile(self): return True
        def as_video_stream_profile(self): return self
        def width(self): return self.w
        def height(self): return self.h
        def fps(self): return 30
        def get_intrinsics(self):
            return SimpleNamespace(width=self.w, height=self.h, fx=400, fy=400, ppx=self.w/2, ppy=self.h/2, coeffs=[0]*5, model='distortion.none')
        def get_extrinsics_to(self, other):
            return SimpleNamespace(rotation=np.eye(3).ravel().tolist(), translation=[0., 0., 0.])

    streams = [Video('color', 1280, 720), Video('color', 640, 480), Video('depth', 640, 480)]
    sensor = SimpleNamespace(get_stream_profiles=lambda: streams, get_depth_scale=lambda: 0.0001)
    device = SimpleNamespace(get_info=lambda key: {'serial': 'MOCK-RS-001', 'name': 'RealSense RGB-D'}[key], query_sensors=lambda: [sensor], first_depth_sensor=lambda: sensor)
    pipeline = SimpleNamespace(started=False, stopped=False, fail_warmup=False, selected=None)
    profile = SimpleNamespace(get_device=lambda: device, get_stream=lambda _: streams[1])

    def start(config):
        assert config.streams == [('color', 640, 480, 'bgr8', 30), ('depth', 640, 480, 'z16', 30)]
        pipeline.started, pipeline.selected = True, config.streams
        return profile

    def stop(): pipeline.stopped = True
    def wait_for_frames(_timeout):
        if pipeline.fail_warmup:
            raise RuntimeError('warmup timeout')

    pipeline.start, pipeline.stop, pipeline.wait_for_frames = start, stop, wait_for_frames

    class Config:
        def __init__(self): self.streams = []
        def enable_device(self, _serial): pass
        def enable_stream(self, *args): self.streams.append(args)
        def can_resolve(self, _wrapper): return True

    module = SimpleNamespace(
        context=lambda: SimpleNamespace(query_devices=lambda: [device]), pipeline=lambda: pipeline,
        pipeline_wrapper=lambda obj: obj, config=Config, align=lambda _: None,
        stream=SimpleNamespace(color='color', depth='depth'), format=SimpleNamespace(bgr8='bgr8', z16='z16'),
        camera_info=SimpleNamespace(serial_number='serial', name='name'),
    )
    monkeypatch.setitem(sys.modules, 'pyrealsense2', module)
    return pipeline


def test_unsupported_requested_resolution_uses_device_profile_and_correct_pointcloud_size(fake_rs):
    camera = RealSenseCamera({'width': 1280, 'height': 720, 'fps': 30})
    metadata = camera.start()
    assert (camera.width, camera.height, camera.fps) == (640, 480, 30)
    assert metadata['requested_stream'] == [1280, 720, 30]
    assert metadata['width'] == 640 and metadata['height'] == 480
    points, colors = camera._deproject_subsampled(np.zeros((480, 640, 3), np.uint8), np.full((480, 640), 0.3, np.float32))
    assert len(points) == len(range(0, 480, 6)) * len(range(0, 640, 6))
    assert points.shape == colors.shape
    camera.close()
    assert fake_rs.stopped and not camera.connected


def test_warmup_failure_stops_pipeline_even_before_connected_flag(fake_rs):
    fake_rs.fail_warmup = True
    camera = RealSenseCamera({'width': 1280, 'height': 720, 'fps': 30})
    with pytest.raises(RuntimeError, match='warmup timeout'):
        camera.start()
    assert fake_rs.started and fake_rs.stopped
    assert not camera.connected


def test_missing_selected_serial_does_not_start_another_device(fake_rs):
    camera = RealSenseCamera({'serial': 'missing'})
    with pytest.raises(CalibrationError, match='指定'):
        camera.start()
    assert not fake_rs.started
