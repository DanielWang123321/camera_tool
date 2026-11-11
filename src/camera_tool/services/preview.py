"""Single-stream preview acquisition, independent of display frequency."""
from camera_tool.devices.registry import create_camera, device_resource
from camera_tool.services.tasks import DEVICES


def preview(context, kind, config):
    camera = create_camera(kind, config)
    with DEVICES.lease([device_resource(kind, config.get("serial", ""))], "相机预览"):
        try:
            context.check()
            metadata = dict(camera.start())
            context.emit("metadata", metadata)
            while True:
                context.check()
                frame = camera.read()
                actual = frame.get("metadata", metadata)
                if actual != metadata:
                    metadata = dict(actual)
                    context.emit("metadata", metadata)
                context.frame("preview", frame)
        finally:
            camera.stop()
