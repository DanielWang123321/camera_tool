"""Qt adapter; domain services never import Qt."""
from PySide6.QtCore import QThread, Signal

from camera_tool.services.tasks import Cancelled, TaskContext, task_scope


class TaskWorker(QThread):
    update = Signal(str, object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.context = TaskContext(self.update.emit)

    def stop(self):
        self.context.cancelled.set()

    def run(self):
        try:
            with task_scope(self.context):
                result = self.operation(self.context)
            self.completed.emit(result)
        except Cancelled:
            self.update.emit("status", "任务已停止，设备已释放")
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
