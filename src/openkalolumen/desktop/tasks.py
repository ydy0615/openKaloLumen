from __future__ import annotations

import traceback
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from openkalolumen.errors import Cancelled


class TaskSignals(QObject):
    """Deliver task results, errors, progress, and completion through Qt."""

    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)
    finished = Signal()


class Task(QRunnable):
    """Execute a cancellable computation on the Qt thread pool."""

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = TaskSignals()
        self.cancel_event = Event()

    def cancel(self):
        """Invalidate the requested computation and signal cooperative cancellation."""
        self.cancel_event.set()

    def run(self):
        """Execute the task and emit signals only while its generation is active."""
        try:
            if self.cancel_event.is_set():
                return
            result = self.function(self.signals.progress.emit, self.cancel_event.is_set)
            if not self.cancel_event.is_set():
                self.signals.completed.emit(result)
        except Cancelled:
            pass
        except Exception:
            if not self.cancel_event.is_set():
                self.signals.failed.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()


class TaskManager(QObject):
    """Run background tasks and reject results from obsolete generations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.active = {}
        self.versions = {}

    def submit(self, key, function, completed, failed=None, progress=None):
        """Replace a keyed task and deliver only results from its latest generation."""
        self.cancel(key)
        version = self.versions.get(key, 0) + 1
        self.versions[key] = version
        task = Task(function)
        self.active[key] = task

        def deliver(result):
            if self.versions.get(key) == version:
                completed(result)

        task.signals.completed.connect(deliver)
        if failed:
            task.signals.failed.connect(
                lambda e: failed(e) if self.versions.get(key) == version else None
            )
        if progress:
            task.signals.progress.connect(
                lambda n, s: (
                    progress(n, s) if self.versions.get(key) == version else None
                )
            )
        task.signals.finished.connect(
            lambda: self.active.pop(key, None) if self.active.get(key) is task else None
        )
        self.pool.start(task)
        return task

    def cancel(self, key):
        """Invalidate the requested computation and signal cooperative cancellation."""
        if key in self.active:
            self.active[key].cancel()
        self.versions[key] = self.versions.get(key, 0) + 1

    def cancel_all(self):
        """Invalidate every active task without discarding completed results."""
        for key in list(self.active):
            self.cancel(key)
