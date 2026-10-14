from threading import Event

from PySide6.QtCore import QCoreApplication

from openkalolumen.desktop.tasks import TaskManager


def test_cancelled_result_cannot_replace_latest():
    app = QCoreApplication.instance() or QCoreApplication([])
    manager = TaskManager()
    gate = Event()
    entered = Event()
    received = []

    def old(progress, cancelled):
        entered.set()
        gate.wait(2)
        return "old"

    manager.submit("geometry", old, received.append)
    assert entered.wait(2)
    manager.submit("geometry", lambda p, c: "latest", received.append)
    gate.set()
    assert manager.pool.waitForDone(5000)
    app.processEvents()
    assert received == ["latest"]
