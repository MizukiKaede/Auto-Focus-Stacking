"""Run one background read at a time and deliver only the latest request."""

from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, Signal, Slot


class _Result(QObject):
    finished = Signal(int, object, str)


class _Read(QRunnable):
    def __init__(self, generation, function, args, cancelled, result):
        super().__init__()
        self.request = generation, function, args, cancelled, result

    def run(self):
        generation, function, args, cancelled, result = self.request
        value, error = None, ""
        try:
            value = function(*args, cancelled=cancelled)
        except Exception as exc:
            error = str(exc)
        try:
            result.finished.emit(generation, value, error)
        except RuntimeError:
            # Qt may destroy signal objects during QApplication shutdown.
            from shiboken6 import isValid
            if isValid(result):
                raise


class LatestRead(QObject):
    ready = Signal(object, str)

    def __init__(self, parent):
        super().__init__(parent)
        self._generation = 0
        self._active = None
        self._pending = None
        self._result = _Result()
        self._result.finished.connect(self._finished, Qt.QueuedConnection)

    def invalidate(self):
        self._generation += 1
        self._pending = None
        if self._active is not None:
            self._active.set()

    def submit(self, function, *args):
        self.invalidate()
        self._pending = function, args
        if self._active is None:
            self._start()

    def _start(self):
        function, args = self._pending
        self._pending = None
        self._active = Event()
        QThreadPool.globalInstance().start(
            _Read(self._generation, function, args, self._active, self._result)
        )

    @Slot(int, object, str)
    def _finished(self, generation, value, error):
        self._active = None
        if generation == self._generation:
            self.ready.emit(value, error)
        if self._pending is not None and self._active is None:
            self._start()
