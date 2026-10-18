from __future__ import annotations

from collections.abc import Callable
import logging
from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

logger = logging.getLogger(__name__)


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)
    finished = Signal()


class TaskRunner(QRunnable):
    def __init__(self, fn: Callable, parent: QObject | None = None):
        super().__init__()
        self.signals = TaskSignals(parent)
        self._fn = fn
        self._cancelled = Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @Slot()
    def run(self) -> None:
        try:
            if not self.is_cancelled():
                result = self._fn(self._progress, self.is_cancelled)
                if not self.is_cancelled():
                    self.signals.result.emit(result)
        except Exception as exc:
            if not self.is_cancelled():
                logger.exception("Ошибка фоновой обработки")
                message = "Недостаточно оперативной памяти." if isinstance(exc, MemoryError) else str(exc)
                self.signals.error.emit(message or "Неизвестная ошибка обработки.")
        finally:
            self.signals.finished.emit()

    def _progress(self, value: int, message: str) -> None:
        if not self.is_cancelled():
            self.signals.progress.emit(max(0, min(100, int(value))), message)


def submit(fn: Callable, parent: QObject | None = None) -> TaskRunner:
    task = TaskRunner(fn, parent)
    QThreadPool.globalInstance().start(task)
    return task
