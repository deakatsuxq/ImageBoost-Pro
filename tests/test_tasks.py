from threading import Event

from PySide6.QtCore import QThreadPool

from imageboost_pro.workers.tasks import TaskRunner


def test_background_task_progress_result_and_error(qtbot):
    pool = QThreadPool()
    outputs, progress = [], []

    def calculate(update, cancelled):
        update(25, "Анализ")
        assert not cancelled()
        return 42

    task = TaskRunner(calculate)
    task.signals.result.connect(outputs.append)
    task.signals.progress.connect(lambda value, message: progress.append((value, message)))
    with qtbot.waitSignal(task.signals.finished, timeout=3000):
        pool.start(task)
    assert outputs == [42]
    assert progress == [(25, "Анализ")]

    def fail(update, cancelled):
        raise ValueError("Повреждённый файл")

    error_task = TaskRunner(fail)
    errors = []
    error_task.signals.error.connect(errors.append)
    with qtbot.waitSignals([error_task.signals.error, error_task.signals.finished], timeout=3000):
        pool.start(error_task)
    assert errors == ["Повреждённый файл"]
    pool.waitForDone()


def test_cancel_running_task_suppresses_stale_result(qtbot):
    entered, release = Event(), Event()
    outputs = []

    def calculate(update, cancelled):
        entered.set()
        release.wait(2)
        assert cancelled()
        return "устаревший результат"

    task = TaskRunner(calculate)
    task.signals.result.connect(outputs.append)
    pool = QThreadPool()
    pool.start(task)
    qtbot.waitUntil(entered.is_set)
    task.cancel()
    with qtbot.waitSignal(task.signals.finished, timeout=3000):
        release.set()
    assert outputs == []
    pool.waitForDone()
