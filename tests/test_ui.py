"""Real Qt regression tests for async editor state, export and settings."""
from dataclasses import replace
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QSettings, QTimer
from PySide6.QtWidgets import QMessageBox, QSlider
import pytest

import imageboost_pro.ui.main_window as main_module
from imageboost_pro.ui.export_dialog import ExportDialog


@pytest.fixture
def editor(qtbot, tmp_path, monkeypatch):
    preferences = QSettings(str(tmp_path / "preferences.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(main_module, "QSettings", lambda *args: preferences)
    window = main_module.MainWindow()
    qtbot.addWidget(window)
    window.show()
    yield window
    window.close()
    qtbot.waitUntil(lambda: not window._tasks, timeout=15000)


def add_image(editor, qtbot, tmp_path, name="Тёмное фото.png", size=(160, 100)):
    path = tmp_path / name
    y, x = np.mgrid[:size[1], :size[0]]
    rgb = np.stack((15 + x % 23, 10 + y % 20, 20 + (x + y) % 17), axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    editor.load_paths([path])
    qtbot.waitUntil(lambda: editor.current_document is not None, timeout=15000)
    return path


def test_real_async_open_settings_history_full_result(editor, qtbot, tmp_path):
    add_image(editor, qtbot, tmp_path)
    original = editor.current_document.rgb.copy()
    assert editor.current_analysis.metrics["width"] == 160
    assert editor.image_list.count() == 1
    assert not editor.image_list.item(0).icon().isNull()
    assert len(editor.controls.findChildren(QSlider)) >= 10
    editor.change_settings({"night_mode": "balanced", "scale": 1.5})
    assert editor.current_settings.night_mode == "balanced"
    editor.undo()
    assert editor.current_settings.night_mode == "off"
    editor.redo()
    assert editor.current_settings.scale == 1.5
    editor.process_full()
    qtbot.waitUntil(lambda: editor.full_result is not None and editor._active_task is None, timeout=15000)
    assert editor.full_result.shape == (150, 240, 3)
    assert editor.full_result.mean() > original.mean()
    assert np.array_equal(editor.current_document.rgb, original)
    assert editor.progress.value() == 100
    assert "полном разрешении" in editor.preview_notice.text()
    editor.reset_settings()
    assert editor.current_settings.night_mode == "off"


def test_analysis_reports_original_large_dimensions(editor, qtbot, tmp_path):
    add_image(editor, qtbot, tmp_path, size=(1400, 150))
    assert editor.preview_source.shape[1] == 1200
    assert editor.current_analysis.metrics["width"] == 1400


def test_stale_full_result_cannot_replace_new_settings(editor, qtbot, tmp_path, monkeypatch):
    add_image(editor, qtbot, tmp_path)
    qtbot.waitUntil(lambda: editor._preview_task is None, timeout=15000)
    started = []

    def delayed(rgb, settings, progress=None, cancelled=None):
        started.append(settings.exposure)
        time.sleep(0.08)
        if progress:
            progress(100, "Готово")
        return rgb.copy()

    monkeypatch.setattr(main_module, "process_image", delayed)
    editor.process_full()
    qtbot.waitUntil(lambda: bool(started), timeout=5000)
    editor.change_settings({"exposure": 0.7})
    qtbot.waitUntil(lambda: editor._active_task is None, timeout=5000)
    assert editor.full_result is None
    assert editor.current_settings.exposure == 0.7


def test_fast_slider_change_cancels_stale_preview(editor, qtbot, tmp_path, monkeypatch):
    add_image(editor, qtbot, tmp_path)
    qtbot.waitUntil(lambda: editor._preview_task is None, timeout=15000)
    applied = []
    original_set_images = editor.viewer.set_images

    def set_images(original, result):
        applied.append(int(result[0, 0, 0]))
        original_set_images(original, result)

    def delayed(rgb, settings, progress=None, cancelled=None):
        time.sleep(0.08 if settings.brightness == 10 else 0.01)
        return np.full_like(rgb, int(settings.brightness))

    monkeypatch.setattr(main_module, "process_image", delayed)
    monkeypatch.setattr(editor.viewer, "set_images", set_images)
    editor.change_settings({"brightness": 10})
    editor._debounce.stop()
    editor._preview()
    editor.change_settings({"brightness": 20})
    editor._debounce.stop()
    editor._preview()
    qtbot.waitUntil(lambda: 20 in applied, timeout=5000)
    qtbot.wait(150)
    assert applied == [20]


def test_corrupted_input_is_visible_without_crash(editor, qtbot, tmp_path):
    bad = tmp_path / "Повреждено.png"
    bad.write_bytes(b"this is not a photograph")
    editor.load_paths([bad])
    qtbot.waitUntil(lambda: "Не удалось" in editor.analysis_text.toPlainText(), timeout=10000)
    assert editor.current_document is None
    assert not editor.save_action.isEnabled()
    assert "Ошибка" in editor.image_list.item(0).text()


def test_export_never_overwrites_original_without_permission(qtbot, tmp_path, monkeypatch):
    source = tmp_path / "Оригинал.png"
    source.write_bytes(b"original bytes")
    preferences = QSettings(str(tmp_path / "export.ini"), QSettings.Format.IniFormat)
    preferences.setValue("export/folder", str(tmp_path))
    dialog = ExportDialog(source, (160, 100), preferences)
    qtbot.addWidget(dialog)
    dialog.name.setText(source.stem)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    dialog.accept()
    assert dialog.result() == 0
    assert not dialog.overwrite
    assert source.read_bytes() == b"original bytes"
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    dialog.accept()
    assert dialog.result() == 1
    assert dialog.overwrite
    assert dialog.output_path == source


def test_per_document_settings_remain_individual(editor, qtbot, tmp_path):
    first = add_image(editor, qtbot, tmp_path)
    editor.change_settings({"night_mode": "natural"})
    second = add_image(editor, qtbot, tmp_path, "Другое.png")
    assert editor.current_state.path == second
    assert editor.current_settings.night_mode == "off"
    editor.image_list.setCurrentRow(0)
    qtbot.waitUntil(lambda: editor.current_document is not None, timeout=10000)
    assert editor.current_state.path == first
    assert editor.current_settings.night_mode == "natural"


def test_reset_section_preserves_other_tools(editor, qtbot, tmp_path):
    add_image(editor, qtbot, tmp_path)
    editor.change_settings({"denoise_strength": 35, "night_mode": "natural"})
    editor.reset_section(("denoise_strength", "denoise_method"))
    assert editor.current_settings.denoise_strength == 0
    assert editor.current_settings.night_mode == "natural"
    assert editor.current_state.can_undo
    editor.undo()
    assert editor.current_settings == replace(editor.current_settings, denoise_strength=35)


def test_real_export_uses_original_resolution(editor, qtbot, tmp_path, monkeypatch):
    source = add_image(editor, qtbot, tmp_path)
    editor.preferences.setValue("export/folder", str(tmp_path))
    editor.change_settings({"scale": 1.5, "night_mode": "balanced"})
    original_exec = ExportDialog.exec

    def accept_export(dialog):
        QTimer.singleShot(0, dialog.accept)
        return original_exec(dialog)

    notices = []
    monkeypatch.setattr(ExportDialog, "exec", accept_export)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))
    editor.export_current()
    output = tmp_path / (source.stem + "_улучшено.png")
    qtbot.waitUntil(lambda: output.exists() and editor._active_task is None, timeout=15000)
    with Image.open(output) as result:
        assert result.size == (240, 150)
        assert np.asarray(result).mean() > editor.current_document.rgb.mean()
    with Image.open(source) as original:
        assert original.size == (160, 100)
    assert any("сохранено" in text for text in notices)


def prepare_batch_dialogs(monkeypatch, folder, notices, individual=True):
    monkeypatch.setattr(main_module.QFileDialog, "getExistingDirectory", lambda *args: str(folder))
    original_exec = QMessageBox.exec

    def choose_settings(dialog):
        label = "Индивидуальные" if individual else "Текущие для всех"
        button = next(button for button in dialog.buttons() if button.text() == label)
        QTimer.singleShot(0, button.click)
        return original_exec(dialog)

    monkeypatch.setattr(QMessageBox, "exec", choose_settings)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: notices.append(args[2]))


def test_batch_continues_corrupt_files_and_preserves_collisions(editor, qtbot, tmp_path, monkeypatch):
    source = add_image(editor, qtbot, tmp_path)
    editor.change_settings({"scale": 1.5})
    bad = tmp_path / "Ошибка.png"
    bad.write_bytes(b"not an image")
    editor.load_paths([bad])
    qtbot.waitUntil(lambda: "Не удалось" in editor.analysis_text.toPlainText(), timeout=10000)
    folder = tmp_path / "batch"
    folder.mkdir()
    collision = folder / (source.stem + "_улучшено.png")
    collision.write_bytes(b"do not overwrite")
    notices = []
    prepare_batch_dialogs(monkeypatch, folder, notices)
    editor.export_batch()
    qtbot.waitUntil(lambda: editor._active_task is None, timeout=15000)
    assert collision.read_bytes() == b"do not overwrite"
    with Image.open(folder / (source.stem + "_улучшено_2.png")) as saved:
        assert saved.size == (240, 150)
    assert "Сохранено: 1. Ошибок: 1." in notices[0]
    assert "Ошибка.png" in notices[0]
    assert "Ошибка:" in editor.image_list.item(1).text()


def test_batch_cancel_reports_and_retains_completed_files(editor, qtbot, tmp_path, monkeypatch):
    paths = []
    for i in range(4):
        path = tmp_path / f"Файл{i}.png"
        Image.fromarray(np.full((30, 50, 3), 40 + i, np.uint8)).save(path)
        paths.append(path)
    editor.load_paths(paths)
    qtbot.waitUntil(lambda: editor.current_document is not None, timeout=15000)
    qtbot.waitUntil(lambda: editor._preview_task is None, timeout=15000)
    folder = tmp_path / "cancelled_batch"
    folder.mkdir()
    notices = []
    prepare_batch_dialogs(monkeypatch, folder, notices)
    calls = []
    original_process = main_module.process_image

    def slow_process(rgb, settings, progress=None, cancelled=None):
        calls.append(1)
        time.sleep(0.08)
        return original_process(rgb, settings, progress=progress, cancelled=cancelled)

    monkeypatch.setattr(main_module, "process_image", slow_process)
    editor.export_batch()
    qtbot.waitUntil(lambda: len(calls) >= 2, timeout=5000)
    editor.cancel_processing()
    qtbot.waitUntil(lambda: editor._active_task is None, timeout=15000)
    assert len(list(folder.glob("*.png"))) == 1
    assert "Сохранено: 1." in notices[0]
    assert "остановлена" in notices[0]
