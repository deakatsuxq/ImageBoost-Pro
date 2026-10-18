"""Mask painting, genuine repair, cancellation and safe copy export in Qt."""
import time

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, QTimer, Qt
from PySide6.QtWidgets import QMessageBox
import pytest

import imageboost_pro.ui.restoration_dialog as restoration_module
from imageboost_pro.ui.export_dialog import ExportDialog
from imageboost_pro.ui.restoration_dialog import BrushStroke, MaskView, RestorationDialog, rasterize_mask
from imageboost_pro.utils.image_io import load_image


@pytest.fixture
def repair_dialog(qtbot, tmp_path):
    rgb = np.full((60, 100, 3), 125, dtype=np.uint8)
    rgb[29:33, 20:80] = 0
    alpha = np.full((60, 100, 1), 180, dtype=np.uint8)
    source = tmp_path / "Царапина.png"
    Image.fromarray(np.concatenate((rgb, alpha), axis=2)).save(source)
    document = load_image(source)
    preferences = QSettings(str(tmp_path / "repair.ini"), QSettings.Format.IniFormat)
    preferences.setValue("export/folder", str(tmp_path))
    dialog = RestorationDialog(document, preferences)
    qtbot.addWidget(dialog)
    dialog.show()
    yield dialog
    dialog.close()
    qtbot.waitUntil(lambda: not dialog._tasks, timeout=10000)


def test_vector_mask_undo_redo_and_one_channel_rasterization(repair_dialog):
    view = repair_dialog.view
    assert not repair_dialog.apply_button.isEnabled()
    view.add_stroke(((30, 20),), radius=2)
    assert repair_dialog.apply_button.isEnabled()
    mask = rasterize_mask((60, 100, 3), view.snapshot())
    assert mask.shape == (60, 100)
    assert mask.dtype == np.uint8
    assert mask[20, 30] == 255
    assert mask[20, 35] == 0
    view.undo_stroke()
    assert view.snapshot() == ()
    assert not repair_dialog.apply_button.isEnabled()
    view.redo_stroke()
    assert len(view.snapshot()) == 1
    view.clear_mask()
    assert view.snapshot() == ()
    assert not view._redo_strokes


def test_brush_coordinates_stay_in_original_pixels_when_zoom_changes(qtbot):
    view = MaskView(np.full((80, 120, 3), 100, dtype=np.uint8))
    qtbot.addWidget(view)
    view.resize(700, 450)
    view.show()
    qtbot.waitExposed(view)
    for zoom in (1.0, 4.0):
        view.set_zoom(zoom)
        rect = view._image_rect(view._panes()[0])
        point = QPoint(round(rect.left() + 40 / 120 * rect.width()),
                       round(rect.top() + 30 / 80 * rect.height()))
        qtbot.mouseClick(view, Qt.MouseButton.LeftButton, pos=point)
        x, y = view.strokes[-1].points[0]
        assert abs(x - 40) <= 1
        assert abs(y - 30) <= 1
    assert len(view.snapshot()) == 2
    assert not view.grab().isNull()


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_real_worker_repairs_scratch_and_preserves_unmasked_pixels(repair_dialog, qtbot, method):
    dialog = repair_dialog
    original = dialog.document.rgb.copy()
    dialog.method.setCurrentIndex(dialog.method.findData(method))
    dialog.view.add_stroke(((20, 31), (79, 31)), radius=3)
    mask = rasterize_mask(original.shape, dialog.view.snapshot())
    dialog.process_mask()
    qtbot.waitUntil(lambda: dialog.restored_rgb is not None and dialog._task is None, timeout=10000)
    repaired = dialog.restored_rgb
    assert repaired[30:32, 30:70].mean() > 100
    assert np.array_equal(repaired[mask == 0], original[mask == 0])
    assert np.array_equal(dialog.document.rgb, original)
    assert dialog.save_button.isEnabled()
    assert not dialog.paint_toggle.isChecked()
    assert dialog.view._mode == "split"
    assert dialog.progress.value() == 100


def test_new_mask_invalidates_inflight_result(repair_dialog, qtbot, monkeypatch):
    dialog = repair_dialog
    started = []
    original_inpaint = restoration_module.inpaint_mask

    def delayed(*args):
        started.append(True)
        time.sleep(0.1)
        return original_inpaint(*args)

    monkeypatch.setattr(restoration_module, "inpaint_mask", delayed)
    dialog.view.add_stroke(((20, 31), (79, 31)), radius=3)
    dialog.process_mask()
    qtbot.waitUntil(lambda: bool(started), timeout=5000)
    dialog.view.add_stroke(((10, 10),), radius=2)
    qtbot.waitUntil(lambda: dialog._task is None, timeout=5000)
    assert dialog.restored_rgb is None
    assert not dialog.save_button.isEnabled()
    assert dialog.paint_toggle.isChecked()
    assert dialog.apply_button.isEnabled()


def test_save_copy_preserves_original_and_alpha(repair_dialog, qtbot, monkeypatch):
    dialog = repair_dialog
    original_file = dialog.document.path.read_bytes()
    dialog.view.add_stroke(((20, 31), (79, 31)), radius=3)
    dialog.process_mask()
    qtbot.waitUntil(lambda: dialog.restored_rgb is not None and dialog._task is None, timeout=10000)
    original_exec = ExportDialog.exec

    def accept_export(export):
        QTimer.singleShot(0, export.accept)
        return original_exec(export)

    monkeypatch.setattr(ExportDialog, "exec", accept_export)
    dialog.save_copy()
    qtbot.waitUntil(lambda: dialog.saved_path is not None, timeout=10000)
    assert dialog.saved_path.name == "Царапина_ретушь.png"
    assert dialog.saved_path != dialog.document.path
    assert dialog.document.path.read_bytes() == original_file
    copy = load_image(dialog.saved_path)
    assert np.array_equal(copy.rgb, dialog.restored_rgb)
    assert np.array_equal(copy.alpha, dialog.document.alpha)


def test_mask_rasterization_can_cancel_without_output():
    strokes = (BrushStroke(2, ((10, 10), (20, 20))),)
    assert rasterize_mask((100, 100, 3), strokes, cancelled=lambda: True) is None


def test_closing_cancels_worker_without_updating_closed_dialog(repair_dialog, qtbot, monkeypatch):
    started = []

    def delayed(rgb, mask, radius, method):
        started.append(True)
        time.sleep(0.08)
        return rgb.copy()

    monkeypatch.setattr(restoration_module, "inpaint_mask", delayed)
    repair_dialog.view.add_stroke(((20, 31),), radius=3)
    repair_dialog.process_mask()
    qtbot.waitUntil(lambda: bool(started), timeout=5000)
    repair_dialog.close()
    assert repair_dialog._closing
    qtbot.waitUntil(lambda: not repair_dialog._tasks, timeout=5000)
    assert repair_dialog.restored_rgb is None


def test_retouch_cannot_overwrite_original_even_after_export_permission(repair_dialog, qtbot, monkeypatch):
    dialog = repair_dialog
    original_file = dialog.document.path.read_bytes()
    dialog.view.add_stroke(((20, 31), (79, 31)), radius=3)
    dialog.process_mask()
    qtbot.waitUntil(lambda: dialog.restored_rgb is not None and dialog._task is None, timeout=10000)

    def choose_original(export):
        export.name.setText(dialog.document.path.stem)
        export.overwrite = True
        return 1

    messages = []
    monkeypatch.setattr(ExportDialog, "exec", choose_original)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    dialog.save_copy()
    assert dialog._task is None
    assert dialog.saved_path is None
    assert dialog.document.path.read_bytes() == original_file
    assert "новый файл" in messages[0]
