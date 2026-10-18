"""Interaction checks for the actual Qt canvas, including scaled results."""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from imageboost_pro.ui.comparison_view import ComparisonView


@pytest.fixture
def viewer(qtbot):
    widget = ComparisonView()
    widget.resize(800, 600)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(widget.isVisible)
    return widget


def solid(color, width=200, height=150):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def color_at(widget, x, y):
    pixel = widget.grab().toImage().pixelColor(x, y)
    return pixel.red(), pixel.green(), pixel.blue()


def wheel(widget, position, delta):
    event = QWheelEvent(
        QPointF(position),
        QPointF(widget.mapToGlobal(position)),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_split_can_be_dragged_and_qimage_owns_pixels(viewer, qtbot):
    original = solid((210, 30, 15))
    result = solid((15, 40, 220), width=400, height=300)
    viewer.set_images(original[:, ::-1], result)
    original[:] = 0
    result[:] = 0
    assert color_at(viewer, 200, 250) == (210, 30, 15)
    assert color_at(viewer, 600, 250) == (15, 40, 220)

    qtbot.mousePress(viewer, Qt.MouseButton.LeftButton, pos=QPoint(400, 300))
    qtbot.mouseMove(viewer, QPoint(640, 300))
    qtbot.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=QPoint(640, 300))
    assert viewer.divider_position == pytest.approx(0.8)
    assert color_at(viewer, 600, 250) == (210, 30, 15)
    assert color_at(viewer, 700, 250) == (15, 40, 220)


def test_wheel_keeps_image_point_under_cursor(viewer):
    viewer.set_images(solid((80, 90, 100), width=1000, height=800))
    viewer.set_mode("result")
    viewer.set_zoom(2.0)
    point = QPoint(500, 350)
    pane = viewer._panes()[0]
    before = viewer._normalized_at(QPointF(point), pane)
    wheel(viewer, point, 120)
    after = viewer._normalized_at(QPointF(point), pane)
    assert viewer.zoom == pytest.approx(2.4)
    assert after.x() == pytest.approx(before.x())
    assert after.y() == pytest.approx(before.y())


def test_side_panes_stay_aligned_when_zoomed_and_panned(viewer, qtbot):
    # Each quadrant identifies a different part of the photograph.
    original = solid((40, 160, 70), width=200, height=150)
    original[:75, :100] = (200, 50, 30)
    original[:75, 100:] = (30, 70, 200)
    original[75:, :100] = (190, 150, 40)
    result = original.repeat(4, axis=0).repeat(4, axis=1)
    viewer.set_images(original, result)
    viewer.set_mode("side")
    viewer.set_zoom(1.5)

    qtbot.mousePress(viewer, Qt.MouseButton.LeftButton, pos=QPoint(180, 300))
    qtbot.mouseMove(viewer, QPoint(240, 340))
    qtbot.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=QPoint(240, 340))
    assert viewer._center.x() < 0.5
    assert viewer._center.y() < 0.5
    # A 406 px pane offset must select the same normalized source point.
    for x, y in [(120, 200), (260, 200), (120, 420), (320, 420)]:
        assert color_at(viewer, x, y) == color_at(viewer, x + 406, y)


def test_new_upscale_preserves_scene_size_in_manual_zoom(viewer):
    original = solid((20, 90, 120), width=200, height=150)
    viewer.set_images(original)
    viewer.set_zoom(2.0)
    before_width = viewer._image_rect(viewer._panes()[0]).width()
    viewer.set_images(original, original.repeat(4, 0).repeat(4, 1))
    assert viewer.zoom == pytest.approx(0.5)
    assert viewer._image_rect(viewer._panes()[0]).width() == pytest.approx(before_width)


def test_hold_original_with_space_and_button_restores_comparison(viewer, qtbot):
    viewer.set_images(solid((210, 30, 15)), solid((15, 40, 220)))
    assert color_at(viewer, 600, 250) == (15, 40, 220)
    qtbot.keyPress(viewer, Qt.Key.Key_Space)
    assert color_at(viewer, 600, 250) == (210, 30, 15)
    qtbot.keyRelease(viewer, Qt.Key.Key_Space)
    assert color_at(viewer, 600, 250) == (15, 40, 220)
    viewer.show_original(True)
    assert color_at(viewer, 600, 250) == (210, 30, 15)
    viewer.show_original(False)
    assert color_at(viewer, 600, 250) == (15, 40, 220)


def test_fit_tiny_and_extremely_wide_images(viewer):
    viewer.set_images(solid((210, 30, 15), width=1, height=1))
    assert viewer.zoom == viewer.MAX_ZOOM
    assert not viewer.grab().isNull()
    viewer.set_images(solid((30, 100, 190), width=40000, height=1))
    assert 0 < viewer.zoom < viewer.MIN_ZOOM
    image_rect = viewer._image_rect(viewer._panes()[0])
    assert image_rect.left() >= 0
    assert image_rect.right() <= viewer.width()


@pytest.mark.parametrize("mode", ["side", "split", "original", "result"])
def test_all_modes_render_and_zoom_is_bounded(viewer, mode):
    viewer.set_images(solid((90, 80, 70)))
    viewer.set_mode(mode)
    viewer.set_zoom(1000)
    assert viewer.zoom == viewer.MAX_ZOOM
    viewer.set_zoom(0.00001)
    assert viewer.zoom > 0
    assert not viewer.grab().isNull()


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_zoom_is_rejected(viewer, factor):
    with pytest.raises(ValueError, match="Масштаб"):
        viewer.set_zoom(factor)


def test_invalid_image_or_mode_is_rejected_without_replacing_image(viewer):
    viewer.set_images(solid((90, 80, 70)))
    with pytest.raises(ValueError):
        viewer.set_images(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        viewer.set_images(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        viewer.set_mode("unknown")
    assert not viewer.grab().isNull()
