"""Interactive comparison with a shared, normalized image viewport.

The result defines the displayed aspect ratio and the 100% pixel scale.  Both
images map their normalized corners to that same rectangle, so a 2x/4x result
stays aligned with its original during zoom and pan.  Painting only scales
cached QImages; it never runs image processing.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget


class ComparisonView(QWidget):
    """Compare RGB images using a draggable split or synchronized side panes."""

    zoom_changed = Signal(float)
    MIN_ZOOM = 0.05
    MAX_ZOOM = 16.0
    MODES = frozenset({"split", "side", "original", "result"})
    PANE_GAP = 12.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("comparisonView")
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Интерактивное сравнение оригинала и результата")
        self.setToolTip(
            "Колесо — масштаб под курсором. Перетаскивание — перемещение.\n"
            "Перетащите разделитель для сравнения. Пробел — показать оригинал."
        )
        self._original = QImage()
        self._result = QImage()
        self._canvas_width = 1
        self._canvas_height = 1
        self._mode = "split"
        self._original_held = False
        self._space_held = False
        self._zoom = 1.0
        self._center = QPointF(0.5, 0.5)
        self._divider = 0.5
        self._drag: str | None = None
        self._last_mouse = QPointF()
        self._fit_active = True

    @staticmethod
    def _to_qimage(rgb: np.ndarray) -> QImage:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise ValueError("Изображение должно быть массивом RGB uint8")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Ожидается непустое изображение размера H×W×3")
        pixels = np.ascontiguousarray(rgb)
        height, width = pixels.shape[:2]
        # QImage must own the bytes after the caller releases or edits its array.
        return QImage(
            pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGB888
        ).copy()

    def set_images(self, original: np.ndarray, result: np.ndarray | None = None) -> None:
        original_image = self._to_qimage(original)
        result_image = self._to_qimage(result) if result is not None else original_image
        first_image = self._original.isNull()
        new_original_size = original_image.size() != self._original.size()
        previous_width = self._canvas_width
        self._original = original_image
        self._result = result_image
        self._canvas_width = result_image.width()
        self._canvas_height = result_image.height()
        if first_image or new_original_size or self._fit_active:
            self.fit_to_window()
        else:
            # An upscaled result must not suddenly magnify the visible scene.
            self._zoom *= previous_width / self._canvas_width
            self._zoom = self._bounded_zoom(self._zoom)
            self._constrain_center()
            self.zoom_changed.emit(self._zoom)
            self.update()

    def clear(self) -> None:
        """Release cached image memory when closing a document."""
        self._original = QImage()
        self._result = QImage()
        self._drag = None
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Неизвестный режим сравнения: {mode}")
        self._mode = mode
        self._drag = None
        if self._fit_active:
            self.fit_to_window()
        else:
            self._constrain_center()
            self.update()

    def show_original(self, visible: bool) -> None:
        self._original_held = bool(visible)
        self.update()

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def divider_position(self) -> float:
        """Divider position as a fraction of the viewport width."""
        return self._divider

    def _effective_mode(self) -> str:
        return "original" if self._original_held or self._space_held else self._mode

    def _panes(self) -> list[QRectF]:
        full = QRectF(self.rect())
        if self._effective_mode() != "side":
            return [full]
        pane_width = max(1.0, (full.width() - self.PANE_GAP) / 2)
        return [
            QRectF(0, 0, pane_width, full.height()),
            QRectF(pane_width + self.PANE_GAP, 0, pane_width, full.height()),
        ]

    def _pane_at(self, position: QPointF) -> QRectF:
        panes = self._panes()
        return min(panes, key=lambda pane: abs(position.x() - pane.center().x()))

    def _fit_zoom(self) -> float:
        pane = self._panes()[0]
        return min(
            max(1.0, pane.width() - 32) / self._canvas_width,
            max(1.0, pane.height() - 32) / self._canvas_height,
            self.MAX_ZOOM,
        )

    def _bounded_zoom(self, factor: float) -> float:
        # Very large photographs can fit below 5%; they must remain viewable.
        minimum = min(self.MIN_ZOOM, self._fit_zoom())
        return max(minimum, min(self.MAX_ZOOM, factor))

    def fit_to_window(self) -> None:
        self._fit_active = True
        self._center = QPointF(0.5, 0.5)
        if not self._original.isNull():
            self._zoom = self._fit_zoom()
            self.zoom_changed.emit(self._zoom)
        self.update()

    def set_zoom(self, factor: float) -> None:
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Масштаб должен быть положительным конечным числом")
        self._fit_active = False
        self._zoom = self._bounded_zoom(float(factor))
        self._constrain_center()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _image_rect(self, pane: QRectF) -> QRectF:
        width = self._canvas_width * self._zoom
        height = self._canvas_height * self._zoom
        return QRectF(
            pane.center().x() - self._center.x() * width,
            pane.center().y() - self._center.y() * height,
            width,
            height,
        )

    def _normalized_at(self, position: QPointF, pane: QRectF) -> QPointF:
        image_rect = self._image_rect(pane)
        return QPointF(
            (position.x() - image_rect.left()) / image_rect.width(),
            (position.y() - image_rect.top()) / image_rect.height(),
        )

    def _constrain_center(self) -> None:
        pane = self._panes()[0]
        limits = (
            pane.width() / (2 * self._canvas_width * self._zoom),
            pane.height() / (2 * self._canvas_height * self._zoom),
        )
        coordinates = (self._center.x(), self._center.y())
        values = [
            0.5 if half >= 0.5 else max(half, min(1.0 - half, value))
            for value, half in zip(coordinates, limits)
        ]
        self._center = QPointF(*values)

    @staticmethod
    def _draw_label(painter: QPainter, text: str, position: QPointF) -> None:
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        rect = QRectF(position, metrics.size(Qt.TextFlag.TextSingleLine, text).toSizeF())
        rect.adjust(-10, -5, 10, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(17, 21, 28, 220))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QColor("#e4e9f2"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_image(self, painter: QPainter, image: QImage, pane: QRectF) -> None:
        painter.save()
        painter.setClipRect(pane, Qt.ClipOperation.IntersectClip)
        painter.drawImage(self._image_rect(pane), image)
        painter.restore()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161b23"))
        if self._original.isNull():
            painter.setPen(QColor("#d7deeb"))
            font = QFont(painter.font())
            font.setPointSize(18)
            painter.setFont(font)
            painter.drawText(
                self.rect().adjusted(16, 16, -16, -16),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "Откройте фотографию\nдля просмотра и улучшения",
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 4)
        panes = self._panes()
        mode = self._effective_mode()
        if mode == "side":
            self._draw_image(painter, self._original, panes[0])
            self._draw_image(painter, self._result, panes[1])
            self._draw_label(painter, "Оригинал", QPointF(panes[0].left() + 22, 20))
            self._draw_label(painter, "Результат", QPointF(panes[1].left() + 22, 20))
        elif mode == "split":
            split_x = self.width() * self._divider
            self._draw_image(painter, self._result, panes[0])
            painter.save()
            painter.setClipRect(QRectF(0, 0, split_x, self.height()))
            self._draw_image(painter, self._original, panes[0])
            painter.restore()
            self._draw_label(painter, "Оригинал", QPointF(22, 20))
            label_width = painter.fontMetrics().horizontalAdvance("Результат")
            self._draw_label(painter, "Результат", QPointF(self.width() - label_width - 22, 20))
            painter.setPen(QPen(QColor("#70c9db"), 2))
            painter.drawLine(QPointF(split_x, 0), QPointF(split_x, self.height()))
            handle = QRectF(split_x - 12, self.height() / 2 - 22, 24, 44)
            painter.setBrush(QColor("#253b46"))
            painter.drawRoundedRect(handle, 8, 8)
            painter.setPen(QPen(QColor("#d5f5fa"), 2))
            for offset in (-3, 3):
                painter.drawLine(
                    QPointF(split_x + offset, self.height() / 2 - 8),
                    QPointF(split_x + offset, self.height() / 2 + 8),
                )
        else:
            image = self._original if mode == "original" else self._result
            self._draw_image(painter, image, panes[0])
            self._draw_label(painter, "Оригинал" if mode == "original" else "Результат", QPointF(22, 20))

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._fit_active:
            self.fit_to_window()
        else:
            self._constrain_center()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._original.isNull():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if not delta:
            delta = event.pixelDelta().y() * 8
        if not delta:
            event.ignore()
            return
        position = event.position()
        pane = self._pane_at(position)
        point = self._normalized_at(position, pane)
        self._zoom = self._bounded_zoom(self._zoom * 1.2 ** (delta / 120))
        self._center = QPointF(
            point.x() - (position.x() - pane.center().x()) / (self._canvas_width * self._zoom),
            point.y() - (position.y() - pane.center().y()) / (self._canvas_height * self._zoom),
        )
        self._constrain_center()
        self._fit_active = False
        self.zoom_changed.emit(self._zoom)
        self.update()
        event.accept()

    def _near_divider(self, position: QPointF) -> bool:
        return (
            self._effective_mode() == "split"
            and abs(position.x() - self.width() * self._divider) <= 12
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._original.isNull():
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._drag = "divider" if self._near_divider(event.position()) else "pan"
        self._last_mouse = event.position()
        self.setCursor(Qt.CursorShape.SplitHCursor if self._drag == "divider" else Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag == "divider":
            self._divider = max(0.0, min(1.0, event.position().x() / max(1, self.width())))
            self.update()
        elif self._drag == "pan":
            movement = event.position() - self._last_mouse
            self._center -= QPointF(
                movement.x() / (self._canvas_width * self._zoom),
                movement.y() / (self._canvas_height * self._zoom),
            )
            self._constrain_center()
            self._fit_active = False
            self.update()
        else:
            self.setCursor(
                Qt.CursorShape.SplitHCursor if self._near_divider(event.position()) else Qt.CursorShape.OpenHandCursor
            )
        self._last_mouse = event.position()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_window()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.update()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        # A focus change must not leave the temporary original view stuck on.
        self._space_held = False
        self._drag = None
        self.update()
        super().focusOutEvent(event)
