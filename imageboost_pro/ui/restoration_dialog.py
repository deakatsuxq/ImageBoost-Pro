"""Original-pixel brush masks and genuine, asynchronous OpenCV inpainting."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from imageboost_pro.core.restoration import inpaint_mask
from imageboost_pro.ui.comparison_view import ComparisonView
from imageboost_pro.ui.export_dialog import ExportDialog
from imageboost_pro.utils.image_io import save_image
from imageboost_pro.workers.tasks import submit


@dataclass(frozen=True)
class BrushStroke:
    radius: int
    points: tuple[tuple[float, float], ...]


def rasterize_mask(shape, strokes, cancelled=lambda: False):
    """Allocate one channel once, keeping the interactive history vector-only."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for stroke in strokes:
        if cancelled():
            return None
        points = [(round(x), round(y)) for x, y in stroke.points]
        if not points:
            continue
        for first, second in zip(points, points[1:]):
            cv2.line(mask, first, second, 255, stroke.radius * 2, cv2.LINE_8)
        for point in (points[0], points[-1]):
            cv2.circle(mask, point, stroke.radius, 255, -1)
    return mask


class MaskView(ComparisonView):
    mask_changed = Signal()

    def __init__(self, rgb, parent=None):
        super().__init__(parent)
        self._image_height, self._image_width = rgb.shape[:2]
        self.brush_radius = 8
        self.painting_enabled = True
        self.strokes = []
        self._redo_strokes = []
        self._live_points = None
        self._live_radius = 8
        self.set_images(rgb)
        self.set_mode("original")
        self.setToolTip("Левая кнопка — маска повреждений; правая — перемещение; колесо — масштаб. "
                        "После восстановления отключите кисть для сравнения разделителем.")

    def set_painting_enabled(self, enabled):
        self.painting_enabled = bool(enabled)
        self._live_points = None
        self.set_mode("original" if enabled else "split")
        self.update()

    def snapshot(self):
        strokes = tuple(self.strokes)
        if self._live_points:
            strokes += (BrushStroke(self._live_radius, tuple(self._live_points)),)
        return strokes

    def add_stroke(self, points, radius=None):
        """The same original-coordinate action is used by mouse input and tests."""
        if not points:
            return
        self.strokes.append(BrushStroke(int(radius or self.brush_radius), tuple(points)))
        self._redo_strokes.clear()
        self.mask_changed.emit()
        self.update()

    def undo_stroke(self):
        if self.strokes:
            self._redo_strokes.append(self.strokes.pop())
            self.mask_changed.emit()
            self.update()

    def redo_stroke(self):
        if self._redo_strokes:
            self.strokes.append(self._redo_strokes.pop())
            self.mask_changed.emit()
            self.update()

    def clear_mask(self):
        if self.strokes or self._live_points:
            self.strokes.clear()
            self._redo_strokes.clear()
            self._live_points = None
            self.mask_changed.emit()
            self.update()

    def _original_point(self, position):
        pane = self._pane_at(position)
        normalized = self._normalized_at(position, pane)
        if not 0 <= normalized.x() <= 1 or not 0 <= normalized.y() <= 1:
            return None
        return (min(self._image_width - 1, normalized.x() * self._image_width),
                min(self._image_height - 1, normalized.y() * self._image_height))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._drag = "pan"
            self._last_mouse = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif self.painting_enabled and event.button() == Qt.MouseButton.LeftButton:
            point = self._original_point(event.position())
            if point is not None:
                self._live_points = [point]
                self._live_radius = self.brush_radius
                self._redo_strokes.clear()
                self.mask_changed.emit()
                self.update()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._live_points is not None:
            point = self._original_point(event.position())
            if point is not None and point != self._live_points[-1]:
                self._live_points.append(point)
                self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._live_points is not None:
            points = self._live_points
            self._live_points = None
            self.add_stroke(points, self._live_radius)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.painting_enabled or not self.snapshot():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        image_rect = self._image_rect(self._panes()[0])
        painter.setClipRect(image_rect)
        painter.translate(image_rect.left(), image_rect.top())
        painter.scale(image_rect.width() / self._image_width, image_rect.height() / self._image_height)
        painter.setOpacity(0.48)
        for stroke in self.snapshot():
            pen = QPen(QColor("#FF4E66"), stroke.radius * 2,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            if len(stroke.points) == 1:
                painter.drawPoint(QPointF(*stroke.points[0]))
            else:
                path = QPainterPath(QPointF(*stroke.points[0]))
                for point in stroke.points[1:]:
                    path.lineTo(QPointF(*point))
                painter.drawPath(path)


class RestorationDialog(QDialog):
    def __init__(self, document, preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ретушь повреждений по маске")
        self.resize(1040, 760)
        self.document = document
        self.preferences = preferences
        self.restored_rgb = None
        self.saved_path = None
        self._version = 0
        self._task = None
        self._tasks = []
        self._closing = False
        layout = QVBoxLayout(self)
        note = QLabel("Инструмент работает с оригиналом, отдельно от текущих эффектов. "
                      "Закрасьте небольшие царапины или пятна; сохранение создаст новую копию. "
                      "Левая кнопка — кисть, правая — перемещение, колесо — масштаб.")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        self.brush = QSpinBox()
        self.brush.setRange(1, 80)
        self.brush.setValue(8)
        row.addWidget(QLabel("Радиус кисти, px"))
        row.addWidget(self.brush)
        self.method = QComboBox()
        self.method.addItem("Telea", "telea")
        self.method.addItem("Навье — Стокс", "navier_stokes")
        row.addWidget(self.method)
        row.addWidget(QLabel("Радиус восстановления"))
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 20)
        self.radius.setValue(3)
        self.radius.setSingleStep(0.5)
        row.addWidget(self.radius)
        self.paint_toggle = QCheckBox("Рисовать маску")
        self.paint_toggle.setChecked(True)
        row.addWidget(self.paint_toggle)
        fit = QPushButton("Вписать")
        row.addWidget(fit)
        layout.addLayout(row)
        self.view = MaskView(document.rgb)
        layout.addWidget(self.view, 1)
        self.brush.valueChanged.connect(lambda value: setattr(self.view, "brush_radius", value))
        self.paint_toggle.toggled.connect(self.view.set_painting_enabled)
        fit.clicked.connect(self.view.fit_to_window)
        self.view.mask_changed.connect(self._mask_changed)
        self.method.currentIndexChanged.connect(self._parameters_changed)
        self.radius.valueChanged.connect(self._parameters_changed)
        row = QHBoxLayout()
        self.undo = QPushButton("Отменить мазок")
        self.redo = QPushButton("Повторить мазок")
        clear = QPushButton("Очистить маску")
        self.undo.clicked.connect(self.view.undo_stroke)
        self.redo.clicked.connect(self.view.redo_stroke)
        clear.clicked.connect(self.view.clear_mask)
        row.addWidget(self.undo)
        row.addWidget(self.redo)
        row.addWidget(clear)
        row.addStretch()
        self.apply_button = QPushButton("Восстановить")
        self.apply_button.clicked.connect(self.process_mask)
        row.addWidget(self.apply_button)
        self.cancel_button = QPushButton("Отменить обработку")
        self.cancel_button.clicked.connect(self.cancel_processing)
        row.addWidget(self.cancel_button)
        self.save_button = QPushButton("Сохранить копию…")
        self.save_button.clicked.connect(self.save_copy)
        row.addWidget(self.save_button)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)
        bottom = QHBoxLayout()
        self.status = QLabel("Нарисуйте маску повреждений. Оригинал сохраняется без изменений.")
        bottom.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        layout.addLayout(bottom)
        self._update_buttons()

    def _parameters_changed(self, *args):
        self._mask_changed()

    def _mask_changed(self):
        self._version += 1
        self.restored_rgb = None
        self.paint_toggle.setChecked(True)
        if self._task:
            self._task.cancel()
        self.status.setText("Маска или параметры изменены. Нажмите «Восстановить».")
        self._update_buttons()

    def _update_buttons(self):
        busy = self._task is not None
        self.apply_button.setEnabled(bool(self.view.snapshot()) and not busy)
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(self.restored_rgb is not None and not busy)
        self.undo.setEnabled(bool(self.view.strokes))
        self.redo.setEnabled(bool(self.view._redo_strokes))

    def _submit(self, function, callback):
        task = submit(function)
        self._task = task
        self._tasks.append(task)
        task.signals.result.connect(lambda payload: callback(payload) if not self._closing else None)
        task.signals.error.connect(self._failed)
        task.signals.progress.connect(self._progress)
        task.signals.finished.connect(lambda: self._finished(task))
        self._update_buttons()
        return task

    def process_mask(self):
        if self._task is not None or not self.view.snapshot():
            return
        version = self._version
        strokes = self.view.snapshot()
        source = self.document.rgb
        method, radius = self.method.currentData(), self.radius.value()
        self.progress.setValue(0)
        self.status.setText("Подготовка маски…")

        def run(progress, cancelled):
            mask = rasterize_mask(source.shape, strokes, cancelled)
            if mask is None or cancelled():
                return None
            progress(15, "Восстановление повреждённых пикселей…")
            restored = inpaint_mask(source, mask, radius, method)
            if cancelled():
                return None
            progress(100, "Восстановление завершено")
            return restored

        def done(result):
            if version == self._version and result is not None:
                self.restored_rgb = result
                self.view.set_images(source, result)
                self.paint_toggle.setChecked(False)
                self.status.setText("Результат готов. Сравните разделителем или сохраните копию.")
        self._submit(run, done)

    def _progress(self, percent, stage):
        if not self._closing:
            self.progress.setValue(percent)
            self.status.setText(stage)

    def _finished(self, task):
        if task in self._tasks:
            self._tasks.remove(task)
        if self._task is task:
            self._task = None
            if not self._closing:
                if task.is_cancelled():
                    self.status.setText("Обработка отменена. Маску можно исправить и запустить снова.")
                self._update_buttons()

    def _failed(self, error):
        if not self._closing:
            self.status.setText("Восстановление не завершено")
            QMessageBox.warning(self, "Ошибка ретуши", error)

    def cancel_processing(self):
        if self._task:
            self._task.cancel()
            self.status.setText("Отмена после текущей операции OpenCV…")

    def save_copy(self):
        if self.restored_rgb is None or self._task is not None:
            return
        dialog = ExportDialog(self.document.path, self.document.size, self.preferences, self)
        dialog.name.setText(self.document.path.stem + "_ретушь")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path = dialog.output_path
        if path.resolve() == Path(self.document.path).resolve():
            QMessageBox.warning(self, "Защита оригинала", "Ретушь сохраняется в новый файл. Выберите другое имя.")
            return
        rgb = self.restored_rgb
        quality, metadata, overwrite = dialog.quality.value(), dialog.metadata.isChecked(), dialog.overwrite

        def run(progress, cancelled):
            if cancelled():
                return None
            progress(0, "Сохранение копии…")
            saved = save_image(self.document, rgb, path, quality=quality,
                               keep_metadata=metadata, overwrite=overwrite)
            progress(100, "Копия сохранена")
            return saved

        def done(saved):
            if saved is not None:
                self.saved_path = Path(saved)
                self.accept()
        self._submit(run, done)

    def done(self, code):
        self._closing = True
        for task in self._tasks:
            task.cancel()
        super().done(code)
