"""Russian desktop editor: asynchronous, non-destructive, per-document processing."""
from dataclasses import asdict, replace
from pathlib import Path
import logging
import time
from threading import Event

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton,
    QSplitter, QTextBrowser, QToolBar, QVBoxLayout, QWidget,
)

from imageboost_pro.app_state import DocumentState
from imageboost_pro.core.image_analysis import analyze_image
from imageboost_pro.core.pipeline import output_size, process_image
from imageboost_pro.core.settings import ProcessingSettings
from imageboost_pro.ui.comparison_view import ComparisonView
from imageboost_pro.ui.controls import ControlsPanel
from imageboost_pro.ui.export_dialog import ExportDialog
from imageboost_pro.ui.settings_dialog import SettingsDialog
from imageboost_pro.utils.image_io import load_image, save_image
from imageboost_pro.utils.presets import BUILTIN_PRESETS, load_preset, save_preset
from imageboost_pro.workers.tasks import submit


LOGGER = logging.getLogger(__name__)
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_FILTER = "Изображения (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)"


def preview_copy(rgb, maximum=1200):
    """Run in workers; never resize full-size images on the GUI thread."""
    height, width = rgb.shape[:2]
    factor = min(1.0, maximum / max(height, width))
    if factor == 1:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, round(width * factor)), max(1, round(height * factor))),
                      interpolation=cv2.INTER_AREA)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageBoost Pro — улучшение фотографий")
        self.setMinimumSize(1060, 680)
        self.resize(1440, 920)
        self.preferences = QSettings("ImageBoostPro", "ImageBoostPro")
        self.documents = {}
        self.current_state = None
        self.current_document = None
        self.current_analysis = None
        self.preview_source = None
        self.full_result = None
        self._tasks = []
        self._load_task = None
        self._preview_task = None
        self._active_task = None
        self._cancel_callback = None
        self._generation = 0
        self._preview_generation = 0
        self._started = 0
        self._closing = False
        self._build_ui()
        geometry = self.preferences.value("ui/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._apply_theme()
        self._apply_performance()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(280)
        self._debounce.timeout.connect(self._preview)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self.setAcceptDrops(True)
        self._update_actions()

    @property
    def current_settings(self):
        return self.current_state.settings if self.current_state else ProcessingSettings()

    def _build_ui(self):
        toolbar = QToolBar("Основные действия", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(toolbar)
        brand = QLabel("  ImageBoost Pro  ")
        brand.setObjectName("brand")
        toolbar.addWidget(brand)
        self.open_action = self._action("Открыть", self.open_images, "Ctrl+O", toolbar)
        self._action("Добавить", lambda: self.open_images(add=True), "Ctrl+Shift+O", toolbar)
        self._action("Папка", self.add_folder, None, toolbar)
        toolbar.addSeparator()
        self.save_action = self._action("Сохранить", self.export_current, "Ctrl+S", toolbar)
        self.undo_action = self._action("Отменить", self.undo, "Ctrl+Z", toolbar)
        self.redo_action = self._action("Повторить", self.redo, "Ctrl+Y", toolbar)
        toolbar.addSeparator()
        self._action("Настройки", self.show_settings, "Ctrl+,", toolbar)
        self._action("Тема", self.toggle_theme, None, toolbar)
        self._action("Справка", self.show_help, "F1", toolbar)
        files_menu = self.menuBar().addMenu("Файл")
        files_menu.addAction(self.open_action)
        files_menu.addAction(self.save_action)
        self._action("Пакетная обработка…", self.export_batch, "Ctrl+Shift+S", files_menu)
        files_menu.addSeparator()
        self._action("Выход", self.close, "Alt+F4", files_menu)
        edit_menu = self.menuBar().addMenu("Обработка")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        self._action("Вернуться к оригиналу", self.reset_settings, "Ctrl+R", edit_menu)
        self._action("Анализ качества", self.analyze_current, "Ctrl+I", edit_menu)
        self.retouch_action = self._action("Ретушь повреждений по маске…", self.retouch_original,
                                           "Ctrl+Shift+R", edit_menu)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(330)
        layout = QVBoxLayout(left)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(QLabel("ФОТОГРАФИИ И ОЧЕРЕДЬ"))
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(76, 60))
        self.image_list.setSpacing(3)
        self.image_list.currentItemChanged.connect(self._selected)
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self._image_context_menu)
        layout.addWidget(self.image_list, 3)
        layout.addWidget(QLabel("ПРЕСЕТЫ"))
        self.presets = QComboBox()
        self.presets.addItem("Выберите пресет…", None)
        for name, settings in BUILTIN_PRESETS.items():
            self.presets.addItem(name, settings)
        self.presets.activated.connect(self._apply_preset)
        layout.addWidget(self.presets)
        row = QHBoxLayout()
        import_preset = QPushButton("Импорт…")
        import_preset.clicked.connect(self.import_preset)
        save_current = QPushButton("Сохранить…")
        save_current.clicked.connect(self.export_preset)
        row.addWidget(import_preset)
        row.addWidget(save_current)
        layout.addLayout(row)
        layout.addWidget(QLabel("АНАЛИЗ КАЧЕСТВА"))
        self.analysis_text = QTextBrowser()
        self.analysis_text.setOpenExternalLinks(False)
        self.analysis_text.setPlainText("Откройте фотографию. Анализ оценит освещение, шум, резкость и цвет.")
        layout.addWidget(self.analysis_text, 2)
        splitter.addWidget(left)

        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(4, 8, 4, 8)
        viewer_tools = QHBoxLayout()
        self.mode = QComboBox()
        for title, value in [("Разделитель До / После", "split"), ("Бок о бок", "side"),
                             ("Оригинал", "original"), ("Результат", "result")]:
            self.mode.addItem(title, value)
        self.mode.currentIndexChanged.connect(lambda _: self.viewer.set_mode(self.mode.currentData()))
        viewer_tools.addWidget(self.mode)
        fit = QPushButton("Вписать")
        fit.clicked.connect(lambda: self.viewer.fit_to_window())
        viewer_tools.addWidget(fit)
        self.zoom = QComboBox()
        for percent in (25, 50, 100, 200, 400, 800):
            self.zoom.addItem(f"{percent}%", percent / 100)
        self.zoom.setEditable(True)
        self.zoom.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.zoom.setMaximumWidth(90)
        self.zoom.activated.connect(lambda _: self.viewer.set_zoom(self.zoom.currentData() or 1.0))
        self.zoom.lineEdit().editingFinished.connect(self._custom_zoom)
        viewer_tools.addWidget(self.zoom)
        hold = QPushButton("Оригинал (удерживать)")
        hold.setToolTip("Удерживайте эту кнопку или пробел в области просмотра.")
        hold.pressed.connect(lambda: self.viewer.show_original(True))
        hold.released.connect(lambda: self.viewer.show_original(False))
        viewer_tools.addWidget(hold)
        fullscreen = QPushButton("⛶")
        fullscreen.setToolTip("Полный экран — F11")
        fullscreen.clicked.connect(self.toggle_fullscreen)
        viewer_tools.addWidget(fullscreen)
        layout.addLayout(viewer_tools)
        self.viewer = ComparisonView()
        self.viewer.zoom_changed.connect(lambda value: self.zoom.setEditText(f"{value * 100:.0f}%"))
        layout.addWidget(self.viewer, 1)
        self.preview_notice = QLabel("Предпросмотр до 1200 px. Полный результат вычисляется из оригинала при экспорте.")
        self.preview_notice.setWordWrap(True)
        self.preview_notice.setProperty("secondary", True)
        layout.addWidget(self.preview_notice)
        row = QHBoxLayout()
        self.improve_button = QPushButton("Улучшить изображение — полное разрешение")
        self.improve_button.setObjectName("primary")
        self.improve_button.setMinimumHeight(40)
        self.improve_button.clicked.connect(self.process_full)
        self.cancel_button = QPushButton("Отменить обработку")
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.improve_button, 1)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        splitter.addWidget(center)
        self.controls = ControlsPanel()
        self.controls.changed.connect(self.change_settings)
        self.controls.requested.connect(self._control_action)
        self.controls.reset_requested.connect(self.reset_section)
        splitter.addWidget(self.controls)
        splitter.setSizes([230, 840, 330])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.dimensions = QLabel("Нет изображения")
        self.statusBar().addWidget(self.dimensions, 1)
        self.stage_label = QLabel("Готово")
        self.statusBar().addPermanentWidget(self.stage_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(140)
        self.progress.setValue(0)
        self.statusBar().addPermanentWidget(self.progress)
        self.elapsed_label = QLabel("CPU · 0.0 с")
        self.statusBar().addPermanentWidget(self.elapsed_label)
        self._action("Полный экран", self.toggle_fullscreen, "F11", self)
        self._action("Вписать изображение", lambda: self.viewer.fit_to_window(), "Ctrl+0", self)

    def _action(self, text, callback, shortcut, target):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(callback)
        target.addAction(action)
        return action

    def open_images(self, checked=False, add=False):
        paths, _ = QFileDialog.getOpenFileNames(self, "Открыть изображения", "", IMAGE_FILTER)
        if paths:
            if not add and not self._active_task:
                self.image_list.clear()
                self.documents.clear()
            self.load_paths(paths)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Добавить папку с фотографиями")
        if folder:
            self.load_paths(sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED))

    def load_paths(self, paths):
        first = None
        for raw in paths:
            path = Path(raw).expanduser().resolve()
            if path.suffix.lower() not in SUPPORTED or not path.is_file() or path in self.documents:
                continue
            self.documents[path] = DocumentState(path)
            item = QListWidgetItem(path.name + "\nВ очереди")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(str(path))
            self.image_list.addItem(item)
            first = first or item
        if first is not None:
            self.image_list.setCurrentItem(first)
        self._update_actions()

    def _selected(self, item, previous=None):
        if item is None:
            return
        self._generation += 1
        version = self._generation
        self._cancel_preview()
        if self._load_task:
            self._load_task.cancel()
        self.current_state = self.documents[item.data(Qt.ItemDataRole.UserRole)]
        self.current_document = None
        self.current_analysis = None
        self.full_result = None
        self.preview_source = None
        self.viewer.clear()
        self.controls.set_settings(self.current_settings)
        self.analysis_text.setPlainText("Чтение фотографии и вычисление показателей качества…")
        self.stage_label.setText("Чтение изображения…")
        self._update_actions()
        path = self.current_state.path

        def load(progress, cancelled):
            document = load_image(path)
            if cancelled():
                return None
            preview = preview_copy(document.rgb)
            report = analyze_image(document.rgb)
            thumbnail = preview_copy(preview, 100)
            return document, preview, report, thumbnail
        task = submit(load)
        task.signals.result.connect(lambda data: self._loaded(version, item, data))
        task.signals.error.connect(lambda error: self._load_failed(version, item, error))
        self._load_task = task
        self._retain(task)

    def _loaded(self, version, item, data):
        if self._closing or version != self._generation or data is None:
            return
        self.current_document, self.preview_source, self.current_analysis, thumbnail = data
        self._load_task = None
        document = self.current_document
        height, width = document.rgb.shape[:2]
        item.setText(f"{document.path.name}\n{width} × {height} · {document.format}")
        rgb = np.ascontiguousarray(thumbnail)
        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        item.setIcon(QIcon(QPixmap.fromImage(qimage)))
        warning_text = "\n\n" + "\n".join(document.warnings) if document.warnings else ""
        self.analysis_text.setPlainText(self.current_analysis.to_text() + warning_text)
        self.viewer.set_images(self.preview_source, self.preview_source)
        self.viewer.fit_to_window()
        self.stage_label.setText("Изображение загружено")
        self._update_actions()
        self._update_dimensions()
        self._preview()

    def _load_failed(self, version, item, error):
        if version == self._generation and not self._closing:
            item.setText(item.data(Qt.ItemDataRole.UserRole).name + "\nОшибка чтения")
            self.analysis_text.setPlainText("Не удалось прочитать файл:\n" + error)
            self.stage_label.setText("Ошибка чтения")
            self._update_actions()

    def change_settings(self, values):
        if not self.current_state:
            return
        try:
            settings = replace(self.current_settings, **values)
            # One common validation path is also used by imported presets.
            settings = ProcessingSettings.from_dict(asdict(settings))
        except (ValueError, TypeError) as exc:
            self._error("Некорректные параметры", str(exc))
            self.controls.set_settings(self.current_settings)
            return
        self._set_settings(settings)

    def _set_settings(self, settings):
        if self.current_state and self.current_state.set_settings(settings):
            self._settings_changed()

    def _settings_changed(self):
        self._generation += 1
        self.full_result = None
        self.controls.set_settings(self.current_settings)
        self._cancel_preview()
        self._debounce.start()
        self._update_actions()
        self._update_dimensions()

    def undo(self):
        if self.current_state and self.current_state.can_undo:
            self.current_state.undo()
            self._settings_changed()

    def redo(self):
        if self.current_state and self.current_state.can_redo:
            self.current_state.redo()
            self._settings_changed()

    def reset_settings(self):
        if self.current_state:
            self.current_state.reset()
            self._settings_changed()

    def reset_section(self, keys):
        defaults = asdict(ProcessingSettings())
        self.change_settings({key: defaults[key] for key in keys})

    def _runtime_settings(self, settings, preview=False):
        values = {"ai_model_path": str(self.preferences.value("ai/model_path", settings.ai_model_path)),
                  "ai_device": str(self.preferences.value("ai/device", settings.ai_device)),
                  "ai_tile_size": int(self.preferences.value("ai/tile_size", settings.ai_tile_size)),
                  "max_output_pixels": int(self.preferences.value("performance/max_megapixels", 64)) * 1_000_000}
        if preview:
            # Explicit pixel dimensions refer to the original, not its small proxy.
            if self.current_document is not None:
                h, w = self.current_document.rgb.shape[:2]
                width, height = output_size(w, h, settings)
                factor = min(1.0, 1200 / max(width, height))
                values.update(target_width=max(1, round(width * factor)), target_height=max(1, round(height * factor)),
                              keep_aspect=False)
            # AI inference is always performed on original pixels in full processing/export.
            # Preview honestly uses classical geometry and leaves the AI stage to that action.
            if settings.upscaler == "realesrgan":
                values.update(upscaler="classical")
        return replace(settings, **values)

    def _cancel_preview(self):
        self._preview_generation += 1
        if self._preview_task:
            self._preview_task.cancel()
            self._preview_task = None

    def _preview(self):
        if self.preview_source is None or self._active_task or self._closing:
            return
        self._cancel_preview()
        version = self._generation
        preview_version = self._preview_generation
        source = self.preview_source
        try:
            settings = self._runtime_settings(self.current_settings, preview=True)
        except ValueError as exc:
            self._error("Размер результата", str(exc))
            return
        self.stage_label.setText("Предпросмотр…")
        self.preview_notice.setText("Предпросмотр до 1200 px. " +
            ("AI не вычисляется в быстром предпросмотре; нажмите «Улучшить изображение» для реального AI-результата."
             if self.current_settings.upscaler == "realesrgan" else
             "Шум и резкость на уменьшенной копии могут отличаться от экспорта; полный результат — из оригинала."))

        def run(progress, cancelled):
            return process_image(source, settings, cancelled=cancelled)
        task = submit(run)
        task.signals.result.connect(lambda result: self._preview_done(version, preview_version, source, result))
        task.signals.error.connect(lambda error: self._preview_failed(version, preview_version, error))
        self._preview_task = task
        self._retain(task)

    def _preview_done(self, version, preview_version, source, result):
        if not self._closing and version == self._generation and preview_version == self._preview_generation:
            self.viewer.set_images(source, result)
            self.stage_label.setText("Предпросмотр готов")
            self._preview_task = None

    def _preview_failed(self, version, preview_version, error):
        if version == self._generation and preview_version == self._preview_generation and not self._closing:
            self.stage_label.setText("Ошибка предпросмотра")
            self.analysis_text.append("\nОшибка предпросмотра: " + error)

    def process_full(self):
        if self.current_document is None or self._active_task:
            return
        source = self.current_document.rgb
        settings = self._runtime_settings(self.current_settings)
        version = self._generation

        def run(progress, cancelled):
            return process_image(source, settings, progress=progress, cancelled=cancelled)

        def done(result):
            if version == self._generation:
                self.full_result = result
                self.viewer.set_images(source, result)
                self.preview_notice.setText("Показаны оригинал и результат в полном разрешении. "
                                            "Масштаб 100% соответствует пикселям результата.")
        self._start_operation(run, done, "Обработка полного разрешения")

    def export_current(self):
        if self.current_document is None or self._active_task:
            return
        document = self.current_document
        settings = self._runtime_settings(self.current_settings)
        try:
            size = output_size(document.rgb.shape[1], document.rgb.shape[0], settings)
        except ValueError as exc:
            self._error("Размер результата", str(exc))
            return
        dialog = ExportDialog(document.path, size, self.preferences, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        path, quality = dialog.output_path, dialog.quality.value()
        metadata, overwrite = dialog.metadata.isChecked(), dialog.overwrite

        def run(progress, cancelled):
            result = process_image(document.rgb, settings,
                                   progress=lambda pct, stage: progress(round(pct * 0.95), stage),
                                   cancelled=cancelled)
            if cancelled():
                return None
            progress(95, "Запись файла")
            saved = save_image(document, result, path, quality=quality,
                               keep_metadata=metadata, overwrite=overwrite)
            progress(100, "Файл сохранён")
            return saved

        def done(saved):
            if saved is not None:
                QMessageBox.information(self, "Экспорт завершён", f"Изображение сохранено:\n{saved}")
        self._start_operation(run, done, "Экспорт полного разрешения")

    def export_batch(self):
        if not self.documents or self._active_task:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка экспорта очереди",
                                                  str(self.preferences.value("export/folder", "")))
        if not folder:
            return
        choice = QMessageBox(self)
        choice.setWindowTitle("Настройки очереди")
        choice.setText("Какие параметры применить к изображениям очереди?")
        common = choice.addButton("Текущие для всех", QMessageBox.ButtonRole.AcceptRole)
        individual = choice.addButton("Индивидуальные", QMessageBox.ButtonRole.ActionRole)
        choice.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        choice.exec()
        if choice.clickedButton() not in (common, individual):
            return
        settings_list = [(path, self._runtime_settings(self.current_settings if choice.clickedButton() == common
                                                      else state.settings)) for path, state in self.documents.items()]
        target_dir = Path(folder)
        suffix = str(self.preferences.value("export/format", "png"))
        quality = int(self.preferences.value("export/quality", 95))
        metadata = self.preferences.value("export/metadata", True, type=bool)
        self.preferences.setValue("export/folder", folder)
        batch_cancelled = Event()

        def run(progress, cancelled):
            worker_cancelled = cancelled

            def cancelled():
                return batch_cancelled.is_set() or worker_cancelled()
            successes, errors = [], []
            total = len(settings_list)
            for index, (path, settings) in enumerate(settings_list):
                if cancelled():
                    break
                progress(round(index * 100 / total), f"{index + 1}/{total} · {path.name}")
                try:
                    document = load_image(path)
                    result = process_image(document.rgb, settings,
                        progress=lambda pct, stage: progress(round((index + pct / 100) * 100 / total),
                                                              f"{index + 1}/{total} · {path.name} · {stage}"),
                        cancelled=cancelled)
                    if cancelled():
                        break
                    target = target_dir / f"{path.stem}_улучшено.{suffix}"
                    counter = 2
                    while target.exists() or target.resolve() == path.resolve():
                        target = target_dir / f"{path.stem}_улучшено_{counter}.{suffix}"
                        counter += 1
                    save_image(document, result, target, quality=quality, keep_metadata=metadata)
                    successes.append((path, target))
                    del document, result
                except Exception as exc:
                    if cancelled():
                        break
                    LOGGER.exception("Batch file failed: %s", path)
                    errors.append((path, str(exc)))
            return successes, errors, cancelled()

        def done(payload):
            successes, errors, was_cancelled = payload
            for path, target in successes:
                self._set_queue_status(path, "Сохранено → " + target.name)
            for path, error in errors:
                self._set_queue_status(path, "Ошибка: " + error)
            text = f"Сохранено: {len(successes)}. Ошибок: {len(errors)}."
            if was_cancelled:
                text += " Очередь остановлена. Уже сохранённые файлы доступны в папке экспорта."
            if errors:
                text += "\n\n" + "\n".join(f"{path.name}: {error}" for path, error in errors)
            QMessageBox.information(self, "Пакетная обработка", text)
        self._start_operation(run, done, "Пакетная обработка", cancel_callback=batch_cancelled.set)

    def _set_queue_status(self, path, status):
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(path.name + "\n" + status)
                item.setToolTip(str(path) + "\n" + status)

    def _start_operation(self, fn, done, title, cancel_callback=None):
        self._cancel_preview()
        self._debounce.stop()
        self._started = time.monotonic()
        self._elapsed_timer.start()
        self.progress.setValue(0)
        self.stage_label.setText(title)
        task = submit(fn)
        self._active_task = task
        self._cancel_callback = cancel_callback
        task.signals.progress.connect(self._operation_progress)
        task.signals.result.connect(lambda payload: done(payload) if not self._closing else None)
        task.signals.error.connect(self._operation_error)
        task.signals.finished.connect(lambda: self._operation_finished(task))
        self._retain(task)
        self._update_actions()

    def _operation_progress(self, value, stage):
        if not self._closing:
            self.progress.setValue(value)
            self.stage_label.setText(stage)

    def _operation_error(self, error):
        if not self._closing:
            self.stage_label.setText("Обработка не завершена")
            self._error("Ошибка обработки", error)

    def _operation_finished(self, task):
        if self._active_task is task:
            self._active_task = None
            self._cancel_callback = None
            self._elapsed_timer.stop()
            if self._closing:
                return
            self._update_elapsed()
            self._update_actions()

    def _retain(self, task):
        self._tasks.append(task)
        task.signals.finished.connect(lambda: self._release_task(task))

    def _release_task(self, task):
        if task in self._tasks:
            self._tasks.remove(task)

    def cancel_processing(self):
        if self._active_task:
            if self._cancel_callback is not None:
                self._cancel_callback()
            else:
                self._active_task.cancel()
            self.stage_label.setText("Отмена после текущего этапа…")
            self.cancel_button.setEnabled(False)

    def analyze_current(self):
        if self.current_analysis:
            self.analysis_text.setPlainText(self.current_analysis.to_text())

    def apply_smart(self, mode, noise_only=False):
        if self.current_analysis is None:
            return
        settings = self.current_analysis.suggest_settings(mode)
        if noise_only:
            keys = ("denoise_method", "denoise_strength", "denoise_luma", "denoise_chroma",
                    "detail_preservation", "edge_protection")
            self.change_settings({key: getattr(settings, key) for key in keys})
        else:
            self._set_settings(settings)
        mode_name = {"natural": "Естественное", "balanced": "Сбалансированное", "maximum": "Максимальное"}[mode]
        selected = asdict(settings)
        defaults = asdict(ProcessingSettings())
        names = {"denoise_strength": "Шумоподавление", "sharpen_amount": "Резкость", "exposure": "Экспозиция",
                 "gamma": "Гамма", "clahe": "Локальный контраст", "saturation": "Насыщенность",
                 "temperature": "Температура", "night_mode": "Ночное восстановление", "shadows": "Тени"}
        details = "\n".join(f"• {names[key]}: {value}" for key, value in selected.items()
                            if key in names and value != defaults[key])
        self.analysis_text.setPlainText(self.current_analysis.to_text() +
            f"\n\nВыбрано: {mode_name} улучшение.\nПараметры основаны на измерениях выше.\n{details}")

    def _control_action(self, action):
        actions = {"analyze": self.analyze_current, "auto_noise": lambda: self.apply_smart("balanced", True),
                   "reset": self.reset_settings, "export": self.export_current, "batch": self.export_batch,
                   "why": self.explain_analysis, "retouch": self.retouch_original}
        if action.startswith("smart:"):
            self.apply_smart(action.split(":", 1)[1])
        elif action in actions:
            actions[action]()

    def explain_analysis(self):
        if self.current_analysis is not None:
            QMessageBox.information(self, "Почему выбраны эти параметры?", self.analysis_text.toPlainText())

    def retouch_original(self):
        if self.current_document is None or self._active_task:
            return
        from imageboost_pro.ui.restoration_dialog import RestorationDialog
        dialog = RestorationDialog(self.current_document, self.preferences, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved_path is not None:
            self.load_paths([dialog.saved_path])

    def _apply_preset(self, index):
        preset = self.presets.itemData(index)
        if preset is not None:
            if self.presets.itemText(index) == "Автоматическое улучшение":
                self.apply_smart("balanced")
            else:
                self._set_settings(preset)

    def export_preset(self):
        if not self.current_state:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить пресет", "Мой пресет.json", "Пресет JSON (*.json)")
        if path:
            try:
                save_preset(path, self.current_settings)
            except (OSError, ValueError) as exc:
                self._error("Не удалось сохранить пресет", str(exc))

    def import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Импортировать пресет", "", "Пресет JSON (*.json)")
        if path:
            try:
                settings = load_preset(path)
                self.presets.addItem(Path(path).stem, settings)
                self.presets.setCurrentIndex(self.presets.count() - 1)
                self._set_settings(settings)
            except (OSError, ValueError) as exc:
                self._error("Не удалось загрузить пресет", str(exc))

    def show_settings(self):
        dialog = SettingsDialog(self.preferences, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_theme()
            self._apply_performance()
            if self.current_document is not None:
                self._settings_changed()

    def _apply_performance(self):
        cv2.setNumThreads(int(self.preferences.value("performance/threads", 2)))

    def toggle_theme(self):
        self.preferences.setValue("ui/theme", "light" if self.preferences.value("ui/theme", "dark") == "dark" else "dark")
        self._apply_theme()

    def _apply_theme(self):
        dark = self.preferences.value("ui/theme", "dark") == "dark"
        background, panel, surface = (("#17191D", "#22252B", "#2B2F36") if dark else
                                      ("#F3F4F6", "#FFFFFF", "#E8EBEF"))
        text, muted, border = (("#F3F4F6", "#A8AFBA", "#383D45") if dark else
                              ("#202630", "#536171", "#C7CDD6"))
        app = QApplication.instance()
        app.setStyle("Fusion")
        palette = QPalette()
        for role, color in [(QPalette.ColorRole.Window, background), (QPalette.ColorRole.WindowText, text),
                            (QPalette.ColorRole.Base, panel), (QPalette.ColorRole.AlternateBase, surface),
                            (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, surface),
                            (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.Highlight, "#4F8EF7"),
                            (QPalette.ColorRole.HighlightedText, "#FFFFFF"),
                            (QPalette.ColorRole.ToolTipBase, surface), (QPalette.ColorRole.ToolTipText, text)]:
            palette.setColor(role, QColor(color))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(muted))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(muted))
        app.setPalette(palette)
        app.setStyleSheet(f"""
            QWidget {{ font-family: 'Segoe UI', 'DejaVu Sans', sans-serif; font-size: 12px; }}
            QMainWindow, QDialog {{ background: {background}; }}
            QToolBar {{ background: {panel}; border-bottom: 1px solid {border}; spacing: 5px; padding: 7px; }}
            QPushButton, QToolButton {{ padding: 7px 9px; border: 1px solid {border}; border-radius: 4px; }}
            QPushButton:hover, QToolButton:hover {{ border-color: #4F8EF7; }}
            QPushButton#primary {{ background: #326DD0; color: white; font-weight: 600; border-color: #4F8EF7; }}
            QLabel#brand {{ font-size: 17px; font-weight: 700; color: #4F8EF7; }}
            QLabel[secondary='true'] {{ color: {muted}; font-size: 11px; }}
            QListWidget, QTextBrowser, QScrollArea {{ border: 1px solid {border}; border-radius: 3px; }}
            QListWidget::item {{ padding: 6px; }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ min-height: 25px; }}
            QProgressBar {{ border: 1px solid {border}; text-align: center; border-radius: 3px; }}
            QProgressBar::chunk {{ background: #4F8EF7; }}
            QSplitter::handle {{ background: {border}; width: 1px; }}
        """)

    def _update_actions(self):
        ready = self.current_document is not None and self._active_task is None
        busy = self._active_task is not None
        self.save_action.setEnabled(ready)
        self.retouch_action.setEnabled(ready)
        self.improve_button.setEnabled(ready)
        self.undo_action.setEnabled(not busy and bool(self.current_state and self.current_state.can_undo))
        self.redo_action.setEnabled(not busy and bool(self.current_state and self.current_state.can_redo))
        self.cancel_button.setEnabled(busy)
        self.controls.setEnabled(not busy and self.current_document is not None)
        self.image_list.setEnabled(not busy)
        self.presets.setEnabled(not busy and self.current_state is not None)
        self.open_action.setEnabled(not busy)

    def _update_dimensions(self):
        if self.current_document is None:
            self.dimensions.setText("Нет изображения")
            return
        height, width = self.current_document.rgb.shape[:2]
        try:
            new_width, new_height = output_size(width, height, self.current_settings)
            algorithm = ("Real-ESRGAN" if self.current_settings.upscaler == "realesrgan" else
                         {"nearest": "Ближайший", "bilinear": "Билинейный", "bicubic": "Бикубический",
                          "lanczos": "Lanczos", "area": "По площади"}[self.current_settings.interpolation])
            self.dimensions.setText(f"{width} × {height} → {new_width} × {new_height} px · "
                                    f"{self.current_document.format} · {algorithm}")
        except ValueError as exc:
            self.dimensions.setText(str(exc))

    def _update_elapsed(self):
        elapsed = max(0, time.monotonic() - self._started)
        device = str(self.preferences.value("ai/device", "auto"))
        name = ("CUDA (AI)" if device == "cuda" else "CPU (AI)" if device == "cpu" else "AI · авто") if (
            self.current_settings.upscaler == "realesrgan") else "CPU"
        self.elapsed_label.setText(f"{name} · {elapsed:.1f} с")

    def _custom_zoom(self):
        try:
            value = float(self.zoom.currentText().strip().rstrip("%")) / 100
            self.viewer.set_zoom(max(0.01, min(32, value)))
        except ValueError:
            pass

    def toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    def _image_context_menu(self, position):
        item = self.image_list.itemAt(position)
        if item is None or self._active_task:
            return
        menu = QMenu(self)
        remove = menu.addAction("Убрать из очереди")
        if menu.exec(self.image_list.mapToGlobal(position)) == remove:
            path = item.data(Qt.ItemDataRole.UserRole)
            self.image_list.takeItem(self.image_list.row(item))
            self.documents.pop(path, None)
            if not self.documents:
                self._generation += 1
                self._cancel_preview()
                self.current_document = self.current_state = self.current_analysis = self.preview_source = None
                self.full_result = None
                self.viewer.clear()
                self._update_actions()
                self._update_dimensions()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self._active_task:
            return
        paths = []
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED))
            else:
                paths.append(path)
        self.load_paths(paths)
        event.acceptProposedAction()

    def show_help(self):
        QMessageBox.information(self, "ImageBoost Pro — справка",
            "1. Откройте фото или перетащите файлы в окно.\n"
            "2. Изучите анализ качества и выберите умный режим, пресет или ручные настройки.\n"
            "3. Сравните результат разделителем. Колесо — масштаб, перетаскивание — перемещение, "
            "пробел — оригинал.\n4. «Улучшить изображение» проверит полный результат; "
            "«Сохранить» выполнит обработку оригинала и экспорт.\n\n"
            "Ctrl+O — открыть; Ctrl+S — сохранить; Ctrl+Z / Ctrl+Y — история; "
            "Ctrl+R — исходные параметры; Ctrl+0 — вписать; F11 — полный экран.\n\n"
            "Оригинал сохраняется до явного подтверждения перезаписи. AI требует необязательного "
            "PyTorch и локальных совместимых весов; обычные алгоритмы работают без интернета.")

    def _error(self, title, error):
        LOGGER.error("%s: %s", title, error)
        QMessageBox.warning(self, title, error)

    def closeEvent(self, event):
        self._closing = True
        self._generation += 1
        self._debounce.stop()
        self._elapsed_timer.stop()
        self.preferences.setValue("ui/geometry", self.saveGeometry())
        for task in list(self._tasks):
            task.cancel()
        event.accept()
