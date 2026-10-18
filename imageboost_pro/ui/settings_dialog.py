"""Persistent preferences and opt-in installation of genuine AI model weights."""
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from imageboost_pro.ui.export_dialog import FORMATS
from imageboost_pro.workers.tasks import submit


class SettingsDialog(QDialog):
    def __init__(self, preferences: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки ImageBoost Pro")
        self.setMinimumSize(600, 430)
        self.preferences = preferences
        self._tasks = []
        self._download = None
        self._runtime_ready = False
        self._installed_models = {}
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        general = QWidget()
        form = QFormLayout(general)
        self.theme = QComboBox()
        self.theme.addItem("Тёмная", "dark")
        self.theme.addItem("Светлая", "light")
        self.theme.setCurrentIndex(max(0, self.theme.findData(preferences.value("ui/theme", "dark"))))
        form.addRow("Тема", self.theme)
        form.addRow("Язык", QLabel("Русский"))
        self.folder = QLineEdit(str(preferences.value("export/folder", str(Path.home()))))
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(self._browse_folder)
        row = QHBoxLayout()
        row.addWidget(self.folder)
        row.addWidget(browse)
        form.addRow("Папка экспорта", row)
        self.format = QComboBox()
        for label, key in FORMATS:
            self.format.addItem(label, key)
        self.format.setCurrentIndex(max(0, self.format.findData(preferences.value("export/format", "png"))))
        form.addRow("Формат по умолчанию", self.format)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(int(preferences.value("export/quality", 95)))
        form.addRow("Качество JPEG / WebP", self.quality)
        self.metadata = QCheckBox("Сохранять метаданные и цветовой профиль")
        self.metadata.setChecked(preferences.value("export/metadata", True, type=bool))
        form.addRow(self.metadata)
        tabs.addTab(general, "Общие и экспорт")

        performance = QWidget()
        form = QFormLayout(performance)
        self.threads = QSpinBox()
        self.threads.setRange(1, 32)
        self.threads.setValue(int(preferences.value("performance/threads", 2)))
        form.addRow("Потоки OpenCV", self.threads)
        self.device = QComboBox()
        for label, key in [("Автоматически", "auto"), ("Процессор (CPU)", "cpu"), ("NVIDIA CUDA", "cuda")]:
            self.device.addItem(label, key)
        self.device.setCurrentIndex(max(0, self.device.findData(preferences.value("ai/device", "auto"))))
        form.addRow("Устройство AI", self.device)
        self.tile = QSpinBox()
        self.tile.setRange(32, 1024)
        self.tile.setSingleStep(32)
        self.tile.setValue(int(preferences.value("ai/tile_size", 256)))
        form.addRow("Размер блока AI, px", self.tile)
        self.max_pixels = QSpinBox()
        self.max_pixels.setRange(1, 256)
        self.max_pixels.setValue(int(preferences.value("performance/max_megapixels", 64)))
        form.addRow("Лимит результата, мегапикселей", self.max_pixels)
        note = QLabel("Предпросмотр: до 1200 px. Файлы очереди обрабатываются по одному. "
                      "Уменьшите размер блока при нехватке памяти GPU. Лимит результата "
                      "предотвращает случайное создание слишком больших изображений.")
        note.setWordWrap(True)
        form.addRow(note)
        tabs.addTab(performance, "Производительность")

        ai_page = QWidget()
        form = QFormLayout(ai_page)
        self.model_path = QLineEdit(str(preferences.value("ai/model_path", "")))
        model_browse = QPushButton("Выбрать .pth…")
        model_browse.clicked.connect(self._browse_model)
        row = QHBoxLayout()
        row.addWidget(self.model_path, 1)
        row.addWidget(model_browse)
        form.addRow("Активные веса", row)
        self.model_combo = QComboBox()
        self.model_combo.activated.connect(self._model_selected)
        form.addRow("Официальная модель", self.model_combo)
        self.download_button = QPushButton("Скачать выбранную модель…")
        self.download_button.clicked.connect(self._download_model)
        form.addRow(self.download_button)
        self.ai_status = QLabel("Проверка PyTorch и установленных моделей…")
        self.ai_status.setWordWrap(True)
        form.addRow(self.ai_status)
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.hide()
        form.addRow(self.download_progress)
        notice = QLabel("Нейросетевой модуль необязателен. Лёгкая Windows-сборка работает "
                        "классическими алгоритмами без PyTorch. Скачивание весов требует интернета. "
                        "AI может добавлять правдоподобные, но неточные детали. "
                        "Восстановление лиц GFPGAN в этой версии не включено.")
        notice.setWordWrap(True)
        form.addRow(notice)
        tabs.addTab(ai_page, "Нейросети")
        buttons = QDialogButtonBox()
        buttons.addButton("Сохранить", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._check_ai()

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Папка экспорта", self.folder.text())
        if path:
            self.folder.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Совместимая модель Real-ESRGAN", "", "Веса модели (*.pth)")
        if path:
            self.model_path.setText(path)

    def _check_ai(self):
        def check(progress, cancelled):
            from imageboost_pro.ai import ModelManager, runtime_status
            manager = ModelManager()
            return runtime_status(), manager.list_models(), manager.installed_models()
        task = submit(check)
        task.signals.result.connect(self._ai_checked)
        task.signals.error.connect(self._ai_error)
        self._tasks.append(task)

    def _ai_checked(self, result):
        status, models, installed = result
        self._runtime_ready = status.get("available", False)
        self._installed_models = installed
        self.model_combo.clear()
        for spec in models:
            suffix = " — установлена" if spec.key in installed else ""
            self.model_combo.addItem(spec.label + suffix, spec)
        device = "CUDA" if status.get("cuda") else "CPU"
        self.ai_status.setText((f"PyTorch доступен, устройство: {device}. " if self._runtime_ready else
                               "PyTorch недоступен: " + str(status.get("error", "не установлен")) + ". ") +
                              f"Установлено моделей: {len(installed)}.")
        self.download_button.setEnabled(self._runtime_ready)
        if not self._runtime_ready:
            self.download_button.setToolTip("Для загрузки и проверки весов используйте AI-дистрибутив с PyTorch.")

    def _model_selected(self, index):
        spec = self.model_combo.itemData(index)
        if spec is not None and spec.key in self._installed_models:
            self.model_path.setText(str(self._installed_models[spec.key]))

    def _ai_error(self, error):
        self.ai_status.setText("Модуль AI недоступен: " + error)
        self.download_button.setEnabled(False)

    def _download_model(self):
        spec = self.model_combo.currentData()
        if spec is None:
            return
        size = getattr(spec, "size_mb", 70)
        text = (f"Скачать официальные веса «{spec.label}» (около {size} МБ) из GitHub? "
                "Загрузка не устанавливает PyTorch и может занять несколько минут.")
        if QMessageBox.question(self, "Загрузка модели", text, QMessageBox.StandardButton.Yes |
                                QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
                                ) != QMessageBox.StandardButton.Yes:
            return
        self.download_progress.show()
        self.download_progress.setValue(0)
        self.download_button.setEnabled(False)

        def download(progress, cancelled):
            from imageboost_pro.ai import ModelManager
            return ModelManager().download_model(spec.key,
                progress=lambda percent: progress(int(percent), "Загрузка весов"), cancelled=cancelled)
        task = submit(download)
        task.signals.progress.connect(lambda value, _: self.download_progress.setValue(value))
        task.signals.result.connect(self._download_done)
        task.signals.error.connect(self._download_error)
        self._download = task
        self._tasks.append(task)

    def _download_done(self, path):
        self.model_path.setText(str(path))
        self.download_button.setEnabled(True)
        self.ai_status.setText("Веса сохранены. Сохраните настройки, чтобы использовать модель.")
        self._download = None

    def _download_error(self, error):
        self.download_button.setEnabled(True)
        self.ai_status.setText("Загрузка не завершена: " + error)
        self._download = None

    def accept(self):
        if self._download:
            QMessageBox.information(self, "Загрузка модели", "Дождитесь загрузки или нажмите «Отмена».")
            return
        values = {"ui/theme": self.theme.currentData(), "export/folder": self.folder.text(),
                  "export/format": self.format.currentData(), "export/quality": self.quality.value(),
                  "export/metadata": self.metadata.isChecked(), "performance/threads": self.threads.value(),
                  "performance/max_megapixels": self.max_pixels.value(), "ai/device": self.device.currentData(),
                  "ai/tile_size": self.tile.value(), "ai/model_path": self.model_path.text().strip()}
        for key, value in values.items():
            self.preferences.setValue(key, value)
        super().accept()

    def reject(self):
        for task in self._tasks:
            task.cancel()
        super().reject()
