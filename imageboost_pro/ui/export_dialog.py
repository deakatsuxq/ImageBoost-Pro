"""Export options without silently overwriting originals or other files."""
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)


FORMATS = [("PNG — без потерь", "png"), ("JPEG", "jpg"), ("WebP", "webp"),
           ("TIFF — без потерь", "tiff"), ("BMP", "bmp")]


class ExportDialog(QDialog):
    def __init__(self, source_path, output_size, preferences: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Экспорт изображения")
        self.setMinimumWidth(490)
        self.preferences = preferences
        self.source_path = Path(source_path)
        self.overwrite = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.folder = QLineEdit(str(preferences.value("export/folder", str(self.source_path.parent))))
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow("Папка", row)
        self.name = QLineEdit(self.source_path.stem + "_улучшено")
        form.addRow("Имя без расширения", self.name)
        self.format = QComboBox()
        for label, key in FORMATS:
            self.format.addItem(label, key)
        index = self.format.findData(preferences.value("export/format", "png"))
        self.format.setCurrentIndex(max(0, index))
        form.addRow("Формат", self.format)
        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(int(preferences.value("export/quality", 95)))
        form.addRow("Качество JPEG / WebP", self.quality)
        self.metadata = QCheckBox("Сохранять доступные метаданные и цветовой профиль")
        self.metadata.setChecked(preferences.value("export/metadata", True, type=bool))
        form.addRow(self.metadata)
        form.addRow("Разрешение", QLabel(f"{output_size[0]} × {output_size[1]} px"))
        note = QLabel("Обработка выполняется заново из оригинала в полном разрешении. "
                      "Прозрачность сохраняется в PNG, WebP и TIFF; для JPEG фон белый. "
                      "Размер файла зависит от изображения и сжатия.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox()
        buttons.addButton("Экспортировать", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def output_path(self):
        return Path(self.folder.text()).expanduser() / (self.name.text().strip() + "." + self.format.currentData())

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Папка экспорта", self.folder.text())
        if path:
            self.folder.setText(path)

    def accept(self):
        name = self.name.text().strip()
        if not name or any(c in name for c in '<>:"/\\|?*') or name.endswith((".", " ")):
            QMessageBox.warning(self, "Некорректное имя", "Введите имя файла без запрещённых символов.")
            return
        folder = Path(self.folder.text()).expanduser()
        if not folder.is_dir():
            QMessageBox.warning(self, "Папка не найдена", "Выберите существующую папку экспорта.")
            return
        target = self.output_path
        if target.exists():
            is_original = target.resolve() == self.source_path.resolve()
            question = ("Это исходный файл. Перезаписать оригинал? Это действие нельзя отменить."
                        if is_original else f"Файл «{target.name}» уже существует. Перезаписать его?")
            if QMessageBox.question(self, "Подтверждение перезаписи", question,
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            self.overwrite = True
        self.preferences.setValue("export/folder", str(folder))
        self.preferences.setValue("export/format", self.format.currentData())
        self.preferences.setValue("export/quality", self.quality.value())
        self.preferences.setValue("export/metadata", self.metadata.isChecked())
        super().accept()
