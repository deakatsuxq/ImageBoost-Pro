"""Reusable, real processing controls. Values match ProcessingSettings exactly."""
from dataclasses import asdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSlider, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from imageboost_pro.core.settings import ProcessingSettings


class ControlsPanel(QScrollArea):
    changed = Signal(dict)
    requested = Signal(str)
    reset_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)
        self.setMaximumWidth(420)
        self._updating = False
        self.editors = {}
        self._groups = {}
        body = QWidget()
        self.layout = QVBoxLayout(body)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)
        self.setWidget(body)

        f = self._category("Автоматическое улучшение", expanded=True)
        self._action(f, "Анализировать фотографию", "analyze")
        for title, mode in [("Естественно", "natural"), ("Сбалансированно", "balanced"),
                            ("Максимальное улучшение", "maximum")]:
            self._action(f, title, "smart:" + mode)
        self._action(f, "Почему выбраны эти параметры?", "why")
        note = QLabel("Умный режим выбирает параметры по измеренным дефектам снимка. Объяснение — слева.")
        note.setWordWrap(True)
        note.setProperty("secondary", True)
        f.addRow(note)

        f = self._category("Увеличение изображения")
        self._combo(f, "upscaler", "Обработка", [("Классическая", "classical"),
                    ("Real-ESRGAN (нужны веса)", "realesrgan")])
        self._combo(f, "interpolation", "Интерполяция", [("Ближайший сосед", "nearest"),
                    ("Билинейная", "bilinear"), ("Бикубическая", "bicubic"),
                    ("Lanczos", "lanczos"), ("По площади", "area")])
        self._number(f, "scale", "Масштаб ×", 0.1, 8, 0.5, 2)
        self._number(f, "target_width", "Ширина, px (0 = авто)", 0, 30000)
        self._number(f, "target_height", "Высота, px (0 = авто)", 0, 30000)
        self._check(f, "keep_aspect", "Сохранять пропорции")
        note = QLabel("Явные размеры имеют приоритет над масштабом. AI требует локальную модель в настройках.")
        note.setWordWrap(True)
        f.addRow(note)

        f = self._category("Удаление шума")
        self._combo(f, "denoise_method", "Алгоритм", [("Non-Local Means", "nlm"),
                    ("Билатеральный", "bilateral"), ("Гауссов", "gaussian"),
                    ("Медианный", "median"), ("Вейвлетный", "wavelet")])
        for key, label in [("denoise_strength", "Сила, %"), ("denoise_luma", "Яркостный шум, %"),
                           ("denoise_chroma", "Цветной шум, %"),
                           ("detail_preservation", "Сохранение деталей, %"),
                           ("edge_protection", "Защита контуров, %")]:
            self._number(f, key, label, 0, 100)
        self._action(f, "Определить шум автоматически", "auto_noise")

        f = self._category("Резкость и детали")
        self._combo(f, "sharpen_method", "Алгоритм", [("Контурная резкость", "unsharp"),
                    ("High-Pass", "highpass"), ("Лапласиан", "laplacian"), ("Адаптивный", "adaptive")])
        self._number(f, "sharpen_amount", "Интенсивность, %", 0, 100)
        self._number(f, "sharpen_radius", "Радиус, px", 0.1, 10, 0.1, 1)
        self._number(f, "sharpen_threshold", "Порог", 0, 100)
        for key, label in [("detail_amount", "Локальные детали, %"), ("microcontrast", "Микроконтраст, %"),
                           ("halo_protection", "Защита от ореолов, %")]:
            self._number(f, key, label, 0, 100)

        f = self._category("Восстановление фотографии")
        self._action(f, "Ретушь повреждений по маске…", "retouch")
        self._number(f, "deblur", "Уменьшение размытия, %", 0, 100)
        self._number(f, "jpeg_artifacts", "JPEG-артефакты, %", 0, 100)
        self._number(f, "banding_reduction", "Полосы на градиентах, %", 0, 100)
        note = QLabel("Умеренное восстановление повышает читаемость деталей; утраченные данные не воссоздаются.")
        note.setWordWrap(True)
        f.addRow(note)

        f = self._category("Ночное восстановление", expanded=True)
        self._combo(f, "night_mode", "Режим", [("Выключено", "off"), ("Естественный", "natural"),
                    ("Сбалансированный", "balanced"), ("Максимальная видимость", "maximum")])
        self._number(f, "night_strength", "Сила восстановления, %", 0, 100)
        self._number(f, "highlight_protection", "Защита светов, %", 0, 100)

        f = self._category("Цвет и освещение")
        self._number(f, "exposure", "Экспозиция, EV", -5.0, 5.0, 0.1, 1)
        self._number(f, "gamma", "Гамма", 0.1, 5, 0.05, 2)
        for key, label in [("brightness", "Яркость"), ("contrast", "Контраст"),
                           ("saturation", "Насыщенность"), ("vibrance", "Сочность"),
                           ("temperature", "Температура"), ("tint", "Оттенок"),
                           ("shadows", "Тени"), ("highlights", "Света"),
                           ("black_point", "Точка чёрного"), ("white_point", "Точка белого")]:
            self._number(f, key, label, -100, 100)
        self._check(f, "white_balance", "Автоматический баланс белого")

        f = self._category("Дополнительная обработка")
        self._number(f, "clahe", "Локальный контраст CLAHE, %", 0, 100)
        self._number(f, "retinex", "Выравнивание света Retinex, %", 0, 100)
        self._action(f, "Вернуть исходные параметры", "reset")

        f = self._category("Экспорт")
        self._action(f, "Сохранить текущее изображение…", "export")
        self._action(f, "Обработать очередь в папку…", "batch")
        self.layout.addStretch()
        self.set_settings(ProcessingSettings())

    def _category(self, title, expanded=False):
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QToolButton()
        header.setText(title)
        header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        header.setMinimumHeight(34)
        contents = QWidget()
        form = QFormLayout(contents)
        form.setContentsMargins(8, 8, 8, 12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        contents.setVisible(expanded)
        header.toggled.connect(lambda checked: (contents.setVisible(checked), header.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)))
        outer.addWidget(header)
        outer.addWidget(contents)
        self.layout.addWidget(container)
        keys = []
        self._groups[id(form)] = keys
        if title not in ("Автоматическое улучшение", "Экспорт"):
            button = QPushButton("Сбросить раздел")
            button.clicked.connect(lambda: self.reset_requested.emit(tuple(keys)))
            form.addRow(button)
        return form

    def _register(self, form, key, editor):
        self.editors[key] = editor
        self._groups[id(form)].append(key)
        editor.setObjectName(key)
        editor.setToolTip("Изменения применяются к предпросмотру; оригинал сохраняется до экспорта.")

    def _number(self, form, key, label, minimum, maximum, step=1, decimals=0):
        editor = QDoubleSpinBox() if decimals else QSpinBox()
        if decimals:
            editor.setDecimals(decimals)
        editor.setRange(minimum, maximum)
        editor.setSingleStep(step)
        editor.setKeyboardTracking(False)
        editor.valueChanged.connect(lambda value: self._emit(key, value))
        self._register(form, key, editor)
        if not decimals and maximum == 100:
            row = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(minimum, maximum)
            slider.setMinimumWidth(80)
            slider.valueChanged.connect(editor.setValue)
            editor.valueChanged.connect(slider.setValue)
            editor.setFixedWidth(64)
            row.addWidget(slider, 1)
            row.addWidget(editor)
            form.addRow(label, row)
        else:
            form.addRow(label, editor)

    def _combo(self, form, key, label, choices):
        editor = QComboBox()
        for text, value in choices:
            editor.addItem(text, value)
        editor.currentIndexChanged.connect(lambda _: self._emit(key, editor.currentData()))
        self._register(form, key, editor)
        form.addRow(label, editor)

    def _check(self, form, key, label):
        editor = QCheckBox(label)
        editor.toggled.connect(lambda value: self._emit(key, value))
        self._register(form, key, editor)
        form.addRow(editor)

    def _action(self, form, title, action):
        button = QPushButton(title)
        button.clicked.connect(lambda: self.requested.emit(action))
        form.addRow(button)

    def _emit(self, key, value):
        if not self._updating:
            self.changed.emit({key: value})

    def set_settings(self, settings):
        self._updating = True
        try:
            for key, value in asdict(settings).items():
                editor = self.editors.get(key)
                if isinstance(editor, QComboBox):
                    editor.setCurrentIndex(editor.findData(value))
                elif isinstance(editor, QCheckBox):
                    editor.setChecked(value)
                elif editor is not None:
                    editor.setValue(value)
        finally:
            self._updating = False
