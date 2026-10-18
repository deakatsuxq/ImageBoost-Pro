# Архитектура ImageBoost Pro

Приложение Python 3.11+ / PySide6. Все вычисления работают с RGB uint8 H×W×3;
альфа-канал и метаданные хранятся отдельно. UI не импортирует тяжёлые AI-зависимости при запуске.

* `core.settings.ProcessingSettings` — неизменяемый набор параметров конвейера (dataclass).
* `core.pipeline.process_image(rgb, settings, progress=None, cancelled=None)` — неразрушающая
  обработка, возвращает RGB uint8. `progress(percent, stage_ru)` вызывается на реальных границах этапов;
  `cancelled()` приводит к `ProcessingCancelled`. `core.pipeline.output_size(width, height, settings)`
  вычисляет итоговое разрешение. Нейтральные параметры сохраняют исходные пиксели.
* `core.image_analysis.analyze_image(rgb)` — AnalysisReport с численными метриками,
  русскими рекомендациями, `to_text()` и `suggest_settings(mode)` для natural/balanced/maximum.
* `utils.image_io.load_image(path)` — ImageDocument: rgb, alpha, path, format, icc_profile, exif.
  EXIF-ориентация нормализуется при чтении. `save_image(document, rgb, path, quality=95,
  keep_metadata=True, overwrite=False)` — экспорт с масштабированием альфы под результат.
* `utils.presets.BUILTIN_PRESETS` — словарь русских имён → ProcessingSettings;
  `save_preset`/`load_preset` — безопасный JSON без объектов Python.
* `app_state.DocumentState` — путь, параметры и история параметров без копий изображений.
* `workers.tasks.TaskRunner` — QRunnable с сигналами result(object), error(str), progress(int,str),
  finished(); `submit(fn)` передаёт в `fn(progress, cancelled)` колбэки. Отмена кооперативная.
* `ui.comparison_view.ComparisonView` — `set_images(original, result)`, `set_mode(mode)`
  (split/side/original/result), `fit_to_window()`, `set_zoom(factor)`, `show_original(bool)`,
  сигнал `zoom_changed(float)`. Масштаб/позиция общие в нормализованных координатах.
* `ai.realesrgan_engine.upscale(rgb, model_path, scale, tile_size=256, device='auto',
  progress=None, cancelled=None)` — настоящая RRDB inference официальных x2/x4 весов,
  без молчаливой подмены интерполяцией. Модели подключаются локально, не скачиваются при запуске.
* `ui.main_window.MainWindow` связывает документы, настройки, анализ, просмотр и фоновые задачи.

Предпросмотр использует уменьшенную копию (до 1200 px); полный результат всегда вычисляется
из оригинала при обработке/экспорте. Устаревшие задачи не заменяют актуальное изображение.
Пакетная обработка освобождает документ после каждого файла и продолжает очередь после ошибки.

Windows: PyInstaller onedir, EXE внутри ZIP. Linux служит средой разработки и headless-проверок;
Windows EXE собирается и проверяется только Windows runner в GitHub Actions.
