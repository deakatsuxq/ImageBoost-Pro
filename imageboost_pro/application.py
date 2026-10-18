"""Desktop entry point and an executable-level functional readiness check."""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import tempfile
import time


def _configure_logging() -> None:
    from PySide6.QtCore import QStandardPaths
    folder = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            folder / "imageboost.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
    except OSError:
        handler = logging.NullHandler()
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _smoke_test(app, window, output: Path) -> int:
    """Exercise real Qt, async loading/processing, display and lossless export.

    Runs inside the bundled executable on Windows as well as from source. A
    watchdog turns a missing signal / blocked worker into a failed build.
    """
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import QTimer

    from imageboost_pro.core.image_analysis import analyze_image
    from imageboost_pro.core.pipeline import process_image
    from imageboost_pro.core.settings import ProcessingSettings
    from imageboost_pro.utils.image_io import ImageDocument, load_image, save_image
    from imageboost_pro.ui.restoration_dialog import RestorationDialog

    started = time.monotonic()
    temporary = tempfile.TemporaryDirectory(prefix="imageboost-smoke-")
    folder = Path(temporary.name)
    y, x = np.mgrid[:96, :160]
    dark = np.stack((10 + x // 5, 8 + y // 4, 12 + (x + y) // 8), axis=2).astype(np.uint8)
    dark[15:25, 120:140] = 235
    fixture = folder / "Ночное фото.png"
    Image.fromarray(dark).save(fixture)
    result_code = 1
    phase = "load"
    retouch = None

    def finish(report: dict, code: int) -> None:
        nonlocal result_code
        result_code = code
        timer.stop()
        report["seconds"] = round(time.monotonic() - started, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if retouch is not None:
            retouch.close()
        window.close()
        app.exit(code)

    def check() -> None:
        nonlocal phase, retouch
        try:
            if time.monotonic() - started > 90:
                raise TimeoutError(f"Проверка приложения превысила 90 секунд: {phase}")
            if phase == "load" and window.current_document is not None:
                assert window.current_document.size == (160, 96), "Неверные размеры исходного файла"
                report = analyze_image(window.current_document.rgb)
                assert report.metrics["brightness"] < 100, "Анализ не распознал тёмный снимок"
                window.change_settings({"night_mode": "balanced", "denoise_strength": 15, "scale": 1.5})
                window.process_full()
                phase = "process"
            elif phase == "process" and window.full_result is not None:
                result = window.full_result
                assert result.shape == (144, 240, 3), f"Неверный размер результата: {result.shape}"
                assert float(result.mean()) > float(dark.mean()), "Ночное восстановление не осветлило фото"
                exported = folder / "Результат.png"
                save_image(window.current_document, result, exported)
                restored = load_image(exported)
                assert np.array_equal(restored.rgb, result), "PNG экспорт изменил пиксели"
                # Exercise lazily imported SciPy/skimage/PyWavelets code inside
                # the actual frozen runtime; source tests alone cannot catch a
                # missing PyInstaller hidden import or bundled data file.
                noisy = np.clip(115 + np.random.default_rng(7).normal(0, 15, (48, 48, 3)),
                                0, 255).astype(np.uint8)
                denoised = process_image(noisy, ProcessingSettings(
                    denoise_method="wavelet", denoise_strength=80,
                    detail_preservation=0, edge_protection=0))
                assert np.mean((denoised.astype(float) - 115) ** 2) < np.mean((noisy.astype(float) - 115) ** 2)
                deblurred = process_image(dark, ProcessingSettings(deblur=25))
                assert deblurred.shape == dark.shape and deblurred.dtype == np.uint8
                for extension in ("jpg", "webp", "tiff", "bmp"):
                    converted = folder / f"Экспорт.{extension}"
                    save_image(window.current_document, result, converted)
                    assert load_image(converted).size == restored.size, f"Ошибка экспорта {extension}"
                scratched = np.full((48, 64, 3), 90, dtype=np.uint8)
                scratched[8:40, 30:34] = 230
                retouch = RestorationDialog(ImageDocument(scratched, path=folder / "Царапина.png"),
                                            window.preferences, window)
                retouch.show()
                retouch.view.add_stroke([(32, 8), (32, 40)], radius=3)
                retouch.process_mask()
                phase = "retouch"
            elif phase == "retouch" and retouch.restored_rgb is not None and retouch._task is None:
                assert float(retouch.restored_rgb[10:38, 30:34].mean()) < 120, "Ретушь не убрала царапину"
                assert int(retouch.document.rgb[20, 32, 0]) == 230, "Ретушь изменила исходник"
                assert retouch.grab().save(str(output.with_name(output.stem + "-retouch.png")))
                retouch.close()
                screenshot = window.grab()
                assert not screenshot.isNull(), "Qt не отрисовал окно"
                assert screenshot.save(str(output.with_suffix(".png"))), "Не удалось сохранить снимок окна"
                finish({"status": "ok", "checks": ["Qt window", "async image load", "quality analysis",
                        "night recovery", "async full processing", "1.5x resize", "lossless Unicode PNG export",
                        "wavelet denoise", "Richardson-Lucy deconvolution", "JPEG/WebP/TIFF/BMP export",
                        "original-pixel brush mask", "async OpenCV inpainting", "window render"],
                        "source_size": [160, 96], "result_size": [240, 144]}, 0)
        except Exception as exc:
            logging.exception("Проверка приложения не пройдена")
            finish({"status": "failed", "phase": phase, "error": str(exc)}, 1)

    timer = QTimer()
    timer.timeout.connect(check)
    timer.start(30)
    window.load_paths([fixture])
    app.exec()
    temporary.cleanup()
    return result_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ImageBoost Pro — улучшение фотографий")
    parser.add_argument("files", nargs="*", help="Фотографии для открытия")
    parser.add_argument("--smoke-test", action="store_true", help="Выполнить функциональную проверку")
    parser.add_argument("--smoke-output", type=Path, default=Path("smoke-result.json"))
    args = parser.parse_args(argv)
    import cv2
    from PySide6.QtCore import QLibraryInfo, QLocale, QThreadPool, QTranslator, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication
    from imageboost_pro.ui.main_window import MainWindow

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    app = QApplication([sys.argv[0]])
    app.setOrganizationName("ImageBoostPro")
    app.setApplicationName("ImageBoost Pro")
    app.setApplicationVersion("0.1.0")
    resource_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    app.setWindowIcon(QIcon(str(resource_root / "assets" / "app.ico")))
    QLocale.setDefault(QLocale("ru_RU"))
    translator = QTranslator(app)
    translator.load("qtbase_ru", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
    app.installTranslator(translator)
    _configure_logging()
    # Limit concurrent memory-heavy full-image jobs. Individual OpenCV operations
    # can still use the configured native thread budget.
    cv2.setNumThreads(2)
    QThreadPool.globalInstance().setMaxThreadCount(2)
    window = MainWindow()
    window.show()
    if args.smoke_test:
        return _smoke_test(app, window, args.smoke_output.resolve())
    if args.files:
        window.load_paths(args.files)
    return app.exec()


def safe_main(argv: list[str] | None = None) -> int:
    """Make frozen startup failures machine-readable instead of hiding a dialog."""
    arguments = sys.argv[1:] if argv is None else argv
    try:
        return main(arguments)
    except Exception:
        if "--smoke-test" not in arguments:
            raise
        import traceback
        try:
            index = arguments.index("--smoke-output")
            output = Path(arguments[index + 1])
        except (ValueError, IndexError):
            output = Path("smoke-result.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "failed", "phase": "startup",
                                     "error": traceback.format_exc()}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return 1
