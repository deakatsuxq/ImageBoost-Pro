# Сторонние компоненты

ImageBoost Pro использует следующие открытые компоненты. Их авторские права и
лицензии сохраняются; лицензии моделей проверяются отдельно перед их распространением.

| Компонент | Версия базовой сборки | Лицензия / сведения |
|---|---|---|
| Python | 3.11 | [PSF](https://docs.python.org/3.11/license.html) |
| PySide6 / Qt | 6.8.3 | [LGPLv3 / GPLv3 / коммерческая](https://doc.qt.io/qtforpython-6/licenses.html) |
| NumPy | 2.2.6 | [BSD-3-Clause](https://numpy.org/doc/stable/license.html) и лицензии встроенных компонентов |
| OpenCV Python Headless | 4.11.0.86 | [Apache-2.0](https://github.com/opencv/opencv/blob/4.11.0/LICENSE) и лицензии встроенных компонентов |
| Pillow | 11.2.1 | [HPND](https://github.com/python-pillow/Pillow/blob/11.2.1/LICENSE) |
| SciPy | 1.15.3 | [BSD-3-Clause](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt) и лицензии встроенных компонентов |
| scikit-image | 0.25.2 | [BSD-3-Clause](https://github.com/scikit-image/scikit-image/blob/v0.25.2/LICENSE.txt) |
| PyWavelets | 1.8.0 | [MIT](https://github.com/PyWavelets/pywt/blob/v1.8.0/LICENSE) |
| PyInstaller | 6.14.1 | [GPLv2+ с исключением для собранных приложений](https://pyinstaller.org/en/stable/license.html) |

Qt поставляется как отдельные динамические библиотеки в каталоге `_internal`;
исходники соответствующей версии доступны в [архиве Qt 6.8.3](https://download.qt.io/archive/qt/6.8/6.8.3/)
и [репозитории PySide 6.8.3](https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3).
Тексты LGPLv3 и GPLv3 лежат в `assets/licenses`. Сборка сохраняет метаданные
и прилагаемые лицензии установленных Python-пакетов. При изменении состава поставки
необходимо проверить лицензии новых библиотек и моделей.

Необязательный AI-режим использует PyTorch (BSD) и совместимую архитектуру
Real-ESRGAN / RRDBNet с официальными весами проекта (BSD-3-Clause).
Весовые файлы не включаются в облегчённый ZIP и не загружаются при запуске.
Источники: [PyTorch](https://github.com/pytorch/pytorch),
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN),
[BasicSR](https://github.com/XPixelGroup/BasicSR).
GFPGAN и facexlib не входят в текущую версию приложения.
