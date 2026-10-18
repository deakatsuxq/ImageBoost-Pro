# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build. Weights are deliberately kept outside the distribution."""
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project = Path(SPECPATH)
with_ai = os.environ.get("IMAGEBOOST_INCLUDE_AI", "0") == "1"

datas = [(str(project / "assets"), "assets")]
# scikit-image uses lazy imports and ships data/stub files needed at runtime.
datas += collect_data_files("skimage")
hiddenimports = collect_submodules("skimage.restoration")
hiddenimports += collect_submodules("scipy.ndimage")
hiddenimports += ["scipy.special._special_ufuncs"]
excludes = [
    "tkinter", "matplotlib", "IPython", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtMultimedia",
    "PySide6.Qt3DCore", "PySide6.QtDesigner", "PySide6.QtTest",
]
if with_ai:
    # These imports fail clearly if the optional dependencies were not installed.
    hiddenimports += ["torch", "torch.nn", "torch.nn.functional", "torch.serialization"]
    datas += copy_metadata("torch")
    excludes += ["torchvision", "basicsr", "realesrgan", "gfpgan", "facexlib"]
else:
    excludes += ["torch", "torchvision", "basicsr", "realesrgan", "gfpgan", "facexlib"]

# Keep distribution-provided notices alongside the frozen runtime.
for package in (
    "numpy", "opencv-python-headless", "Pillow", "scipy", "scikit-image", "PyWavelets",
    "PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6",
    "imageio", "tifffile", "networkx", "packaging", "lazy_loader",
):
    datas += copy_metadata(package)

a = Analysis(
    [str(project / "main.py")],
    pathex=[str(project)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ImageBoostPro",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(project / "assets" / "app.ico"),
    version=str(project / "assets" / "version_info.txt"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ImageBoostPro",
)
