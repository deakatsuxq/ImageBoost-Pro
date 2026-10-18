"""Explicit downloads of official Real-ESRGAN weights; no automatic downloads."""

from __future__ import annotations

import hashlib
import os
import ssl
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    filename: str
    url: str
    scale: int
    num_blocks: int
    kind: str
    size_mb: int
    sha256: str | None = None


_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
MODELS = (
    ModelSpec("realesrgan-x2plus", "Real-ESRGAN x2plus — фотографии, ×2 / ×4 / ×8",
              "RealESRGAN_x2plus.pth", f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
              2, 23, "photo", 67),
    ModelSpec("realesrgan-x4plus", "Real-ESRGAN x4plus — фотографии, ×4",
              "RealESRGAN_x4plus.pth", f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
              4, 23, "photo", 67),
    ModelSpec("realesrgan-anime-6b", "Real-ESRGAN anime 6B — рисунки, ×4",
              "RealESRGAN_x4plus_anime_6B.pth",
              f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
              4, 6, "illustration", 18),
)


def default_models_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "ImageBoostPro" / "models"


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        allowed = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
        if parsed.scheme != "https" or parsed.hostname not in allowed:
            raise ValueError("Перенаправление загрузки вне официального HTTPS-хранилища отклонено.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


class ModelManager:
    """Manage known weight names. Local files are strictly checked when inference loads them.

    Upstream releases currently do not publish SHA256 checksums for these assets.
    TLS authenticates the download source; ``sha256_file`` computes a local digest
    for reproducibility, which must not be described as publisher verification.
    """

    def __init__(self, models_dir: str | Path | None = None):
        self.models_dir = Path(models_dir) if models_dir is not None else default_models_dir()

    def list_models(self) -> tuple[ModelSpec, ...]:
        return MODELS

    def installed_models(self) -> dict[str, Path]:
        return {m.key: self.models_dir / m.filename for m in MODELS
                if (self.models_dir / m.filename).is_file()}

    def resolve_model(self, key: str) -> Path:
        spec = self._spec(key)
        path = self.models_dir / spec.filename
        if not path.is_file():
            raise FileNotFoundError(f"Модель {spec.filename} не установлена: {self.models_dir}")
        return path

    @staticmethod
    def _spec(key: str) -> ModelSpec:
        for model in MODELS:
            if model.key == key:
                return model
        raise ValueError(f"Неизвестная модель: {key}")

    def download_model(self, key: str, progress: Callable[[float], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Path:
        """Download only after an explicit UI/user request. Preserve an existing file on failure."""
        from .realesrgan_engine import ProcessingCancelled, runtime_status, validate_model

        spec = self._spec(key)
        if cancelled and cancelled():
            raise ProcessingCancelled("Загрузка модели отменена.")
        status = runtime_status()
        if not status["available"]:
            raise RuntimeError(status["error"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        destination = self.models_dir / spec.filename
        temporary: Path | None = None
        opener = urllib.request.build_opener(
            _OfficialRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        try:
            request = urllib.request.Request(spec.url, headers={"User-Agent": "ImageBoostPro/0.1"})
            with opener.open(request, timeout=30) as response:
                expected_size = int(response.headers.get("Content-Length", 0))
                if expected_size > 100 * 1024 * 1024:
                    raise ValueError("Файл модели превышает допустимый размер 100 МиБ.")
                with tempfile.NamedTemporaryFile(dir=self.models_dir, suffix=".part", delete=False) as stream:
                    temporary = Path(stream.name)
                    digest = hashlib.sha256()
                    received = 0
                    while True:
                        if cancelled and cancelled():
                            raise ProcessingCancelled("Загрузка модели отменена.")
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > 100 * 1024 * 1024:
                            raise ValueError("Файл модели превышает допустимый размер 100 МиБ.")
                        stream.write(chunk)
                        digest.update(chunk)
                        if progress and expected_size:
                            progress(min(95.0, received / expected_size * 95.0))
            if expected_size and received != expected_size:
                raise ValueError("Загрузка модели прервана: размер файла не совпадает.")
            if spec.sha256 and digest.hexdigest() != spec.sha256:
                raise ValueError("Контрольная сумма модели не совпадает с опубликованной SHA256.")
            if cancelled and cancelled():
                raise ProcessingCancelled("Загрузка модели отменена.")
            info = validate_model(temporary)
            if (info["scale"], info["num_blocks"]) != (spec.scale, spec.num_blocks):
                raise ValueError("Архитектура загруженной модели не соответствует выбранной.")
            temporary.replace(destination)
            if progress:
                progress(100.0)
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def sha256_file(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
