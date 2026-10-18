"""Bounded, non-destructive parameter history; no full-image history copies."""
from __future__ import annotations

from pathlib import Path

from imageboost_pro.core.settings import ProcessingSettings


class DocumentState:
    def __init__(self, path: str | Path, settings: ProcessingSettings | None = None):
        self.path = Path(path)
        self._history = [settings or ProcessingSettings()]
        self._index = 0

    @property
    def settings(self) -> ProcessingSettings:
        return self._history[self._index]

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return self._index + 1 < len(self._history)

    def set_settings(self, value: ProcessingSettings) -> bool:
        if not isinstance(value, ProcessingSettings):
            raise TypeError("Ожидаются параметры обработки изображения.")
        value.validate()
        if value == self.settings:
            return False
        self._history = self._history[:self._index + 1]
        self._history.append(value)
        self._history = self._history[-100:]
        self._index = len(self._history) - 1
        return True

    def undo(self) -> ProcessingSettings:
        self._index = max(0, self._index - 1)
        return self.settings

    def redo(self) -> ProcessingSettings:
        self._index = min(len(self._history) - 1, self._index + 1)
        return self.settings

    def reset(self) -> ProcessingSettings:
        self.set_settings(ProcessingSettings())
        return self.settings
