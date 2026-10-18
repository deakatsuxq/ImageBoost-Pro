from dataclasses import replace
import json

import pytest

from imageboost_pro.app_state import DocumentState
from imageboost_pro.core.settings import ProcessingSettings
from imageboost_pro.utils.presets import BUILTIN_PRESETS, load_preset, save_preset


def test_history_undo_redo_new_branch_and_reset():
    state = DocumentState("image.png")
    original = state.settings
    state.set_settings(replace(original, brightness=10))
    state.set_settings(replace(original, scale=2))
    assert state.undo().brightness == 10
    assert state.redo().scale == 2
    state.undo()
    state.set_settings(replace(original, contrast=20))
    assert not state.can_redo
    state.reset()
    assert state.settings == original
    assert state.undo().contrast == 20


def test_history_is_bounded_settings_only():
    state = DocumentState("image.png")
    for index in range(150):
        state.set_settings(replace(state.settings, brightness=index % 100))
    assert len(state._history) == 100
    for _ in range(120):
        state.undo()
    assert not state.can_undo


def test_json_preset_roundtrip_and_all_builtins_validate(tmp_path):
    for name, settings in BUILTIN_PRESETS.items():
        path = tmp_path / f"{name}.json"
        save_preset(path, settings)
        assert load_preset(path) == settings


@pytest.mark.parametrize("data", [
    {"version": 8, "settings": {}},
    {"version": 1, "settings": {"scale": float("nan")}},
    {"version": 1, "settings": {"imaginary": 1}},
    {"version": 1, "settings": {"scale": True}},
    {"version": 1, "settings": {"upscaler": "fake-ai"}},
    [1, 2],
])
def test_untrusted_preset_rejected(tmp_path, data):
    path = tmp_path / "untrusted.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_preset(path)


def test_preset_payload_limit(tmp_path):
    path = tmp_path / "huge.json"
    path.write_text(" " * 128_001)
    with pytest.raises(ValueError, match="слишком большой"):
        load_preset(path)


def test_default_settings_validates():
    ProcessingSettings().validate()
