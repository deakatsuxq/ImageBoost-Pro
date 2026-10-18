"""No network or pretrained downloads in automated tests.

Synthetic RRDB weights verify architecture/inference, not pretrained image quality.
An explicit official-checkpoint smoke check is reported separately in validation.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from imageboost_pro.ai import ModelManager, ProcessingCancelled, runtime_status, upscale
from imageboost_pro.ai import model_manager as manager_module
from imageboost_pro.ai import realesrgan_engine as engine


def test_registry_has_only_official_https_model_urls(tmp_path):
    manager = ModelManager(tmp_path)
    models = manager.list_models()
    assert len(models) == 3
    assert all(m.url.startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/") for m in models)
    assert {(m.scale, m.num_blocks) for m in models} == {(2, 23), (4, 23), (4, 6)}
    assert manager.installed_models() == {}
    with pytest.raises(FileNotFoundError):
        manager.resolve_model(models[0].key)
    with pytest.raises(ValueError):
        manager.resolve_model("https://untrusted.invalid/model.pth")


def test_runtime_explains_missing_optional_torch():
    with patch.object(engine.importlib.util, "find_spec", return_value=None):
        status = runtime_status()
    assert not status["available"]
    assert "PyTorch" in status["error"]


def test_download_cancellation_preserves_existing_model(tmp_path):
    manager = ModelManager(tmp_path)
    existing = tmp_path / manager.list_models()[0].filename
    existing.write_bytes(b"existing-local-weights")
    with pytest.raises(ProcessingCancelled):
        manager.download_model("realesrgan-x2plus", cancelled=lambda: True)
    assert existing.read_bytes() == b"existing-local-weights"
    assert not list(tmp_path.glob("*.part"))


def test_download_validates_before_replacing_local_weights(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path)
    destination = tmp_path / manager.list_models()[0].filename
    destination.write_bytes(b"existing")

    class Response(io.BytesIO):
        headers = {"Content-Length": "13"}

    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"invalid model")

    monkeypatch.setattr(manager_module.urllib.request, "build_opener", lambda *a: Opener())
    monkeypatch.setattr(engine, "runtime_status", lambda: {"available": True})
    monkeypatch.setattr(engine, "validate_model", lambda path: (_ for _ in ()).throw(ValueError("invalid")))
    with pytest.raises(ValueError, match="invalid"):
        manager.download_model("realesrgan-x2plus")
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.part"))


def test_missing_torch_prevents_wasted_download(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "runtime_status", lambda: {"available": False, "error": "PyTorch absent"})
    with pytest.raises(RuntimeError, match="PyTorch absent"):
        ModelManager(tmp_path / "not-created").download_model("realesrgan-x2plus")
    assert not (tmp_path / "not-created").exists()


def test_redirect_rejects_http_and_untrusted_hosts():
    handler = manager_module._OfficialRedirect()
    for url in ("http://github.com/weights", "https://other.invalid/weights"):
        with pytest.raises(ValueError):
            handler.redirect_request(None, None, 302, "Found", {}, url)


def test_local_digest_is_reproducible(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    assert ModelManager.sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture(scope="module")
def torch_module():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield torch
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_network(torch_module):
    from imageboost_pro.ai.rrdbnet import RRDBNet

    torch_module.manual_seed(7)
    return RRDBNet(scale=2, num_feat=4, num_block=1, num_grow_ch=2).eval()


@pytest.mark.parametrize("native_scale", [2, 4])
def test_full_receptive_field_tiles_match_whole_inference(torch_module, native_scale):
    from imageboost_pro.ai.rrdbnet import RRDBNet

    torch_module.manual_seed(4)
    network = RRDBNet(scale=native_scale, num_feat=4, num_block=1, num_grow_ch=2).eval()
    image = torch_module.rand(1, 3, 64, 66)
    reports = []
    with torch_module.inference_mode():
        whole = network(image)
        tiled = engine._run_tiled(network, image, 16, reports.append)
    torch_module.testing.assert_close(tiled, whole, rtol=1e-5, atol=2e-6)
    assert reports[-1] == 100
    assert reports == sorted(reports)


def test_neural_x8_uses_three_actual_x2_passes(torch_module, tiny_network, monkeypatch):
    monkeypatch.setattr(engine, "_load_network", lambda path: tiny_network)
    passes = []
    handle = tiny_network.register_forward_hook(lambda *args: passes.append(True))
    image = np.random.default_rng(12).integers(0, 256, (3, 5, 3), dtype=np.uint8)
    reports = []
    result = upscale(image, "synthetic-fixture", 8, tile_size=256, device="cpu", progress=reports.append)
    handle.remove()
    assert result.shape == (24, 40, 3)
    assert result.dtype == np.uint8
    assert len(passes) == 3
    assert reports[0] == 0 and reports[-1] == 100
    assert reports == sorted(reports)
    import cv2

    assert not np.array_equal(result, cv2.resize(image, (40, 24)))


@pytest.mark.parametrize("factor,shape", [(1.5, (4, 8, 3)), (0.5, (2, 2, 3)), (3, (9, 15, 3))])
def test_fractional_target_keeps_real_neural_inference(torch_module, tiny_network, monkeypatch, factor, shape):
    monkeypatch.setattr(engine, "_load_network", lambda path: tiny_network)
    calls = []
    handle = tiny_network.register_forward_hook(lambda *args: calls.append(True))
    result = upscale(np.full((3, 5, 3), 80, np.uint8), "fixture", factor)
    handle.remove()
    assert result.shape == shape
    assert len(calls) >= 1


def test_pretrained_compatible_checkpoint_strict_safe_loading(torch_module, tmp_path, monkeypatch):
    from imageboost_pro.ai.rrdbnet import RRDBNet

    # Official anime6B architecture, synthetic weights: structural check only.
    network = RRDBNet(scale=4, num_block=6)
    path = tmp_path / "synthetic-anime6b.pth"
    torch_module.save({"params_ema": network.state_dict()}, path)
    actual_load = torch_module.load
    seen = []

    def safe_load(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch_module, "load", safe_load)
    assert engine.validate_model(path) == {
        "scale": 4, "num_blocks": 6, "architecture": "RRDBNet", "pretrained_quality_verified": False}
    result = upscale(np.full((2, 3, 3), 100, np.uint8), path, 4, tile_size=64, device="cpu")
    assert result.shape == (8, 12, 3)
    assert seen and all(seen)
    del network


def test_invalid_checkpoint_never_falls_back_to_unsafe_pickle(torch_module, tmp_path):
    path = tmp_path / "unsupported.pth"
    torch_module.save({"params": {"unexpected": torch_module.zeros(3)}}, path)
    with pytest.raises(ValueError, match="официальные"):
        engine.validate_model(path)


def test_intermediate_resource_guard_prevents_oversized_native_pass(torch_module, tiny_network, monkeypatch):
    monkeypatch.setattr(engine, "_load_network", lambda path: tiny_network)
    with pytest.raises(ValueError, match="Промежуточный"):
        upscale(np.zeros((2001, 2001, 3), np.uint8), "fixture", 3)


def test_cancelled_before_load_is_pipeline_compatible():
    from imageboost_pro.core.pipeline import ProcessingCancelled as PipelineCancelled

    with pytest.raises(PipelineCancelled):
        upscale(np.zeros((2, 3, 3), np.uint8), "missing.pth", 2, cancelled=lambda: True)


def test_cancelled_during_tiles(torch_module, tiny_network):
    calls = [False, True]
    with torch_module.inference_mode(), pytest.raises(ProcessingCancelled):
        engine._run_tiled(tiny_network, torch_module.zeros(1, 3, 24, 24), 16,
                          cancelled=lambda: calls.pop(0))


@pytest.mark.parametrize("factor", [float("nan"), float("inf"), True, 0, 17])
def test_invalid_scales_fail_before_model_load(factor):
    with pytest.raises(ValueError, match="масштаб"):
        upscale(np.zeros((2, 3, 3), np.uint8), Path("missing.pth"), factor)
