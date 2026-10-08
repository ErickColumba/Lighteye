"""Pruebas de las herramientas de IA. Se saltan si faltan PyTorch o los modelos."""

import numpy as np
import pytest

from app.ai import runtime

needs = lambda *keys: pytest.mark.skipif(  # noqa: E731
    not all(runtime.model_available(k) for k in keys), reason="faltan PyTorch o los modelos")


@needs("realesrgan_x2", "realesrgan_x4")
def test_upscale_sizes_and_range():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (40, 60, 3)).astype(np.float32)
    for factor in (2, 4):
        out = runtime.upscale(img, factor)
        assert out.shape == (40 * factor, 60 * factor, 3)
        assert out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1


@needs("realesrgan_x2")
def test_tiled_matches_single_pass_away_from_seams():
    y, x = np.mgrid[0:300, 0:300].astype(np.float32)
    img = np.stack([np.sin(x / 13), np.cos(y / 17), np.sin((x + y) / 23)], axis=2) * 0.4 + 0.5
    model = runtime.load("realesrgan_x2", "cpu")
    whole = runtime.run_tiled(model, img, tile=1024)
    tiled = runtime.run_tiled(model, img, tile=192, overlap=32)
    runtime.release()
    assert np.abs(whole - tiled).mean() < 0.01


@needs("realesrgan_x2")
def test_export_with_ai_scale(tmp_path):
    import cv2

    from app.core.exporter import ExportOptions, export_image
    from app.core.loader import load_image
    from app.core.settings import Settings

    src = tmp_path / "a.png"
    cv2.imencode(".png", np.full((50, 70, 3), 128, np.uint8))[1].tofile(src)
    out = export_image(src, load_image(src).image, Settings(), tmp_path / "b.png",
                       ExportOptions("png8", ai_scale=2))
    assert cv2.imread(str(out)).shape[:2] == (100, 140)


def test_device_name_is_text():
    assert isinstance(runtime.device_name(), str)
