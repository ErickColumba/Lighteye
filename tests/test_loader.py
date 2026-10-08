import cv2
import numpy as np
import pytest

from app.core.color import linear_to_srgb, srgb_to_linear, to_display_u8
from app.core.loader import load_image, make_preview


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_load_png_8bit_is_linear_float(tmp_path):
    bgr = np.zeros((10, 20, 3), np.uint8)
    bgr[..., 2] = 255  # rojo puro
    bgr[..., 1] = 128
    path = tmp_path / "foto con espacios ñ.png"
    cv2.imencode(".png", bgr)[1].tofile(path)

    loaded = load_image(path)
    assert loaded.image.dtype == np.float32
    assert loaded.image.shape == (10, 20, 3)
    assert not loaded.is_raw
    assert loaded.info["bits"] == 8
    assert np.allclose(loaded.image[..., 0], 1.0)  # canal R
    assert np.allclose(loaded.image[..., 2], 0.0)  # canal B
    # 128 en sRGB ≈ 0.216 en lineal
    assert np.allclose(loaded.image[..., 1], 0.2158, atol=1e-3)
    # Al volver a mostrar debe recuperar los mismos valores.
    assert np.array_equal(to_display_u8(loaded.image)[0, 0], [255, 128, 0])


def test_load_tiff_16bit(tmp_path):
    img = np.full((4, 4, 3), 65535, np.uint16)
    path = tmp_path / "a.tif"
    cv2.imencode(".tif", img)[1].tofile(path)
    loaded = load_image(path)
    assert loaded.info["bits"] == 16
    assert np.allclose(loaded.image, 1.0)


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path / "nota.txt")


def test_make_preview_limits_long_side():
    img = np.zeros((3000, 4500, 3), np.float32)
    prev = make_preview(img, 1600)
    assert max(prev.shape[:2]) == 1600
    assert prev.shape[0] == round(3000 * 1600 / 4500)
    small = np.zeros((100, 50, 3), np.float32)
    assert make_preview(small, 1600).shape == small.shape
