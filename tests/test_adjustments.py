import numpy as np
import pytest

from app.core import adjustments as adj
from app.core.color import luminance
from app.core.pipeline import process
from app.core.settings import PARAMS, Settings, load_sidecar, save_sidecar, sidecar_path


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, (32, 48, 3)).astype(np.float32)


def test_exposure_zero_is_identity(img):
    assert np.array_equal(adj.exposure(img, 0.0), img)


def test_exposure_plus_one_doubles_linear_values(img):
    assert np.allclose(adj.exposure(img, 1.0), img * 2)
    assert np.allclose(adj.exposure(img, -2.0), img / 4)


def test_contrast_zero_is_identity(img):
    # La LUT de 16 bits introduce como mucho medio escalón de error.
    assert np.allclose(adj.contrast(img, 0.0), img, atol=1e-5)


def test_contrast_keeps_anchors_and_is_monotonic():
    x = np.linspace(0, 1, 1001, dtype=np.float32)[:, None, None].repeat(3, axis=2)
    for amount in (-100, -50, 50, 100):
        y = adj.contrast(x, amount)[:, 0, 0]
        assert np.all(np.diff(y) >= 0)
        assert y[0] == pytest.approx(0.0, abs=1e-6)
        assert y[-1] == pytest.approx(1.0, abs=1e-6)
    grey = np.full((1, 1, 3), adj.MID_GREY, np.float32)
    assert np.allclose(adj.contrast(grey, 80), grey, atol=1e-5)


def test_positive_contrast_darkens_shadows_and_brightens_highlights():
    dark = np.full((1, 1, 3), 0.05, np.float32)
    bright = np.full((1, 1, 3), 0.6, np.float32)
    assert adj.contrast(dark, 50)[0, 0, 0] < 0.05
    assert adj.contrast(bright, 50)[0, 0, 0] > 0.6


def test_contrast_leaves_values_above_white():
    hot = np.full((1, 1, 3), 2.5, np.float32)
    assert np.allclose(adj.contrast(hot, 60), hot)


def test_saturation_zero_is_identity(img):
    assert np.allclose(adj.saturation(img, 0.0), img, atol=1e-6)


def test_saturation_minus_100_is_grey_with_same_luminance(img):
    out = adj.saturation(img, -100)
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-6)
    assert np.allclose(out[..., 1], out[..., 2], atol=1e-6)
    assert np.allclose(luminance(out), luminance(img), atol=1e-5)


def test_white_balance_zero_is_identity(img):
    assert np.allclose(adj.white_balance(img, 0, 0), img, atol=1e-6)


def test_white_balance_warm_and_keeps_grey_luminance():
    grey = np.full((1, 1, 3), 0.5, np.float32)
    warm = adj.white_balance(grey, 50, 0)[0, 0]
    assert warm[0] > warm[2]
    assert float(warm @ np.array([0.2126, 0.7152, 0.0722])) == pytest.approx(0.5, abs=1e-5)
    magenta = adj.white_balance(grey, 0, 50)[0, 0]
    assert magenta[1] < magenta[0]


def test_pipeline_defaults_is_identity_and_does_not_mutate(img):
    original = img.copy()
    out = process(img, Settings())
    assert out is not img
    assert np.array_equal(out, img)
    process(img, Settings({"exposure": 1, "contrast": 30, "saturation": 20}))
    assert np.array_equal(img, original)


def test_pipeline_applies_exposure(img):
    assert np.allclose(process(img, Settings({"exposure": 1.0})), img * 2)


def test_settings_clamp_and_ignore_unknown():
    s = Settings({"exposure": 99, "desconocido": 5})
    assert s["exposure"] == 4.0
    assert s.get("desconocido") is None
    assert all(Settings().is_default(p.key) for p in PARAMS)


def test_sidecar_roundtrip(tmp_path):
    photo = tmp_path / "foto.jpg"
    photo.write_bytes(b"")
    assert load_sidecar(photo) is None
    s = Settings({"exposure": 0.7, "saturation": -20})
    path = save_sidecar(photo, s)
    assert path == sidecar_path(photo) == tmp_path / "foto.jpg.json"
    assert load_sidecar(photo) == s


def test_corrupt_sidecar_is_ignored(tmp_path):
    photo = tmp_path / "foto.jpg"
    sidecar_path(photo).write_text("{no es json")
    assert load_sidecar(photo) is None
