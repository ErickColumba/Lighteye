import numpy as np
import pytest

from app.core import geometry as geo
from app.core.settings import Settings


@pytest.fixture
def img():
    a = np.zeros((40, 60, 3), np.float32)
    a[:10, :20] = (1, 0, 0)  # marca roja arriba a la izquierda
    return a


def test_orient_quarter_turns_and_flips(img):
    r = geo.orient(img, 1, False, False)  # 90° horario
    assert r.shape == (60, 40, 3)
    assert r[0, -1, 0] == 1  # la marca pasa a arriba a la derecha
    assert geo.orient(img, 4, False, False) is img
    assert geo.orient(img, 0, True, False)[0, -1, 0] == 1
    assert geo.orient(img, 0, False, True)[-1, 0, 0] == 1
    assert r.flags["C_CONTIGUOUS"]


def test_straighten_zero_is_identity_and_crops_borders():
    flat = np.full((400, 600, 3), 0.5, np.float32)
    assert geo.straighten(flat, 0) is flat
    out = geo.straighten(flat, 10)
    h, w = out.shape[:2]
    assert w < 600 and h < 400
    assert w / h == pytest.approx(1.5, rel=0.01)  # misma proporción
    # Sin esquinas vacías: todo sigue siendo el gris original.
    assert np.allclose(out, 0.5, atol=1e-4)


def test_inscribed_scale_symmetry():
    assert geo.inscribed_scale(600, 400, 0) == pytest.approx(1)
    assert geo.inscribed_scale(600, 400, 7) == pytest.approx(geo.inscribed_scale(600, 400, -7))


def test_crop_normalized(img):
    out = geo.crop(img, (0.0, 0.0, 0.5, 0.25))
    assert out.shape == (10, 30, 3)
    assert geo.crop(img, geo.FULL_CROP) is img
    assert geo.crop(img, (0.99, 0.99, 0.01, 0.01)).shape[:2] == (1, 1)


def test_apply_geometry_and_settings(img):
    s = Settings({"rotate": 1, "crop": [0, 0, 1, 0.5]})
    out = geo.apply_geometry(img, s)
    assert out.shape == (30, 40, 3)
    assert geo.apply_geometry(img, s, with_crop=False).shape == (60, 40, 3)
    assert geo.is_identity(Settings())
    assert not geo.is_identity(s)
    assert geo.geometry_signature(s) != geo.geometry_signature(Settings())


def test_crop_setting_is_clamped():
    assert Settings({"crop": [0.5, 0.5, 0.9, 2]})["crop"] == (0.5, 0.5, 0.5, 0.5)
    assert Settings({"crop": [-1, 0, 0, 1]})["crop"] == (0.0, 0.0, 0.01, 1.0)
    assert Settings({"crop": "mal"}) == Settings()


def _stepwise(img, s):
    """Referencia lenta: cada paso por separado."""
    out = geo.orient(img, s["rotate"], s["flip_h"], s["flip_v"])
    out = geo.straighten(out, s["angle"])
    return geo.crop(out, s["crop"])


@pytest.mark.parametrize("rotate", [0, 1, 2, 3])
@pytest.mark.parametrize("flip_h,flip_v", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_single_pass_matches_stepwise_exactly_without_angle(rotate, flip_h, flip_v):
    rng = np.random.default_rng(rotate * 4 + flip_h * 2 + flip_v)
    img = rng.uniform(0, 1, (37, 53, 3)).astype(np.float32)
    s = Settings({"rotate": rotate, "flip_h": flip_h, "flip_v": flip_v,
                  "crop": [0.13, 0.21, 0.5, 0.6]})
    assert np.array_equal(geo.apply_geometry(img, s), _stepwise(img, s))
    s_full = Settings({"rotate": rotate, "flip_h": flip_h, "flip_v": flip_v})
    assert np.array_equal(geo.apply_geometry(img, s_full), geo.orient(img, rotate, flip_h, flip_v))


@pytest.mark.parametrize("rotate,angle", [(0, 4.0), (1, -7.5), (3, 12.0)])
def test_single_pass_matches_stepwise_with_angle(rotate, angle):
    y, x = np.mgrid[0:300, 0:420].astype(np.float32)
    img = np.stack([np.sin(x / 23), np.cos(y / 31), np.sin((x + y) / 41)], axis=2) * 0.4 + 0.5
    s = Settings({"rotate": rotate, "angle": angle, "crop": [0.1, 0.2, 0.6, 0.5]})
    a, b = geo.apply_geometry(img, s), _stepwise(img, s)
    assert a.shape == b.shape
    assert np.abs(a - b)[2:-2, 2:-2].max() < 0.01


def test_geometry_preview_matches_full_then_reduce():
    from app.core.loader import make_preview
    y, x = np.mgrid[0:900, 0:1500].astype(np.float32)
    img = np.stack([np.sin(x / 40), np.cos(y / 50), np.sin((x + y) / 70)], axis=2) * 0.4 + 0.5
    for s in (Settings({"crop": [0.1, 0.1, 0.8, 0.8]}), Settings({"rotate": 1, "angle": 3})):
        ref = make_preview(geo.apply_geometry(img, s), 400)
        fast = geo.geometry_preview(img, s, 400)
        assert fast.shape == ref.shape
        assert np.abs(fast - ref)[3:-3, 3:-3].mean() < 0.01
    assert geo.final_size(1500, 900, Settings({"rotate": 1, "crop": [0, 0, 0.5, 1]})) == (450, 1500)
