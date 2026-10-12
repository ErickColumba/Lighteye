import json

import numpy as np
import pytest

from app.core.geometry import apply_geometry, geometry_matrix
from app.core.masks import (
    MaskContext,
    final_mask,
    local_adjustments,
    new_mask,
    normalize_mask,
    shape_mask,
)
from app.core.pipeline import process, render_full
from app.core.settings import Settings


def _ctx(w=200, h=100):
    return MaskContext(np.eye(3), (w, h), (w, h))


def test_normalize_clamps_and_drops_zero_adjusts():
    m = normalize_mask({"type": "radial", "opacity": 3, "adjust": {"exposure": 9, "contrast": 0, "foo": 1},
                        "range": {"lo": 0.8, "hi": 0.2}})
    assert m["opacity"] == 1.0
    assert m["adjust"] == {"exposure": 4.0}
    assert m["range"]["lo"] == 0.2 and m["range"]["hi"] == 0.8
    with pytest.raises(ValueError):
        normalize_mask({"type": "nope"})


def test_linear_full_before_p0_none_after_p1():
    m = normalize_mask({"type": "linear", "p0": (0.5, 0.2), "p1": (0.5, 0.6)})
    s = shape_mask(m, _ctx())
    assert s[5, 100] == pytest.approx(1.0)
    assert s[90, 100] == pytest.approx(0.0)
    assert s[40, 100] > s[50, 100]


def test_radial_inside_and_invert():
    m = normalize_mask({"type": "radial", "center": (0.5, 0.5), "radius": (0.1, 0.1), "feather": 0.2})
    s = shape_mask(m, _ctx())
    assert s[50, 100] == pytest.approx(1.0)
    assert s[5, 5] == pytest.approx(0.0)
    inv = shape_mask({**m, "invert": True}, _ctx())
    assert inv[50, 100] == pytest.approx(0.0) and inv[5, 5] == pytest.approx(1.0)


def test_brush_paints_and_erases():
    m = normalize_mask({"type": "brush", "hardness": 1.0, "strokes": [
        {"r": 0.05, "pts": [(0.2, 0.5), (0.8, 0.5)]},
        {"r": 0.05, "pts": [(0.5, 0.5)], "erase": True},
    ]})
    s = shape_mask(m, _ctx())
    assert s[50, 40] == pytest.approx(1.0)
    assert s[50, 100] == pytest.approx(0.0)  # borrado
    assert s[5, 40] == 0.0


def test_subject_without_alpha_is_skipped():
    img = np.full((50, 50, 3), 0.2, np.float32)
    m = normalize_mask({"type": "subject", "adjust": {"exposure": 1}})
    assert final_mask(m, img) is None
    assert local_adjustments(img, (m,)) is img


def test_luminance_range_limits_mask():
    img = np.zeros((10, 20, 3), np.float32)
    img[:, 10:] = 0.9
    m = normalize_mask({"type": "radial", "radius": (5, 5), "range": {"lo": 0.6, "hi": 1.0, "smooth": 0.05}})
    f = final_mask(m, img)
    assert f[5, 2] == pytest.approx(0.0)
    assert f[5, 15] == pytest.approx(1.0)


def test_local_adjustment_only_inside_mask():
    img = np.full((100, 200, 3), 0.18, np.float32)
    m = normalize_mask({"type": "linear", "p0": (0.5, 0.1), "p1": (0.5, 0.2), "adjust": {"exposure": 1}})
    out = local_adjustments(img, (m,))
    assert out[2, 100, 0] == pytest.approx(0.36, rel=0.02)
    assert out[90, 100, 0] == pytest.approx(0.18)
    assert img[2, 100, 0] == pytest.approx(0.18)  # no modifica la entrada


def test_masks_survive_json_and_settings():
    s = Settings()
    s["masks"] = [new_mask("radial") | {"adjust": {"exposure": 0.5}}]
    back = Settings(json.loads(json.dumps(s.non_default())))
    assert back["masks"] == s["masks"]
    assert back["masks"][0]["adjust"] == {"exposure": 0.5}


def test_mask_follows_geometry():
    """La máscara está pegada a la foto: con la foto girada 180°, el degradado
    superior queda abajo."""
    img = np.full((100, 200, 3), 0.18, np.float32)
    s = Settings()
    s["masks"] = [normalize_mask({"type": "linear", "p0": (0.5, 0.1), "p1": (0.5, 0.2),
                                  "adjust": {"exposure": 1}})]
    s["rotate"] = 2
    out = render_full(img, s)
    assert out[95, 100, 0] > out[5, 100, 0] * 1.5
    # Igual con la vista previa reducida a la mitad.
    small = apply_geometry(img[::2, ::2].copy(), s)
    ctx = MaskContext(geometry_matrix(200, 100, s) @ np.eye(3), (200, 100), (200, 100))
    half = process(small, s, mask_context=ctx)
    assert half[47, 50, 0] > half[2, 50, 0] * 1.5


def test_heal_strokes_work_without_model(monkeypatch):
    """El corrector rápido no carga LaMa: funciona aunque falte el modelo."""
    from app.ai import inpaint, runtime

    monkeypatch.setattr(runtime, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sin modelo")))
    img = np.full((80, 120, 3), 0.5, np.float32)
    img[38:42, 58:62] = 0.0  # una mancha
    patches = inpaint.erase(img, [{"r": 0.03, "pts": [(0.5, 0.5)], "heal": True}])
    assert len(patches) == 1
    out = inpaint.paste_patches(np.full_like(img, 0.2), patches)
    assert out.shape == img.shape
    p = patches[0]
    assert abs(float(p.image[40 - p.y0, 60 - p.x0].mean()) - 0.5) < 0.05  # mancha rellenada
