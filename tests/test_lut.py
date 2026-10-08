import numpy as np
import pytest

from app.core import adjustments as adj
from app.core.lut import apply_cube, load_cube, parse_cube


def make_cube(n, fn, header=""):
    lines = [f"TITLE \"prueba\"", header, f"LUT_3D_SIZE {n}", "# comentario"]
    grid = np.linspace(0, 1, n)
    for b in grid:
        for g in grid:
            for r in grid:  # el rojo varía más rápido
                lines.append("%.6f %.6f %.6f" % tuple(fn(np.array([r, g, b]))))
    return "\n".join(lines) + "\n"


@pytest.fixture
def rgb():
    return np.random.default_rng(0).uniform(0, 1, (20, 30, 3)).astype(np.float32)


def test_identity_cube(rgb):
    lut = parse_cube(make_cube(17, lambda c: c))
    assert lut.title == "prueba" and lut.size == 17 and lut.is_3d
    assert np.allclose(apply_cube(lut, rgb), rgb, atol=1e-5)


def test_channel_order_and_trilinear(rgb):
    # Intercambia rojo y azul: detecta si el orden de los datos está mal.
    lut = parse_cube(make_cube(9, lambda c: c[[2, 1, 0]]))
    assert np.allclose(apply_cube(lut, rgb), rgb[..., ::-1], atol=1e-5)
    inv = parse_cube(make_cube(5, lambda c: 1 - c))
    assert np.allclose(apply_cube(inv, rgb), 1 - rgb, atol=1e-5)


def test_1d_cube(rgb):
    n = 32
    rows = "\n".join("%f %f %f" % (v**2, v, 1 - v) for v in np.linspace(0, 1, n))
    lut = parse_cube(f"LUT_1D_SIZE {n}\n{rows}\n")
    out = apply_cube(lut, rgb)
    assert np.allclose(out[..., 1], rgb[..., 1], atol=1e-5)
    assert np.allclose(out[..., 2], 1 - rgb[..., 2], atol=1e-5)
    assert np.allclose(out[..., 0], rgb[..., 0] ** 2, atol=1e-3)


def test_domain(rgb):
    lut = parse_cube(make_cube(3, lambda c: c, "DOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2"))
    assert np.allclose(apply_cube(lut, rgb), rgb / 2, atol=1e-5)


@pytest.mark.parametrize("text", ["", "LUT_3D_SIZE 2\n0 0 0\n", "LUT_3D_SIZE 2\n" + "a b c\n" * 8])
def test_invalid_cubes(text):
    with pytest.raises(ValueError):
        parse_cube(text)


def test_lut_adjustment_amount_and_missing_file(tmp_path, rgb):
    path = tmp_path / "invertir.cube"
    path.write_text(make_cube(5, lambda c: 1 - c))
    lin = rgb ** 2.2
    assert adj.lut(lin, None, 100) is lin
    assert adj.lut(lin, str(path), 0) is lin
    full = adj.lut(lin, str(path), 100)
    half = adj.lut(lin, str(path), 50)
    assert not np.allclose(full, lin, atol=0.05)
    # Al 50 % de una inversión, todo tiende al gris medio en sRGB.
    assert np.ptp(half) < np.ptp(lin)
    with pytest.raises(ValueError):
        load_cube(tmp_path / "no-existe.cube")


def test_builtin_luts_are_listed_and_load():
    from app.core.lut import builtin_luts, display_name, resolve_path

    luts = builtin_luts()
    assert len(luts) >= 5
    titles = [t for t, _ in luts]
    assert "Cálido" in titles
    for title, key in luts:
        assert key.startswith("lighteye:")
        assert resolve_path(key).is_file()
        assert display_name(key) == title
        assert load_cube(key).size == 17


def test_builtin_lut_in_settings_and_pipeline(rgb):
    from app.core.pipeline import process
    from app.core.settings import Settings

    s = Settings({"lut_path": "lighteye:blanco_negro_contraste.cube"})
    out = process(rgb ** 2.2, s)
    assert np.allclose(out[..., 0], out[..., 1], atol=1e-3)  # blanco y negro
