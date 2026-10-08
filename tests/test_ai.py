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


# --- Pegar rostros (sin modelos: caras sintéticas) ------------------------------

def _fake_face(center_xy, size, color):
    """Un 'rostro restaurado' de color liso cuyo recorte cubre `size` píxeles
    alrededor de `center_xy` en la foto original."""
    from app.ai.faces import FACE_SIZE, RestoredFace

    k = FACE_SIZE / size
    cx, cy = center_xy
    matrix = np.array([[k, 0, FACE_SIZE / 2 - k * cx], [0, k, FACE_SIZE / 2 - k * cy]], np.float32)
    return RestoredFace(np.full((FACE_SIZE, FACE_SIZE, 3), color, np.float32), matrix)


def _centroid(img, channel=0):
    ys, xs = np.nonzero(img[..., channel] > 0.5)
    return xs.mean(), ys.mean()


def test_paste_faces_strength_and_position():
    from app.ai.faces import paste_faces

    img = np.zeros((300, 400, 3), np.float32)
    face = _fake_face((120, 100), 80, (1.0, 0.0, 0.0))
    assert paste_faces(img, [face], 0.0) is img
    out = paste_faces(img, [face], 1.0)
    cx, cy = _centroid(out)
    assert abs(cx - 120) < 2 and abs(cy - 100) < 2
    half = paste_faces(img, [face], 0.5)
    assert abs(half[100, 120, 0] - 0.5) < 0.05
    assert np.array_equal(out[250:, 300:], img[250:, 300:])  # fuera de la cara, intacta


def test_paste_faces_follows_geometry_and_scale():
    from app.ai.faces import paste_faces
    from app.core.geometry import apply_geometry, geometry_matrix
    from app.core.settings import Settings

    img = np.zeros((300, 400, 3), np.float32)
    face = _fake_face((300, 80), 60, (1.0, 0.0, 0.0))
    s = Settings({"rotate": 1, "angle": 5, "crop": [0.0, 0.3, 0.9, 0.6]})
    geo = apply_geometry(img, s)
    m = geometry_matrix(400, 300, s)
    out = paste_faces(geo, [face], 1.0, m)
    expected = m @ np.array([300, 80, 1.0])
    cx, cy = _centroid(out)
    assert abs(cx - expected[0]) < 2 and abs(cy - expected[1]) < 2
    # Reducido a la mitad (vista previa)
    small = np.zeros((geo.shape[0] // 2, geo.shape[1] // 2, 3), np.float32)
    out_small = paste_faces(small, [face], 1.0, np.diag([0.5, 0.5, 1]) @ m)
    sx, sy = _centroid(out_small)
    assert abs(sx - expected[0] / 2) < 2 and abs(sy - expected[1] / 2) < 2


def test_face_cache_roundtrip(tmp_path, monkeypatch):
    from app.ai import faces as fx

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"x")
    key = fx.cache_key(photo, True, 0.7)
    assert key != fx.cache_key(photo, False, 0.7) != fx.cache_key(photo, True, 0.5)
    assert fx.load_cached(key) is None
    face = _fake_face((50, 50), 40, (0.2, 0.4, 0.6))
    fx.save_cached(key, [face])
    loaded = fx.load_cached(key)
    assert len(loaded) == 1
    assert np.allclose(loaded[0].face, face.face, atol=1e-3)
    assert np.allclose(loaded[0].matrix, face.matrix)


def test_render_full_pastes_faces_only_when_enabled():
    from app.core.pipeline import render_full
    from app.core.settings import Settings

    img = np.zeros((200, 200, 3), np.float32)
    face = _fake_face((100, 100), 50, (1.0, 0.0, 0.0))
    assert render_full(img, Settings(), faces=[face]).max() == 0
    out = render_full(img, Settings({"face_restore": 100}), faces=[face])
    assert out[100, 100, 0] > 0.9


# --- Retoque con máscaras (sin modelos: análisis sintético) ----------------------

def _fake_parse():
    """Cara de 512 px en (0,0)–(512,512): mitad izquierda piel, un 'ojo' y pelo arriba."""
    from app.ai.faces import FaceParse

    labels = np.zeros((512, 512), np.uint8)
    labels[200:512, 0:256] = 1  # piel
    labels[250:280, 300:360] = 4  # ojo
    labels[0:150, :] = 17  # pelo
    return FaceParse(labels, np.array([[1, 0, 0], [0, 1, 0]], np.float32))


def test_retouch_only_changes_its_zone():
    from app.ai.retouch import apply_retouch
    from app.core.settings import Settings

    rng = np.random.default_rng(0)
    img = (0.2 + 0.1 * rng.random((512, 512, 3))).astype(np.float32)
    parse = [_fake_parse()]
    assert apply_retouch(img, parse, Settings()) is img

    smooth = apply_retouch(img, parse, Settings({"skin_smooth": 100}))
    assert smooth[300:500, 20:230].std() < img[300:500, 20:230].std() * 0.8  # piel más lisa
    assert np.allclose(smooth[300:500, 320:500], img[300:500, 320:500])  # fuera: intacto

    eyes = apply_retouch(img, parse, Settings({"eyes_brighten": 100}))
    assert eyes[265, 330].mean() > img[265, 330].mean() * 1.3
    assert np.allclose(eyes[400, 450], img[400, 450])


def test_retouch_follows_transform():
    from app.ai.retouch import apply_retouch
    from app.core.settings import Settings

    img = np.full((256, 256, 3), 0.2, np.float32)
    half = np.diag([0.5, 0.5, 1.0])  # la imagen es la foto a la mitad de tamaño
    out = apply_retouch(img, [_fake_parse()], Settings({"eyes_brighten": 100}), half)
    assert out[132, 165].mean() > 0.25  # el ojo (265, 330) queda en (132, 165)
    assert np.allclose(out[60, 60], 0.2)


# --- Borrar objetos -----------------------------------------------------------------

def test_strokes_mask_and_settings():
    from app.ai.inpaint import strokes_mask
    from app.core.settings import Settings

    strokes = [{"r": 0.05, "pts": [[0.25, 0.5], [0.75, 0.5]]}]
    mask = strokes_mask(100, 200, strokes)
    assert mask[50, 100] == 255 and mask[50, 50] == 255  # a lo largo del trazo
    assert mask[5, 5] == 0
    s = Settings({"erase_strokes": [{"r": 9, "pts": [[2, -1]]}]})
    assert s["erase_strokes"] == ({"r": 0.5, "pts": ((1.0, 0.0),)},)  # normalizado
    assert Settings({"erase_strokes": "basura"}) == Settings()


def test_erase_strokes_are_photo_specific():
    from app.core.presets import apply_look, look_values
    from app.core.settings import Settings

    s = Settings({"exposure": 1, "erase_strokes": [{"r": 0.01, "pts": [[0.5, 0.5]]}]})
    assert "erase_strokes" not in look_values(s)
    other = Settings({"erase_strokes": [{"r": 0.02, "pts": [[0.1, 0.1]]}]})
    assert apply_look(other, look_values(s))["erase_strokes"] == other["erase_strokes"]


def test_paste_patches_follows_transform():
    from app.ai.inpaint import Patch, paste_patches

    patch = Patch(np.ones((20, 30, 3), np.float32), np.ones((20, 30), np.float32), 100, 40)
    img = np.zeros((200, 300, 3), np.float32)
    out = paste_patches(img, [patch])
    assert out[50, 115, 0] > 0.9 and out[10, 10, 0] == 0
    half = paste_patches(np.zeros((100, 150, 3), np.float32), [patch], np.diag([0.5, 0.5, 1.0]))
    assert half[25, 57, 0] > 0.9 and half[70, 120, 0] == 0


@needs("lama")
def test_erase_fills_a_hole_with_surroundings():
    from app.ai.inpaint import erase

    y, x = np.mgrid[0:256, 0:256].astype(np.float32)
    img = np.stack([x / 255, y / 255, np.full_like(x, 0.3)], axis=2)  # degradado suave
    img[110:146, 110:146] = (1.0, 0.0, 1.0)  # "objeto" magenta
    patches = erase(img, [{"r": 0.1, "pts": [[0.5, 0.5]]}], device="cpu")
    assert len(patches) == 1
    p = patches[0]
    filled = p.image[128 - p.y0, 128 - p.x0]
    assert abs(filled[0] - 0.5) < 0.15 and abs(filled[1] - 0.5) < 0.15  # sigue el degradado
    assert filled[2] < 0.6  # el magenta ha desaparecido


# --- Quitar el fondo --------------------------------------------------------------

def test_background_helpers():
    from app.ai import background as bg

    alpha = np.zeros((64, 64), np.float32)
    alpha[16:48, 16:48] = 1.0
    out = bg.warp_alpha(alpha, 200, 100, None, 200, 100)  # cuadrada → foto 200×100
    assert out.shape == (100, 200)
    assert out[50, 100] == 1.0 and out[5, 5] == 0.0
    half = bg.warp_alpha(alpha, 200, 100, np.diag([0.5, 0.5, 1.0]), 100, 50)
    assert half[25, 50] == 1.0
    soft = np.linspace(0, 1, 11, dtype=np.float32)
    assert bg.adjust_edge(soft, 0.5)[3] > soft[3]  # agrandar
    assert bg.adjust_edge(soft, -0.5)[7] < soft[7]  # encoger
    rgb = np.full((2, 2, 3), 0.8, np.float32)
    a = np.array([[1, 0], [0.5, 0]], np.float32)
    comp = bg.composite(rgb, a, (0.0, 0.0, 1.0))
    assert np.allclose(comp[0, 0], 0.8) and np.allclose(comp[0, 1], (0, 0, 1))
    assert np.allclose(comp[1, 0], (0.4, 0.4, 0.9))
    board = bg.checkerboard(24, 24, 12)
    assert board[0, 0].tolist() != board[0, 12].tolist()


def test_export_with_transparent_or_color_background(tmp_path, monkeypatch):
    import cv2

    from app.ai import background as bg
    from app.core.exporter import ExportOptions, export_image
    from app.core.loader import load_image
    from app.core.settings import Settings

    src = tmp_path / "a.png"
    cv2.imencode(".png", np.full((60, 80, 3), 128, np.uint8))[1].tofile(src)
    mask = np.zeros((32, 32), np.float32)
    mask[8:24, 8:24] = 1.0  # sujeto en el centro
    monkeypatch.setattr(bg, "alpha_for", lambda photo, linear: mask)
    image = load_image(src).image

    out = export_image(src, image, Settings({"bg_remove": 1}), tmp_path / "t.png", ExportOptions("png8"))
    rgba = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert rgba.shape == (60, 80, 4) and rgba[30, 40, 3] == 255 and rgba[2, 2, 3] == 0

    out = export_image(src, image, Settings({"bg_remove": 1, "bg_color": (1, 0, 0)}),
                       tmp_path / "c.jpg", ExportOptions("jpeg"))
    bgr = cv2.imread(str(out))
    assert bgr[2, 2, 2] > 230 and bgr[2, 2, 0] < 30  # fondo rojo
    assert abs(int(bgr[30, 40, 1]) - 128) < 10  # el sujeto no cambia

    out = export_image(src, image, Settings({"bg_remove": 1}), tmp_path / "w.jpg", ExportOptions("jpeg"))
    assert cv2.imread(str(out))[2, 2].min() > 240  # JPG transparente → blanco
