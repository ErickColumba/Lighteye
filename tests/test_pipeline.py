import cv2
import numpy as np

from app.core.color import linear_to_srgb
from app.core.geometry import GEOMETRY_KEYS
from app.core.loader import make_preview
from app.core.pipeline import STEPS, process
from app.core.settings import PARAMS, Settings

EVERYTHING = {
    "temperature": 15, "tint": -5, "exposure": 0.4, "nr_luma": 30, "nr_color": 40,
    "highlights": -40, "shadows": 35, "whites": 10, "blacks": -10, "contrast": 20,
    "curves": {"rgb": [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]]},
    "hsl_s_blue": 30, "hsl_l_green": -20, "vibrance": 25, "saturation": 10,
    "clarity": 30, "dehaze": 20, "split_shadow_sat": 20, "split_high_sat": 15,
    "vignette": -30, "sharpen": 40, "grain": 20,
}


def _scene(h, w):
    """Escena suave con bordes: degradados + rectángulos, determinista."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    img = np.stack([0.2 + 0.6 * x * y, 0.3 + 0.4 * y + 0 * x, 0.8 - 0.6 * x + 0 * y], axis=2)
    for i in range(6):
        x0, x1 = int(w * (0.05 + i * 0.15)), int(w * (0.15 + i * 0.15))
        img[int(h * 0.6):int(h * 0.9), x0:x1] = [(i * 0.17) % 1, 0.5, 1 - (i * 0.13) % 1]
    return np.ascontiguousarray(img ** 2.2, dtype=np.float32)


def test_every_param_belongs_to_a_pipeline_step():
    used = {k for keys, _ in STEPS for k in keys}
    from app.core.settings import FACE_KEYS

    assert {p.key for p in PARAMS} | {"curves", "lut_path", "crop"} == \
        used | set(GEOMETRY_KEYS) | set(FACE_KEYS)


def test_full_resolution_matches_preview():
    full = _scene(800, 3200)
    preview = make_preview(full)
    settings = Settings(EVERYTHING)
    out_full = make_preview(process(full, settings))  # reducido al tamaño de la vista previa
    out_prev = process(preview, settings)
    a = linear_to_srgb(out_full)
    b = linear_to_srgb(out_prev)
    # Se compara a escala de detalle media: el grano y la nitidez son de alta
    # frecuencia y cambian algo con el remuestreo, lo demás debe coincidir.
    blur = lambda im: cv2.GaussianBlur(im, (0, 0), 2)
    assert np.abs(blur(a) - blur(b)).mean() < 0.01
    assert np.abs(blur(a) - blur(b)).max() < 0.08


def test_cache_gives_identical_results_while_dragging_sliders():
    from app.core.pipeline import PipelineCache

    img = _scene(300, 450)
    cache = PipelineCache()
    base = dict(EVERYTHING)
    # Simula arrastrar varios sliders, volver atrás y cambiar de slider.
    sequence = [("contrast", v) for v in (20, 25, 30)] + [("vignette", v) for v in (-30, -50)] \
        + [("exposure", 0.6), ("contrast", 25), ("grain", 0), ("grain", 20)]
    for key, value in sequence:
        base[key] = value
        s = Settings(base)
        cached = process(img, s, cache=cache)
        assert np.array_equal(cached, process(img, s)), (key, value)
    assert 0 < len(cache.checkpoints) <= cache.max_checkpoints


def test_cache_resets_with_new_image_and_output_is_independent():
    from app.core.pipeline import PipelineCache

    cache = PipelineCache()
    a, b = _scene(100, 150), _scene(100, 150) * 0.5
    s = Settings({"exposure": 1, "contrast": 10})
    out_a = process(a, s, cache=cache)
    out_b = process(b, s, cache=cache)
    assert not np.allclose(out_a, out_b)
    again = process(b, s, cache=cache)  # todo desde la caché
    again += 1  # modificar el resultado no debe estropear la caché
    assert np.array_equal(process(b, s, cache=cache), out_b)


def test_detail_scale_halves_radii_for_draft():
    full = _scene(400, 600)
    half = cv2.resize(full, (300, 200), interpolation=cv2.INTER_AREA)
    s = Settings({"clarity": 60, "sharpen": 80})
    ref = cv2.resize(process(full, s), (300, 200), interpolation=cv2.INTER_AREA)
    good = process(half, s, detail_scale=0.5)
    naive = process(half, s)
    assert np.abs(good - ref).mean() < np.abs(naive - ref).mean()
