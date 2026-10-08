import numpy as np

from app.core.histogram import clipping_overlay, compute_histogram


def test_histogram_counts_and_clipping():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5] = (255, 128, 10)  # mitad: rojo quemado
    img[5:, :5] = (40, 40, 40)
    h = compute_histogram(img)
    assert h.r.sum() == 100 and h.luma.sum() == 100
    assert h.r[255] == 50 and h.g[128] == 50
    assert h.clipped_highlights == 0.5
    assert h.clipped_shadows == 0.25  # 25 píxeles negros puros


def test_clipping_overlay_marks_only_clipped_pixels():
    img = np.full((2, 3, 3), 100, np.uint8)
    img[0, 0] = (255, 200, 200)
    img[1, 2] = (0, 0, 0)
    out = clipping_overlay(img)
    assert tuple(out[0, 0]) == (255, 0, 0)
    assert tuple(out[1, 2]) == (0, 80, 255)
    assert tuple(out[0, 1]) == (100, 100, 100)
    assert tuple(img[0, 0]) == (255, 200, 200)  # no modifica la original
