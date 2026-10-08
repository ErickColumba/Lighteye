import cv2
import numpy as np
import pytest

from app.core import thumbnails as th
from app.core.settings import Settings, save_sidecar


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _jpg(path, w=900, h=600, value=120):
    cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))[1].tofile(path)
    return path


def test_list_images_natural_order_and_filter(tmp_path):
    for name in ("foto10.jpg", "foto2.JPG", "foto1.png", "nota.txt", ".oculta.jpg", "b.NEF"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "carpeta.jpg").mkdir()
    names = [p.name for p in th.list_images(tmp_path)]
    assert names == ["b.NEF", "foto1.png", "foto2.JPG", "foto10.jpg"]


def test_thumbnail_is_cached(tmp_path, cache):
    photo = _jpg(tmp_path / "a.jpg")
    first = th.load_thumbnail(photo, 200)
    assert max(first.shape[:2]) == 200
    assert len(list(cache.rglob("*.jpg"))) == 1
    second = th.load_thumbnail(photo, 200)
    assert np.allclose(first, second, atol=0.02)
    # Si la foto cambia, la caché no se reutiliza.
    _jpg(photo, 300, 300)
    import os
    os.utime(photo, ns=(1, 1))
    assert th.load_thumbnail(photo, 200).shape[:2] == (200, 200)


def test_render_thumbnail_applies_sidecar(tmp_path):
    photo = _jpg(tmp_path / "b.jpg")
    plain = th.render_thumbnail(photo, 120)
    save_sidecar(photo, Settings({"exposure": 1.5, "rotate": 1}))
    edited = th.render_thumbnail(photo, 120)
    assert edited.shape[:2] == (plain.shape[1], plain.shape[0])  # girada
    assert edited.mean() > plain.mean() + 20
