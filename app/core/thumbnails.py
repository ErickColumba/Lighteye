"""Miniaturas para el navegador de carpeta, con caché en ~/.cache/lighteye.

La caché guarda la foto original reducida (sin ajustes). Los ajustes de cada
foto (su sidecar) se aplican al mostrarla, que a este tamaño es instantáneo;
así editar una foto no invalida la caché.
"""

import hashlib
import os
import re
from pathlib import Path

import cv2
import numpy as np

from app.core.color import int_srgb_to_linear, to_display_u8
from app.core.geometry import apply_geometry
from app.core.loader import RAW_EXTENSIONS, SUPPORTED_EXTENSIONS, load_image, make_preview
from app.core.pipeline import process
from app.core.settings import Settings, load_sidecar

THUMB_SIDE = 320


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "lighteye" / "thumbs"


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    try:
        files = [p for p in folder.iterdir()
                 if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    except OSError:
        return []
    return sorted(files, key=_natural_key)


def _cache_file(path: Path, side: int) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{side}"
    return cache_dir() / (hashlib.sha1(key.encode()).hexdigest() + ".jpg")


def _raw_thumbnail(path: Path, side: int) -> np.ndarray:
    """Miniatura de un RAW: la JPEG incrustada por la cámara si existe (rápido),
    o un revelado a media resolución."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        flip = raw.sizes.flip
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                bgr = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.asarray(thumb.data)
            img = int_srgb_to_linear(np.ascontiguousarray(rgb, dtype=np.uint8))
            # La miniatura incrustada no viene girada: se aplica la orientación.
            img = {3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
                   5: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
                   6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)}.get(flip, lambda a: a)(img)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
                                  gamma=(1, 1), output_bps=16)
            img = rgb.astype(np.float32) / 65535.0
    return make_preview(img, side)


def _raster_thumbnail(path: Path, side: int) -> np.ndarray:
    """Los JPG se pueden decodificar ya reducidos a 1/2, 1/4 u 1/8: mucho más
    rápido que abrir la foto entera. Se usa la mayor reducción que aún deja
    al menos `side` píxeles."""
    if path.suffix.lower() in (".jpg", ".jpeg"):
        data = np.fromfile(path, dtype=np.uint8)
        for flag in (cv2.IMREAD_REDUCED_COLOR_8, cv2.IMREAD_REDUCED_COLOR_4,
                     cv2.IMREAD_REDUCED_COLOR_2):
            bgr = cv2.imdecode(data, flag)
            if bgr is not None and max(bgr.shape[:2]) >= side:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return make_preview(int_srgb_to_linear(rgb), side)
    return make_preview(load_image(path).image, side)


def load_thumbnail(path: str | Path, side: int = THUMB_SIDE) -> np.ndarray:
    """Foto original reducida (lineal float32), desde la caché si es posible."""
    path = Path(path)
    cached = _cache_file(path, side)
    if cached.is_file():
        bgr = cv2.imread(str(cached), cv2.IMREAD_COLOR)
        if bgr is not None:
            return int_srgb_to_linear(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if path.suffix.lower() in RAW_EXTENSIONS:
        img = _raw_thumbnail(path, side)
    else:
        img = _raster_thumbnail(path, side)

    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(to_display_u8(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(cached), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    except OSError:
        pass  # sin caché (p. ej. disco lleno): se recalculará la próxima vez
    return img


def render_thumbnail(path: str | Path, side: int = THUMB_SIDE,
                     settings: Settings | None = None) -> np.ndarray:
    """Miniatura sRGB uint8 con los ajustes de la foto aplicados."""
    img = load_thumbnail(path, side)
    if settings is None:
        settings = load_sidecar(path) or Settings()
    if settings != Settings():
        img = process(apply_geometry(img, settings), settings)
    return to_display_u8(img)
