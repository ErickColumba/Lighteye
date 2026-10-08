"""Carga de imágenes (JPG/PNG/TIFF/WebP y RAW) a float32 lineal en rango 0–1."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.core.color import srgb_to_linear

RAW_EXTENSIONS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf",
    ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl", ".kdc",
}
RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | RASTER_EXTENSIONS

PREVIEW_LONG_SIDE = 1600


@dataclass
class LoadedImage:
    path: Path
    image: np.ndarray  # (H, W, 3) float32 RGB lineal, 0–1
    is_raw: bool
    info: dict = field(default_factory=dict)


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    ext = path.suffix.lower()
    if ext in RAW_EXTENSIONS:
        return _load_raw(path)
    if ext in RASTER_EXTENSIONS:
        return _load_raster(path)
    raise ValueError(f"Formato no soportado: {ext or path.name}")


def _load_raster(path: Path) -> LoadedImage:
    # imdecode en vez de imread para no depender de la codificación de la ruta.
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {path.name}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bits = 16 if img.dtype == np.uint16 else 8
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = np.clip(img.astype(np.float32), 0.0, 1.0)

    return LoadedImage(path, srgb_to_linear(img), False, {"bits": bits})


def _load_raw(path: Path) -> LoadedImage:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),            # salida lineal
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB,
        )
    img = rgb.astype(np.float32) / 65535.0
    return LoadedImage(path, img, True, {"bits": 16})


def make_preview(img: np.ndarray, long_side: int = PREVIEW_LONG_SIDE) -> np.ndarray:
    """Versión reducida para editar con fluidez; la completa solo al exportar."""
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    if scale >= 1.0:
        return img.copy()
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
