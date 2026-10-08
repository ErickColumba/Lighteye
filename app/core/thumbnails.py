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
            size = (raw.sizes.width, raw.sizes.height) if flip not in (5, 6) else \
                (raw.sizes.height, raw.sizes.width)
            img = {3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
                   5: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
                   6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE)}.get(flip, lambda a: a)(img)
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
                                  gamma=(1, 1), output_bps=16)
            img = rgb.astype(np.float32) / 65535.0
            size = (img.shape[1] * 2, img.shape[0] * 2)
    return make_preview(img, side), size


def _exact_size(data: np.ndarray, approx: tuple[int, int]) -> tuple[int, int]:
    """Tamaño real de un JPG (la decodificación reducida redondea), con la
    orientación EXIF aplicada como hace load_image."""
    from io import BytesIO

    try:
        from PIL import Image

        with Image.open(BytesIO(data.tobytes())) as im:
            w, h = im.size
            if im.getexif().get(274, 1) in (5, 6, 7, 8):  # girada 90°
                w, h = h, w
            return w, h
    except Exception:  # noqa: BLE001
        return approx


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
                factor = {cv2.IMREAD_REDUCED_COLOR_8: 8, cv2.IMREAD_REDUCED_COLOR_4: 4,
                          cv2.IMREAD_REDUCED_COLOR_2: 2}[flag]
                h, w = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return make_preview(int_srgb_to_linear(rgb), side), _exact_size(data, (w * factor, h * factor))
    full = load_image(path).image
    return make_preview(full, side), (full.shape[1], full.shape[0])


def original_size(path: str | Path, side: int = THUMB_SIDE) -> tuple[int, int] | None:
    """Tamaño (ancho, alto) de la foto original, guardado junto a su miniatura."""
    meta = _cache_file(Path(path), side).with_suffix(".json")
    try:
        import json

        data = json.loads(meta.read_text())
        return int(data["w"]), int(data["h"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_thumbnail(path: str | Path, side: int = THUMB_SIDE) -> np.ndarray:
    """Foto original reducida (lineal float32), desde la caché si es posible."""
    path = Path(path)
    cached = _cache_file(path, side)
    if cached.is_file() and cached.with_suffix(".json").is_file():
        bgr = cv2.imread(str(cached), cv2.IMREAD_COLOR)
        if bgr is not None:
            return int_srgb_to_linear(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if path.suffix.lower() in RAW_EXTENSIONS:
        img, size = _raw_thumbnail(path, side)
    else:
        img, size = _raster_thumbnail(path, side)

    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(to_display_u8(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(cached), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        import json

        cached.with_suffix(".json").write_text(json.dumps({"w": size[0], "h": size[1]}))
    except OSError:
        pass  # sin caché (p. ej. disco lleno): se recalculará la próxima vez
    return img


def render_thumbnail(path: str | Path, side: int = THUMB_SIDE,
                     settings: Settings | None = None) -> np.ndarray:
    """Miniatura sRGB uint8 con los ajustes de la foto aplicados."""
    img = load_thumbnail(path, side)
    if settings is None:
        settings = load_sidecar(path) or Settings()
    if settings == Settings():
        return to_display_u8(img)
    th, tw = img.shape[:2]
    out = apply_geometry(img, settings)
    size = original_size(path, side)
    ai = _cached_ai(Path(path), settings) if size else {}
    transform = None
    if ai:
        from app.core.geometry import geometry_matrix

        w, h = size
        transform = geometry_matrix(tw, th, settings) @ np.diag([tw / w, th / h, 1.0])
        out = _paste_cached_ai(out, ai, settings, transform)
    rgb = to_display_u8(process(out, settings))
    if "alpha" in ai:
        from app.ai import background as bg

        oh, ow = rgb.shape[:2]
        alpha = bg.adjust_edge(bg.warp_alpha(ai["alpha"], w, h, transform, ow, oh), settings["bg_edge"] / 100)
        color = settings["bg_color"]
        backdrop = bg.checkerboard(oh, ow, 6) if color is None else np.array(color, np.float32) * 255
        rgb = np.clip(bg.composite(rgb.astype(np.float32), alpha, backdrop) + 0.5, 0, 255).astype(np.uint8)
    return rgb


def _cached_ai(path: Path, settings: Settings) -> dict:
    """Resultados de IA ya calculados para esta foto (nunca se calcula nada
    aquí: las miniaturas deben ser rápidas)."""
    from app.ai import runtime

    if not runtime.torch_available():
        return {}
    from app.core.settings import RETOUCH_KEYS

    found = {}
    try:
        if settings["erase_strokes"]:
            from app.ai import inpaint

            if (p := inpaint.load_cached(path, list(settings["erase_strokes"]))) is not None:
                found["patches"] = p
        if settings["face_restore"] > 0:
            from app.ai import faces

            key = faces.cache_key(path, bool(settings["face_codeformer"]), settings["face_fidelity"] / 100)
            if (f := faces.load_cached(key)) is not None:
                found["faces"] = f
        if any(not settings.is_default(k) for k in RETOUCH_KEYS):
            from app.ai import faces

            if (p := faces.load_parses_cached(path)) is not None:
                found["parses"] = p
        if settings["bg_remove"]:
            from app.ai import background

            if (a := background.load_cached(path)) is not None:
                found["alpha"] = a
    except OSError:
        return {}
    return found


def _paste_cached_ai(img: np.ndarray, ai: dict, settings: Settings, transform: np.ndarray) -> np.ndarray:
    if "patches" in ai:
        from app.ai.inpaint import paste_patches

        img = paste_patches(img, ai["patches"], transform)
    if "faces" in ai:
        from app.ai.faces import paste_faces

        img = paste_faces(img, ai["faces"], settings["face_restore"] / 100, transform)
    if "parses" in ai:
        from app.ai.retouch import apply_retouch

        img = apply_retouch(img, ai["parses"], settings, transform)
    return img
