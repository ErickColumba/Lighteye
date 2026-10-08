"""Exportación a resolución completa: JPG, PNG (8/16 bits) o TIFF 16 bits.

Se usa el mismo pipeline que la vista previa, sobre la foto original, y se
conservan los metadatos EXIF del original cuando el formato lo permite
(JPG y PNG).
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.color import linear_to_srgb_u16, to_display_u8
from app.core.pipeline import render_full
from app.core.settings import Settings

# clave: (nombre, extensión, bits)
FORMATS = {
    "jpeg": ("JPG", ".jpg", 8),
    "png8": ("PNG 8 bits", ".png", 8),
    "png16": ("PNG 16 bits", ".png", 16),
    "tiff16": ("TIFF 16 bits", ".tif", 16),
}


@dataclass
class ExportOptions:
    fmt: str = "jpeg"
    quality: int = 92  # solo JPG
    long_side: int | None = None  # None = tamaño original
    ai_scale: int = 1  # 2 o 4: escalar con Real-ESRGAN antes de guardar

    @property
    def extension(self) -> str:
        return FORMATS[self.fmt][1]

    @property
    def bits(self) -> int:
        return FORMATS[self.fmt][2]


def default_output_path(src: Path, options: ExportOptions) -> Path:
    return src.with_name(f"{src.stem}_lighteye{options.extension}")


def resize_long_side(img: np.ndarray, long_side: int | None) -> np.ndarray:
    h, w = img.shape[:2]
    if not long_side or max(h, w) <= long_side:
        return img
    scale = long_side / max(h, w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def to_output_bits(linear: np.ndarray, bits: int) -> np.ndarray:
    """Lineal float32 -> sRGB entero de 8 o 16 bits (RGB), con tablas."""
    if bits == 16:
        return linear_to_srgb_u16(linear)
    return to_display_u8(linear)


def encode(rgb: np.ndarray, options: ExportOptions) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    params = []
    if options.fmt == "jpeg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(options.quality),
                  cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    elif options.fmt.startswith("png"):
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif options.fmt == "tiff16":
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]  # LZW, sin pérdida
    ok, buf = cv2.imencode(options.extension, bgr, params)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {FORMATS[options.fmt][0]}")
    return buf.tobytes()


# --- EXIF ----------------------------------------------------------------------


def build_exif(src: Path, width: int, height: int) -> bytes | None:
    """EXIF del original adaptado a la imagen exportada, o None si no hay.

    - Orientación = 1: el giro ya está aplicado en los píxeles.
    - Dimensiones nuevas y sin miniatura (la del original ya no coincide).
    """
    import piexif

    try:
        exif = piexif.load(str(src))
    except Exception:  # noqa: BLE001 — formato sin EXIF legible: se exporta sin él
        return None
    if not any(exif.get(ifd) for ifd in ("0th", "Exif", "GPS")):
        return None

    exif["0th"][piexif.ImageIFD.Orientation] = 1
    exif["0th"][piexif.ImageIFD.Software] = b"Lighteye"
    for tag in (piexif.ImageIFD.ImageWidth, piexif.ImageIFD.ImageLength,
                piexif.ImageIFD.StripOffsets, piexif.ImageIFD.StripByteCounts):
        exif["0th"].pop(tag, None)  # datos de la imagen original (RAW/TIFF)
    exif.setdefault("Exif", {})
    exif["Exif"][piexif.ExifIFD.PixelXDimension] = width
    exif["Exif"][piexif.ExifIFD.PixelYDimension] = height
    exif["1st"] = {}
    exif["thumbnail"] = None
    try:
        return piexif.dump(exif)
    except Exception:  # noqa: BLE001
        # Algunas MakerNote de fabricante no se pueden reescribir: sin ella.
        exif["Exif"].pop(piexif.ExifIFD.MakerNote, None)
        try:
            return piexif.dump(exif)
        except Exception:  # noqa: BLE001
            return None


def _png_with_exif(png: bytes, exif: bytes) -> bytes:
    """Inserta un bloque eXIf (estándar PNG) justo después de la cabecera."""
    data = exif[6:] if exif.startswith(b"Exif\x00\x00") else exif
    chunk = b"eXIf" + data
    block = struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    ihdr_end = 8 + 8 + 13 + 4  # firma + (longitud, tipo, datos, crc) de IHDR
    return png[:ihdr_end] + block + png[ihdr_end:]


def _jpeg_with_exif(jpeg: bytes, exif: bytes) -> bytes:
    import io

    import piexif

    out = io.BytesIO()
    piexif.insert(exif, jpeg, out)
    return out.getvalue()


# --- Exportar ------------------------------------------------------------------


MAX_AI_MEGAPIXELS = 150  # más que esto no cabe razonablemente en memoria


def ai_upscale(linear: np.ndarray, factor: int, progress=None) -> np.ndarray:
    """Escala con Real-ESRGAN (trabaja en sRGB) y devuelve lineal."""
    from app.ai import runtime
    from app.core.color import linear_to_srgb, srgb_to_linear

    h, w = linear.shape[:2]
    megapixels = h * w * factor * factor / 1e6
    if megapixels > MAX_AI_MEGAPIXELS:
        raise ValueError(f"El resultado tendría {megapixels:.0f} MP (máximo {MAX_AI_MEGAPIXELS}). "
                         "Usa ×2 o recorta la foto.")
    srgb = runtime.upscale(linear_to_srgb(linear), factor, progress)
    return srgb_to_linear(srgb)


def export_image(src: Path, image: np.ndarray, settings: Settings, out_path: Path,
                 options: ExportOptions, progress=None) -> Path:
    """Procesa `image` (lineal, resolución completa) y la guarda en out_path.

    `progress(fracción 0–1)` informa del avance; si lanza una excepción, la
    exportación se cancela sin dejar archivo a medias.
    """
    upscaling = options.ai_scale > 1
    share = 0.4 if upscaling else 0.9  # parte de la barra para el pipeline

    def step(done, total):
        if progress:
            progress(share * done / max(total, 1))

    faces = None
    if settings["face_restore"] > 0:
        from app.ai.faces import faces_for

        faces = faces_for(Path(src), image, bool(settings["face_codeformer"]),
                          settings["face_fidelity"] / 100)
    parses = None
    from app.core.settings import RETOUCH_KEYS

    if any(not settings.is_default(k) for k in RETOUCH_KEYS):
        from app.ai.faces import parses_for

        parses = parses_for(Path(src), image)
    linear = render_full(image, settings, step, faces, parses)
    if upscaling:
        linear = ai_upscale(linear, options.ai_scale,
                            lambda f: progress(share + 0.5 * f) if progress else None)
    linear = resize_long_side(linear, options.long_side)
    rgb = to_output_bits(linear, options.bits)
    data = encode(rgb, options)

    exif = build_exif(src, rgb.shape[1], rgb.shape[0])
    if exif:
        if options.fmt == "jpeg":
            data = _jpeg_with_exif(data, exif)
        elif options.fmt.startswith("png"):
            data = _png_with_exif(data, exif)

    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(out_path)  # el archivo final aparece completo o no aparece
    if progress:
        progress(1.0)
    return out_path


# --- Por lotes -------------------------------------------------------------------


def unique_path(path: Path) -> Path:
    """`path`, o `nombre_2.ext`, `nombre_3.ext`… si ya existe (nunca sobrescribe)."""
    path = Path(path)
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


def export_many(paths: list[Path], out_dir: Path, options: ExportOptions,
                progress=None) -> list[tuple[Path, str]]:
    """Exporta varias fotos, cada una con sus propios ajustes (su sidecar).

    `progress(fracción total 0–1, índice, ruta)` informa del avance; si lanza
    una excepción se detiene todo. Un error en una foto no detiene el resto:
    se devuelve la lista de (foto, mensaje de error).
    """
    from app.core.loader import load_image
    from app.core.settings import load_sidecar

    errors = []
    total = len(paths)
    for i, src in enumerate(paths):
        src = Path(src)

        def step(fraction, i=i, src=src):
            if progress:
                progress((i + fraction) / total, i, src)

        step(0.0)
        try:
            loaded = load_image(src)
            settings = load_sidecar(src) or Settings()
            out = unique_path(Path(out_dir) / default_output_path(src, options).name)
            export_image(src, loaded.image, settings, out, options, step)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 — se informa al final
            errors.append((src, str(exc) or exc.__class__.__name__))
    if progress:
        progress(1.0, total - 1, Path(paths[-1]) if paths else Path())
    return errors
