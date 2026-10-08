"""Borrar objetos con LaMa (relleno inteligente).

El usuario pinta trazos sobre lo que quiere borrar. Los trazos se guardan en
los ajustes de la foto, normalizados respecto a la foto original (0–1), así
sobreviven a recortes y giros y el original nunca se modifica.

Cada zona pintada se rellena con LaMa usando a su alrededor el contexto
justo (no la foto entera), y el resultado se guarda en caché.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ai import runtime

MAX_SIDE = 1024  # LaMa trabaja a esta resolución como máximo; luego se amplía
CONTEXT = 0.6  # margen de contexto alrededor de cada zona (proporción de su tamaño)


@dataclass
class Patch:
    image: np.ndarray  # (h, w, 3) sRGB float32: la zona ya rellenada
    mask: np.ndarray  # (h, w) float32 0–1: cuánto se usa del parche (bordes suaves)
    x0: int  # posición en la foto original
    y0: int


def strokes_mask(h: int, w: int, strokes) -> np.ndarray:
    """Máscara uint8 (255 = borrar) de los trazos, a tamaño h×w.

    Cada trazo: {"r": radio relativo al lado largo, "pts": [[x, y], …] en 0–1}.
    """
    mask = np.zeros((h, w), np.uint8)
    long_side = max(h, w)
    for stroke in strokes:
        r = max(1, round(stroke["r"] * long_side))
        pts = [(round(x * w), round(y * h)) for x, y in stroke["pts"]]
        for p in pts:
            cv2.circle(mask, p, r, 255, -1, cv2.LINE_AA)
        for a, b in zip(pts, pts[1:]):
            cv2.line(mask, a, b, 255, 2 * r, cv2.LINE_AA)
    return mask


def _inpaint(model, srgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import torch

    h, w = srgb.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8  # LaMa necesita múltiplos de 8
    img = np.pad(srgb, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    m = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
    with torch.inference_mode():
        t = runtime._to_tensor(img, model)
        mt = torch.from_numpy(m)[None, None].to(t.device, t.dtype)
        out = runtime._to_array(model(t, mt))
    return out[:h, :w]


def erase(srgb: np.ndarray, strokes, progress=None, device: str | None = None) -> list[Patch]:
    """Rellena las zonas pintadas de una foto (sRGB 0–1, resolución completa)."""
    h, w = srgb.shape[:2]
    full = strokes_mask(h, w, strokes)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((full > 0).astype(np.uint8))
    patches = []
    if count <= 1:
        return patches
    model = runtime.load("lama", device)
    try:
        for i in range(1, count):
            bx, by, bw, bh = stats[i, :4]
            margin = int(max(32, CONTEXT * max(bw, bh)))
            x0, y0 = max(0, bx - margin), max(0, by - margin)
            x1, y1 = min(w, bx + bw + margin), min(h, by + bh + margin)
            region = srgb[y0:y1, x0:x1]
            hole = ((labels[y0:y1, x0:x1] == i) & (full[y0:y1, x0:x1] > 0)).astype(np.float32)
            # Agrandar un poco el hueco evita que quede un halo del objeto borrado.
            grow = max(2, int(0.01 * max(bw, bh)))
            hole = cv2.dilate(hole, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
            scale = min(1.0, MAX_SIDE / max(region.shape[:2]))
            if scale < 1:
                size = (round(region.shape[1] * scale), round(region.shape[0] * scale))
                small = cv2.resize(region, size, interpolation=cv2.INTER_AREA)
                small_hole = (cv2.resize(hole, size, interpolation=cv2.INTER_LINEAR) > 0.1).astype(np.float32)
                filled = cv2.resize(_inpaint(model, small, small_hole), region.shape[1::-1],
                                    interpolation=cv2.INTER_CUBIC)
            else:
                filled = _inpaint(model, region, hole)
            soft = cv2.GaussianBlur(hole, (0, 0), max(1.0, grow / 2))
            patches.append(Patch(np.clip(filled, 0, 1).astype(np.float32), soft, int(x0), int(y0)))
            if progress:
                progress(i / (count - 1))
    finally:
        runtime.release()
    return patches


def paste_patches(img: np.ndarray, patches: list[Patch], transform: np.ndarray | None = None) -> np.ndarray:
    """Pega los parches sobre `img` (lineal). `transform` (3×3): foto original → img."""
    if not patches:
        return img
    from app.core.color import srgb_to_linear

    out = img.copy()
    h, w = img.shape[:2]
    t = np.eye(3) if transform is None else transform
    for p in patches:
        ph, pw = p.mask.shape
        place = t @ np.array([[1, 0, p.x0], [0, 1, p.y0], [0, 0, 1]], float)  # parche → img
        corners = place @ np.array([[0, pw, 0, pw], [0, 0, ph, ph], [1, 1, 1, 1]])
        x0, y0 = max(0, int(corners[0].min())), max(0, int(corners[1].min()))
        x1, y1 = min(w, int(np.ceil(corners[0].max())) + 1), min(h, int(np.ceil(corners[1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi_m = (np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ place)[:2]
        size = (x1 - x0, y1 - y0)
        patch = cv2.warpAffine(p.image, roi_m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(p.mask, roi_m, size, flags=cv2.INTER_LINEAR)[..., None]
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = roi + (srgb_to_linear(patch) - roi) * mask
    return out


# --- Caché ------------------------------------------------------------------------


def _cache_dir() -> Path:
    from app.paths import cache_dir

    return cache_dir() / "erase"


def cache_key(photo: Path, strokes) -> str:
    st = Path(photo).stat()
    raw = f"{Path(photo).resolve()}|{st.st_mtime_ns}|{st.st_size}|" + json.dumps(strokes, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def load_cached(photo: Path, strokes) -> list[Patch] | None:
    """Zonas borradas guardadas en caché, o None si no se han calculado."""
    path = _cache_dir() / f"{cache_key(photo, strokes)}.npz"
    if path.is_file():
        try:
            d = np.load(path)
            return [Patch(d[f"img{i}"].astype(np.float32), d[f"mask{i}"].astype(np.float32),
                          int(d[f"x{i}"]), int(d[f"y{i}"])) for i in range(int(d["count"]))]
        except (OSError, ValueError, KeyError):
            pass
    return None


def patches_for(photo: Path, linear_full: np.ndarray, strokes, progress=None) -> list[Patch]:
    """Zonas borradas de una foto: de la caché o calculándolas con LaMa."""
    from app.core.color import to_srgb_fast

    if not strokes:
        return []
    cached = load_cached(photo, strokes)
    if cached is not None:
        return cached
    path = _cache_dir() / f"{cache_key(photo, strokes)}.npz"
    patches = erase(to_srgb_fast(linear_full), strokes, progress)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"count": np.array(len(patches))}
        for i, p in enumerate(patches):
            arrays.update({f"img{i}": p.image.astype(np.float16), f"mask{i}": p.mask.astype(np.float16),
                           f"x{i}": np.array(p.x0), f"y{i}": np.array(p.y0)})
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except OSError:
        pass
    return patches
