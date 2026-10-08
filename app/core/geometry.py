"""Geometría: giros de 90°, volteos, enderezado y recorte.

Orden: orientar (giro 90° + volteos) → enderezar (rotación libre con recorte
automático para no dejar esquinas vacías) → recortar.

El recorte se guarda normalizado (x, y, ancho, alto en 0–1) respecto a la
imagen ya orientada y enderezada, así sirve igual para la vista previa que
para la resolución completa.

La geometría se aplica antes que el resto de ajustes (no al final como en el
orden original del documento): así la viñeta y el grano siguen el encuadre
final y se procesan menos píxeles.
"""

import math

import cv2
import numpy as np

FULL_CROP = (0.0, 0.0, 1.0, 1.0)
GEOMETRY_KEYS = ("rotate", "flip_h", "flip_v", "angle", "crop")


def orient(img: np.ndarray, quarter_turns: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Giro en pasos de 90° en sentido horario y volteos."""
    # cv2.rotate / cv2.flip son mucho más rápidos que np.rot90 + copia en
    # imágenes grandes (una foto de 24 MP en float32 ocupa ~290 MB).
    rotations = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}
    out = img
    k = int(quarter_turns) % 4
    if k:
        out = cv2.rotate(out, rotations[k])
    if flip_h and flip_v:
        out = cv2.flip(out, -1)
    elif flip_h:
        out = cv2.flip(out, 1)
    elif flip_v:
        out = cv2.flip(out, 0)
    return out


def inscribed_scale(w: int, h: int, angle: float) -> float:
    """Escala del mayor rectángulo con la misma proporción que cabe dentro de
    la imagen girada `angle` grados (sin esquinas vacías)."""
    a = math.radians(abs(angle))
    c, s = math.cos(a), math.sin(a)
    return min(w / (w * c + h * s), h / (w * s + h * c))


def straightened_size(w: int, h: int, angle: float) -> tuple[int, int]:
    if angle == 0:
        return w, h
    k = inscribed_scale(w, h, angle)
    return max(1, round(w * k)), max(1, round(h * k))


def straighten(img: np.ndarray, angle: float) -> np.ndarray:
    """Rota `angle` grados (positivo = horario) y recorta los bordes vacíos."""
    if angle == 0:
        return img
    h, w = img.shape[:2]
    m, out_w, out_h = _straighten_matrix(w, h, angle)
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], float)
    m = np.linalg.inv(half) @ m @ half
    return cv2.warpAffine(img, m[:2], (out_w, out_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def crop_pixels(w: int, h: int, rect) -> tuple[int, int, int, int]:
    """Rectángulo normalizado -> (x0, y0, x1, y1) en píxeles, al menos 1×1."""
    x, y, cw, ch = rect
    x0 = min(w - 1, max(0, round(x * w)))
    y0 = min(h - 1, max(0, round(y * h)))
    x1 = min(w, max(x0 + 1, round((x + cw) * w)))
    y1 = min(h, max(y0 + 1, round((y + ch) * h)))
    return x0, y0, x1, y1


def crop(img: np.ndarray, rect) -> np.ndarray:
    if tuple(rect) == FULL_CROP:
        return img
    x0, y0, x1, y1 = crop_pixels(img.shape[1], img.shape[0], rect)
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _orientation_matrix(w: int, h: int, quarter_turns: int, flip_h: bool,
                        flip_v: bool) -> tuple[np.ndarray, int, int]:
    """Matriz 3×3 (coordenadas de borde de píxel) de la orientación, y el
    tamaño resultante."""
    m = np.eye(3)
    k = int(quarter_turns) % 4
    for _ in range(k):  # cada giro de 90° horario: (x, y) → (alto − y, x)
        m = np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], float) @ m
        w, h = h, w
    if flip_h:
        m = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], float) @ m
    if flip_v:
        m = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], float) @ m
    return m, w, h


def _straighten_matrix(w: int, h: int, angle: float) -> tuple[np.ndarray, int, int]:
    """Matriz 3×3 que gira alrededor del centro y centra el resultado en el
    rectángulo inscrito (sin esquinas vacías)."""
    out_w, out_h = straightened_size(w, h, angle)
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    to_origin = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], float)
    rotate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)  # horario con y hacia abajo
    to_out = np.array([[1, 0, out_w / 2], [0, 1, out_h / 2], [0, 0, 1]], float)
    return to_out @ rotate @ to_origin, out_w, out_h


def apply_geometry(img: np.ndarray, settings, with_crop: bool = True) -> np.ndarray:
    """Orientar → enderezar → recortar, en una sola pasada.

    Solo se calculan los píxeles del resultado: recortar una foto de 24 MP
    no obliga a girar antes la foto entera.
    """
    h, w = img.shape[:2]
    orient_m, ow, oh = _orientation_matrix(w, h, settings["rotate"], settings["flip_h"],
                                           settings["flip_v"])
    angle = settings["angle"]
    rect = settings["crop"] if with_crop else FULL_CROP

    if angle == 0:
        # Sin enderezar: el recorte se lleva a la foto original, se corta (sin
        # copiar) y solo ese trozo se gira/voltea. El resultado es exacto.
        x0, y0, x1, y1 = crop_pixels(ow, oh, rect)
        inv = np.linalg.inv(orient_m)
        corners = inv @ np.array([[x0, x1, x0, x1], [y0, y0, y1, y1], [1, 1, 1, 1]], float)
        sx0, sx1 = round(corners[0].min()), round(corners[0].max())
        sy0, sy1 = round(corners[1].min()), round(corners[1].max())
        region = img[sy0:sy1, sx0:sx1]
        out = orient(region, settings["rotate"], settings["flip_h"], settings["flip_v"])
        return out if out is not img else img

    straight_m, sw, sh = _straighten_matrix(ow, oh, angle)
    x0, y0, x1, y1 = crop_pixels(sw, sh, rect)
    crop_m = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    m = crop_m @ straight_m @ orient_m
    # De coordenadas de borde a índices de píxel (centros en enteros).
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], float)
    m = np.linalg.inv(half) @ m @ half
    return cv2.warpAffine(img, m[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def geometry_matrix(w: int, h: int, settings, with_crop: bool = True) -> np.ndarray:
    """Matriz 3×3 que lleva un punto de la foto original (w×h) a su posición
    en el resultado de apply_geometry. Sirve para colocar sobre la imagen ya
    recortada cosas calculadas en la original (p. ej. rostros restaurados)."""
    orient_m, ow, oh = _orientation_matrix(w, h, settings["rotate"], settings["flip_h"],
                                           settings["flip_v"])
    m, sw, sh = orient_m, ow, oh
    if settings["angle"] != 0:
        straight_m, sw, sh = _straighten_matrix(ow, oh, settings["angle"])
        m = straight_m @ m
    if with_crop:
        x0, y0, _, _ = crop_pixels(sw, sh, settings["crop"])
        m = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ m
    return m


def final_size(w: int, h: int, settings) -> tuple[int, int]:
    """Tamaño en píxeles del resultado de apply_geometry para una foto w×h."""
    if settings["rotate"] % 2:
        w, h = h, w
    w, h = straightened_size(w, h, settings["angle"])
    x0, y0, x1, y1 = crop_pixels(w, h, settings["crop"])
    return x1 - x0, y1 - y0


def geometry_preview(img: np.ndarray, settings, long_side: int) -> np.ndarray:
    """Igual que apply_geometry + reducir a `long_side`, pero sin crear nunca
    la imagen girada a resolución completa: primero se reduce la foto lo
    justo para que el recorte siga teniendo `long_side` píxeles."""
    import app.core.loader as loader  # evita import circular

    fw, fh = final_size(img.shape[1], img.shape[0], settings)
    scale = long_side / max(fw, fh)
    if scale < 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return loader.make_preview(apply_geometry(img, settings), long_side)


def geometry_signature(settings, with_crop: bool = True) -> tuple:
    """Valores que determinan la geometría (para saber si hay que recalcular)."""
    keys = GEOMETRY_KEYS if with_crop else GEOMETRY_KEYS[:-1]
    return tuple(settings[k] for k in keys) + (with_crop,)


def is_identity(settings) -> bool:
    return (settings["rotate"] % 4 == 0 and not settings["flip_h"] and not settings["flip_v"]
            and settings["angle"] == 0 and tuple(settings["crop"]) == FULL_CROP)
