"""Máscaras para ajustes locales (pincel, degradados, sujeto/fondo) y su aplicación.

Cada máscara guarda su forma en coordenadas normalizadas (0–1) respecto a la
foto original sin girar ni recortar, igual que «Borrar objetos»: así sigue en
su sitio aunque cambien el recorte, los giros o el tamaño (vista previa,
borrador o exportación).

Estructura de una máscara (dict):
    type      "brush" | "linear" | "radial" | "subject" | "background"
    invert    True para invertirla
    opacity   0–1
    range     None o {"lo": 0–1, "hi": 0–1, "smooth": 0–1}: limita la máscara a
              un rango de luminancia (p. ej. solo las luces)
    adjust    ajustes locales: exposure, contrast, highlights, shadows,
              temperature, tint, saturation, clarity, dehaze, sharpen
    brush:    strokes [{"r": radio/lado largo, "pts": [[x, y], …], "erase": bool}],
              hardness 0–1
    linear:   p0, p1 [x, y]: efecto completo antes de p0, nada después de p1
    radial:   center [x, y], radius [rx, ry] (/lado largo), angle (grados),
              feather 0–1: efecto dentro de la elipse
"""

from contextvars import ContextVar
from dataclasses import dataclass

import cv2
import numpy as np

# --- Ajustes locales disponibles --------------------------------------------------

# (clave, nombre, mínimo, máximo, paso)
LOCAL_PARAMS = [
    ("exposure", "Exposición", -4.0, 4.0, 0.01),
    ("contrast", "Contraste", -100, 100, 1),
    ("highlights", "Altas luces", -100, 100, 1),
    ("shadows", "Sombras", -100, 100, 1),
    ("temperature", "Temperatura", -100, 100, 1),
    ("tint", "Tinte", -100, 100, 1),
    ("saturation", "Saturación", -100, 100, 1),
    ("clarity", "Claridad", -100, 100, 1),
    ("dehaze", "Neblina", -100, 100, 1),
    ("sharpen", "Nitidez", 0, 100, 1),
]
LOCAL_KEYS = {k: (lo, hi) for k, _, lo, hi, _ in LOCAL_PARAMS}
TYPES = ("brush", "linear", "radial", "subject", "background")
TYPE_NAMES = {"brush": "Pincel", "linear": "Degradado lineal", "radial": "Degradado radial",
              "subject": "Sujeto (IA)", "background": "Fondo (IA)"}


def _clamp01(v) -> float:
    return min(1.0, max(0.0, float(v)))


def _point(p) -> tuple[float, float]:
    x, y = p
    return (float(x), float(y))  # puede salir de 0–1 (degradados que empiezan fuera)


def normalize_mask(m: dict) -> dict:
    """Valida y limpia una máscara (para guardarla en los ajustes)."""
    kind = m.get("type")
    if kind not in TYPES:
        raise ValueError(f"Tipo de máscara desconocido: {kind}")
    out = {"type": kind, "invert": bool(m.get("invert", False)),
           "opacity": _clamp01(m.get("opacity", 1.0)), "name": str(m.get("name") or TYPE_NAMES[kind])}
    adjust = {}
    for key, value in dict(m.get("adjust") or {}).items():
        if key in LOCAL_KEYS:
            lo, hi = LOCAL_KEYS[key]
            v = min(hi, max(lo, float(value)))
            if v != 0:
                adjust[key] = v
    out["adjust"] = adjust
    rng = m.get("range")
    if rng:
        lo, hi = _clamp01(rng.get("lo", 0)), _clamp01(rng.get("hi", 1))
        out["range"] = {"lo": min(lo, hi), "hi": max(lo, hi), "smooth": _clamp01(rng.get("smooth", 0.1))}
    else:
        out["range"] = None
    if kind == "brush":
        strokes = []
        for s in m.get("strokes") or ():
            pts = tuple(_point(p) for p in s["pts"])
            if pts:
                strokes.append({"r": min(0.5, max(0.0005, float(s["r"]))), "pts": pts,
                                "erase": bool(s.get("erase", False))})
        out["strokes"] = tuple(strokes)
        out["hardness"] = _clamp01(m.get("hardness", 0.5))
    elif kind == "linear":
        out["p0"] = _point(m.get("p0", (0.5, 0.2)))
        out["p1"] = _point(m.get("p1", (0.5, 0.6)))
    elif kind == "radial":
        out["center"] = _point(m.get("center", (0.5, 0.5)))
        rx, ry = m.get("radius", (0.25, 0.25))
        out["radius"] = (max(0.005, float(rx)), max(0.005, float(ry)))
        out["angle"] = float(m.get("angle", 0.0)) % 360.0
        out["feather"] = _clamp01(m.get("feather", 0.5))
    return out


def new_mask(kind: str) -> dict:
    """Máscara nueva con valores razonables (el usuario la ajusta después)."""
    return normalize_mask({"type": kind})


# --- Contexto: cómo se relaciona la foto original con la imagen procesada ---------


@dataclass(frozen=True)
class MaskContext:
    """Transformación 3×3 de la foto base (sin girar ni recortar, de tamaño
    `base`) a una imagen de tamaño `target`; y la máscara del sujeto (BiRefNet,
    cuadrada sobre la foto base) si se ha calculado."""

    transform: np.ndarray
    base: tuple[int, int]  # (ancho, alto)
    target: tuple[int, int]  # (ancho, alto)
    subject: np.ndarray | None = None

    def key(self) -> tuple:
        """Identifica el contexto (para la caché del pipeline)."""
        return (tuple(np.round(self.transform, 4).ravel()), self.base, self.target,
                self.subject is not None)


MASK_CONTEXT: ContextVar[MaskContext | None] = ContextVar("mask_context", default=None)


def _context_for(img: np.ndarray) -> MaskContext:
    """Contexto ajustado al tamaño real de `img` (p. ej. el borrador a la mitad)."""
    h, w = img.shape[:2]
    ctx = MASK_CONTEXT.get()
    if ctx is None:  # sin contexto: la imagen es la propia foto base
        return MaskContext(np.eye(3), (w, h), (w, h))
    sx, sy = w / ctx.target[0], h / ctx.target[1]
    if abs(sx - 1) < 1e-6 and abs(sy - 1) < 1e-6:
        return ctx
    return MaskContext(np.diag([sx, sy, 1.0]) @ ctx.transform, ctx.base, (w, h), ctx.subject)


# --- Dibujar cada tipo de máscara ---------------------------------------------------


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _base_coords(ctx: MaskContext) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas (en píxeles de la foto base) de cada píxel de la imagen."""
    w, h = ctx.target
    inv = np.linalg.inv(ctx.transform)
    xs = np.arange(w, dtype=np.float32) + 0.5
    ys = np.arange(h, dtype=np.float32) + 0.5
    bx = inv[0, 0] * xs[None, :] + inv[0, 1] * ys[:, None] + inv[0, 2]
    by = inv[1, 0] * xs[None, :] + inv[1, 1] * ys[:, None] + inv[1, 2]
    return bx.astype(np.float32), by.astype(np.float32)


def _linear(m: dict, ctx: MaskContext) -> np.ndarray:
    bw, bh = ctx.base
    bx, by = _base_coords(ctx)
    x0, y0 = m["p0"][0] * bw, m["p0"][1] * bh
    x1, y1 = m["p1"][0] * bw, m["p1"][1] * bh
    dx, dy = x1 - x0, y1 - y0
    t = ((bx - x0) * dx + (by - y0) * dy) / max(dx * dx + dy * dy, 1e-6)
    return 1.0 - _smoothstep(0.0, 1.0, t)


def _radial(m: dict, ctx: MaskContext) -> np.ndarray:
    bw, bh = ctx.base
    long_side = max(bw, bh)
    bx, by = _base_coords(ctx)
    cx, cy = m["center"][0] * bw, m["center"][1] * bh
    a = np.radians(m["angle"])
    dx, dy = bx - cx, by - cy
    u = (dx * np.cos(a) + dy * np.sin(a)) / (m["radius"][0] * long_side)
    v = (-dx * np.sin(a) + dy * np.cos(a)) / (m["radius"][1] * long_side)
    r = np.sqrt(u * u + v * v)
    feather = max(m["feather"], 0.02)
    return 1.0 - _smoothstep(1.0 - feather, 1.0, r)


def _brush(m: dict, ctx: MaskContext) -> np.ndarray:
    w, h = ctx.target
    bw, bh = ctx.base
    long_side = max(bw, bh)
    t = ctx.transform
    scale = float(np.sqrt(abs(np.linalg.det(t[:2, :2]))))
    mask = np.zeros((h, w), np.float32)
    radii = []
    for stroke in m["strokes"]:
        r = max(1, round(stroke["r"] * long_side * scale))
        radii.append(r)
        pts = []
        for x, y in stroke["pts"]:
            px, py, _ = t @ np.array([x * bw, y * bh, 1.0])
            pts.append((round(px), round(py)))
        value = 0.0 if stroke["erase"] else 1.0
        for p in pts:
            cv2.circle(mask, p, r, value, -1, cv2.LINE_AA)
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(mask, p0, p1, value, 2 * r, cv2.LINE_AA)
    if radii and m["hardness"] < 1:
        sigma = float(np.mean(radii)) * (1.0 - m["hardness"]) * 0.5
        if sigma > 0.3:
            mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    return mask


def _subject(m: dict, ctx: MaskContext) -> np.ndarray | None:
    if ctx.subject is None:
        return None
    from app.ai.background import warp_alpha

    w, h = ctx.target
    alpha = warp_alpha(ctx.subject, ctx.base[0], ctx.base[1], ctx.transform, w, h)
    return alpha if m["type"] == "subject" else 1.0 - alpha


_SHAPES = {"brush": _brush, "linear": _linear, "radial": _radial,
           "subject": _subject, "background": _subject}


def shape_mask(m: dict, ctx: MaskContext) -> np.ndarray | None:
    """La forma de la máscara (sin rango de luminancia), 0–1 del tamaño de ctx.target.
    None si depende de la IA y aún no está calculada."""
    shape = _SHAPES[m["type"]](m, ctx)
    if shape is None:
        return None
    if m["invert"]:
        shape = 1.0 - shape
    return shape.astype(np.float32, copy=False)


def luminance_range(img: np.ndarray, rng: dict) -> np.ndarray:
    """Peso 0–1 según la luminancia (perceptual) de cada píxel."""
    from app.core.color import LUMA

    y = cv2.pow(np.maximum(cv2.transform(img, LUMA[None, :]), 0.0), 1 / 2.2)
    s = max(rng["smooth"], 0.01)
    return _smoothstep(rng["lo"] - s, rng["lo"], y) * (1.0 - _smoothstep(rng["hi"], rng["hi"] + s, y))


def final_mask(m: dict, img: np.ndarray) -> np.ndarray | None:
    """Máscara completa (forma × rango × opacidad) para la imagen `img`."""
    shape = shape_mask(m, _context_for(img))
    if shape is None:
        return None
    if m["range"]:
        shape = shape * luminance_range(img, m["range"])
    return shape * np.float32(m["opacity"])


# --- Aplicar los ajustes locales -----------------------------------------------------


def _adjust(img: np.ndarray, a: dict) -> np.ndarray:
    from app.core import adjustments as adj

    out = img
    if a.get("temperature") or a.get("tint"):
        out = adj.white_balance(out, a.get("temperature", 0), a.get("tint", 0))
    if a.get("exposure"):
        out = adj.exposure(out, a["exposure"])
    if a.get("highlights") or a.get("shadows"):
        out = adj.tones(out, a.get("highlights", 0), a.get("shadows", 0), 0, 0)
    if a.get("contrast"):
        out = adj.contrast(out, a["contrast"])
    if a.get("saturation"):
        out = adj.saturation(out, a["saturation"])
    if a.get("clarity"):
        out = adj.clarity(out, a["clarity"])
    if a.get("dehaze"):
        out = adj.dehaze(out, a["dehaze"])
    if a.get("sharpen"):
        out = adj.sharpen(out, a["sharpen"])
    return out


def local_adjustments(img: np.ndarray, masks) -> np.ndarray:
    """Paso del pipeline: aplica cada máscara con sus ajustes, una tras otra."""
    out = img
    for m in masks:
        if not m["adjust"]:
            continue
        mask = final_mask(m, out)
        if mask is None or float(mask.max()) < 1e-3:
            continue
        # Solo se procesa el rectángulo donde actúa la máscara (con margen para
        # los ajustes que miran alrededor: claridad, nitidez, neblina).
        ys, xs = np.nonzero(mask > 1e-3)
        pad = 48
        y0, y1 = max(0, ys.min() - pad), min(out.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(out.shape[1], xs.max() + pad + 1)
        region = out[y0:y1, x0:x1]
        adjusted = _adjust(region, m["adjust"])
        if out is img:
            out = img.copy()
        out[y0:y1, x0:x1] = region + (adjusted - region) * mask[y0:y1, x0:x1, None]
    return out
