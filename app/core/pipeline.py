"""Aplica los ajustes en un orden fijo (ver docs/funciones.md, sección 4).

Orden: balance de blancos → exposición → tonos → color → detalle → efectos.
Un paso se salta si todos sus ajustes están en el valor por defecto.
"""

import numpy as np

from app.core import adjustments as adj
from app.core.settings import HSL_KEYS, Settings


# (ajustes que usa el paso, función). Los valores se pasan en ese orden.
STEPS = [
    (("temperature", "tint"), adj.white_balance),
    (("exposure",), adj.exposure),
    (("highlights", "shadows", "whites", "blacks"), adj.tones),
    (("contrast",), adj.contrast),
    (("curves",), adj.curves),
    (tuple(HSL_KEYS), adj.hsl),
    (("vibrance",), adj.vibrance),
    (("saturation",), adj.saturation),
]


def process(img: np.ndarray, settings: Settings) -> np.ndarray:
    """Devuelve una imagen lineal float32 nueva; `img` no se modifica."""
    out = img
    for keys, fn in STEPS:
        if all(settings.is_default(k) for k in keys):
            continue
        out = fn(out, *(settings[k] for k in keys))
    if out is img:
        out = img.copy()
    return out.astype(np.float32, copy=False)
