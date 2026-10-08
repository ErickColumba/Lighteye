"""Aplica los ajustes en un orden fijo (ver docs/funciones.md, sección 4).

La geometría (recorte, giros) no está aquí: se aplica antes, al preparar la
imagen de origen (ver geometry.py y render_full).

Orden: balance de blancos → exposición → tonos → color → detalle → efectos.
Un paso se salta si todos sus ajustes están en el valor por defecto.
"""

import numpy as np

from app.core import adjustments as adj
from app.core.geometry import apply_geometry
from app.core.settings import HSL_KEYS, Settings


# (ajustes que usa el paso, función). Los valores se pasan en ese orden.
STEPS = [
    (("temperature", "tint"), adj.white_balance),
    (("exposure",), adj.exposure),
    (("nr_luma", "nr_color"), adj.noise_reduction),
    (("highlights", "shadows", "whites", "blacks"), adj.tones),
    (("contrast",), adj.contrast),
    (("curves",), adj.curves),
    (tuple(HSL_KEYS), adj.hsl),
    (("vibrance",), adj.vibrance),
    (("saturation",), adj.saturation),
    (("lut_path", "lut_amount"), adj.lut),
    (("clarity",), adj.clarity),
    (("dehaze",), adj.dehaze),
    (("bw", "bw_red", "bw_green", "bw_blue"), adj.black_and_white),
    (("split_shadow_hue", "split_shadow_sat", "split_high_hue", "split_high_sat",
      "split_balance"), adj.split_toning),
    (("vignette", "vignette_size", "vignette_feather"), adj.vignette),
    # Nitidez de salida: sobre la imagen ya terminada.
    (("sharpen",), adj.sharpen),
    # El grano va después de la nitidez para que no se realce.
    (("grain", "grain_size"), adj.grain),
]


def process(img: np.ndarray, settings: Settings, progress=None) -> np.ndarray:
    """Devuelve una imagen lineal float32 nueva; `img` no se modifica.

    `progress(hecho, total)` se llama tras cada paso (para la barra de
    progreso de la exportación); si lanza una excepción, el proceso se corta.
    """
    active = [(keys, fn) for keys, fn in STEPS if not all(settings.is_default(k) for k in keys)]
    out = img
    for i, (keys, fn) in enumerate(active, start=1):
        out = fn(out, *(settings[k] for k in keys))
        if progress:
            progress(i, len(active))
    if out is img:
        out = img.copy()
    return out.astype(np.float32, copy=False)


def render_full(img: np.ndarray, settings: Settings, progress=None) -> np.ndarray:
    """Geometría + ajustes sobre la imagen completa (para exportar)."""
    return process(apply_geometry(img, settings), settings, progress)
