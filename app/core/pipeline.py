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
