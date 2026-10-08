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


def _freeze(value):
    """Valor de un ajuste -> algo comparable y usable como clave (curvas, etc.)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


class PipelineCache:
    """Resultados intermedios de la última ejecución, para no repetir trabajo.

    Al mover un slider solo cambia un paso: lo anterior a ese paso es igual
    que en el cálculo previo. Se guarda la entrada de ese paso y las
    siguientes ejecuciones arrancan desde ahí. Solo para la vista previa
    (en la exportación cada imagen se procesa una vez).
    """

    def __init__(self, max_checkpoints: int = 4):
        self.max_checkpoints = max_checkpoints
        self.source: np.ndarray | None = None
        self.checkpoints: dict[tuple, np.ndarray] = {}
        self.last_sigs: list = []

    def _reset(self, source: np.ndarray) -> None:
        self.source = source
        self.checkpoints.clear()
        self.last_sigs = []

    def _store(self, key: tuple, image: np.ndarray) -> None:
        self.checkpoints.pop(key, None)
        self.checkpoints[key] = image
        while len(self.checkpoints) > self.max_checkpoints:
            del self.checkpoints[next(iter(self.checkpoints))]  # el más antiguo


def process(img: np.ndarray, settings: Settings, progress=None,
            cache: PipelineCache | None = None, detail_scale: float = 1.0) -> np.ndarray:
    """Devuelve una imagen lineal float32 nueva; `img` no se modifica.

    `progress(hecho, total)` se llama tras cada paso (para la barra de
    progreso de la exportación); si lanza una excepción, el proceso se corta.
    `cache` reutiliza resultados intermedios entre llamadas sobre la misma
    imagen. `detail_scale` < 1 indica una versión reducida de la vista previa
    (borrador): los radios de nitidez, claridad, etc. se reducen igual.
    """
    active = [(keys, fn) for keys, fn in STEPS if not all(settings.is_default(k) for k in keys)]
    sigs = [(fn.__name__, _freeze(tuple(settings[k] for k in keys))) for keys, fn in active]

    out, start = img, 0
    if cache is not None:
        if cache.source is not img:
            cache._reset(img)
        for i in range(len(sigs), 0, -1):
            hit = cache.checkpoints.get(tuple(sigs[:i]))
            if hit is not None:
                out, start = hit, i
                break
        # Primer paso que cambió respecto a la vez anterior: su entrada es la
        # que conviene guardar para el siguiente movimiento del mismo slider.
        prev = cache.last_sigs
        changed = next((i for i, sig in enumerate(sigs) if i >= len(prev) or prev[i] != sig),
                       len(sigs))
        cache.last_sigs = sigs

    token = adj.DETAIL_SCALE.set(detail_scale)
    try:
        for i in range(start, len(active)):
            if cache is not None and i == changed and i > 0:
                cache._store(tuple(sigs[:i]), out)
            keys, fn = active[i]
            out = fn(out, *(settings[k] for k in keys))
            if progress:
                progress(i + 1, len(active))
    finally:
        adj.DETAIL_SCALE.reset(token)

    if out is img or (cache is not None and start == len(active)):
        out = out.copy()  # nunca se devuelve la entrada ni algo guardado en la caché
    return out.astype(np.float32, copy=False)


def render_full(img: np.ndarray, settings: Settings, progress=None) -> np.ndarray:
    """Geometría + ajustes sobre la imagen completa (para exportar)."""
    return process(apply_geometry(img, settings), settings, progress)
