"""Ajustes de edición: definición de cada parámetro y sidecar JSON.

La interfaz solo modifica un objeto Settings; el pipeline lo lee para
procesar. El original nunca se toca: los ajustes viven en `foto.ext.json`.

Hay dos tipos de ajustes:
- Numéricos (PARAMS): un slider o una casilla cada uno.
- Especiales (EXTRA_DEFAULTS): curvas y archivo LUT, con su propio widget.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

SIDECAR_VERSION = 1


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    group: str
    minimum: float
    maximum: float
    default: float = 0.0
    step: float = 1.0  # resolución del slider
    kind: str = "slider"  # "slider" o "toggle" (casilla 0/1)
    tab: str = ""  # pestaña dentro del grupo (p. ej. HSL: Tono/Saturación/…)


def _toggle(key: str, label: str, group: str) -> Param:
    return Param(key, label, group, 0, 1, kind="toggle")


# Grupos en el orden en que aparecen en el panel.
GROUPS = ["Luz", "Curvas", "Color", "HSL", "Detalle", "Efectos"]

PARAMS: list[Param] = [
    Param("exposure", "Exposición", "Luz", -4.0, 4.0, step=0.01),
    Param("contrast", "Contraste", "Luz", -100, 100),
    Param("highlights", "Altas luces", "Luz", -100, 100),
    Param("shadows", "Sombras", "Luz", -100, 100),
    Param("whites", "Blancos", "Luz", -100, 100),
    Param("blacks", "Negros", "Luz", -100, 100),
    Param("temperature", "Temperatura", "Color", -100, 100),
    Param("tint", "Tinte", "Color", -100, 100),
    Param("vibrance", "Intensidad", "Color", -100, 100),
    Param("saturation", "Saturación", "Color", -100, 100),
]

PARAMS += [
    Param("sharpen", "Nitidez", "Detalle", 0, 100),
    Param("nr_luma", "Reducción de ruido", "Detalle", 0, 100),
    Param("nr_color", "Ruido de color", "Detalle", 0, 100),
    Param("clarity", "Claridad", "Detalle", -100, 100),
    Param("dehaze", "Neblina", "Detalle", -100, 100),
]

# HSL por color: (clave, nombre, centro del rango de tono en grados).
HSL_COLORS = [
    ("red", "Rojo", 0), ("orange", "Naranja", 30), ("yellow", "Amarillo", 60),
    ("green", "Verde", 120), ("aqua", "Aguamarina", 180), ("blue", "Azul", 240),
    ("purple", "Morado", 270), ("magenta", "Magenta", 300),
]
HSL_KINDS = [("h", "Tono"), ("s", "Saturación"), ("l", "Luminancia")]
HSL_KEYS = [f"hsl_{k}_{c}" for k, _ in HSL_KINDS for c, _, _ in HSL_COLORS]
PARAMS += [
    Param(f"hsl_{k}_{c}", name, "HSL", -100, 100, tab=tab)
    for k, tab in HSL_KINDS
    for c, name, _ in HSL_COLORS
]

PARAMS_BY_KEY = {p.key: p for p in PARAMS}

# Curva identidad: puntos (x, y) en espacio perceptual 0–1.
IDENTITY_CURVE = ((0.0, 0.0), (1.0, 1.0))
CURVE_CHANNELS = ("rgb", "r", "g", "b")

EXTRA_DEFAULTS: dict = {
    "curves": {ch: IDENTITY_CURVE for ch in CURVE_CHANNELS},
}


class Settings:
    """Valores de todos los ajustes; los que faltan toman su valor por defecto."""

    def __init__(self, values: dict | None = None):
        self._values = {p.key: p.default for p in PARAMS}
        self._values.update(copy.deepcopy(EXTRA_DEFAULTS))
        if values:
            self.update(values)

    def __getitem__(self, key: str):
        return self._values[key]

    def __setitem__(self, key: str, value) -> None:
        if key in PARAMS_BY_KEY:
            p = PARAMS_BY_KEY[key]
            self._values[key] = float(min(p.maximum, max(p.minimum, value)))
        elif key in EXTRA_DEFAULTS:
            self._values[key] = _normalize_extra(key, value)
        else:
            raise KeyError(key)

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self._values == other._values

    def get(self, key: str, default=None):
        return self._values.get(key, default)

    def update(self, values: dict) -> None:
        # Se ignoran claves desconocidas o inválidas (p. ej. de otras versiones).
        for key, value in values.items():
            try:
                self[key] = value
            except (KeyError, TypeError, ValueError):
                continue

    def reset(self, key: str) -> None:
        self._values[key] = copy.deepcopy(_default(key))

    def is_default(self, key: str) -> bool:
        return self._values[key] == _default(key)

    def copy(self) -> "Settings":
        new = Settings.__new__(Settings)
        new._values = copy.deepcopy(self._values)
        return new

    def to_dict(self) -> dict:
        return copy.deepcopy(self._values)

    def non_default(self) -> dict:
        return {k: copy.deepcopy(v) for k, v in self._values.items() if not self.is_default(k)}


def _default(key: str):
    return PARAMS_BY_KEY[key].default if key in PARAMS_BY_KEY else EXTRA_DEFAULTS[key]


def normalize_curve(points) -> tuple:
    """Lista de puntos -> tupla de (x, y) dentro de 0–1, ordenada por x y sin
    x repetidas. Siempre con al menos dos puntos."""
    pts = sorted((min(1.0, max(0.0, float(x))), min(1.0, max(0.0, float(y)))) for x, y in points)
    unique = []
    for x, y in pts:
        if unique and abs(x - unique[-1][0]) < 1e-4:
            unique[-1] = (x, y)
        else:
            unique.append((x, y))
    if len(unique) < 2:
        raise ValueError("Una curva necesita al menos dos puntos")
    return tuple(unique)


def _normalize_extra(key: str, value):
    if key == "curves":
        if not isinstance(value, dict):
            raise TypeError("curves debe ser un diccionario")
        return {ch: normalize_curve(value.get(ch, IDENTITY_CURVE)) for ch in CURVE_CHANNELS}
    if key == "lut_path":
        return str(value) if value else None
    return value


def sidecar_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    return image_path.with_name(image_path.name + ".json")


def save_sidecar(image_path: str | Path, settings: Settings) -> Path:
    path = sidecar_path(image_path)
    data = {"lighteye": SIDECAR_VERSION, "settings": settings.non_default()}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_sidecar(image_path: str | Path) -> Settings | None:
    path = sidecar_path(image_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Settings(data.get("settings", {}))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
