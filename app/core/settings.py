"""Ajustes de edición: definición de cada parámetro y sidecar JSON.

La interfaz solo modifica un objeto Settings; el pipeline lo lee para
procesar. El original nunca se toca: los ajustes viven en `foto.ext.json`.
"""

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


# Grupos en el orden en que aparecen en el panel.
GROUPS = ["Luz", "Color", "Detalle", "Efectos"]

PARAMS: list[Param] = [
    Param("exposure", "Exposición", "Luz", -4.0, 4.0, step=0.01),
    Param("contrast", "Contraste", "Luz", -100, 100),
    Param("temperature", "Temperatura", "Color", -100, 100),
    Param("tint", "Tinte", "Color", -100, 100),
    Param("saturation", "Saturación", "Color", -100, 100),
]

PARAMS_BY_KEY = {p.key: p for p in PARAMS}


class Settings:
    """Valores de todos los ajustes; los que faltan toman su valor por defecto."""

    def __init__(self, values: dict | None = None):
        self._values = {p.key: p.default for p in PARAMS}
        if values:
            self.update(values)

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __setitem__(self, key: str, value: float) -> None:
        p = PARAMS_BY_KEY[key]
        self._values[key] = float(min(p.maximum, max(p.minimum, value)))

    def __eq__(self, other) -> bool:
        return isinstance(other, Settings) and self._values == other._values

    def get(self, key: str, default=None):
        return self._values.get(key, default)

    def update(self, values: dict) -> None:
        # Se ignoran claves desconocidas (p. ej. de versiones futuras).
        for key, value in values.items():
            if key in PARAMS_BY_KEY:
                self[key] = value

    def reset(self, key: str) -> None:
        self._values[key] = PARAMS_BY_KEY[key].default

    def is_default(self, key: str) -> bool:
        return self._values[key] == PARAMS_BY_KEY[key].default

    def copy(self) -> "Settings":
        return Settings(self._values)

    def to_dict(self) -> dict:
        return dict(self._values)

    def non_default(self) -> dict:
        return {k: v for k, v in self._values.items() if not self.is_default(k)}


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
