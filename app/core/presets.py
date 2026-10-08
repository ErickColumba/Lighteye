"""Presets: conjuntos de ajustes guardados en ~/.config/lighteye/presets/.

Un preset guarda el "look" (luz, color, efectos…) pero no la geometría:
el recorte y los giros son propios de cada foto.

Lighteye trae además presets incluidos (app/presets/, generados con
tools/make_default_presets.py); aparecen primero y no se pueden borrar.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.geometry import GEOMETRY_KEYS
from app.core.settings import Settings

PRESET_VERSION = 1


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "lighteye"


def presets_dir() -> Path:
    return config_dir() / "presets"


BUILTIN_DIR = Path(__file__).resolve().parent.parent / "presets"


@dataclass(frozen=True)
class Preset:
    name: str
    path: Path
    values: dict  # solo los ajustes distintos del valor por defecto
    builtin: bool = False  # incluido con Lighteye (no se puede borrar)


def look_values(settings: Settings) -> dict:
    """Ajustes que forman parte de un preset (todo menos la geometría)."""
    return {k: v for k, v in settings.non_default().items() if k not in GEOMETRY_KEYS}


def apply_look(settings: Settings, values: dict) -> Settings:
    """Ajustes resultantes de aplicar un preset (o ajustes copiados) a una foto.

    Se conserva la geometría de la foto; todo lo demás pasa a ser exactamente
    lo del preset (lo que el preset no incluye vuelve a su valor por defecto),
    así el resultado es predecible.
    """
    result = Settings(values)
    for key in GEOMETRY_KEYS:
        result[key] = settings[key]
    return result


def _file_name(name: str) -> str:
    safe = re.sub(r"[^\w\-]+", " ", name, flags=re.UNICODE)
    safe = re.sub(r"\s+", " ", safe).strip() or "preset"
    return safe[:80] + ".json"


def save_preset(name: str, settings: Settings, folder: Path | None = None) -> Preset:
    folder = folder or presets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    values = look_values(settings)
    path = folder / _file_name(name)
    data = {"lighteye_preset": PRESET_VERSION, "name": name.strip(), "settings": values}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Preset(name.strip(), path, values)


def load_preset(path: Path, builtin: bool = False) -> Preset | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        values = data["settings"]
        if not isinstance(values, dict):
            return None
        # Se normaliza pasando por Settings (descarta claves desconocidas).
        clean = look_values(Settings(values))
        return Preset(str(data.get("name") or Path(path).stem), Path(path), clean, builtin)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _load_folder(folder: Path, builtin: bool) -> list[Preset]:
    if not folder.is_dir():
        return []
    presets = [p for p in (load_preset(f, builtin) for f in folder.glob("*.json")) if p]
    return sorted(presets, key=lambda p: p.name.lower())


def list_presets(folder: Path | None = None) -> list[Preset]:
    """Presets incluidos y luego los del usuario. Con `folder`, solo esa carpeta."""
    if folder is not None:
        return _load_folder(folder, builtin=False)
    return _load_folder(BUILTIN_DIR, builtin=True) + _load_folder(presets_dir(), builtin=False)


def delete_preset(preset: Preset) -> None:
    if preset.builtin:
        raise ValueError("Los presets incluidos con Lighteye no se pueden eliminar")
    preset.path.unlink(missing_ok=True)
