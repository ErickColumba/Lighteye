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
    category: str = "ajustes"  # "ajustes" o "ia" (pestaña del panel)


# Preset especial: quita todos los ajustes de color/luz (vuelve al original),
# conservando el recorte y los giros. No es un archivo.
NONE_PRESET = Preset("Ninguno (original)", Path(), {}, builtin=True)


# Propios de cada foto: no se copian ni se guardan en presets.
PHOTO_KEYS = (*GEOMETRY_KEYS, "erase_strokes")
# Registro de presets aplicados: no es un ajuste en sí.
BOOKKEEPING_KEYS = ("preset_stack",)


def look_values(settings: Settings) -> dict:
    """Ajustes que forman parte de un preset (todo menos geometría y borrados)."""
    return {k: v for k, v in settings.non_default().items()
            if k not in PHOTO_KEYS and k not in BOOKKEEPING_KEYS}


# --- Presets acumulados ------------------------------------------------------------
#
# Aplicar un preset suma sus ajustes a los que ya tiene la foto (si dos presets
# tocan el mismo ajuste, gana el último). Se recuerda la lista para poder
# mostrar de qué preset viene cada ajuste y quitar uno solo.


def stack_preset(settings: Settings, preset: "Preset") -> Settings:
    """Ajustes tras añadir `preset` encima de los actuales. «Ninguno» lo quita todo."""
    if not preset.values:
        return apply_look(settings, {})
    result = settings.copy()
    previous = {k: settings[k] for k in preset.values if not settings.is_default(k)}
    for key, value in preset.values.items():
        result[key] = value
    stack = [e for e in result["preset_stack"] if e["name"] != preset.name]
    stack.append({"name": preset.name, "values": dict(preset.values), "previous": previous})
    result["preset_stack"] = stack
    return result


def preset_owners(settings: Settings) -> dict[str, str | None]:
    """De qué preset viene cada ajuste cambiado (None = cambiado a mano)."""
    owners = {}
    stack = settings["preset_stack"]
    for key, value in look_values(settings).items():
        owners[key] = next((e["name"] for e in reversed(stack)
                            if key in e["values"] and e["values"][key] == value), None)
    return owners


def remove_stacked_preset(settings: Settings, name: str) -> Settings:
    """Quita un preset aplicado. Sus ajustes vuelven al valor del preset
    anterior que también los tocaba (o al original); lo cambiado a mano se
    respeta."""
    owners = preset_owners(settings)
    result = settings.copy()
    removed = next((e for e in settings["preset_stack"] if e["name"] == name), None)
    stack = [e for e in settings["preset_stack"] if e["name"] != name]
    before = removed.get("previous", {}) if removed else {}
    for key, owner in owners.items():
        if owner != name:
            continue
        if key in before:
            result[key] = before[key]  # lo que había justo antes (a mano u otro preset)
        else:
            result.reset(key)
    result["preset_stack"] = stack
    return result


def apply_look(settings: Settings, values: dict) -> Settings:
    """Ajustes resultantes de aplicar un preset (o ajustes copiados) a una foto.

    Se conserva la geometría de la foto; todo lo demás pasa a ser exactamente
    lo del preset (lo que el preset no incluye vuelve a su valor por defecto),
    así el resultado es predecible.
    """
    result = Settings(values)
    for key in PHOTO_KEYS:
        result[key] = settings[key]
    return result


def category_of(values: dict) -> str:
    """"ia" si el preset usa alguna herramienta de IA; si no, "ajustes"."""
    from app.core.settings import BACKGROUND_KEYS, FACE_KEYS

    ai_keys = set(FACE_KEYS) | set(BACKGROUND_KEYS)
    return "ia" if ai_keys & set(values) else "ajustes"


def _file_name(name: str) -> str:
    safe = re.sub(r"[^\w\-]+", " ", name, flags=re.UNICODE)
    safe = re.sub(r"\s+", " ", safe).strip() or "preset"
    return safe[:80] + ".json"


def save_preset(name: str, settings: Settings, folder: Path | None = None) -> Preset:
    folder = folder or presets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    values = look_values(settings)
    path = folder / _file_name(name)
    category = category_of(values)
    data = {"lighteye_preset": PRESET_VERSION, "name": name.strip(), "category": category,
            "settings": values}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return Preset(name.strip(), path, values, category=category)


def load_preset(path: Path, builtin: bool = False) -> Preset | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        values = data["settings"]
        if not isinstance(values, dict):
            return None
        # Se normaliza pasando por Settings (descarta claves desconocidas).
        clean = look_values(Settings(values))
        return Preset(str(data.get("name") or Path(path).stem), Path(path), clean, builtin,
                      category_of(clean))
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
