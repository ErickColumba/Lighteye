"""Historial de deshacer / rehacer.

Guarda copias de los ajustes (no de imágenes), así que cada paso ocupa unos
pocos KB y el historial puede ser largo.
"""

from app.core.settings import PARAMS_BY_KEY, Settings

EXTRA_LABELS = {"curves": "Curvas", "lut_path": "LUT", "crop": "Recorte",
                "erase_strokes": "Borrar objetos", "bg_color": "Color de fondo",
                "preset_stack": "Presets"}


def describe_changes(before: Settings, after: Settings) -> str:
    """Nombre legible de lo que cambió entre dos estados ("Exposición", …)."""
    labels = []
    for key in after.to_dict():
        if before[key] != after[key]:
            if key in PARAMS_BY_KEY:
                p = PARAMS_BY_KEY[key]
                labels.append(f"{p.label} ({p.tab})" if p.tab else p.label)
            else:
                labels.append(EXTRA_LABELS.get(key, key))
    if not labels:
        return ""
    if len(labels) > 3:
        return f"{len(labels)} ajustes"
    return ", ".join(labels)


class History:
    def __init__(self, initial: Settings, limit: int = 200):
        self._states = [initial.copy()]
        self._index = 0
        self._limit = limit

    @property
    def current(self) -> Settings:
        return self._states[self._index].copy()

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return self._index < len(self._states) - 1

    def push(self, settings: Settings) -> bool:
        """Añade un estado. Devuelve False si no había cambios."""
        if settings == self._states[self._index]:
            return False
        # Un cambio nuevo después de deshacer descarta los pasos "rehacibles".
        del self._states[self._index + 1:]
        self._states.append(settings.copy())
        if len(self._states) > self._limit:
            del self._states[0]
        self._index = len(self._states) - 1
        return True

    def undo(self) -> tuple[Settings, str] | None:
        """Vuelve al estado anterior. Devuelve (estado, qué se deshizo)."""
        if not self.can_undo():
            return None
        undone = describe_changes(self._states[self._index - 1], self._states[self._index])
        self._index -= 1
        return self.current, undone

    def redo(self) -> tuple[Settings, str] | None:
        if not self.can_redo():
            return None
        self._index += 1
        redone = describe_changes(self._states[self._index - 1], self._states[self._index])
        return self.current, redone
