"""Tareas de IA en segundo plano (no bloquean la interfaz)."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal


class FaceWorker(QThread):
    """Detecta y restaura los rostros de una foto y los guarda en caché."""

    done = Signal(str, list)  # clave, rostros
    failed = Signal(str)

    def __init__(self, photo: Path, linear: np.ndarray, use_codeformer: bool, fidelity: float,
                 key: str, parent=None):
        super().__init__(parent)
        self.photo, self.linear = photo, linear
        self.use_codeformer, self.fidelity, self.key = use_codeformer, fidelity, key

    def run(self) -> None:
        from app.ai.faces import faces_for

        try:
            faces = faces_for(self.photo, self.linear, self.use_codeformer, self.fidelity)
        except Exception as exc:  # noqa: BLE001 — se muestra al usuario
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.done.emit(self.key, faces)
