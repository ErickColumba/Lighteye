"""Tareas de IA en segundo plano (no bloquean la interfaz)."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal


class BackgroundWorker(QThread):
    """Máscara del sujeto (quitar fondo) con BiRefNet, con caché."""

    done = Signal(str, object)  # ruta de la foto, máscara
    failed = Signal(str)

    def __init__(self, photo: Path, linear: np.ndarray, parent=None):
        super().__init__(parent)
        self.photo, self.linear = photo, linear

    def run(self) -> None:
        from app.ai.background import alpha_for

        try:
            alpha = alpha_for(self.photo, self.linear)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.done.emit(str(self.photo), alpha)


class EraseWorker(QThread):
    """Rellena con LaMa las zonas pintadas con «Borrar objetos» (con caché)."""

    done = Signal(str, list)  # clave, parches
    failed = Signal(str)

    def __init__(self, photo: Path, linear: np.ndarray, strokes, key: str, parent=None):
        super().__init__(parent)
        self.photo, self.linear, self.strokes, self.key = photo, linear, list(strokes), key

    def run(self) -> None:
        from app.ai.inpaint import patches_for

        try:
            patches = patches_for(self.photo, self.linear, self.strokes)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.done.emit(self.key, patches)


class ParseWorker(QThread):
    """Análisis facial (máscaras de piel, ojos, labios, pelo) con caché."""

    done = Signal(str, list)  # ruta de la foto, análisis
    failed = Signal(str)

    def __init__(self, photo: Path, linear: np.ndarray, parent=None):
        super().__init__(parent)
        self.photo, self.linear = photo, linear

    def run(self) -> None:
        from app.ai.faces import parses_for

        try:
            parses = parses_for(self.photo, self.linear)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.done.emit(str(self.photo), parses)


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
