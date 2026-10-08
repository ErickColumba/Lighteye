"""Procesa la vista previa en un hilo aparte para no congelar la interfaz.

Solo hay un cálculo en marcha a la vez. Si llegan peticiones mientras tanto,
se guarda únicamente la última; las intermedias se descartan. Así, al mover un
slider rápido, la imagen siempre acaba mostrando el valor final.
"""

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.core.color import to_display_u8
from app.core.pipeline import process
from app.core.settings import Settings


class _WorkerSignals(QObject):
    done = Signal(int, object)  # generación, imagen sRGB uint8 (o Exception)


class _Worker(QRunnable):
    def __init__(self, generation: int, image: np.ndarray, settings: Settings):
        super().__init__()
        self.generation = generation
        self.image = image
        self.settings = settings
        self.signals = _WorkerSignals()

    def run(self):
        try:
            result = to_display_u8(process(self.image, self.settings))
        except Exception as exc:  # noqa: BLE001 — se informa en el hilo principal
            result = exc
        self.signals.done.emit(self.generation, result)


class PreviewRenderer(QObject):
    rendered = Signal(object)  # imagen sRGB uint8 lista para el visor
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._generation = 0
        # Referencia al trabajo en curso: si Python lo liberara antes de
        # terminar, sus señales se destruirían a mitad del cálculo.
        self._running: _Worker | None = None
        self._pending: _Worker | None = None

    def request(self, image: np.ndarray, settings: Settings) -> None:
        self._generation += 1
        # Copia de settings: la interfaz puede seguir cambiándolos mientras tanto.
        self._pending = _Worker(self._generation, image, settings.copy())
        self._start_pending()

    def cancel(self) -> None:
        """Descarta lo pendiente y el resultado del cálculo en curso."""
        self._generation += 1
        self._pending = None

    def shutdown(self) -> None:
        """Espera a que termine el cálculo en curso (al cerrar la ventana)."""
        self.cancel()
        self._pool.waitForDone()

    def _start_pending(self) -> None:
        if self._running is not None or self._pending is None:
            return
        worker, self._pending = self._pending, None
        worker.setAutoDelete(False)
        worker.signals.done.connect(self._on_done)
        self._running = worker
        self._pool.start(worker)

    def _on_done(self, generation: int, result) -> None:
        self._running = None
        if generation == self._generation:
            if isinstance(result, Exception):
                self.failed.emit(str(result))
            else:
                self.rendered.emit(result)
        self._start_pending()
