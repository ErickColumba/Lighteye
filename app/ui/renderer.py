"""Procesa la vista previa en un hilo aparte para no congelar la interfaz.

Solo hay un cálculo en marcha a la vez. Si llegan peticiones mientras tanto,
se guarda únicamente la última; las intermedias se descartan. Así, al mover un
slider rápido, la imagen siempre acaba mostrando el valor final. Los resultados
que llegan mientras tanto sí se muestran (aunque ya haya otra petición en
cola): si no, durante un arrastre continuo no se vería nada hasta soltar.

Para que arrastrar un slider sea fluido:
- Se reutilizan los resultados intermedios del cálculo anterior (PipelineCache).
- Con `draft=True` se calcula a media resolución y se amplía (borrador); al
  soltar el slider se pide la versión completa.
"""

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.core.color import to_display_u8
from app.core.histogram import compute_histogram
from app.core.pipeline import PipelineCache, process
from app.core.settings import Settings


class _WorkerSignals(QObject):
    done = Signal(int, object)  # generación, (imagen sRGB uint8, histograma) o Exception


DRAFT_SCALE = 0.5


class _State:
    """Datos que se conservan entre cálculos. Solo los usa el hilo de trabajo
    (hay uno solo), así que no necesitan bloqueos."""

    def __init__(self):
        self.full_cache = PipelineCache()
        self.draft_cache = PipelineCache()
        self.draft_for: np.ndarray | None = None
        self.draft_image: np.ndarray | None = None
        # Origen preparado (rostros + retoque): se reutiliza mientras no cambie,
        # así la caché del pipeline sigue sirviendo al mover otros sliders.
        self.prep_key = None
        self.prep_image: np.ndarray | None = None

    def prepared(self, image: np.ndarray, prep) -> np.ndarray:
        if prep is None:
            return image
        key, fn = prep
        if self.prep_key != key or self.prep_image is None:
            self.prep_image = fn(image)
            self.prep_key = key
        return self.prep_image

    def draft_of(self, image: np.ndarray) -> np.ndarray:
        if self.draft_for is not image:
            h, w = image.shape[:2]
            size = (max(1, round(w * DRAFT_SCALE)), max(1, round(h * DRAFT_SCALE)))
            self.draft_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            self.draft_for = image
        return self.draft_image


class _Worker(QRunnable):
    def __init__(self, generation: int, image: np.ndarray, settings: Settings,
                 draft: bool, state: _State, prep=None, post=None):
        super().__init__()
        self.prep = prep
        self.post = post
        self.generation = generation
        self.image = image
        self.settings = settings
        self.draft = draft
        self.state = state
        self.signals = _WorkerSignals()

    def run(self):
        try:
            self.image = self.state.prepared(self.image, self.prep)
            if self.draft:
                small = self.state.draft_of(self.image)
                rgb = to_display_u8(process(small, self.settings, cache=self.state.draft_cache,
                                            detail_scale=DRAFT_SCALE))
                h, w = self.image.shape[:2]
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb = to_display_u8(process(self.image, self.settings, cache=self.state.full_cache))
            histogram = compute_histogram(rgb)  # del resultado, sin el fondo nuevo
            if self.post is not None:
                rgb = self.post(rgb)
            result = (rgb, histogram)
        except Exception as exc:  # noqa: BLE001 — se informa en el hilo principal
            result = exc
        self.signals.done.emit(self.generation, result)


class PreviewRenderer(QObject):
    rendered = Signal(object, object)  # imagen sRGB uint8 lista para el visor, histograma
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._generation = 0
        self._shown_generation = 0  # el resultado más reciente que se ha mostrado
        self._cancelled_upto = 0  # resultados de esta generación o anteriores: descartar
        # Referencia al trabajo en curso: si Python lo liberara antes de
        # terminar, sus señales se destruirían a mitad del cálculo.
        self._running: _Worker | None = None
        self._pending: _Worker | None = None
        self._state = _State()

    def request(self, image: np.ndarray, settings: Settings, draft: bool = False,
                prep=None, post=None) -> None:
        """`prep` = (clave, función imagen → imagen) que se aplica al origen
        antes del pipeline, en el hilo de trabajo (p. ej. rostros y retoque)."""
        self._generation += 1
        # Copia de settings: la interfaz puede seguir cambiándolos mientras tanto.
        self._pending = _Worker(self._generation, image, settings.copy(), draft, self._state, prep, post)
        self._start_pending()

    def cancel(self) -> None:
        """Descarta lo pendiente y el resultado del cálculo en curso."""
        self._generation += 1
        self._cancelled_upto = self._generation
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
        if generation > max(self._shown_generation, self._cancelled_upto):
            self._shown_generation = generation
            if isinstance(result, Exception):
                self.failed.emit(str(result))
            else:
                self.rendered.emit(*result)
        self._start_pending()
