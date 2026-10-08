"""Tira de miniaturas de la carpeta actual.

- Clic: abre la foto.
- Ctrl/Mayús + clic: selecciona varias (para pegar ajustes, aplicar presets
  o exportar por lotes desde el menú contextual).
"""

import threading
from collections import deque
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QListView, QListWidget, QListWidgetItem, QMenu

from app.core.settings import Settings, load_sidecar
from app.core.thumbnails import list_images, render_thumbnail
from app.ui.icons import pixmap as icon_pixmap
from app.ui.viewer import array_to_qimage

ICON = QSize(150, 100)


class ThumbnailWorker(QThread):
    """Genera miniaturas en segundo plano, en el orden en que se piden."""

    ready = Signal(str, QImage, bool)  # ruta, miniatura, ¿tiene ediciones?

    def __init__(self, parent=None):
        super().__init__(parent)
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False

    def request(self, paths: list[str], front: bool = False) -> None:
        with self._lock:
            for p in paths:
                if p in self._queue:
                    self._queue.remove(p)
            if front:
                self._queue.extendleft(reversed(paths))
            else:
                self._queue.extend(paths)
        self._wake.set()

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()
        self.wait()

    def run(self) -> None:
        while not self._stop:
            with self._lock:
                path = self._queue.popleft() if self._queue else None
            if path is None:
                self._wake.wait()
                self._wake.clear()
                continue
            settings = load_sidecar(path) or Settings()
            try:
                image = array_to_qimage(render_thumbnail(path, settings=settings))
            except Exception:  # noqa: BLE001 — archivo dañado o ilegible
                image = QImage()
            self.ready.emit(path, image, settings != Settings())


def with_edited_badge(image: QImage) -> QPixmap:
    """Miniatura con una insignia redonda (icono de ajustes) en la esquina."""
    pix = QPixmap.fromImage(image)
    size = max(18, round(min(pix.width(), pix.height()) * 0.2))
    margin = max(4, size // 4)
    box = QRectF(pix.width() - size - margin, margin, size, size)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(255, 255, 255, 230), max(1.5, size / 14)))
    p.setBrush(QColor(37, 99, 235, 235))  # azul
    p.drawEllipse(box)
    inner = round(size * 0.62)
    p.drawPixmap(round(box.center().x() - inner / 2), round(box.center().y() - inner / 2),
                 icon_pixmap("edited", "#ffffff", inner))
    p.end()
    return pix


def _placeholder() -> QIcon:
    pix = QPixmap(ICON)
    pix.fill(QColor(50, 50, 50))
    return QIcon(pix)


class FilmStrip(QListWidget):
    open_requested = Signal(str)
    # Acciones sobre la selección: (nombre de la acción, lista de rutas)
    batch_requested = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(ICON)
        self.setGridSize(QSize(ICON.width() + 12, ICON.height() + 30))
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(ICON.height() + 52)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.itemClicked.connect(self._clicked)

        self.folder: Path | None = None
        self._items: dict[str, QListWidgetItem] = {}
        self._placeholder = _placeholder()
        self.presets_menu_provider = None  # función que devuelve [(nombre, preset)]
        self.can_paste = lambda: False

        self.worker = ThumbnailWorker(self)
        self.worker.ready.connect(self._thumbnail_ready)
        self.worker.start()

    # --- Contenido -----------------------------------------------------------

    def set_folder(self, folder: Path) -> None:
        folder = Path(folder)
        self.worker.clear()
        self.clear()
        self._items.clear()
        self.folder = folder
        paths = [str(p) for p in list_images(folder)]
        for p in paths:
            item = QListWidgetItem(self._placeholder, Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.addItem(item)
            self._items[p] = item
        self.worker.request(paths)

    def paths(self) -> list[str]:
        return list(self._items)

    def selected_paths(self) -> list[str]:
        return [i.data(Qt.ItemDataRole.UserRole) for i in self.selectedItems()]

    def mark_current(self, path: str) -> None:
        item = self._items.get(str(path))
        if item is None:
            return
        self.blockSignals(True)
        self.setCurrentItem(item)
        self.blockSignals(False)
        self.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
        self.worker.request([str(path)], front=True)

    def refresh(self, paths: list[str]) -> None:
        """Vuelve a dibujar las miniaturas (p. ej. tras cambiar sus ajustes)."""
        known = [str(p) for p in paths if str(p) in self._items]
        self.worker.request(known, front=True)

    def neighbour(self, path: str, step: int) -> str | None:
        paths = self.paths()
        if str(path) not in paths:
            return None
        i = paths.index(str(path)) + step
        return paths[i] if 0 <= i < len(paths) else None

    def _thumbnail_ready(self, path: str, image: QImage, edited: bool) -> None:
        item = self._items.get(path)
        if item is None:
            return
        if not image.isNull():
            pix = with_edited_badge(image) if edited else QPixmap.fromImage(image)
            item.setIcon(QIcon(pix))
        item.setToolTip(f"{path}\n{'Editada' if edited else 'Sin editar'}")

    # --- Interacción ---------------------------------------------------------

    def _clicked(self, item: QListWidgetItem) -> None:
        if len(self.selectedItems()) <= 1:
            self.open_requested.emit(item.data(Qt.ItemDataRole.UserRole))

    def _menu(self, pos) -> None:
        clicked = self.itemAt(pos)
        if clicked is not None and not clicked.isSelected():
            # Clic derecho sobre una foto no seleccionada: se actúa sobre ella.
            self.setCurrentItem(clicked)
        selected = self.selected_paths()
        if not selected:
            return
        n = len(selected)
        source = clicked.data(Qt.ItemDataRole.UserRole) if clicked else selected[0]
        menu = QMenu(self)
        copy = menu.addAction(f"Copiar ajustes de {Path(source).name}")
        paste = menu.addAction(f"Pegar ajustes ({n})")
        paste.setEnabled(self.can_paste())
        presets = menu.addMenu(f"Aplicar preset ({n})")
        for name, preset in (self.presets_menu_provider() if self.presets_menu_provider else []):
            act = presets.addAction(name)
            act.setData(preset)
        presets.setEnabled(not presets.isEmpty())
        reset = menu.addAction(f"Restablecer ajustes ({n})")
        menu.addSeparator()
        export = menu.addAction(f"Exportar {'seleccionadas' if n > 1 else 'foto'} ({n})…")

        chosen: QAction | None = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is copy:
            self.batch_requested.emit("copy", [source])
        elif chosen is paste:
            self.batch_requested.emit("paste", selected)
        elif chosen is reset:
            self.batch_requested.emit("reset", selected)
        elif chosen is export:
            self.batch_requested.emit("export", selected)
        elif chosen.data() is not None:
            self.batch_requested.emit("preset", [chosen.data()] + selected)

    def shutdown(self) -> None:
        self.worker.stop()
