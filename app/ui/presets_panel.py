"""Lista de presets con miniatura de cómo queda la foto actual con cada uno."""

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.color import to_display_u8
from app.core.pipeline import process
from app.core.presets import Preset, apply_look, list_presets
from app.core.settings import Settings
from app.ui.viewer import array_to_qimage

THUMB_SIZE = QSize(112, 76)


class PresetPanel(QWidget):
    apply_requested = Signal(object)  # Preset
    save_requested = Signal()
    delete_requested = Signal(object)  # Preset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.list = QListWidget()
        self.list.setIconSize(THUMB_SIZE)
        self.list.setSpacing(2)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setWordWrap(True)
        self.list.setToolTip("Clic para aplicar el preset a la foto")
        save = QPushButton("Guardar actual…")
        save.setToolTip("Guarda los ajustes actuales (sin recorte ni giros) como preset")
        self.delete = QPushButton("Eliminar")

        buttons = QHBoxLayout()
        buttons.addWidget(save, 1)
        buttons.addWidget(self.delete)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)

        self.list.itemClicked.connect(lambda item: self.apply_requested.emit(item.data(Qt.ItemDataRole.UserRole)))
        self.list.currentItemChanged.connect(lambda cur, _: self.delete.setEnabled(cur is not None))
        save.clicked.connect(self.save_requested.emit)
        self.delete.clicked.connect(self._delete_current)

        self._small: np.ndarray | None = None  # versión diminuta de la foto actual
        self._geometry = Settings()
        self._pending: list[QListWidgetItem] = []
        # Las miniaturas se calculan de una en una para no bloquear la interfaz.
        self._timer = QTimer(self, interval=0)
        self._timer.timeout.connect(self._render_next)
        self.reload()

    def reload(self) -> None:
        self.list.clear()
        for preset in list_presets():
            item = QListWidgetItem(preset.name)
            item.setData(Qt.ItemDataRole.UserRole, preset)
            item.setSizeHint(QSize(0, THUMB_SIZE.height() + 8))
            self.list.addItem(item)
        if self.list.count() == 0:
            empty = QListWidgetItem("Sin presets todavía.\nAjusta una foto y pulsa\n«Guardar actual…».")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
        self.delete.setEnabled(False)
        self._schedule()

    def set_photo(self, small: np.ndarray | None, geometry: Settings) -> None:
        """Foto (lineal, ya recortada y pequeña) sobre la que se ven los presets."""
        self._small = small
        self._geometry = geometry.copy()
        self._schedule()

    def _schedule(self) -> None:
        self._pending = [self.list.item(i) for i in range(self.list.count())
                         if self.list.item(i).data(Qt.ItemDataRole.UserRole)]
        if self._small is not None and self._pending:
            self._timer.start()

    def _render_next(self) -> None:
        if not self._pending or self._small is None:
            self._timer.stop()
            return
        item = self._pending.pop(0)
        preset: Preset = item.data(Qt.ItemDataRole.UserRole)
        try:
            rgb = to_display_u8(process(self._small, apply_look(self._geometry, preset.values)))
            item.setIcon(QIcon(QPixmap.fromImage(array_to_qimage(rgb))))
        except Exception:  # noqa: BLE001 — p. ej. un LUT que ya no existe: sin miniatura
            item.setIcon(QIcon())

    def _delete_current(self) -> None:
        item = self.list.currentItem()
        preset = item.data(Qt.ItemDataRole.UserRole) if item else None
        if preset:
            self.delete_requested.emit(preset)
