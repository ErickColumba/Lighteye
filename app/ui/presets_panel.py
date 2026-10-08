"""Lista de presets con miniatura de cómo queda la foto actual con cada uno.

Dos pestañas: «Ajustes» (luz, color, efectos) e «IA» (usan restaurar rostros,
retoque con máscaras o quitar el fondo). Las miniaturas de la pestaña IA
muestran solo la parte de ajustes, con una marca «IA» (la parte de IA se
calcula al aplicar el preset).
"""

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.color import to_display_u8
from app.core.pipeline import process
from app.core.presets import NONE_PRESET, Preset, apply_look, list_presets
from app.core.settings import Settings
from app.ui.viewer import array_to_qimage

THUMB_SIZE = QSize(112, 76)


def _ai_badge(pix: QPixmap) -> QPixmap:
    """Marca «IA» en la esquina de la miniatura."""
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont(p.font())
    font.setPixelSize(11)
    font.setBold(True)
    p.setFont(font)
    w = p.fontMetrics().horizontalAdvance("IA") + 10
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(37, 99, 235, 235))
    p.drawRoundedRect(pix.width() - w - 4, 4, w, 16, 4, 4)
    p.setPen(QColor(255, 255, 255))
    p.drawText(pix.width() - w - 4, 4, w, 16, Qt.AlignmentFlag.AlignCenter, "IA")
    p.end()
    return pix


class PresetPanel(QWidget):
    apply_requested = Signal(object)  # Preset
    save_requested = Signal()
    delete_requested = Signal(object)  # Preset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.lists: dict[str, QListWidget] = {}
        for category, title in (("ajustes", "Ajustes"), ("ia", "IA")):
            lst = QListWidget()
            lst.setIconSize(THUMB_SIZE)
            lst.setSpacing(2)
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            lst.setWordWrap(True)
            lst.setToolTip("Clic para aplicar el preset a la foto")
            lst.itemClicked.connect(self._clicked)
            lst.currentItemChanged.connect(lambda cur, _: self._update_delete(cur))
            self.lists[category] = lst
            self.tabs.addTab(lst, title)
        self.tabs.setTabToolTip(1, "Presets que usan IA: restaurar rostros, retoque de piel, ojos, "
                                   "labios y pelo, y quitar el fondo")
        self.tabs.currentChanged.connect(lambda _: self._update_delete(self.list.currentItem()))
        save = QPushButton("Guardar actual…")
        save.setToolTip("Guarda los ajustes actuales (sin recorte ni giros) como preset")
        self.delete = QPushButton("Eliminar")

        buttons = QHBoxLayout()
        buttons.addWidget(save, 1)
        buttons.addWidget(self.delete)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(buttons)

        save.clicked.connect(self.save_requested.emit)
        self.delete.clicked.connect(self._delete_current)

        self._small: np.ndarray | None = None  # versión diminuta de la foto actual
        self._geometry = Settings()
        self._pending: list[QListWidgetItem] = []
        # Las miniaturas se calculan de una en una para no bloquear la interfaz.
        self._timer = QTimer(self, interval=0)
        self._timer.timeout.connect(self._render_next)
        self.reload()

    @property
    def list(self) -> QListWidget:
        """La lista de la pestaña visible."""
        return self.tabs.currentWidget()

    def _header(self, lst: QListWidget, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        lst.addItem(item)

    def reload(self) -> None:
        presets = list_presets()
        for category, lst in self.lists.items():
            lst.clear()
            builtin = [p for p in presets if p.builtin and p.category == category]
            mine = [p for p in presets if not p.builtin and p.category == category]
            if category == "ajustes":
                builtin = [NONE_PRESET] + builtin
            for title, group in (("Incluidos", builtin), ("Mis presets", mine)):
                self._header(lst, title)
                for preset in group:
                    item = QListWidgetItem(preset.name.removeprefix("IA · "))
                    item.setData(Qt.ItemDataRole.UserRole, preset)
                    item.setSizeHint(QSize(0, THUMB_SIZE.height() + 8))
                    if preset is NONE_PRESET:
                        item.setToolTip("Quita los ajustes y vuelve a la foto original "
                                        "(conserva el recorte). También puedes usar Ctrl+Z.")
                    elif category == "ia":
                        item.setToolTip("Usa IA: la primera vez tarda unos segundos en calcularse")
                    elif preset.builtin:
                        item.setToolTip("Incluido con Lighteye · clic para aplicar")
                    lst.addItem(item)
            if not mine:
                text = ("Ajusta una foto y pulsa\n«Guardar actual…» para crear\nlos tuyos."
                        if category == "ajustes" else
                        "Los presets que guardes usando\nherramientas de IA aparecerán aquí.")
                hint = QListWidgetItem(text)
                hint.setFlags(Qt.ItemFlag.NoItemFlags)
                lst.addItem(hint)
        self.delete.setEnabled(False)
        self._schedule()

    def set_photo(self, small: np.ndarray | None, geometry: Settings) -> None:
        """Foto (lineal, ya recortada y pequeña) sobre la que se ven los presets."""
        self._small = small
        self._geometry = geometry.copy()
        self._schedule()

    def _schedule(self) -> None:
        self._pending = [lst.item(i) for lst in self.lists.values() for i in range(lst.count())
                         if lst.item(i).data(Qt.ItemDataRole.UserRole)]
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
            pix = QPixmap.fromImage(array_to_qimage(rgb))
            if preset.category == "ia":
                pix = _ai_badge(pix)
            item.setIcon(QIcon(pix))
        except Exception:  # noqa: BLE001 — p. ej. un LUT que ya no existe: sin miniatura
            item.setIcon(QIcon())

    def _update_delete(self, item: QListWidgetItem | None) -> None:
        preset = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.delete.setEnabled(bool(preset and not preset.builtin))
        self.delete.setToolTip("Los presets incluidos no se pueden eliminar"
                               if preset and preset.builtin else "")

    def _clicked(self, item: QListWidgetItem) -> None:
        preset = item.data(Qt.ItemDataRole.UserRole)
        if preset is not None:  # el aviso "Sin presets todavía" no es un preset
            self.apply_requested.emit(preset)

    def _delete_current(self) -> None:
        item = self.list.currentItem()
        preset = item.data(Qt.ItemDataRole.UserRole) if item else None
        if preset and not preset.builtin:
            self.delete_requested.emit(preset)
