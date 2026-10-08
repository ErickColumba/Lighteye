"""Lista de presets con miniatura de cómo queda la foto actual con cada uno.

- Clic en un preset: vista previa (no cambia la foto ni el historial).
- Botón «+»: añade el preset a la foto (se suma a lo que ya tiene).

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
    QLabel,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.color import to_display_u8
from app.core.pipeline import process
from app.core.presets import NONE_PRESET, Preset, list_presets, stack_preset
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


class PresetRow(QWidget):
    """Fila de un preset: miniatura, nombre y botón «+» para añadirlo."""

    add_clicked = Signal()

    def __init__(self, name: str, add_tip: str, parent=None):
        super().__init__(parent)
        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_SIZE)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)  # fotos verticales, centradas
        self.name = QLabel(name)
        self.name.setWordWrap(True)
        self.add = QToolButton()
        self.add.setText("+")
        font = self.add.font()
        font.setPointSizeF(font.pointSizeF() * 1.5)
        font.setBold(True)
        self.add.setFont(font)
        self.add.setFixedSize(30, 30)
        self.add.setToolTip(add_tip)
        self.add.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add.clicked.connect(self.add_clicked.emit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.addWidget(self.thumb)
        layout.addWidget(self.name, 1)
        layout.addWidget(self.add)
        # Los clics fuera del botón llegan a la lista (vista previa).
        for w in (self.thumb, self.name):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_pixmap(self, pix: QPixmap) -> None:
        self.thumb.setPixmap(pix)


class PresetPanel(QWidget):
    apply_requested = Signal(object)  # Preset: añadirlo a la foto («+»)
    preview_requested = Signal(object)  # Preset o None: vista previa (clic)
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

        self.previewing = None  # preset en vista previa
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
            builtin = [NONE_PRESET] + builtin  # en las dos pestañas
            for title, group in (("Incluidos", builtin), ("Mis presets", mine)):
                self._header(lst, title)
                for preset in group:
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, preset)
                    item.setSizeHint(QSize(0, THUMB_SIZE.height() + 8))
                    if preset is NONE_PRESET:
                        tip = "Clic: ver la foto original · +: quitar todos los ajustes (Ctrl+Z deshace)"
                        add_tip = "Quitar todos los ajustes (conserva el recorte)"
                    else:
                        tip = "Clic: vista previa · +: añadirlo a la foto"
                        if category == "ia":
                            tip += " · usa IA: la primera vez tarda unos segundos"
                        add_tip = "Añadir este preset a la foto (se suma a lo que ya tiene)"
                    item.setToolTip(tip)
                    lst.addItem(item)
                    row = PresetRow(preset.name.removeprefix("IA · "), add_tip)
                    row.setToolTip(tip)
                    row.add_clicked.connect(lambda p=preset: self._add(p))
                    lst.setItemWidget(item, row)
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
            # Cómo quedaría la foto con el preset sumado a lo que ya tiene.
            rgb = to_display_u8(process(self._small, stack_preset(self._geometry, preset)))
            pix = QPixmap.fromImage(array_to_qimage(rgb))
            if preset.category == "ia":
                pix = _ai_badge(pix)
            row = item.listWidget().itemWidget(item) if item.listWidget() else None
            if row is not None:
                row.set_pixmap(pix.scaled(THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation))
        except Exception:  # noqa: BLE001 — p. ej. un LUT que ya no existe: sin miniatura
            pass

    def _update_delete(self, item: QListWidgetItem | None) -> None:
        preset = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.delete.setEnabled(bool(preset and not preset.builtin))
        self.delete.setToolTip("Los presets incluidos no se pueden eliminar"
                               if preset and preset.builtin else "")

    def _clicked(self, item: QListWidgetItem) -> None:
        preset = item.data(Qt.ItemDataRole.UserRole)
        if preset is None:  # encabezados y avisos no son presets
            return
        if preset is self.previewing:
            self.clear_preview()  # segundo clic: se quita la vista previa
            return
        self.previewing = preset
        self.preview_requested.emit(preset)

    def _add(self, preset) -> None:
        self.clear_preview(emit=False)
        self.apply_requested.emit(preset)

    def clear_preview(self, emit: bool = True) -> None:
        """Termina la vista previa (y quita la selección de la lista)."""
        was = self.previewing is not None
        self.previewing = None
        for lst in self.lists.values():
            lst.clearSelection()
            lst.setCurrentItem(None)
        if emit and was:
            self.preview_requested.emit(None)

    def _delete_current(self) -> None:
        item = self.list.currentItem()
        preset = item.data(Qt.ItemDataRole.UserRole) if item else None
        if preset and not preset.builtin:
            self.delete_requested.emit(preset)
