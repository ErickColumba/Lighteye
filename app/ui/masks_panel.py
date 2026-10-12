"""Pestaña «Máscaras»: ajustes locales (solo en una parte de la foto).

Arriba, la lista de máscaras de la foto y los botones para añadir una nueva
(pincel, degradado lineal o radial, sujeto o fondo con IA). Debajo, la forma
de la máscara seleccionada, su rango de luminancia y sus ajustes.

La forma se dibuja sobre la foto (ver MaskOverlay en app/ui/mask_overlay.py);
este panel solo edita los valores.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.masks import LOCAL_PARAMS, TYPE_NAMES, TYPES, new_mask, normalize_mask
from app.core.settings import Param
from app.ui.panels import CollapsibleSection, ParamRow

HINTS = {
    "brush": "Pinta sobre la foto la zona a ajustar. «Borrar» quita partes de la máscara.",
    "linear": "Arrastra sobre la foto: el efecto es total donde empiezas y desaparece donde "
              "sueltas. Arrastra los puntos para moverlo.",
    "radial": "Arrastra desde el centro hacia fuera para dibujar la elipse (Mayús: círculo). "
              "Arrastra el punto central para moverla.",
    "subject": "La IA detecta la persona u objeto principal (la primera vez tarda unos segundos).",
    "background": "Todo menos el sujeto, detectado con IA (la primera vez tarda unos segundos).",
}


def _param(key, label, lo, hi, step, default=0.0) -> Param:
    return Param(key, label, "Máscara", lo, hi, default, step)


class MasksPanel(QWidget):
    changed = Signal(object)  # tupla de máscaras nueva
    selection_changed = Signal(int)  # índice o -1
    tool_changed = Signal()  # pincel pintar/borrar, tamaño u overlay

    def __init__(self, parent=None):
        super().__init__(parent)
        self.masks: list[dict] = []
        self._updating = False

        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(self._row_changed)
        add = QToolButton()
        add.setText("Añadir máscara")
        add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        add.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        menu = QMenu(add)
        for kind in TYPES:
            menu.addAction(TYPE_NAMES[kind], lambda k=kind: self.add_mask(k))
        add.setMenu(menu)
        self.duplicate = QPushButton("Duplicar")
        self.duplicate.clicked.connect(self._duplicate)
        self.remove = QPushButton("Eliminar")
        self.remove.clicked.connect(self._remove)
        buttons = QHBoxLayout()
        buttons.addWidget(add, 1)
        buttons.addWidget(self.duplicate)
        buttons.addWidget(self.remove)

        # --- Forma ---
        self.shape = CollapsibleSection("Forma")
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.shape.body_layout.insertWidget(0, self.hint)
        g = self.shape.grid
        self.paint_btn = QPushButton("Pintar")
        self.erase_btn = QPushButton("Borrar")
        self._brush_mode = QButtonGroup(self)
        for i, b in enumerate((self.paint_btn, self.erase_btn)):
            b.setCheckable(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._brush_mode.addButton(b, i)
        self.paint_btn.setChecked(True)
        self.paint_btn.setToolTip("Pintar la zona de la máscara")
        self.erase_btn.setToolTip("Borrar partes de la máscara (Alt mientras pintas)")
        self._brush_mode.idToggled.connect(lambda *_: self.tool_changed.emit())
        mode_row = QWidget()
        ml = QHBoxLayout(mode_row)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.addWidget(self.paint_btn)
        ml.addWidget(self.erase_btn)
        g.addWidget(mode_row, 0, 0, 1, 3)
        self.brush_size = ParamRow(_param("size", "Tamaño", 2, 300, 1, 40), g, 1,
                                   lambda *_: self.tool_changed.emit())
        self.hardness = ParamRow(_param("hardness", "Dureza", 0, 100, 1, 50), g, 2,
                                 lambda _, v: self._set("hardness", v / 100))
        self.feather = ParamRow(_param("feather", "Difuminado", 0, 100, 1, 50), g, 3,
                                lambda _, v: self._set("feather", v / 100))
        self.opacity = ParamRow(_param("opacity", "Opacidad", 0, 100, 1, 100), g, 4,
                                lambda _, v: self._set("opacity", v / 100))
        self.invert = QCheckBox("Invertir")
        self.invert.toggled.connect(lambda on: self._set("invert", on))
        self.show_overlay = QCheckBox("Ver máscara (O)")
        self.show_overlay.setToolTip("Muestra en rojo dónde actúa la máscara")
        self.show_overlay.toggled.connect(lambda *_: self.tool_changed.emit())
        checks = QWidget()
        cl = QHBoxLayout(checks)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.invert)
        cl.addWidget(self.show_overlay)
        cl.addStretch(1)
        g.addWidget(checks, 5, 0, 1, 3)
        self._brush_widgets = [mode_row, *self._row_widgets(self.brush_size), *self._row_widgets(self.hardness)]

        # --- Rango de luminancia ---
        self.range_section = CollapsibleSection("Rango de luminancia")
        self.range_on = QCheckBox("Solo en un rango de luz")
        self.range_on.setToolTip("Limita la máscara a las sombras, medios o luces dentro de su forma")
        self.range_on.toggled.connect(self._range_toggled)
        rg = self.range_section.grid
        rg.addWidget(self.range_on, 0, 0, 1, 3)
        self.range_lo = ParamRow(_param("lo", "Desde", 0, 100, 1, 0), rg, 1, self._range_changed)
        self.range_hi = ParamRow(_param("hi", "Hasta", 0, 100, 1, 100), rg, 2, self._range_changed)
        self.range_smooth = ParamRow(_param("smooth", "Suavidad", 1, 100, 1, 10), rg, 3, self._range_changed)

        # --- Ajustes ---
        self.adjust_section = CollapsibleSection("Ajustes de la máscara")
        self.adjust_rows: dict[str, ParamRow] = {}
        for row, (key, label, lo, hi, step) in enumerate(LOCAL_PARAMS):
            self.adjust_rows[key] = ParamRow(_param(key, label, lo, hi, step), self.adjust_section.grid,
                                             row, self._adjust_changed)
        reset = QPushButton("Restablecer ajustes de la máscara")
        reset.clicked.connect(self._reset_adjust)
        self.adjust_section.body_layout.addWidget(reset)

        editor = QWidget()
        el = QVBoxLayout(editor)
        el.setContentsMargins(0, 0, 0, 0)
        for w in (self.shape, self.range_section, self.adjust_section):
            el.addWidget(w)
        el.addStretch(1)
        self.editor = editor
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(editor)
        self.empty = QLabel("Las máscaras aplican ajustes solo a una parte de la foto: "
                            "pinta una zona, usa un degradado (cielo, viñeta a medida) o deja "
                            "que la IA separe sujeto y fondo.\n\nPulsa «Añadir máscara».")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.list)
        layout.addLayout(buttons)
        layout.addWidget(self.empty)
        layout.addWidget(scroll, 1)
        self._scroll = scroll
        self._refresh_editor()

    @staticmethod
    def _row_widgets(row: ParamRow) -> list:
        return [row.label, row.slider, row.value_label]

    # --- Estado de las herramientas (lo usa la ventana para el visor) ----------

    @property
    def brush_erasing(self) -> bool:
        return self.erase_btn.isChecked()

    def set_brush_erasing(self, erasing: bool) -> None:
        (self.erase_btn if erasing else self.paint_btn).setChecked(True)

    @property
    def brush_px(self) -> int:
        return self.brush_size.slider.value()

    def current(self) -> int:
        return self.list.currentRow() if 0 <= self.list.currentRow() < len(self.masks) else -1

    def current_mask(self) -> dict | None:
        i = self.current()
        return self.masks[i] if i >= 0 else None

    # --- Lista ----------------------------------------------------------------

    def set_masks(self, masks) -> None:
        """Máscaras de la foto (al abrir, deshacer…). Mantiene la selección."""
        masks = [dict(m) for m in masks]
        if masks == self.masks:
            return
        row = self.current()
        self.masks = masks
        self._fill_list(min(row, len(masks) - 1) if row >= 0 else (0 if masks else -1))

    def _fill_list(self, select: int) -> None:
        self._updating = True
        self.list.clear()
        for m in self.masks:
            text = m["name"]
            if not m["adjust"]:
                text += "  (sin ajustes)"
            item = QListWidgetItem(text)
            self.list.addItem(item)
        self.list.setCurrentRow(select)
        self._updating = False
        self._refresh_editor()
        self.selection_changed.emit(self.current())

    def _row_changed(self, _row: int) -> None:
        if self._updating:
            return
        self._refresh_editor()
        self.selection_changed.emit(self.current())

    def _name_for(self, kind: str) -> str:
        base = TYPE_NAMES[kind]
        names = {m["name"] for m in self.masks}
        n = 1
        while f"{base} {n}" in names:
            n += 1
        return f"{base} {n}"

    def add_mask(self, kind: str) -> None:
        mask = new_mask(kind)
        mask["name"] = self._name_for(kind)
        self.masks.append(mask)
        self._fill_list(len(self.masks) - 1)
        self._emit()

    def _duplicate(self) -> None:
        m = self.current_mask()
        if m is None:
            return
        copy = normalize_mask(m)
        copy["name"] = self._name_for(m["type"])
        self.masks.insert(self.current() + 1, copy)
        self._fill_list(self.current() + 1)
        self._emit()

    def _remove(self) -> None:
        i = self.current()
        if i < 0:
            return
        del self.masks[i]
        self._fill_list(min(i, len(self.masks) - 1))
        self._emit()

    # --- Edición ----------------------------------------------------------------

    def _emit(self) -> None:
        self.changed.emit(tuple(normalize_mask(m) for m in self.masks))

    def update_current(self, **values) -> None:
        """Cambia la máscara seleccionada (p. ej. desde el visor: trazos, puntos)."""
        i = self.current()
        if i < 0:
            return
        self.masks[i] = normalize_mask({**self.masks[i], **values})
        self._refresh_names()
        self._emit()

    def _set(self, key: str, value) -> None:
        if self._updating:
            return
        self.update_current(**{key: value})

    def _refresh_names(self) -> None:
        for i, m in enumerate(self.masks):
            item = self.list.item(i)
            if item is not None:
                item.setText(m["name"] + ("" if m["adjust"] else "  (sin ajustes)"))

    def _adjust_changed(self, key: str, value: float) -> None:
        m = self.current_mask()
        if self._updating or m is None:
            return
        self.update_current(adjust={**m["adjust"], key: value})

    def _reset_adjust(self) -> None:
        if self.current_mask() is not None:
            self.update_current(adjust={})
            self._refresh_editor()

    def _range_values(self) -> dict:
        return {"lo": self.range_lo.slider.value() / 100, "hi": self.range_hi.slider.value() / 100,
                "smooth": self.range_smooth.slider.value() / 100}

    def _range_toggled(self, on: bool) -> None:
        for row in (self.range_lo, self.range_hi, self.range_smooth):
            for w in self._row_widgets(row):
                w.setEnabled(on)
        self._set("range", self._range_values() if on else None)

    def _range_changed(self, *_):
        if self.range_on.isChecked():
            self._set("range", self._range_values())

    def _refresh_editor(self) -> None:
        """Muestra los valores de la máscara seleccionada."""
        m = self.current_mask()
        has = m is not None
        self._scroll.setVisible(has)
        self.empty.setVisible(not self.masks)
        self.duplicate.setEnabled(has)
        self.remove.setEnabled(has)
        if not has:
            return
        self._updating = True
        try:
            kind = m["type"]
            self.hint.setText(HINTS[kind])
            for w in self._brush_widgets:
                w.setVisible(kind == "brush")
            for w in self._row_widgets(self.feather):
                w.setVisible(kind == "radial")
            if kind == "brush":
                self.hardness.set_value(round(m["hardness"] * 100))
            if kind == "radial":
                self.feather.set_value(round(m["feather"] * 100))
            self.opacity.set_value(round(m["opacity"] * 100))
            self.invert.setChecked(m["invert"])
            rng = m["range"]
            self.range_on.setChecked(rng is not None)
            r = rng or {"lo": 0, "hi": 1, "smooth": 0.1}
            self.range_lo.set_value(round(r["lo"] * 100))
            self.range_hi.set_value(round(r["hi"] * 100))
            self.range_smooth.set_value(max(1, round(r["smooth"] * 100)))
            for row in (self.range_lo, self.range_hi, self.range_smooth):
                for w in self._row_widgets(row):
                    w.setEnabled(rng is not None)
            for key, row in self.adjust_rows.items():
                row.set_value(m["adjust"].get(key, 0.0))
        finally:
            self._updating = False
