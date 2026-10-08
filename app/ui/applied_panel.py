"""Pestaña «Aplicados»: solo los ajustes que tiene la foto, agrupados por el
preset del que vienen (con una ✕ para quitar el preset entero) y, aparte,
los cambiados a mano.

Cada ajuste se puede cambiar (número o casilla) o quitar con la ✕ (vuelve a
su valor por defecto). Los ajustes especiales (curvas, LUT, recorte,
borrados, color de fondo) se muestran como texto y se pueden quitar.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.settings import GROUPS, PARAMS, PARAMS_BY_KEY, Settings

# Ajustes especiales: (clave, grupo, nombre)
EXTRAS = [
    ("curves", "Curvas", "Curvas"),
    ("lut_path", "LUT", "LUT"),
    ("bg_color", "Fondo (IA)", "Fondo nuevo"),
    ("crop", "Geometría", "Recorte"),
    ("erase_strokes", "Geometría", "Borrar objetos"),
]
ORDER = [*GROUPS, "Geometría"]


def _describe_extra(key: str, value) -> str:
    if key == "curves":
        changed = [name for ch, name in (("rgb", "RGB"), ("r", "Rojo"), ("g", "Verde"), ("b", "Azul"))
                   if len(value[ch]) > 2 or value[ch] != ((0.0, 0.0), (1.0, 1.0))]
        return "Modificadas: " + ", ".join(changed)
    if key == "lut_path":
        from app.core.lut import display_name

        return display_name(value)
    if key == "bg_color":
        r, g, b = (round(c * 255) for c in value)
        return f"Color ({r}, {g}, {b})"
    if key == "crop":
        x, y, w, h = value
        return f"{w * 100:.0f} % × {h * 100:.0f} % de la foto"
    if key == "erase_strokes":
        n = len(value)
        return f"{n} trazo{'s' if n != 1 else ''}"
    return str(value)


class AppliedPanel(QScrollArea):
    changed = Signal(str, object)  # clave, valor nuevo
    reset_requested = Signal(str)  # clave
    reset_all_requested = Signal()
    remove_preset_requested = Signal(str)  # nombre del preset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._keys: tuple = ()
        self._editors: dict[str, QWidget] = {}
        self._texts: dict[str, QLabel] = {}

        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setContentsMargins(6, 6, 6, 6)
        header = QHBoxLayout()
        self.count_label = QLabel()
        font = self.count_label.font()
        font.setBold(True)
        self.count_label.setFont(font)
        self.clear_all = QPushButton("Quitar todos")
        self.clear_all.setToolTip("Quita todos los ajustes de la foto (Ctrl+Z para deshacer)")
        self.clear_all.clicked.connect(self.reset_all_requested.emit)
        header.addWidget(self.count_label, 1)
        header.addWidget(self.clear_all)
        self._layout.addLayout(header)
        self._grid_holder = QWidget()
        self._layout.addWidget(self._grid_holder)
        self._layout.addStretch(1)
        self.setWidget(content)
        self.set_settings(Settings())

    # --- Contenido -----------------------------------------------------------

    def set_settings(self, settings: Settings) -> None:
        """Muestra los ajustes de la foto. Si siguen siendo los mismos ajustes
        solo se actualizan los valores (así no se pierde el que se está editando)."""
        from app.core.presets import preset_owners

        values = settings.non_default()
        values.pop("preset_stack", None)
        owners = preset_owners(settings)
        stack = [e["name"] for e in settings["preset_stack"]]
        keys = (tuple((k, owners.get(k)) for k in values), tuple(stack))
        if keys != self._keys:
            self._rebuild(values, owners, stack)
            self._keys = keys
        else:
            self._update(values)
        n = len(values)
        self.count_label.setText("Sin ajustes: la foto está como el original" if n == 0 else
                                 f"{n} ajuste{'s' if n != 1 else ''} en esta foto")
        self.clear_all.setEnabled(n > 0)

    def _rebuild(self, values: dict, owners: dict, stack: list) -> None:
        self._editors.clear()
        self._texts.clear()
        old = self._grid_holder
        self._grid_holder = QWidget()
        self._layout.replaceWidget(old, self._grid_holder)
        old.hide()  # si no, se sigue viendo debajo hasta que Qt lo borra
        old.deleteLater()
        grid = QGridLayout(self._grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)

        def ordered(keys) -> list:
            """(tipo, clave) en el orden del panel de ajustes."""
            items = [("param", p.key) for p in PARAMS if p.key in keys]
            return items + [("extra", k) for k, _, _ in EXTRAS if k in keys]

        row = 0
        # Un bloque por preset aplicado, con los ajustes que vienen de él.
        for name in stack:
            mine = [k for k in values if owners.get(k) == name]
            if not mine:
                continue
            row = self._preset_header(grid, row, name)
            for kind, key in ordered(mine):
                self._add_row(grid, row, kind, key, values[key])
                row += 1

        # Lo cambiado a mano, por secciones.
        manual = [k for k in values if owners.get(k) is None]
        if manual and row:
            row = self._title(grid, row, "Ajustes manuales", big=True)
        groups: dict[str, list] = {}
        for kind, key in ordered(manual):
            group = PARAMS_BY_KEY[key].group if kind == "param" else \
                next(g for k, g, _ in EXTRAS if k == key)
            groups.setdefault(group, []).append((kind, key))
        for group in ORDER:
            if group not in groups:
                continue
            row = self._title(grid, row, group)
            for kind, key in groups[group]:
                self._add_row(grid, row, kind, key, values[key])
                row += 1

    def _title(self, grid: QGridLayout, row: int, text: str, big: bool = False) -> int:
        title = QLabel(text)
        font = title.font()
        font.setBold(True)
        if big:
            font.setPointSizeF(font.pointSizeF() * 1.1)
        title.setFont(font)
        title.setContentsMargins(0, 10 if row else 0, 0, 2)
        grid.addWidget(title, row, 0, 1, 3)
        return row + 1

    def _preset_header(self, grid: QGridLayout, row: int, name: str) -> int:
        box = QWidget()
        box.setObjectName("presetHeader")
        box.setStyleSheet("#presetHeader { background: rgba(37, 99, 235, 40); border-radius: 4px; }")
        line = QHBoxLayout(box)
        line.setContentsMargins(6, 3, 2, 3)
        label = QLabel(f"Preset: {name.removeprefix('IA · ')}")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        line.addWidget(label, 1)
        remove = QToolButton()
        remove.setText("✕")
        remove.setAutoRaise(True)
        remove.setToolTip(f"Quitar el preset «{name}» (sus ajustes vuelven a como estaban)")
        remove.clicked.connect(lambda _=False, n=name: self.remove_preset_requested.emit(n))
        line.addWidget(remove)
        if row:
            grid.setRowMinimumHeight(row, 8)
            row += 1
        grid.addWidget(box, row, 0, 1, 3)
        return row + 1

    def _add_row(self, grid: QGridLayout, row: int, kind: str, key: str, value) -> None:
        if kind == "param":
            p = PARAMS_BY_KEY[key]
            name = f"{p.label} ({p.tab})" if p.tab else p.label
        else:
            name = next(n for k, _, n in EXTRAS if k == key)
        label = QLabel(name)
        label.setWordWrap(True)
        grid.addWidget(label, row, 0)

        if kind == "param" and PARAMS_BY_KEY[key].kind == "toggle":
            editor = QCheckBox("Activado")
            editor.setChecked(bool(value))
            editor.toggled.connect(lambda on, k=key: self.changed.emit(k, 1.0 if on else 0.0))
        elif kind == "param" and key == "rotate":
            editor = QLabel(f"{int(value) * 90}°")
        elif kind == "param":
            p = PARAMS_BY_KEY[key]
            editor = QDoubleSpinBox()
            decimals = 2 if p.step < 0.1 else 1 if p.step < 1 else 0
            editor.setDecimals(decimals)
            editor.setRange(p.minimum, p.maximum)
            editor.setSingleStep(p.step if p.step < 1 else 1)
            editor.setValue(value)
            editor.setKeyboardTracking(False)  # aplica al terminar de escribir
            editor.setMinimumWidth(80)
            editor.valueChanged.connect(lambda v, k=key: self.changed.emit(k, v))
        else:
            editor = QLabel(_describe_extra(key, value))
            editor.setWordWrap(True)
            self._texts[key] = editor
        grid.addWidget(editor, row, 1)
        self._editors[key] = editor

        remove = QToolButton()
        remove.setText("✕")
        remove.setAutoRaise(True)
        remove.setToolTip(f"Quitar «{name}»")
        remove.clicked.connect(lambda _=False, k=key: self.reset_requested.emit(k))
        grid.addWidget(remove, row, 2)

    def _update(self, values: dict) -> None:
        for key, value in values.items():
            editor = self._editors.get(key)
            if isinstance(editor, QDoubleSpinBox):
                if not editor.hasFocus() and abs(editor.value() - value) > 1e-9:
                    editor.blockSignals(True)
                    editor.setValue(value)
                    editor.blockSignals(False)
            elif isinstance(editor, QCheckBox):
                editor.blockSignals(True)
                editor.setChecked(bool(value))
                editor.blockSignals(False)
            elif key == "rotate" and isinstance(editor, QLabel):
                editor.setText(f"{int(value) * 90}°")
            elif key in self._texts:
                self._texts[key].setText(_describe_extra(key, value))
