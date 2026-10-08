"""Panel de ajustes: un slider por parámetro, en secciones plegables."""

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.settings import GROUPS, PARAMS, Param, Settings


class ResettableSlider(QSlider):
    """Slider horizontal: doble clic vuelve al valor por defecto.

    La rueda del ratón solo lo mueve si tiene el foco, para que al desplazar
    el panel no se cambien ajustes sin querer.
    """

    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseDoubleClickEvent(self, event):
        self.reset_requested.emit()

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class ParamRow:
    """Etiqueta + slider + valor para un Param. No es un widget propio:
    coloca sus piezas en una fila de la cuadrícula de la sección."""

    def __init__(self, param: Param, grid: QGridLayout, row: int, on_change):
        self.param = param
        self._on_change = on_change
        self._decimals = max(0, -int(math.floor(math.log10(param.step)))) if param.step < 1 else 0

        self.label = QLabel(param.label)
        self.slider = ResettableSlider()
        self.slider.setRange(self._to_ticks(param.minimum), self._to_ticks(param.maximum))
        self.slider.setPageStep(max(1, self._to_ticks(param.maximum) // 10))
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumWidth(44)

        grid.addWidget(self.label, row, 0)
        grid.addWidget(self.slider, row, 1)
        grid.addWidget(self.value_label, row, 2)

        self.slider.valueChanged.connect(self._slider_moved)
        self.slider.reset_requested.connect(lambda: self.set_value(param.default, emit=True))
        self.set_value(param.default)

    def _to_ticks(self, value: float) -> int:
        return round(value / self.param.step)

    def _slider_moved(self, ticks: int) -> None:
        value = round(ticks * self.param.step, self._decimals)
        self._show(value)
        self._on_change(self.param.key, value)

    def _show(self, value: float) -> None:
        signed = self.param.minimum < 0 and value != 0
        text = f"{value:+.{self._decimals}f}" if signed else f"{value:.{self._decimals}f}"
        self.value_label.setText(text)
        bold = value != self.param.default
        font = self.label.font()
        font.setBold(bold)
        self.label.setFont(font)

    def set_value(self, value: float, emit: bool = False) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_ticks(value))
        self.slider.blockSignals(False)
        self._show(value)
        if emit:
            self._on_change(self.param.key, value)


class ToggleRow:
    """Casilla para un Param de tipo "toggle" (valor 0 o 1)."""

    def __init__(self, param: Param, grid: QGridLayout, row: int, on_change):
        self.param = param
        self.checkbox = QCheckBox(param.label)
        grid.addWidget(self.checkbox, row, 0, 1, 3)
        self.checkbox.toggled.connect(lambda on: on_change(param.key, 1.0 if on else 0.0))
        self.set_value(param.default)

    def set_value(self, value: float, emit: bool = False) -> None:
        self.checkbox.blockSignals(not emit)
        self.checkbox.setChecked(bool(value))
        self.checkbox.blockSignals(False)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle.setAutoRaise(True)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 0, 4, 8)
        self.grid = new_grid()
        self.body_layout.addLayout(self.grid)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle)
        layout.addWidget(self.body)

        self.toggle.toggled.connect(self._set_open)

    def _set_open(self, is_open: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if is_open else Qt.ArrowType.RightArrow)
        self.body.setVisible(is_open)


def new_grid() -> QGridLayout:
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setColumnStretch(1, 1)
    return grid


class AdjustmentPanel(QScrollArea):
    changed = Signal(str, object)  # clave, valor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(320)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)

        self.rows: dict[str, ParamRow | ToggleRow] = {}
        self.sections: dict[str, CollapsibleSection] = {}
        for group in GROUPS:
            params = [p for p in PARAMS if p.group == group]
            section = CollapsibleSection(group)
            self._fill_section(section, params)
            section.setVisible(bool(params))  # se muestra al añadirle contenido
            self.sections[group] = section
            layout.addWidget(section)

        layout.addStretch(1)
        self.setWidget(content)

    def _fill_section(self, section: CollapsibleSection, params: list[Param]) -> None:
        tabs: dict[str, QGridLayout] = {}
        tab_widget = None
        for p in params:
            if p.tab:
                if tab_widget is None:
                    tab_widget = QTabWidget()
                    tab_widget.setDocumentMode(True)
                    section.body_layout.addWidget(tab_widget)
                if p.tab not in tabs:
                    page = QWidget()
                    tabs[p.tab] = new_grid()
                    tabs[p.tab].setContentsMargins(0, 6, 0, 0)
                    page.setLayout(tabs[p.tab])
                    tab_widget.addTab(page, p.tab)
                grid = tabs[p.tab]
            else:
                grid = section.grid
            row_cls = ToggleRow if p.kind == "toggle" else ParamRow
            self.rows[p.key] = row_cls(p, grid, grid.rowCount(), self.changed.emit)

    def set_settings(self, settings: Settings) -> None:
        """Refleja unos ajustes en los sliders sin emitir `changed`."""
        for key, row in self.rows.items():
            row.set_value(settings[key])
