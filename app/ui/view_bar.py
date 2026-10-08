"""Barra inferior del visor: ver el original, el resultado o ambos divididos."""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

from app.ui.icons import icon

MODES = [
    ("before", "Antes", "Ver la foto original  (\\ alterna antes / después)"),
    ("after", "Después", "Ver la foto editada"),
    ("compare", "Comparar", "Original a la izquierda y editada a la derecha; arrastra la línea"),
]


class ViewModeBar(QWidget):
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QToolButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addStretch(1)
        for mode, text, tip in MODES:
            b = QToolButton()
            b.setIcon(icon(mode))
            b.setIconSize(QSize(20, 20))
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.clicked.connect(lambda _=False, m=mode: self.mode_changed.emit(m))
            self.group.addButton(b)
            self.buttons[mode] = b
            layout.addWidget(b)
        layout.addStretch(1)
        self.set_mode("after")

    def set_mode(self, mode: str) -> None:
        self.buttons[mode].setChecked(True)
