"""Barra de herramientas del modo recorte."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QToolBar, QWidget, QSizePolicy

from app.ui.panels import ResettableSlider

# (nombre, proporción ancho/alto en horizontal). "orig" = la de la foto.
ASPECTS = [("Libre", None), ("Original", "orig"), ("1:1", 1.0), ("4:5", 5 / 4),
           ("3:2", 3 / 2), ("16:9", 16 / 9)]


class CropToolbar(QToolBar):
    aspect_changed = Signal()
    angle_changed = Signal(float)
    rotate_requested = Signal(int)  # +1 horario, −1 antihorario
    flip_requested = Signal(str)  # "h" o "v"
    reset_requested = Signal()
    cancel_requested = Signal()
    apply_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Recorte", parent)
        self.setMovable(False)

        self.addWidget(QLabel(" Proporción "))
        self.aspect = QComboBox()
        for name, value in ASPECTS:
            self.aspect.addItem(name, value)
        self.aspect.currentIndexChanged.connect(lambda _: self.aspect_changed.emit())
        self.addWidget(self.aspect)
        self.portrait = self._button("Vertical", lambda: self.aspect_changed.emit(), checkable=True)
        self.portrait.setToolTip("Usar la proporción en vertical (p. ej. 2:3 en vez de 3:2)")

        self.addSeparator()
        self.addWidget(QLabel(" Enderezar "))
        self.angle = ResettableSlider()
        self.angle.setRange(-450, 450)  # décimas de grado
        self.angle.setFixedWidth(220)
        self.angle.setToolTip("Doble clic para volver a 0°")
        self.angle_label = QLabel()
        self.angle_label.setMinimumWidth(52)
        self.angle.valueChanged.connect(self._angle_moved)
        self.angle.reset_requested.connect(lambda: self.angle.setValue(0))
        self.addWidget(self.angle)
        self.addWidget(self.angle_label)

        self.addSeparator()
        self._button("⟲ Girar", lambda: self.rotate_requested.emit(-1)).setToolTip("Girar 90° a la izquierda")
        self._button("⟳ Girar", lambda: self.rotate_requested.emit(1)).setToolTip("Girar 90° a la derecha")
        self._button("Voltear ↔", lambda: self.flip_requested.emit("h"))
        self._button("Voltear ↕", lambda: self.flip_requested.emit("v"))

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)
        self._button("Restablecer", self.reset_requested.emit)
        self._button("Cancelar", self.cancel_requested.emit).setToolTip("Esc")
        apply = self._button("Aplicar", self.apply_requested.emit)
        apply.setToolTip("Intro")
        apply.setDefault(True)
        self._set_angle_label(0)

    def _button(self, text: str, slot, checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.clicked.connect(slot)
        self.addWidget(b)
        return b

    def _set_angle_label(self, ticks: int) -> None:
        self.angle_label.setText(f"{ticks / 10:+.1f}°" if ticks else "0.0°")

    def _angle_moved(self, ticks: int) -> None:
        self._set_angle_label(ticks)
        self.angle_changed.emit(ticks / 10)

    def set_angle(self, angle: float) -> None:
        self.angle.blockSignals(True)
        self.angle.setValue(round(angle * 10))
        self.angle.blockSignals(False)
        self._set_angle_label(round(angle * 10))

    def ratio(self, image_w: int, image_h: int) -> float | None:
        """Proporción ancho/alto elegida, o None si es libre."""
        value = self.aspect.currentData()
        if value is None:
            return None
        ratio = image_w / image_h if value == "orig" else float(value)
        if value == "orig":
            # "Vertical" con Original invierte la de la foto.
            return 1 / ratio if self.portrait.isChecked() else ratio
        if self.portrait.isChecked():
            return min(ratio, 1 / ratio)
        return max(ratio, 1 / ratio)
