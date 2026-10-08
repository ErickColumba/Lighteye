"""Editor de curvas: puntos arrastrables sobre un lienzo cuadrado.

- Clic en una zona vacía: añade un punto.
- Arrastrar: mueve el punto (sin pasar a sus vecinos).
- Doble clic o clic derecho sobre un punto: lo elimina (no los extremos).
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from app.core.curves import sample
from app.core.settings import CURVE_CHANNELS, IDENTITY_CURVE, normalize_curve

CHANNEL_NAMES = {"rgb": "RGB", "r": "Rojo", "g": "Verde", "b": "Azul"}
CHANNEL_COLORS = {
    "rgb": QColor(230, 230, 230),
    "r": QColor(235, 80, 80),
    "g": QColor(80, 200, 100),
    "b": QColor(90, 140, 240),
}
HIT_RADIUS = 9  # px
MIN_GAP = 0.01  # separación mínima en x entre puntos vecinos


class CurveCanvas(QWidget):
    points_changed = Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.channel = "rgb"
        self.points: list[tuple[float, float]] = list(IDENTITY_CURVE)
        self.others: dict[str, tuple] = {}  # curvas de los demás canales, en tenue
        self._drag: int | None = None

    # --- Coordenadas -------------------------------------------------------

    def _area(self) -> QRectF:
        side = min(self.width(), self.height()) - 12
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def _to_px(self, x: float, y: float) -> QPointF:
        a = self._area()
        return QPointF(a.left() + x * a.width(), a.bottom() - y * a.height())

    def _from_px(self, pos: QPointF) -> tuple[float, float]:
        a = self._area()
        x = (pos.x() - a.left()) / a.width()
        y = (a.bottom() - pos.y()) / a.height()
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    def _hit(self, pos: QPointF) -> int | None:
        for i, (x, y) in enumerate(self.points):
            d = self._to_px(x, y) - pos
            if d.x() ** 2 + d.y() ** 2 <= HIT_RADIUS**2:
                return i
        return None

    # --- Dibujo ------------------------------------------------------------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = self._area()
        p.fillRect(a, QColor(38, 38, 38))

        p.setPen(QPen(QColor(70, 70, 70), 1))
        for i in range(1, 4):
            t = i / 4
            p.drawLine(self._to_px(t, 0), self._to_px(t, 1))
            p.drawLine(self._to_px(0, t), self._to_px(1, t))
        p.setPen(QPen(QColor(90, 90, 90), 1, Qt.PenStyle.DashLine))
        p.drawLine(self._to_px(0, 0), self._to_px(1, 1))

        for ch, pts in self.others.items():
            if ch != self.channel and pts != IDENTITY_CURVE:
                color = QColor(CHANNEL_COLORS[ch])
                color.setAlpha(90)
                self._draw_curve(p, pts, color, 1)

        color = CHANNEL_COLORS[self.channel]
        self._draw_curve(p, tuple(self.points), color, 2)
        p.setPen(QPen(color, 1.5))
        p.setBrush(QColor(38, 38, 38))
        for x, y in self.points:
            p.drawEllipse(self._to_px(x, y), 4.5, 4.5)
        p.end()

    def _draw_curve(self, p: QPainter, pts: tuple, color: QColor, width: float) -> None:
        ys = sample(pts, 256)
        path = QPainterPath(self._to_px(0, float(ys[0])))
        for i, y in enumerate(ys[1:], start=1):
            path.lineTo(self._to_px(i / 255, float(y)))
        p.setPen(QPen(color, width))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    # --- Ratón -------------------------------------------------------------

    def mousePressEvent(self, event):
        pos = event.position()
        hit = self._hit(pos)
        if event.button() == Qt.MouseButton.RightButton:
            self._remove(hit)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if hit is None:
            x, _ = self._from_px(pos)
            # El nuevo punto se coloca sobre la curva actual: no la deforma.
            y = float(np.interp(x, np.linspace(0, 1, 256), sample(tuple(self.points), 256)))
            if any(abs(x - px) < MIN_GAP for px, _ in self.points):
                return
            self.points.append((x, y))
            self.points.sort()
            hit = self.points.index((x, y))
            self._emit()
        self._drag = hit

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        i = self._drag
        x, y = self._from_px(event.position())
        lo = self.points[i - 1][0] + MIN_GAP if i > 0 else 0.0
        hi = self.points[i + 1][0] - MIN_GAP if i < len(self.points) - 1 else 1.0
        self.points[i] = (min(hi, max(lo, x)), y)
        self._emit()

    def mouseReleaseEvent(self, event):
        self._drag = None

    def mouseDoubleClickEvent(self, event):
        self._remove(self._hit(event.position()))

    def _remove(self, i: int | None) -> None:
        if i is not None and 0 < i < len(self.points) - 1:
            del self.points[i]
            self._drag = None
            self._emit()

    def _emit(self) -> None:
        self.update()
        self.points_changed.emit(normalize_curve(self.points))


class CurveEditor(QWidget):
    changed = Signal(dict)  # todas las curvas

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curves = {ch: IDENTITY_CURVE for ch in CURVE_CHANNELS}

        self.combo = QComboBox()
        for ch in CURVE_CHANNELS:
            self.combo.addItem(CHANNEL_NAMES[ch], ch)
        reset = QPushButton("Restablecer")
        reset.setToolTip("Vuelve recta la curva del canal elegido")
        self.canvas = CurveCanvas()

        top = QHBoxLayout()
        top.addWidget(self.combo, 1)
        top.addWidget(reset)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.canvas)

        self.combo.currentIndexChanged.connect(self._show_channel)
        self.canvas.points_changed.connect(self._on_points)
        reset.clicked.connect(self._reset_channel)

    def set_curves(self, curves: dict) -> None:
        self._curves = dict(curves)
        self._show_channel()

    def _show_channel(self) -> None:
        ch = self.combo.currentData()
        self.canvas.channel = ch
        self.canvas.points = list(self._curves[ch])
        self.canvas.others = self._curves
        self.canvas.update()

    def _on_points(self, points: tuple) -> None:
        self._curves[self.combo.currentData()] = points
        self.canvas.others = self._curves
        self.changed.emit(dict(self._curves))

    def _reset_channel(self) -> None:
        self.canvas.points = list(IDENTITY_CURVE)
        self.canvas._emit()
