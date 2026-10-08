"""Widget del histograma RGB + luminancia con avisos de recorte.

Los triángulos de las esquinas se encienden si hay sombras empastadas
(izquierda) o luces quemadas (derecha). Al pulsarlos se muestran esas zonas
sobre la foto.
"""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.core.histogram import Histogram

CLIP_WARNING = 0.001  # 0.1 % de los píxeles
TRIANGLE = 12


class HistogramWidget(QWidget):
    clipping_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.histogram: Histogram | None = None
        self.show_clipping = False
        self.setToolTip("Pulsa los triángulos para ver las zonas recortadas")

    def set_histogram(self, histogram: Histogram | None) -> None:
        self.histogram = histogram
        self.update()

    def _plot_area(self) -> QRectF:
        return QRectF(4, 4, self.width() - 8, self.height() - 8)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self._plot_area()
        p.fillRect(area, QColor(28, 28, 28))
        p.setPen(QPen(QColor(55, 55, 55), 1))
        for i in range(1, 4):
            x = area.left() + area.width() * i / 4
            p.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))

        h = self.histogram
        if h is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(area, Qt.AlignmentFlag.AlignCenter, "Sin foto")
            return

        # Escala: se ignoran los extremos (0 y 255), que suelen tener picos
        # enormes y aplastarían el resto del histograma.
        peak = max(float(np.max(c[1:-1])) for c in (h.r, h.g, h.b, h.luma)) or 1.0

        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for counts, color in ((h.r, QColor(200, 50, 50)), (h.g, QColor(50, 170, 60)),
                              (h.b, QColor(60, 90, 220))):
            p.fillPath(self._curve(counts, peak, area), color)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setPen(QPen(QColor(220, 220, 220, 160), 1))
        p.drawPath(self._curve(h.luma, peak, area, closed=False))

        self._triangle(p, left=True, on=h.clipped_shadows > CLIP_WARNING, color=QColor(60, 120, 255))
        self._triangle(p, left=False, on=h.clipped_highlights > CLIP_WARNING, color=QColor(255, 70, 70))

    def _curve(self, counts: np.ndarray, peak: float, area: QRectF, closed: bool = True) -> QPainterPath:
        values = np.minimum(counts / peak, 1.0)
        path = QPainterPath(QPointF(area.left(), area.bottom()))
        for i, v in enumerate(values):
            x = area.left() + area.width() * i / 255
            path.lineTo(QPointF(x, area.bottom() - v * area.height()))
        path.lineTo(QPointF(area.right(), area.bottom()))
        if closed:
            path.closeSubpath()
        return path

    def _triangle_rect(self, left: bool) -> QRectF:
        area = self._plot_area()
        x = area.left() + 2 if left else area.right() - TRIANGLE - 2
        return QRectF(x, area.top() + 2, TRIANGLE, TRIANGLE)

    def _triangle(self, p: QPainter, left: bool, on: bool, color: QColor) -> None:
        r = self._triangle_rect(left)
        if left:
            poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomLeft()])
        else:
            poly = QPolygonF([r.topLeft(), r.topRight(), r.bottomRight()])
        p.setPen(QPen(QColor(150, 150, 150), 1))
        p.setBrush(color if on else QColor(60, 60, 60))
        if self.show_clipping:
            p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawPolygon(poly)

    def mousePressEvent(self, event):
        pos = event.position()
        for left in (True, False):
            if self._triangle_rect(left).adjusted(-4, -4, 4, 4).contains(pos):
                self.show_clipping = not self.show_clipping
                self.update()
                self.clipping_toggled.emit(self.show_clipping)
                return
