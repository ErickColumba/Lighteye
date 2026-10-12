"""Guías de los degradados de las máscaras sobre la foto.

La ventana le pasa las líneas y los puntos ya convertidos a coordenadas de la
imagen mostrada; este objeto solo los dibuja y avisa de los arrastres:
- Arrastrar un punto (tirador): lo mueve.
- Arrastrar en otro sitio: dibuja un degradado nuevo.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

HANDLE_PX = 7  # radio de los tiradores, en píxeles de pantalla


class GradientOverlay(QGraphicsObject):
    # ("new", -1) o ("handle", índice); inicio y posición (escena); terminado; Mayús
    dragged = Signal(str, int, QPointF, QPointF, bool, bool)

    def __init__(self):
        super().__init__()
        self.bounds = QRectF()
        self.lines: list[tuple[list[QPointF], bool]] = []  # (puntos, discontinua)
        self.handles: list[QPointF] = []
        self._drag: tuple[str, int, QPointF] | None = None
        self.setZValue(20)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_bounds(self, bounds: QRectF) -> None:
        self.prepareGeometryChange()
        self.bounds = QRectF(bounds)

    def set_guides(self, lines, handles) -> None:
        self.lines = lines
        self.handles = handles
        self.update()

    def _scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        return views[0].transform().m11() if views else 1.0

    def boundingRect(self) -> QRectF:
        # Las guías pueden salir de la foto (un degradado que empieza fuera).
        return self.bounds.adjusted(-self.bounds.width(), -self.bounds.height(),
                                    self.bounds.width(), self.bounds.height())

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self._scale()
        for pts, dashed in self.lines:
            if len(pts) < 2:
                continue
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(QPen(QColor(0, 0, 0, 150), 3 / s))
            p.drawPath(path)
            pen = QPen(QColor(255, 255, 255), 1.5 / s)
            if dashed:
                pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawPath(path)
        r = HANDLE_PX / s
        for h in self.handles:
            p.setPen(QPen(QColor(0, 0, 0, 180), 2 / s))
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(h, r, r)

    def _handle_at(self, pos: QPointF) -> int:
        limit = (HANDLE_PX + 5) / self._scale()
        for i, h in enumerate(self.handles):
            if (h - pos).manhattanLength() <= limit * 1.4:
                return i
        return -1

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        i = self._handle_at(event.pos())
        self._drag = ("handle", i, event.pos()) if i >= 0 else ("new", -1, event.pos())

    def _emit(self, event, finished: bool) -> None:
        if self._drag is None:
            return
        kind, i, start = self._drag
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.dragged.emit(kind, i, start, event.pos(), finished, shift)

    def mouseMoveEvent(self, event):
        self._emit(event, False)

    def mouseReleaseEvent(self, event):
        self._emit(event, True)
        self._drag = None
