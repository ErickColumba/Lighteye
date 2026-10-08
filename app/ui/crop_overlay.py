"""Marco de recorte sobre el visor.

- Arrastrar dentro del marco: moverlo.
- Arrastrar esquinas o lados: cambiar su tamaño (respetando la proporción
  fijada, si la hay).
- Fuera del marco se oscurece la imagen; dentro se dibuja la regla de tercios.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

HANDLE_PX = 10  # tamaño de los tiradores en píxeles de pantalla
MIN_SIZE_PX = 16  # tamaño mínimo del recorte, en píxeles de la imagen


def fit_aspect(rect: QRectF, aspect: float | None, bounds: QRectF) -> QRectF:
    """Mayor rectángulo con proporción `aspect` (ancho/alto) centrado en
    `rect` y dentro de `bounds`."""
    r = QRectF(rect).intersected(bounds)
    if not aspect:
        return r
    w, h = r.width(), r.height()
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    c = r.center()
    return QRectF(c.x() - w / 2, c.y() - h / 2, w, h)


class CropOverlay(QGraphicsObject):
    rect_changed = Signal(QRectF)

    def __init__(self):
        super().__init__()
        self.bounds = QRectF()
        self.rect = QRectF()
        self.aspect: float | None = None
        self._mode: str | None = None  # "move" o combinación de "l", "r", "t", "b"
        self._press = QPointF()
        self._start = QRectF()
        self.setZValue(10)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)

    # --- Estado ----------------------------------------------------------

    def set_bounds(self, bounds: QRectF, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self.bounds = QRectF(bounds)
        self.rect = fit_aspect(rect, self.aspect, self.bounds)
        self.update()

    def set_aspect(self, aspect: float | None) -> None:
        self.aspect = aspect
        if aspect:
            # Con una proporción nueva se usa el mayor marco posible.
            self.rect = fit_aspect(self.bounds, aspect, self.bounds)
        self.update()
        self.rect_changed.emit(self.rect)

    def normalized(self) -> tuple[float, float, float, float]:
        b = self.bounds
        return ((self.rect.x() - b.x()) / b.width(), (self.rect.y() - b.y()) / b.height(),
                self.rect.width() / b.width(), self.rect.height() / b.height())

    # --- Dibujo ----------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return self.bounds.adjusted(-50, -50, 50, 50)

    def _scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        return views[0].transform().m11() if views else 1.0

    def paint(self, p: QPainter, option, widget=None):
        if self.bounds.isEmpty():
            return
        outside = QPainterPath()
        outside.setFillRule(Qt.FillRule.OddEvenFill)
        outside.addRect(self.bounds)
        outside.addRect(self.rect)
        p.fillPath(outside, QColor(0, 0, 0, 150))

        r = self.rect
        thirds = QPen(QColor(255, 255, 255, 110), 0)  # 0 = 1 px de pantalla
        p.setPen(thirds)
        for i in (1, 2):
            x = r.left() + r.width() * i / 3
            y = r.top() + r.height() * i / 3
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor(255, 255, 255), 0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

        size = HANDLE_PX / self._scale()
        p.setBrush(QColor(255, 255, 255))
        for pt in self._handle_points().values():
            p.drawRect(QRectF(pt.x() - size / 2, pt.y() - size / 2, size, size))

    def _handle_points(self) -> dict[str, QPointF]:
        r = self.rect
        cx, cy = r.center().x(), r.center().y()
        return {
            "lt": r.topLeft(), "rt": r.topRight(), "lb": r.bottomLeft(), "rb": r.bottomRight(),
            "t": QPointF(cx, r.top()), "b": QPointF(cx, r.bottom()),
            "l": QPointF(r.left(), cy), "r": QPointF(r.right(), cy),
        }

    # --- Ratón -------------------------------------------------------------

    def _hit(self, pos: QPointF) -> str | None:
        tolerance = HANDLE_PX / self._scale()
        for name, pt in self._handle_points().items():
            if abs(pos.x() - pt.x()) <= tolerance and abs(pos.y() - pt.y()) <= tolerance:
                return name
        return "move" if self.rect.contains(pos) else None

    def hoverMoveEvent(self, event):
        cursors = {
            "lt": Qt.CursorShape.SizeFDiagCursor, "rb": Qt.CursorShape.SizeFDiagCursor,
            "rt": Qt.CursorShape.SizeBDiagCursor, "lb": Qt.CursorShape.SizeBDiagCursor,
            "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
            "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
            "move": Qt.CursorShape.SizeAllCursor,
        }
        hit = self._hit(event.pos())
        if hit:
            self.setCursor(cursors[hit])
        else:
            self.unsetCursor()

    def mousePressEvent(self, event):
        self._mode = self._hit(event.pos())
        if self._mode is None or event.button() != Qt.MouseButton.LeftButton:
            event.ignore()  # fuera del marco: el visor se desplaza
            return
        self._press = event.pos()
        self._start = QRectF(self.rect)

    def mouseMoveEvent(self, event):
        if self._mode is None:
            return
        d = event.pos() - self._press
        self.rect = self._move(d) if self._mode == "move" else self._resize(event.pos())
        self.update()
        self.rect_changed.emit(self.rect)

    def mouseReleaseEvent(self, event):
        self._mode = None

    def _move(self, d: QPointF) -> QRectF:
        r = self._start.translated(d)
        b = self.bounds
        r.moveLeft(min(max(r.left(), b.left()), b.right() - r.width()))
        r.moveTop(min(max(r.top(), b.top()), b.bottom() - r.height()))
        return r

    def _resize(self, pos: QPointF) -> QRectF:
        s, b, m = self._start, self.bounds, self._mode
        left, right, top, bottom = s.left(), s.right(), s.top(), s.bottom()
        x = min(max(pos.x(), b.left()), b.right())
        y = min(max(pos.y(), b.top()), b.bottom())
        if "l" in m:
            left = min(x, right - MIN_SIZE_PX)
        if "r" in m:
            right = max(x, left + MIN_SIZE_PX)
        if "t" in m:
            top = min(y, bottom - MIN_SIZE_PX)
        if "b" in m:
            bottom = max(y, top + MIN_SIZE_PX)
        r = QRectF(QPointF(left, top), QPointF(right, bottom))
        if not self.aspect:
            return r
        return self._constrain_aspect(r, m)

    def _constrain_aspect(self, r: QRectF, m: str) -> QRectF:
        """Ajusta r a la proporción fija, anclado en el lado/esquina opuesto,
        y lo reduce si se sale de la imagen."""
        a, b = self.aspect, self.bounds
        w, h = r.width(), r.height()
        if m in ("l", "r"):
            h = w / a
        elif m in ("t", "b"):
            w = h * a
        elif w / h > a:
            w = h * a
        else:
            h = w / a

        # Punto fijo: el lado o esquina contrarios al que se arrastra.
        s = self._start
        ax = s.right() if "l" in m else s.left() if "r" in m else s.center().x()
        ay = s.bottom() if "t" in m else s.top() if "b" in m else s.center().y()
        # Espacio disponible desde el punto fijo, en la dirección del arrastre.
        if "l" in m:
            max_w = ax - b.left()
        elif "r" in m:
            max_w = b.right() - ax
        else:
            max_w = 2 * min(ax - b.left(), b.right() - ax)
        if "t" in m:
            max_h = ay - b.top()
        elif "b" in m:
            max_h = b.bottom() - ay
        else:
            max_h = 2 * min(ay - b.top(), b.bottom() - ay)
        k = min(1.0, max_w / w if w else 1.0, max_h / h if h else 1.0)
        w, h = w * k, h * k

        x0 = ax - w if "l" in m else ax if "r" in m else ax - w / 2
        y0 = ay - h if "t" in m else ay if "b" in m else ay - h / 2
        return QRectF(x0, y0, w, h)
