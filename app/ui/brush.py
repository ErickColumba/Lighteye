"""Pincel de «Borrar objetos»: se pinta sobre la foto y, al soltar, el trazo
se entrega a la ventana para que lo rellene LaMa."""

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolBar,
    QWidget,
)


class BrushOverlay(QGraphicsObject):
    stroke_finished = Signal(list, float)  # puntos (escena), radio (escena)

    def __init__(self):
        super().__init__()
        self.bounds = QRectF()
        self.radius_px = 30  # radio en píxeles de pantalla
        self._points: list[QPointF] = []
        self._hover: QPointF | None = None
        self.setZValue(20)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption)

    def set_bounds(self, bounds: QRectF) -> None:
        self.prepareGeometryChange()
        self.bounds = QRectF(bounds)

    def _scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        return views[0].transform().m11() if views else 1.0

    def radius(self) -> float:
        """Radio en coordenadas de la imagen mostrada."""
        return self.radius_px / self._scale()

    def boundingRect(self) -> QRectF:
        margin = self.radius() + 4 if self.scene() else 50
        return self.bounds.adjusted(-margin, -margin, margin, margin)

    def paint(self, p: QPainter, option, widget=None):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.radius()
        if self._points:
            pen = QPen(QColor(255, 40, 40, 140), 2 * r, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            path = QPainterPath(self._points[0])
            for pt in self._points[1:]:
                path.lineTo(pt)
            if len(self._points) == 1:
                path.lineTo(self._points[0] + QPointF(0.01, 0))
            p.drawPath(path)
        if self._hover is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 160), 3 / self._scale()))
            p.drawEllipse(self._hover, r, r)
            p.setPen(QPen(QColor(255, 255, 255), 1.5 / self._scale()))
            p.drawEllipse(self._hover, r, r)

    def hoverMoveEvent(self, event):
        self._hover = event.pos()
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        self._points = [event.pos()]
        self.update()

    def mouseMoveEvent(self, event):
        self._hover = event.pos()
        if self._points:
            self._points.append(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        if self._points:
            points = [(pt.x(), pt.y()) for pt in self._points]
            self._points = []
            self.update()
            self.stroke_finished.emit(points, self.radius())


class EraseToolbar(QToolBar):
    size_changed = Signal(int)
    clear_requested = Signal()
    done_requested = Signal()

    def __init__(self, parent=None):
        super().__init__("Borrar objetos", parent)
        self.setMovable(False)
        self.addWidget(QLabel("  Pinta sobre lo que quieras borrar.   Tamaño del pincel "))
        self.size = QSlider(Qt.Orientation.Horizontal)
        self.size.setRange(4, 200)
        self.size.setValue(30)
        self.size.setFixedWidth(200)
        self.size_label = QLabel("30 px")
        self.size_label.setMinimumWidth(50)
        self.size.valueChanged.connect(lambda v: (self.size_label.setText(f"{v} px"), self.size_changed.emit(v)))
        self.addWidget(self.size)
        self.addWidget(self.size_label)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)
        clear = QPushButton("Quitar todos los borrados")
        clear.clicked.connect(self.clear_requested.emit)
        done = QPushButton("Listo")
        done.setToolTip("Intro o Esc")
        done.clicked.connect(self.done_requested.emit)
        for b in (clear, done):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.addWidget(b)
