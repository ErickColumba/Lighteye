"""Visor de imagen con zoom (rueda del ratón) y desplazamiento (arrastrar)."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.ui.brush import BrushOverlay
from app.ui.crop_overlay import CropOverlay

MIN_ZOOM = 0.05
MAX_ZOOM = 16.0


def array_to_qimage(rgb_u8: np.ndarray) -> QImage:
    h, w = rgb_u8.shape[:2]
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    # copy() para que QImage no dependa de la memoria del array de NumPy.
    return QImage(rgb_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class CompareDivider(QGraphicsObject):
    """Línea vertical arrastrable del modo comparar (antes | después)."""

    moved = Signal(float)
    GRAB_PX = 10

    def __init__(self):
        super().__init__()
        self.bounds = QRectF()
        self.position = 0.5  # fracción del ancho
        self._dragging = False
        self.setZValue(5)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity)

    def set_bounds(self, bounds: QRectF) -> None:
        self.prepareGeometryChange()
        self.bounds = QRectF(bounds)

    def _scale(self) -> float:
        views = self.scene().views() if self.scene() else []
        return views[0].transform().m11() if views else 1.0

    def _x(self) -> float:
        return self.bounds.left() + self.position * self.bounds.width()

    def boundingRect(self) -> QRectF:
        return self.bounds.adjusted(-60, -10, 60, 10)

    def paint(self, p: QPainter, option, widget=None):
        if self.bounds.isEmpty():
            return
        k = 1 / self._scale()  # tamaños en píxeles de pantalla
        x = self._x()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 120), 4 * k))
        p.drawLine(QPointF(x, self.bounds.top()), QPointF(x, self.bounds.bottom()))
        p.setPen(QPen(QColor(255, 255, 255), 2 * k))
        p.drawLine(QPointF(x, self.bounds.top()), QPointF(x, self.bounds.bottom()))
        # Tirador redondo en el centro, con flechas.
        c = QPointF(x, self.bounds.center().y())
        p.setBrush(QColor(255, 255, 255))
        p.setPen(QPen(QColor(0, 0, 0, 90), 1 * k))
        p.drawEllipse(c, 14 * k, 14 * k)
        p.setPen(QPen(QColor(40, 40, 40), 2 * k))
        for d in (-1, 1):
            tip = QPointF(x + d * 9 * k, c.y())
            p.drawLine(tip, QPointF(x + d * 4 * k, c.y() - 5 * k))
            p.drawLine(tip, QPointF(x + d * 4 * k, c.y() + 5 * k))
        # Etiquetas arriba.
        font = QFont(p.font())
        font.setPointSizeF(10)
        font.setBold(True)
        p.setFont(font)
        p.save()
        p.translate(x, self.bounds.top())
        p.scale(k, k)
        for text, left in (("Antes", True), ("Después", False)):
            w = p.fontMetrics().horizontalAdvance(text) + 16
            box = QRectF(-w - 8 if left else 8, 8, w, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 160))
            p.drawRoundedRect(box, 5, 5)
            p.setPen(QColor(255, 255, 255))
            p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()

    def _near(self, pos: QPointF) -> bool:
        return abs(pos.x() - self._x()) <= self.GRAB_PX / self._scale()

    def hoverMoveEvent(self, event):
        if self._near(event.pos()):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._near(event.pos()):
            event.ignore()  # lejos de la línea: el visor se desplaza
            return
        self._dragging = True

    def mouseMoveEvent(self, event):
        if not self._dragging or self.bounds.width() <= 0:
            return
        frac = min(1.0, max(0.0, (event.pos().x() - self.bounds.left()) / self.bounds.width()))
        # Al redibujar la foto Qt reenvía un movimiento en la misma posición:
        # si no se ignora, se entra en un bucle sin fin.
        if abs(frac - self.position) < 1e-4:
            return
        self.position = frac
        self.update()
        self.moved.emit(self.position)

    def mouseReleaseEvent(self, event):
        self._dragging = False


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._item)
        self.crop_overlay = CropOverlay()
        self.crop_overlay.setVisible(False)
        self._scene.addItem(self.crop_overlay)
        self.compare_divider = CompareDivider()
        self.compare_divider.setVisible(False)
        self._scene.addItem(self.compare_divider)
        self.brush = BrushOverlay()
        self.brush.setVisible(False)
        self._scene.addItem(self.brush)
        self.setScene(self._scene)

        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        # Se desplaza arrastrando; las barras sobran (y la escena tiene margen).
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._fit_mode = True
        self._label = ""
        # Indicador de carga (tareas de IA): texto y ángulo de la rueda.
        self._busy = ""
        self._spin = 0
        self._spin_timer = QTimer(self, interval=40)
        self._spin_timer.timeout.connect(self._tick)

    def set_label(self, text: str) -> None:
        """Texto fijo en la esquina del visor (p. ej. "Antes")."""
        self._label = text
        self.viewport().update()

    def set_busy(self, text: str) -> None:
        """Muestra (texto) u oculta ("") el indicador de carga en el centro."""
        self._busy = text
        if text and not self._spin_timer.isActive():
            self._spin_timer.start()
        elif not text:
            self._spin_timer.stop()
        self.viewport().update()

    def _tick(self) -> None:
        self._spin = (self._spin + 12) % 360
        self.viewport().update()

    def _draw_busy(self, painter: QPainter) -> None:
        painter.save()
        painter.resetTransform()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        lines = self._busy.split("\n")
        text_w = max(painter.fontMetrics().horizontalAdvance(t) for t in lines)
        w, h = text_w + 90, 30 + 22 * len(lines)
        vp = self.viewport().rect()
        box = QRectF(vp.center().x() - w / 2, vp.center().y() - h / 2, w, h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(20, 20, 20, 215))
        painter.drawRoundedRect(box, 10, 10)
        ring = QRectF(box.left() + 20, box.center().y() - 14, 28, 28)
        painter.setPen(QPen(QColor(255, 255, 255, 60), 4))
        painter.drawEllipse(ring)
        pen = QPen(QColor(80, 160, 255), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(ring, -self._spin * 16, 100 * 16)
        painter.setPen(QColor(255, 255, 255))
        text_box = QRectF(ring.right() + 16, box.top() + 15, text_w + 10, h - 30)
        painter.drawText(text_box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._busy)
        painter.restore()

    def drawForeground(self, painter: QPainter, rect) -> None:
        if self._busy:
            self._draw_busy(painter)
        if not self._label:
            return
        painter.save()
        painter.resetTransform()  # coordenadas del viewport, no de la escena
        font = QFont(painter.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        box = QRectF(12, 12, painter.fontMetrics().horizontalAdvance(self._label) + 20, 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self._label)
        painter.restore()

    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def set_image(self, rgb_u8: np.ndarray, reset_view: bool = False) -> None:
        """Muestra una imagen sRGB uint8. Conserva el zoom salvo reset_view."""
        self._item.setPixmap(QPixmap.fromImage(array_to_qimage(rgb_u8)))
        self.compare_divider.set_bounds(self._item.boundingRect())
        self.brush.set_bounds(self._item.boundingRect())
        # Margen alrededor para poder agarrar los tiradores del recorte en el borde.
        self._scene.setSceneRect(self._item.boundingRect().adjusted(-40, -40, 40, 40))
        if reset_view or self._fit_mode:
            self.fit()

    def clear(self) -> None:
        self._item.setPixmap(QPixmap())

    def fit(self) -> None:
        self._fit_mode = True
        if self.has_image():
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_100(self) -> None:
        self._fit_mode = False
        self.resetTransform()

    def wheelEvent(self, event):
        if not self.has_image():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        current = self.transform().m11()
        factor = max(MIN_ZOOM / current, min(MAX_ZOOM / current, factor))
        self._fit_mode = False
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if not self.crop_overlay.isVisible() and not self.compare_divider.isVisible():
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()
