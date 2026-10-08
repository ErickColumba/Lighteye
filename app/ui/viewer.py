"""Visor de imagen con zoom (rueda del ratón) y desplazamiento (arrastrar)."""

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from app.ui.crop_overlay import CropOverlay

MIN_ZOOM = 0.05
MAX_ZOOM = 16.0


def array_to_qimage(rgb_u8: np.ndarray) -> QImage:
    h, w = rgb_u8.shape[:2]
    rgb_u8 = np.ascontiguousarray(rgb_u8)
    # copy() para que QImage no dependa de la memoria del array de NumPy.
    return QImage(rgb_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


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

    def set_label(self, text: str) -> None:
        """Texto fijo en la esquina del visor (p. ej. "Antes")."""
        self._label = text
        self.viewport().update()

    def drawForeground(self, painter: QPainter, rect) -> None:
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
        if not self.crop_overlay.isVisible():
            self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()
