"""Visor de imagen con zoom (rueda del ratón) y desplazamiento (arrastrar)."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

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
        self.setScene(self._scene)

        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._fit_mode = True

    def has_image(self) -> bool:
        return not self._item.pixmap().isNull()

    def set_image(self, rgb_u8: np.ndarray, reset_view: bool = False) -> None:
        """Muestra una imagen sRGB uint8. Conserva el zoom salvo reset_view."""
        self._item.setPixmap(QPixmap.fromImage(array_to_qimage(rgb_u8)))
        self._scene.setSceneRect(self._item.boundingRect())
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
        self.fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit()
