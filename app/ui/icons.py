"""Iconos de la interfaz, dibujados en SVG (líneas, estilo uniforme).

Se colorean con el color de texto del tema, así se ven bien en modo claro y
oscuro, y no dependen de los iconos instalados en el sistema.
"""

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

# Contenido de cada icono dentro de un viewBox de 24×24.
_SHAPES = {
    "open": '<path d="M4 20h16a1 1 0 0 0 1-1V8a1 1 0 0 0-1-1h-8l-2-3H4a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1z"/>'
            '<path d="M12 11v6M9 14l3-3 3 3"/>',
    "folder": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/>'
              '<path d="M21 16l-5-5-9 9"/>',
    "prev": '<path d="M15 18l-6-6 6-6"/>',
    "next": '<path d="M9 18l6-6-6-6"/>',
    "export": '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M4 14v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"/>',
    "export_many": '<rect x="3" y="7" width="14" height="14" rx="2"/><path d="M7 3h12a2 2 0 0 1 2 2v12"/>'
                   '<path d="M10 17v-6M7.5 13.5L10 11l2.5 2.5"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    "redo": '<path d="M15 14l5-5-5-5"/><path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>',
    "paste": '<rect x="5" y="4" width="14" height="17" rx="2"/><rect x="9" y="2" width="6" height="4" rx="1"/>'
             '<path d="M9 12h6M9 16h4"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "presets": '<path d="M12 2l2.9 6.3 6.6.6-5 4.5 1.5 6.6L12 16.6 6 20l1.5-6.6-5-4.5 6.6-.6z"/>',
    "filmstrip": '<rect x="2" y="5" width="20" height="14" rx="2"/><path d="M7 5v14M17 5v14M2 9h5M2 15h5M17 9h5M17 15h5"/>',
    "fit": '<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3"/>',
    "zoom100": '<circle cx="10.5" cy="10.5" r="7"/><path d="M21 21l-5.5-5.5M8.5 8.5l2-1.5v7"/>',
    "before": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/>'
              '<path d="M3 6a2 2 0 0 1 2-2h7v16H5a2 2 0 0 1-2-2z" fill="currentColor" stroke="none"/>',
    "after": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/>'
             '<path d="M12 4h7a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-7z" fill="currentColor" stroke="none"/>',
    "compare": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 2v20"/>'
               '<path d="M8 10l-2 2 2 2M16 10l2 2-2 2"/>',
    "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "edited": '<path d="M4 7h10M18 7h2M4 17h2M10 17h10"/><circle cx="16" cy="7" r="2"/>'
              '<circle cx="8" cy="17" r="2"/>',
}

_TEMPLATE = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
             'stroke-linejoin="round" color="{color}">{body}</svg>')


def _render(name: str, color: str, size: int) -> QPixmap:
    svg = _TEMPLATE.format(color=color, body=_SHAPES[name]).replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pix


@lru_cache(maxsize=None)
def _icon(name: str, color: str) -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48):
        icon.addPixmap(_render(name, color, size))
    return icon


def pixmap(name: str, color: str, size: int) -> QPixmap:
    """El icono como imagen, en un color fijo (p. ej. blanco sobre una insignia)."""
    return _render(name, color, size)


def icon(name: str) -> QIcon:
    """Icono `name` en el color de texto del tema actual."""
    app = QApplication.instance()
    color = app.palette().color(QPalette.ColorRole.WindowText).name() if app else "#333333"
    return _icon(name, color)
