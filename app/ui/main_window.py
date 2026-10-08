"""Ventana principal de Lighteye."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from app.core.color import to_display_u8
from app.core.loader import (
    SUPPORTED_EXTENSIONS,
    LoadedImage,
    is_supported,
    load_image,
    make_preview,
)
from app.core.settings import Settings
from app.ui.panels import AdjustmentPanel
from app.ui.renderer import PreviewRenderer
from app.ui.viewer import ImageViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lighteye")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.loaded: LoadedImage | None = None
        self.preview = None
        self.settings = Settings()

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        self.panel = AdjustmentPanel()
        self.panel.setEnabled(False)
        self.panel.changed.connect(self._on_param_changed)
        dock = QDockWidget("Ajustes", self)
        dock.setObjectName("ajustes")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(self.panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.renderer = PreviewRenderer(self)
        self.renderer.rendered.connect(self.viewer.set_image)
        self.renderer.failed.connect(lambda msg: self.statusBar().showMessage(f"Error: {msg}"))

        self._build_menus()
        self.statusBar().showMessage("Archivo → Abrir (Ctrl+O) o arrastra una foto aquí")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&Archivo")
        self._add_action(file_menu, "&Abrir…", QKeySequence.StandardKey.Open, self.choose_image)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Salir", QKeySequence.StandardKey.Quit, self.close)

        edit_menu = self.menuBar().addMenu("&Editar")
        self._add_action(edit_menu, "&Restablecer todos los ajustes", "Ctrl+R", self.reset_all)

        view_menu = self.menuBar().addMenu("&Ver")
        self._add_action(view_menu, "&Ajustar a la ventana", "Ctrl+0", self.viewer.fit)
        self._add_action(view_menu, "Tamaño &real (100 %)", "Ctrl+1", self.viewer.zoom_100)

    def _add_action(self, menu, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    # --- Abrir -----------------------------------------------------------

    def choose_image(self) -> None:
        patterns = " ".join(f"*{e} *{e.upper()}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir foto", "", f"Imágenes ({patterns});;Todos los archivos (*)"
        )
        if path:
            self.open_image(path)

    def open_image(self, path: str) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            loaded = load_image(path)
        except Exception as exc:  # noqa: BLE001 — cualquier fallo se muestra al usuario
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Lighteye", f"No se pudo abrir la foto:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        self.renderer.cancel()
        self.loaded = loaded
        self.preview = make_preview(loaded.image)
        self.settings = Settings()
        self.panel.set_settings(self.settings)
        self.panel.setEnabled(True)
        self.viewer.set_image(to_display_u8(self.preview), reset_view=True)

        h, w = loaded.image.shape[:2]
        kind = "RAW" if loaded.is_raw else f"{loaded.info.get('bits', 8)} bits"
        self.setWindowTitle(f"{loaded.path.name} — Lighteye")
        self.statusBar().showMessage(f"{loaded.path.name}  ·  {w} × {h}  ·  {kind}")

    # --- Ajustes ----------------------------------------------------------

    def _on_param_changed(self, key: str, value: float) -> None:
        self.settings[key] = value
        self._request_render()

    def reset_all(self) -> None:
        self.settings = Settings()
        self.panel.set_settings(self.settings)
        self._request_render()

    def _request_render(self) -> None:
        # La interfaz nunca procesa la imagen: solo pide un nuevo cálculo.
        if self.preview is not None:
            self.renderer.request(self.preview, self.settings)

    def closeEvent(self, event):
        self.renderer.shutdown()
        super().closeEvent(event)

    # --- Arrastrar y soltar ------------------------------------------------

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and is_supported(urls[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.open_image(str(Path(event.mimeData().urls()[0].toLocalFile())))
