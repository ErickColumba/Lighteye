"""Ventana principal de Lighteye."""

from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from app.core.color import to_display_u8
from app.core.geometry import (
    FULL_CROP,
    apply_geometry,
    geometry_preview,
    geometry_signature,
    is_identity,
    final_size,
)
from app.core.histogram import clipping_overlay, compute_histogram
from app.core.history import History
from app.core.loader import (
    SUPPORTED_EXTENSIONS,
    PREVIEW_LONG_SIDE,
    LoadedImage,
    is_supported,
    load_image,
    make_preview,
)
from app.core.settings import Settings
from app.ui.crop_tools import CropToolbar
from app.ui.histogram import HistogramWidget
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
        self.base_preview = None  # vista previa de la foto sin geometría
        self.preview = None  # imagen de origen que procesa el pipeline
        self._source_sig = None  # geometría con la que se calculó self.preview
        self.crop_mode = False
        self._crop_backup: Settings | None = None
        self.settings = Settings()
        self.shown_rgb = None  # última imagen calculada (sRGB uint8)
        self.before_rgb = None  # la foto sin ajustes, para comparar
        self.show_before = False
        self.history = History(self.settings)
        # Mover un slider genera decenas de cambios: se guardan en el historial
        # como un solo paso cuando el usuario se detiene un momento.
        self._history_timer = QTimer(self, singleShot=True, interval=400)
        self._history_timer.timeout.connect(self._commit_history)

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)

        self.panel = AdjustmentPanel()
        self.panel.setEnabled(False)
        self.panel.changed.connect(self._on_param_changed)
        self.histogram = HistogramWidget()
        self.histogram.clipping_toggled.connect(lambda _: self._refresh_view())
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.histogram)
        side_layout.addWidget(self.panel, 1)
        dock = QDockWidget("Ajustes", self)
        dock.setObjectName("ajustes")
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(side)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.crop_tools = CropToolbar(self)
        self.crop_tools.setVisible(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.crop_tools)
        self.crop_tools.aspect_changed.connect(self._crop_aspect_changed)
        self.crop_tools.angle_changed.connect(lambda a: self._crop_geometry_changed(angle=a))
        self.crop_tools.rotate_requested.connect(
            lambda d: self._crop_geometry_changed(rotate=(self.settings["rotate"] + d) % 4))
        self.crop_tools.flip_requested.connect(
            lambda axis: self._crop_geometry_changed(**{f"flip_{axis}": 1 - self.settings[f"flip_{axis}"]}))
        self.crop_tools.reset_requested.connect(self._crop_reset)
        self.crop_tools.cancel_requested.connect(self.cancel_crop)
        self.crop_tools.apply_requested.connect(self.apply_crop)
        self.viewer.crop_overlay.rect_changed.connect(self._crop_rect_changed)

        self.renderer = PreviewRenderer(self)
        self.renderer.rendered.connect(self._on_rendered)
        self.renderer.failed.connect(lambda msg: self.statusBar().showMessage(f"Error: {msg}"))

        self._build_menus()
        self.statusBar().showMessage("Archivo → Abrir (Ctrl+O) o arrastra una foto aquí")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&Archivo")
        self._add_action(file_menu, "&Abrir…", QKeySequence.StandardKey.Open, self.choose_image)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Salir", QKeySequence.StandardKey.Quit, self.close)

        edit_menu = self.menuBar().addMenu("&Editar")
        self.undo_action = self._add_action(edit_menu, "&Deshacer", QKeySequence.StandardKey.Undo, self.undo)
        self.redo_action = self._add_action(edit_menu, "&Rehacer", "Ctrl+Shift+Z", self.redo)
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        edit_menu.addSeparator()
        self._add_action(edit_menu, "R&establecer todos los ajustes", "Ctrl+R", self.reset_all)
        edit_menu.addSeparator()
        self.crop_action = self._add_action(edit_menu, "Re&cortar y enderezar", "C", self.toggle_crop)
        self.crop_action.setCheckable(True)
        # Intro / Esc del modo recorte: activas solo mientras se recorta.
        self.crop_apply_action = self._add_action(self, "Aplicar recorte", "Return", self.apply_crop)
        self.crop_apply_action.setShortcuts([QKeySequence("Return"), QKeySequence("Enter")])
        self.crop_cancel_action = self._add_action(self, "Cancelar recorte", "Esc", self.cancel_crop)
        for a in (self.crop_apply_action, self.crop_cancel_action):
            a.setEnabled(False)
        self._update_history_actions()

        view_menu = self.menuBar().addMenu("&Ver")
        self.before_action = self._add_action(view_menu, "Antes / &Después", "\\", self.toggle_before)
        self.before_action.setCheckable(True)
        view_menu.addSeparator()
        self._add_action(view_menu, "&Ajustar a la ventana", "Ctrl+0", self.viewer.fit)
        self._add_action(view_menu, "Tamaño &real (100 %)", "Ctrl+1", self.viewer.zoom_100)

    def _add_action(self, menu, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)  # menu puede ser la ventana: atajo sin entrada de menú
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

        if self.crop_mode:
            self._leave_crop_mode()
        self.renderer.cancel()
        self.loaded = loaded
        self.base_preview = make_preview(loaded.image)
        self.preview = self.base_preview
        self.settings = Settings()
        self._source_sig = geometry_signature(self.settings)
        self._history_timer.stop()
        self.history = History(self.settings)
        self._update_history_actions()
        self.panel.set_settings(self.settings)
        self.panel.setEnabled(True)
        rgb = to_display_u8(self.base_preview)
        self.before_rgb = rgb
        self._set_before(False)
        self._on_rendered(rgb, compute_histogram(rgb))
        self.viewer.fit()

        h, w = loaded.image.shape[:2]
        kind = "RAW" if loaded.is_raw else f"{loaded.info.get('bits', 8)} bits"
        self.setWindowTitle(f"{loaded.path.name} — Lighteye")
        self.statusBar().showMessage(f"{loaded.path.name}  ·  {w} × {h}  ·  {kind}")

    # --- Ajustes ----------------------------------------------------------

    def _on_param_changed(self, key: str, value) -> None:
        self.settings[key] = value
        self._set_before(False)  # al editar se vuelve a ver el resultado
        self._history_timer.start()
        self._update_history_actions()
        self._request_render()

    def reset_all(self) -> None:
        self._apply_settings(Settings())
        self._commit_history()

    def _apply_settings(self, settings: Settings) -> None:
        """Sustituye todos los ajustes (deshacer, restablecer, …)."""
        self.settings = settings.copy()
        self.panel.set_settings(self.settings)
        self._set_before(False)
        self._request_render()

    # --- Historial ----------------------------------------------------------

    def _commit_history(self) -> None:
        self._history_timer.stop()
        if self.preview is not None:
            self.history.push(self.settings)
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(self.history.can_undo() or self._history_timer.isActive())
        self.redo_action.setEnabled(self.history.can_redo())

    def undo(self) -> None:
        if self.crop_mode:
            self.cancel_crop()
            return
        self._commit_history()  # primero se guarda lo que aún estaba pendiente
        self._step_history(self.history.undo(), "Deshecho")

    def redo(self) -> None:
        if self.crop_mode:
            return
        self._commit_history()
        self._step_history(self.history.redo(), "Rehecho")

    def _step_history(self, result, verb: str) -> None:
        if result is None:
            return
        settings, what = result
        self._apply_settings(settings)
        self._update_history_actions()
        self.statusBar().showMessage(f"{verb}: {what}", 3000)

    def _request_render(self) -> None:
        # La interfaz nunca procesa la imagen: solo pide un nuevo cálculo.
        if self.preview is not None:
            self._sync_source()
            self.renderer.request(self.preview, self.settings)

    def _sync_source(self) -> None:
        """Recalcula la imagen de origen si cambió la geometría.

        En modo recorte se usa la vista previa girada pero sin recortar (rápido).
        Con un recorte aplicado se parte de la resolución completa, para que
        un recorte pequeño no se vea borroso.
        """
        sig = geometry_signature(self.settings, with_crop=not self.crop_mode)
        if sig == self._source_sig:
            return
        if self.crop_mode:
            self.preview = apply_geometry(self.base_preview, self.settings, with_crop=False)
        elif is_identity(self.settings):
            self.preview = self.base_preview
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.preview = geometry_preview(self.loaded.image, self.settings, PREVIEW_LONG_SIDE)
            finally:
                QApplication.restoreOverrideCursor()
        self._source_sig = sig

    # --- Recorte y enderezado -------------------------------------------------

    def toggle_crop(self) -> None:
        if self.crop_mode:
            self.apply_crop()
        else:
            self.enter_crop()

    def enter_crop(self) -> None:
        if self.preview is None or self.crop_mode:
            self.crop_action.setChecked(self.crop_mode)
            return
        self._commit_history()
        self._crop_backup = self.settings.copy()
        self.crop_mode = True
        self._set_before(False)
        self.panel.setEnabled(False)
        self.crop_action.setChecked(True)
        for a in (self.crop_apply_action, self.crop_cancel_action):
            a.setEnabled(True)
        self.crop_tools.set_angle(self.settings["angle"])
        self.crop_tools.setVisible(True)
        self._request_render()
        self._update_overlay()
        self.viewer.crop_overlay.setVisible(True)
        self.viewer.fit()

    def apply_crop(self) -> None:
        if not self.crop_mode:
            return
        self._leave_crop_mode()
        self._request_render()
        self.viewer.fit()
        self._commit_history()

    def cancel_crop(self) -> None:
        if not self.crop_mode:
            return
        self._leave_crop_mode()
        self._apply_settings(self._crop_backup)
        self.viewer.fit()

    def _leave_crop_mode(self) -> None:
        self.crop_mode = False
        self.viewer.crop_overlay.setVisible(False)
        self.crop_tools.setVisible(False)
        self.crop_action.setChecked(False)
        for a in (self.crop_apply_action, self.crop_cancel_action):
            a.setEnabled(False)
        self.panel.setEnabled(True)

    def _update_overlay(self) -> None:
        """Coloca el marco según el recorte guardado y la imagen mostrada."""
        h, w = self.preview.shape[:2]
        x, y, cw, ch = self.settings["crop"]
        overlay = self.viewer.crop_overlay
        overlay.aspect = self.crop_tools.ratio(w, h)
        overlay.set_bounds(QRectF(0, 0, w, h), QRectF(x * w, y * h, cw * w, ch * h))
        self.settings["crop"] = overlay.normalized()
        self._show_crop_size()

    def _crop_rect_changed(self, _rect) -> None:
        if self.crop_mode:
            self.settings["crop"] = self.viewer.crop_overlay.normalized()
            self._show_crop_size()

    def _crop_aspect_changed(self) -> None:
        h, w = self.preview.shape[:2]
        self.viewer.crop_overlay.set_aspect(self.crop_tools.ratio(w, h))

    def _crop_geometry_changed(self, **changes) -> None:
        # Girar o voltear invalida el marco: se vuelve a la foto completa.
        if "rotate" in changes or any(k.startswith("flip") for k in changes):
            self.settings["crop"] = FULL_CROP
        for key, value in changes.items():
            self.settings[key] = value
        self._request_render()
        self._update_overlay()

    def _crop_reset(self) -> None:
        for key in ("rotate", "flip_h", "flip_v", "angle"):
            self.settings.reset(key)
        self.settings["crop"] = FULL_CROP
        self.crop_tools.aspect.setCurrentIndex(0)
        self.crop_tools.portrait.setChecked(False)
        self.crop_tools.set_angle(0)
        self._request_render()
        self._update_overlay()

    def _show_crop_size(self) -> None:
        """Tamaño final del recorte en píxeles de la foto original."""
        h, w = self.loaded.image.shape[:2]
        fw, fh = final_size(w, h, self.settings)
        self.statusBar().showMessage(f"Recorte: {fw} × {fh} px")

    def _on_rendered(self, rgb, histogram) -> None:
        self.shown_rgb = rgb
        self.histogram.set_histogram(histogram)
        self._refresh_view()

    def toggle_before(self) -> None:
        self._set_before(not self.show_before)

    def _set_before(self, on: bool) -> None:
        on = on and self.before_rgb is not None
        if on == self.show_before:
            return
        self.show_before = on
        self.before_action.setChecked(on)
        self.viewer.set_label("Antes" if on else "")
        self._refresh_view()

    def _refresh_view(self) -> None:
        """Muestra la última imagen calculada (o la original en modo "Antes"),
        con o sin el aviso de recorte."""
        if self.shown_rgb is None:
            return
        rgb = self.before_rgb if self.show_before else self.shown_rgb
        if self.histogram.show_clipping:
            rgb = clipping_overlay(rgb)
        self.viewer.set_image(rgb)

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
