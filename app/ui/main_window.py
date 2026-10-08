"""Ventana principal de Lighteye."""

from pathlib import Path

import numpy as np

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QSizePolicy,
    QToolBar,
    QToolButton,
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
from app.core.presets import apply_look, delete_preset, list_presets, look_values, save_preset
from app.core.settings import Settings, load_sidecar, save_sidecar, sidecar_path
from app.ui.browser import FilmStrip
from app.ui.crop_tools import CropToolbar
from app.ui.export_dialog import (
    BatchExportWorker,
    ExportDialog,
    ExportWorker,
    ask_output_folder,
    ask_output_path,
)
from app.ui.histogram import HistogramWidget
from app.ui.icons import icon
from app.ui.view_bar import ViewModeBar
from app.ui.panels import AdjustmentPanel
from app.ui.presets_panel import PresetPanel
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
        self._geo_preview = None  # vista previa con la geometría, sin rostros
        self._faces: dict[str, list] = {}  # rostros restaurados por clave de caché
        self._parses: dict = {}  # análisis facial (máscaras) por foto
        self._patches: dict[str, list] = {}  # zonas borradas (LaMa) por clave
        self._face_worker = None
        self._parse_worker = None
        self._erase_worker = None
        self.erase_mode = False
        self.crop_mode = False
        self._crop_backup: Settings | None = None
        self._export_worker: ExportWorker | None = None
        self.copied_look: dict | None = None  # ajustes copiados (sin geometría)
        self.settings = Settings()
        self.shown_rgb = None  # última imagen calculada (sRGB uint8)
        # Modo de vista: "after" (resultado), "before" (original) o "compare".
        self.view_mode = "after"
        self._before_rgb = None  # la foto de origen sin ajustes (para comparar)
        self._before_for = None  # imagen de origen con la que se calculó
        self.history = History(self.settings)
        # Mover un slider genera decenas de cambios: se guardan en el historial
        # como un solo paso cuando el usuario se detiene un momento.
        self._history_timer = QTimer(self, singleShot=True, interval=400)
        self._history_timer.timeout.connect(self._commit_history)
        # Mientras se arrastra se muestra un borrador; al parar, la versión completa.
        self._settle_timer = QTimer(self, singleShot=True, interval=150)
        self._settle_timer.timeout.connect(self._request_render)

        self.viewer = ImageViewer(self)
        # Redibujar al momento desde el evento del ratón provoca otro evento
        # (bucle); se agrupa en un único redibujado en la siguiente vuelta.
        self._view_refresh = QTimer(self, singleShot=True, interval=0)
        self._view_refresh.timeout.connect(self._refresh_view)
        self.viewer.compare_divider.moved.connect(lambda _: self._view_refresh.start())
        self.view_bar = ViewModeBar()
        self.view_bar.mode_changed.connect(self.set_view_mode)
        self.view_bar.setEnabled(False)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.viewer, 1)
        center_layout.addWidget(self.view_bar)
        self.setCentralWidget(center)
        self.viewer.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.viewer.customContextMenuRequested.connect(self._viewer_menu)

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

        self.presets = PresetPanel()
        self.presets.setEnabled(False)
        self.presets.apply_requested.connect(self.apply_preset)
        self.presets.save_requested.connect(self.save_preset)
        self.presets.delete_requested.connect(self.delete_preset)
        presets_dock = QDockWidget("Presets", self)
        presets_dock.setObjectName("presets")
        presets_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                                 | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        presets_dock.setWidget(self.presets)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, presets_dock)
        self.presets_dock = presets_dock

        self.filmstrip = FilmStrip()
        self.filmstrip.open_requested.connect(self._open_from_strip)
        self.filmstrip.batch_requested.connect(self._batch_action)
        self.filmstrip.presets_menu_provider = lambda: [(p.name, p) for p in list_presets()]
        self.filmstrip.can_paste = lambda: self.copied_look is not None
        strip_dock = QDockWidget("Carpeta", self)
        strip_dock.setObjectName("carpeta")
        strip_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                               | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        strip_dock.setWidget(self.filmstrip)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, strip_dock)
        self.strip_dock = strip_dock

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

        from app.ui.brush import EraseToolbar

        self.erase_tools = EraseToolbar(self)
        self.erase_tools.setVisible(False)
        self.erase_tools.size_changed.connect(self._brush_size_changed)
        self.erase_tools.clear_requested.connect(self._clear_erase)
        self.erase_tools.done_requested.connect(self.leave_erase)
        self.viewer.brush.stroke_finished.connect(self._erase_stroke)

        self.renderer = PreviewRenderer(self)
        self.renderer.rendered.connect(self._on_rendered)
        self.renderer.failed.connect(lambda msg: self.statusBar().showMessage(f"Error: {msg}"))

        self._build_menus()
        self.statusBar().showMessage("Abre una foto (Ctrl+O) o arrástrala aquí")

    def _build_menus(self) -> None:
        """Barra superior de iconos (sin menú de texto). Todas las acciones se
        añaden también a la ventana, así sus atajos funcionan siempre."""
        A = self._make_action
        self.open_action = A("open", "Abrir foto", QKeySequence.StandardKey.Open, self.choose_image)
        self.folder_action = A("folder", "Abrir carpeta", "Ctrl+Shift+O", self.choose_folder)
        self.prev_action = A("prev", "Foto anterior", "Ctrl+Left", lambda: self._step_photo(-1))
        self.next_action = A("next", "Foto siguiente", "Ctrl+Right", lambda: self._step_photo(1))
        self.export_action = A("export", "Exportar", "Ctrl+E", self.export)
        self.export_many_action = A("export_many", "Exportar seleccionadas", "Ctrl+Shift+E",
                                    lambda: self.export_batch(self.filmstrip.selected_paths()))
        self.undo_action = A("undo", "Deshacer", QKeySequence.StandardKey.Undo, self.undo)
        self.redo_action = A("redo", "Rehacer", "Ctrl+Shift+Z", self.redo)
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.copy_action = A("copy", "Copiar ajustes", "Ctrl+Shift+C", self.copy_look)
        self.paste_action = A("paste", "Pegar ajustes", "Ctrl+Shift+V", self.paste_look)
        self.reset_action = A("reset", "Restablecer todos los ajustes", "Ctrl+R", self.reset_all)
        self.crop_action = A("crop", "Recortar y enderezar", "C", self.toggle_crop)
        self.crop_action.setCheckable(True)
        self.erase_action = A("eraser", "Borrar objetos (IA)", "B", self.toggle_erase)
        self.erase_action.setCheckable(True)
        self.fit_action = A("fit", "Ajustar a la ventana", "Ctrl+0", self.viewer.fit)
        self.zoom_action = A("zoom100", "Tamaño real (100 %)", "Ctrl+1", self.viewer.zoom_100)
        self.before_action = A(None, "Alternar antes / después", "\\", self.toggle_before)
        self.quit_action = A(None, "Salir", QKeySequence.StandardKey.Quit, self.close)
        for name, dock, text in (("presets", self.presets_dock, "Panel de presets"),
                                 ("filmstrip", self.strip_dock, "Tira de la carpeta")):
            toggle = dock.toggleViewAction()
            toggle.setIcon(icon(name))
            toggle.setText(text)
            toggle.setToolTip(text)
            setattr(self, f"{name}_toggle", toggle)
        # Intro / Esc del modo recorte: activas solo mientras se recorta.
        self.crop_apply_action = A(None, "Aplicar recorte", "Return", self.apply_crop)
        self.crop_apply_action.setShortcuts([QKeySequence("Return"), QKeySequence("Enter")])
        self.crop_cancel_action = A(None, "Cancelar recorte", "Esc", self.cancel_crop)
        self.erase_done_action = A(None, "Terminar de borrar", "Return", self.leave_erase)
        self.erase_done_action.setShortcuts([QKeySequence("Return"), QKeySequence("Enter"),
                                             QKeySequence("Esc")])
        for a in (self.crop_apply_action, self.crop_cancel_action, self.export_action,
                  self.copy_action, self.paste_action, self.reset_action, self.crop_action,
                  self.erase_action, self.erase_done_action):
            a.setEnabled(False)

        bar = QToolBar("Principal", self)
        bar.setObjectName("principal")
        bar.setMovable(False)
        bar.setIconSize(QSize(22, 22))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        groups = [
            [self.open_action, self.folder_action],
            [self.prev_action, self.next_action],
            [self.export_action, self.export_many_action],
            [self.undo_action, self.redo_action],
            [self.copy_action, self.paste_action, self.reset_action],
            [self.crop_action, self.erase_action],
        ]
        for i, group in enumerate(groups):
            if i:
                bar.addSeparator()
            for a in group:
                bar.addAction(a)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        for a in (self.fit_action, self.zoom_action):
            bar.addAction(a)
        bar.addSeparator()
        bar.addAction(self.presets_toggle)
        bar.addAction(self.filmstrip_toggle)
        bar.addSeparator()

        # Menú "≡" con todo (con texto), por si no se recuerda un icono.
        menu = QMenu(self)
        for group in groups + [[self.fit_action, self.zoom_action, self.before_action],
                               [self.presets_toggle, self.filmstrip_toggle], [self.quit_action]]:
            for a in group:
                menu.addAction(a)
            menu.addSeparator()
        more = QToolButton()
        more.setIcon(icon("menu"))
        more.setToolTip("Todas las opciones")
        more.setMenu(menu)
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bar.addWidget(more)

        self.menuBar().hide()
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, bar)
        # La barra de recorte aparece en una fila propia, debajo.
        self.insertToolBar(self.crop_tools, bar)
        self.removeToolBar(self.crop_tools)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.crop_tools)
        self.crop_tools.setVisible(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.erase_tools)
        self.erase_tools.setVisible(False)
        self._update_history_actions()

    def _make_action(self, icon_name, text, shortcut, slot) -> QAction:
        action = QAction(text, self)
        if icon_name:
            action.setIcon(icon(icon_name))
        action.setShortcut(shortcut)
        keys = action.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        action.setToolTip(f"{text}  ({keys})" if keys else text)
        action.triggered.connect(slot)
        self.addAction(action)  # atajo activo aunque no haya menú visible
        return action

    # --- Abrir -----------------------------------------------------------

    def choose_image(self) -> None:
        patterns = " ".join(f"*{e} *{e.upper()}" for e in sorted(SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir foto", "", f"Imágenes ({patterns});;Todos los archivos (*)"
        )
        if path:
            self.open_image(path)

    def choose_folder(self) -> None:
        start = str(self.loaded.path.parent) if self.loaded else ""
        folder = QFileDialog.getExistingDirectory(self, "Abrir carpeta", start)
        if not folder:
            return
        self.filmstrip.set_folder(Path(folder))
        paths = self.filmstrip.paths()
        if paths:
            self.open_image(paths[0])
        else:
            self.statusBar().showMessage("La carpeta no tiene fotos compatibles", 4000)

    def _open_from_strip(self, path: str) -> None:
        if self.loaded is None or Path(path) != self.loaded.path:
            self.open_image(path)

    def _step_photo(self, step: int) -> None:
        if self.loaded is not None:
            target = self.filmstrip.neighbour(str(self.loaded.path), step)
            if target:
                self.open_image(target)

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
            self.cancel_crop()
        if self.erase_mode:
            self.leave_erase()
        self._commit_history()  # guarda la edición de la foto anterior
        self.renderer.cancel()
        if self.filmstrip.folder != loaded.path.parent:
            self.filmstrip.set_folder(loaded.path.parent)
        self.filmstrip.mark_current(str(loaded.path))
        self.loaded = loaded
        self.base_preview = make_preview(loaded.image)
        self.preview = self._geo_preview = self.base_preview
        self._faces.clear()
        self._parses.clear()
        self._patches.clear()
        self._source_sig = geometry_signature(Settings())
        # Si la foto ya se había editado, se recuperan sus ajustes.
        self.settings = load_sidecar(loaded.path) or Settings()
        self._history_timer.stop()
        self.history = History(self.settings)
        self._update_history_actions()
        self.panel.set_settings(self.settings)
        self.panel.setEnabled(True)
        self.presets.setEnabled(True)
        self.export_action.setEnabled(True)
        self.copy_action.setEnabled(True)
        self.paste_action.setEnabled(self.copied_look is not None)
        self.reset_action.setEnabled(True)
        self.crop_action.setEnabled(True)
        from app.ai import runtime

        lama = runtime.model_available("lama")
        self.erase_action.setEnabled(lama)
        if not lama:
            self.erase_action.setToolTip("Borrar objetos: no disponible (faltan PyTorch o el modelo LaMa)")
        self.view_bar.setEnabled(True)
        rgb = to_display_u8(self.base_preview)
        self._before_rgb, self._before_for = rgb, self.base_preview
        self.set_view_mode("after")
        self._on_rendered(rgb, compute_histogram(rgb))
        self._request_render()
        self._update_preset_thumbs()
        self.viewer.fit()

        h, w = loaded.image.shape[:2]
        kind = "RAW" if loaded.is_raw else f"{loaded.info.get('bits', 8)} bits"
        self.setWindowTitle(f"{loaded.path.name} — Lighteye")
        edited = "  ·  ajustes recuperados" if self.settings != Settings() else ""
        self.statusBar().showMessage(f"{loaded.path.name}  ·  {w} × {h}  ·  {kind}{edited}")

    # --- Ajustes ----------------------------------------------------------

    def _on_param_changed(self, key: str, value) -> None:
        self.settings[key] = value
        self._leave_before()  # al editar se vuelve a ver el resultado
        self._history_timer.start()
        self._update_history_actions()
        # Botón del ratón pulsado = arrastrando un slider o un punto de la curva.
        dragging = bool(QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        if dragging:
            self._settle_timer.start()
        self._request_render(draft=dragging)

    def reset_all(self) -> None:
        self._apply_settings(Settings())
        self._commit_history()

    def _apply_settings(self, settings: Settings) -> None:
        """Sustituye todos los ajustes (deshacer, restablecer, …)."""
        self.settings = settings.copy()
        self.panel.set_settings(self.settings)
        self._leave_before()
        self._request_render()

    # --- Historial ----------------------------------------------------------

    def _commit_history(self) -> None:
        self._history_timer.stop()
        if self.preview is not None and self.history.push(self.settings):
            self._save_sidecar()
            self._update_preset_thumbs()
        self._update_history_actions()

    def _save_sidecar(self) -> None:
        """Guarda los ajustes junto a la foto (foto.jpg.json). El original no se toca."""
        if self.loaded is None:
            return
        # Una foto sin editar no necesita archivo, salvo que ya existiera uno.
        if self.settings == Settings() and not sidecar_path(self.loaded.path).exists():
            return
        try:
            save_sidecar(self.loaded.path, self.settings)
        except OSError as exc:
            self.statusBar().showMessage(f"No se pudieron guardar los ajustes: {exc}", 5000)
            return
        self.filmstrip.refresh([str(self.loaded.path)])

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
        self._save_sidecar()
        self._update_history_actions()
        self.statusBar().showMessage(f"{verb}: {what}", 3000)

    def _request_render(self, draft: bool = False) -> None:
        # La interfaz nunca procesa la imagen: solo pide un nuevo cálculo.
        if self.preview is not None:
            if not draft:
                self._settle_timer.stop()
            self._sync_source()
            self.renderer.request(self.preview, self.settings, draft=draft, prep=self._source_prep())

    def _sync_source(self) -> None:
        """Recalcula la imagen de origen si cambió la geometría.

        En modo recorte se usa la vista previa girada pero sin recortar (rápido).
        Con un recorte aplicado se parte de la resolución completa, para que
        un recorte pequeño no se vea borroso.
        """
        sig = geometry_signature(self.settings, with_crop=not self.crop_mode)
        if sig != self._source_sig:
            if self.crop_mode:
                self._geo_preview = apply_geometry(self.base_preview, self.settings, with_crop=False)
            elif is_identity(self.settings):
                self._geo_preview = self.base_preview
            else:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self._geo_preview = geometry_preview(self.loaded.image, self.settings,
                                                         PREVIEW_LONG_SIDE)
                finally:
                    QApplication.restoreOverrideCursor()
            self._source_sig = sig
            self.preview = self._geo_preview

    # --- Rostros (IA) -------------------------------------------------------------

    def _face_key(self) -> str | None:
        if self.loaded is None or self.settings["face_restore"] <= 0:
            return None
        from app.ai.faces import cache_key

        return cache_key(self.loaded.path, bool(self.settings["face_codeformer"]),
                         self.settings["face_fidelity"] / 100)

    def _source_prep(self):
        """Trabajo de IA sobre el origen (pegar rostros, retoque), para que el
        renderizador lo haga en segundo plano: (clave, función) o None."""
        from app.core.settings import RETOUCH_KEYS

        face_key = self._face_key()
        faces = self._faces.get(face_key) if face_key else None
        if face_key and faces is None:
            self._request_faces(face_key)
        retouch = {k: self.settings[k] for k in RETOUCH_KEYS if not self.settings.is_default(k)}
        parses = None
        if retouch and self.loaded is not None:
            parses = self._parses.get(self.loaded.path)
            if parses is None:
                self._request_parses()
        strokes = self.settings["erase_strokes"]
        patches, erase_key = None, None
        if strokes and self.loaded is not None:
            from app.ai.inpaint import cache_key as erase_cache_key

            erase_key = erase_cache_key(self.loaded.path, list(strokes))
            patches = self._patches.get(erase_key)
            if patches is None:
                self._request_erase(erase_key)
        strength = self.settings["face_restore"] / 100
        if not faces and not parses and not patches:
            return None

        from app.ai.faces import paste_faces
        from app.ai.inpaint import paste_patches
        from app.ai.retouch import apply_retouch

        transform = self._preview_transform()
        snapshot = self.settings.copy()

        def prepare(image, faces=faces, parses=parses, patches=patches):
            out = paste_patches(image, patches, transform) if patches else image
            out = paste_faces(out, faces, strength, transform) if faces else out
            return apply_retouch(out, parses, snapshot, transform) if parses else out

        key = (id(self.preview), face_key if faces else None, strength if faces else 0,
               tuple(sorted(retouch.items())) if parses else None, self.crop_mode,
               erase_key if patches else None)
        return key, prepare

    def _preview_transform(self) -> np.ndarray:
        """Matriz 3×3: foto original → imagen que se muestra (vista previa)."""
        from app.core.geometry import geometry_matrix

        h, w = self.loaded.image.shape[:2]
        geo = geometry_matrix(w, h, self.settings, with_crop=not self.crop_mode)
        shown = self.settings
        if self.crop_mode:  # en modo recorte se ve la foto entera (sin recortar)
            shown = self.settings.copy()
            shown["crop"] = FULL_CROP
        fw, fh = final_size(w, h, shown)
        ph, pw = self.preview.shape[:2]
        return np.diag([pw / fw, ph / fh, 1.0]) @ geo

    # --- Borrar objetos (LaMa) ------------------------------------------------------

    def toggle_erase(self) -> None:
        if self.erase_mode:
            self.leave_erase()
        else:
            self.enter_erase()

    def enter_erase(self) -> None:
        if self.preview is None or self.erase_mode:
            self.erase_action.setChecked(self.erase_mode)
            return
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()
        self.set_view_mode("after")
        self.erase_mode = True
        self.erase_action.setChecked(True)
        self.erase_done_action.setEnabled(True)
        self.erase_tools.setVisible(True)
        self.viewer.brush.setVisible(True)
        self.viewer.setDragMode(self.viewer.DragMode.NoDrag)
        self.statusBar().showMessage("Pinta sobre lo que quieras borrar; al soltar se rellena con IA")

    def leave_erase(self) -> None:
        if not self.erase_mode:
            return
        self.erase_mode = False
        self.erase_action.setChecked(False)
        self.erase_done_action.setEnabled(False)
        self.erase_tools.setVisible(False)
        self.viewer.brush.setVisible(False)
        self.viewer.setDragMode(self.viewer.DragMode.ScrollHandDrag)

    def _brush_size_changed(self, px: int) -> None:
        self.viewer.brush.radius_px = px
        self.viewer.brush.update()

    def _erase_stroke(self, points: list, radius: float) -> None:
        """Trazo terminado (en coordenadas de la vista previa) → ajustes."""
        if self.loaded is None:
            return
        inv = np.linalg.inv(self._preview_transform())
        h, w = self.loaded.image.shape[:2]
        orig = [inv @ np.array([x, y, 1.0]) for x, y in points]
        scale = np.sqrt(abs(np.linalg.det(inv[:2, :2])))
        stroke = {"r": radius * scale / max(w, h), "pts": [(p[0] / w, p[1] / h) for p in orig]}
        self.settings["erase_strokes"] = tuple(self.settings["erase_strokes"]) + (stroke,)
        self._commit_history()
        self._request_render()

    def _clear_erase(self) -> None:
        if self.settings["erase_strokes"]:
            self.settings["erase_strokes"] = ()
            self._commit_history()
            self._request_render()

    def _request_erase(self, key: str) -> None:
        from app.ai.inpaint import _cache_dir

        if self._erase_worker is not None:
            return  # al terminar el actual se vuelve a comprobar
        from app.ai import runtime
        from app.ui.ai_worker import EraseWorker

        if not runtime.model_available("lama"):
            return
        worker = EraseWorker(self.loaded.path, self.loaded.image, self.settings["erase_strokes"], key, self)
        worker.done.connect(self._erase_ready)
        worker.failed.connect(lambda msg: (self.statusBar().showMessage(f"Error al borrar: {msg}", 8000),
                                           self._clear_erase_worker()))
        self._erase_worker = worker
        cached = (_cache_dir() / f"{key}.npz").is_file()
        if not cached:
            self.statusBar().showMessage(f"Borrando con IA… ({runtime.device_name()})")
        worker.start()

    def _clear_erase_worker(self) -> None:
        if self._erase_worker is not None:
            self._erase_worker.deleteLater()
        self._erase_worker = None

    def _erase_ready(self, key: str, patches: list) -> None:
        self._patches[key] = patches
        self._clear_erase_worker()
        if self.erase_mode:
            self.statusBar().showMessage("Listo. Sigue pintando para borrar más (Ctrl+Z deshace)", 5000)
        self._request_render()

    def _request_faces(self, key: str) -> None:
        from app.ai.faces import load_cached

        cached = load_cached(key)
        if cached is not None:
            self._faces[key] = cached
            self._report_faces(cached)
            return
        if self._face_worker is not None:
            return  # ya hay uno calculando; al terminar se vuelve a comprobar
        from app.ai import runtime
        from app.ui.ai_worker import FaceWorker

        if not (runtime.model_available("yunet") and runtime.model_available(
                "codeformer" if self.settings["face_codeformer"] else "gfpgan")):
            self.statusBar().showMessage("Restaurar rostros no disponible: faltan PyTorch o los modelos "
                                         "(python tools/download_models.py)", 8000)
            return
        worker = FaceWorker(self.loaded.path, self.loaded.image, bool(self.settings["face_codeformer"]),
                            self.settings["face_fidelity"] / 100, key, self)
        worker.done.connect(self._faces_ready)
        worker.failed.connect(lambda msg: (self.statusBar().showMessage(f"Error al restaurar rostros: {msg}",
                                                                        8000), self._clear_face_worker()))
        self._face_worker = worker
        self.statusBar().showMessage(f"Restaurando rostros con IA… ({runtime.device_name()})")
        worker.start()

    def _clear_face_worker(self) -> None:
        if self._face_worker is not None:
            self._face_worker.deleteLater()
        self._face_worker = None

    def _faces_ready(self, key: str, faces: list) -> None:
        self._faces[key] = faces
        self._clear_face_worker()
        self._report_faces(faces)
        self._request_render()

    def _request_parses(self) -> None:
        if self._parse_worker is not None or self.loaded is None:
            return
        from app.ai import runtime
        from app.ui.ai_worker import ParseWorker

        if not (runtime.model_available("bisenet") and runtime.model_available("yunet")):
            self.statusBar().showMessage("Retoque no disponible: faltan PyTorch o los modelos "
                                         "(python tools/download_models.py)", 8000)
            return
        worker = ParseWorker(self.loaded.path, self.loaded.image, self)
        worker.done.connect(self._parses_ready)
        worker.failed.connect(lambda msg: (self.statusBar().showMessage(f"Error en el análisis facial: {msg}",
                                                                        8000), self._clear_parse_worker()))
        self._parse_worker = worker
        self.statusBar().showMessage("Analizando el rostro (piel, ojos, labios, pelo)…")
        worker.start()

    def _clear_parse_worker(self) -> None:
        if self._parse_worker is not None:
            self._parse_worker.deleteLater()
        self._parse_worker = None

    def _parses_ready(self, path: str, parses: list) -> None:
        self._parses[Path(path)] = parses
        self._clear_parse_worker()
        self.statusBar().showMessage("No se encontraron rostros en la foto" if not parses else
                                     f"Rostros analizados: {len(parses)}", 5000)
        self._request_render()

    def _report_faces(self, faces: list) -> None:
        n = len(faces)
        self.statusBar().showMessage("No se encontraron rostros en la foto" if n == 0 else
                                     f"Rostros restaurados: {n}", 5000)

    # --- Recorte y enderezado -------------------------------------------------

    def toggle_crop(self) -> None:
        if self.crop_mode:
            self.apply_crop()
        else:
            self.enter_crop()

    def enter_crop(self) -> None:
        if self.erase_mode:
            self.leave_erase()
        if self.preview is None or self.crop_mode:
            self.crop_action.setChecked(self.crop_mode)
            return
        self._commit_history()
        self._crop_backup = self.settings.copy()
        self.set_view_mode("after")
        self.crop_mode = True
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
        self.set_view_mode("after" if self.view_mode == "before" else "before")

    def _leave_before(self) -> None:
        if self.view_mode == "before":
            self.set_view_mode("after")

    def set_view_mode(self, mode: str) -> None:
        """"after" (resultado), "before" (original) o "compare" (dividido)."""
        if self.crop_mode and mode != "after":
            mode = "after"
        self.view_mode = mode
        self.view_bar.set_mode(mode)
        self.viewer.set_label("Antes" if mode == "before" else "")
        self.viewer.compare_divider.setVisible(mode == "compare")
        self._refresh_view()

    def _before(self):
        """La imagen de origen (con su recorte) sin ajustes, en sRGB."""
        if self.preview is None:
            return None
        source = self._geo_preview if self._geo_preview is not None else self.preview
        if self._before_for is not source:
            self._before_rgb, self._before_for = to_display_u8(source), source
        return self._before_rgb

    def _refresh_view(self) -> None:
        """Muestra el resultado, la original o ambas divididas, con o sin el
        aviso de recorte."""
        if self.shown_rgb is None:
            return
        rgb = self.shown_rgb
        before = self._before() if self.view_mode != "after" else None
        if before is not None and before.shape == rgb.shape:
            if self.view_mode == "before":
                rgb = before
            elif self.view_mode == "compare":
                x = round(self.viewer.compare_divider.position * rgb.shape[1])
                rgb = rgb.copy()
                rgb[:, :x] = before[:, :x]
        if self.histogram.show_clipping:
            rgb = clipping_overlay(rgb)
        self.viewer.set_image(rgb)

    def _viewer_menu(self, pos) -> None:
        if self.preview is None or self.crop_mode:
            return
        menu = QMenu(self)
        for a in (self.copy_action, self.paste_action, None, self.reset_action, None,
                  self.crop_action, self.export_action):
            menu.addSeparator() if a is None else menu.addAction(a)
        menu.exec(self.viewer.viewport().mapToGlobal(pos))

    # --- Copiar / pegar ajustes ------------------------------------------------

    def copy_look(self) -> None:
        if self.preview is None:
            return
        self._commit_history()
        self.copied_look = look_values(self.settings)
        self.paste_action.setEnabled(True)
        n = len(self.copied_look)
        self.statusBar().showMessage(f"Ajustes copiados ({n} cambiados, sin recorte ni giros)", 4000)

    def copy_look_from(self, path: str) -> None:
        """Copia los ajustes de una foto de la tira (abierta o no)."""
        if self.loaded is not None and Path(path) == self.loaded.path:
            self.copy_look()
            return
        self.copied_look = look_values(load_sidecar(path) or Settings())
        self.paste_action.setEnabled(self.preview is not None)
        self.statusBar().showMessage(
            f"Ajustes de {Path(path).name} copiados ({len(self.copied_look)} cambiados)", 4000)

    def paste_look(self) -> None:
        if self.preview is None or self.copied_look is None:
            return
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()
        self._apply_settings(apply_look(self.settings, self.copied_look))
        self._commit_history()
        self.statusBar().showMessage("Ajustes pegados", 3000)

    # --- Acciones sobre varias fotos ---------------------------------------------

    def _batch_action(self, action: str, items: list) -> None:
        if action == "copy":
            self.copy_look_from(items[0])
            return
        if action == "export":
            self.export_batch(items)
            return
        if action == "paste":
            look = dict(self.copied_look or {})
            change, label = (lambda s: apply_look(s, look)), "Ajustes pegados"
        elif action == "preset":
            preset, items = items[0], items[1:]
            change, label = (lambda s: apply_look(s, preset.values)), f"Preset «{preset.name}» aplicado"
        elif action == "reset":
            change, label = (lambda s: Settings()), "Ajustes restablecidos"
        else:
            return
        self._edit_many(items, change)
        self.statusBar().showMessage(f"{label} a {len(items)} foto(s)", 5000)

    def _edit_many(self, paths: list[str], change) -> None:
        """Cambia los ajustes de varias fotos escribiendo sus sidecars. La foto
        abierta se cambia en vivo (y se puede deshacer)."""
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()
        failed = []
        for p in paths:
            if self.loaded is not None and Path(p) == self.loaded.path:
                self._apply_settings(change(self.settings))
                self._commit_history()
                continue
            try:
                save_sidecar(p, change(load_sidecar(p) or Settings()))
            except OSError:
                failed.append(Path(p).name)
        self.filmstrip.refresh(paths)
        if failed:
            QMessageBox.warning(self, "Lighteye", "No se pudieron guardar los ajustes de:\n"
                                + "\n".join(failed))

    def export_batch(self, paths: list[str]) -> None:
        if not paths:
            self.statusBar().showMessage("Selecciona fotos en la tira de la carpeta (Ctrl/Mayús + clic)", 5000)
            return
        if self._export_worker is not None:
            return
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()  # la foto abierta exporta sus últimos ajustes
        dialog = ExportDialog(self)
        dialog.setWindowTitle(f"Exportar {len(paths)} fotos")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        out_dir = ask_output_folder(self, Path(paths[0]).parent)
        if out_dir is None:
            return

        progress = QProgressDialog("Preparando…", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Exportar por lotes")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(420)
        progress.setValue(0)

        worker = BatchExportWorker([Path(p) for p in paths], out_dir, dialog.options(), self)

        def update(value: int, text: str) -> None:
            progress.setValue(value)
            progress.setLabelText(text)

        def done(errors: list, cancelled: bool) -> None:
            progress.reset()
            self._export_worker = None
            worker.deleteLater()
            if cancelled:
                self.statusBar().showMessage("Exportación cancelada", 4000)
            elif errors:
                lines = "\n".join(f"• {Path(p).name}: {msg}" for p, msg in errors[:10])
                QMessageBox.warning(self, "Lighteye",
                                    f"Se exportaron {len(paths) - len(errors)} de {len(paths)} fotos.\n"
                                    f"Fallaron:\n{lines}")
            else:
                self.statusBar().showMessage(f"{len(paths)} fotos exportadas en {out_dir}", 8000)

        worker.progress.connect(update)
        progress.canceled.connect(worker.cancel)
        worker.finished_all.connect(done)
        self._export_worker = worker
        worker.start()

    # --- Presets ----------------------------------------------------------------

    def _update_preset_thumbs(self) -> None:
        """Las miniaturas de los presets muestran la foto actual con su recorte."""
        if self.preview is None or self.crop_mode:
            return
        self.presets.set_photo(make_preview(self.preview, 112), self.settings)

    def apply_preset(self, preset) -> None:
        if self.preview is None or preset is None:
            return
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()
        self._apply_settings(apply_look(self.settings, preset.values))
        self._commit_history()
        if preset.values:
            self.statusBar().showMessage(f"Preset aplicado: {preset.name}", 4000)
        else:
            self.statusBar().showMessage("Ajustes quitados: foto original (Ctrl+Z para deshacer)", 4000)

    def save_preset(self) -> None:
        self._commit_history()
        name, ok = QInputDialog.getText(self, "Guardar preset", "Nombre del preset:")
        name = name.strip()
        if not ok or not name:
            return
        if any(p.name.lower() == name.lower() for p in list_presets() if not p.builtin):
            answer = QMessageBox.question(self, "Lighteye", f"Ya existe «{name}». ¿Reemplazarlo?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            save_preset(name, self.settings)
        except OSError as exc:
            QMessageBox.warning(self, "Lighteye", f"No se pudo guardar el preset:\n{exc}")
            return
        self.presets.reload()
        self.statusBar().showMessage(f"Preset guardado: {name}", 4000)

    def delete_preset(self, preset) -> None:
        answer = QMessageBox.question(self, "Lighteye", f"¿Eliminar el preset «{preset.name}»?")
        if answer == QMessageBox.StandardButton.Yes:
            delete_preset(preset)
            self.presets.reload()

    # --- Exportar ---------------------------------------------------------------

    def export(self) -> None:
        if self.loaded is None:
            return
        if self.crop_mode:
            self.apply_crop()
        self._commit_history()
        dialog = ExportDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        options = dialog.options()
        out_path = ask_output_path(self, self.loaded.path, options)
        if out_path is None:
            return
        if out_path.resolve() == self.loaded.path.resolve():
            QMessageBox.warning(self, "Lighteye", "No se puede sobrescribir la foto original.")
            return

        progress = QProgressDialog("Exportando…", "Cancelar", 0, 100, self)
        progress.setWindowTitle("Exportar")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = ExportWorker(self.loaded.path, self.loaded.image, self.settings, out_path, options, self)
        worker.progress.connect(progress.setValue)
        progress.canceled.connect(worker.cancel)

        def done(message: str, ok: bool) -> None:
            progress.reset()
            self._export_worker = None
            worker.deleteLater()
            if ok:
                self.statusBar().showMessage(f"Exportada: {message}", 8000)
            elif message:
                QMessageBox.warning(self, "Lighteye", f"No se pudo exportar:\n{message}")
            else:
                self.statusBar().showMessage("Exportación cancelada", 4000)

        worker.succeeded.connect(lambda path: done(path, True))
        worker.failed.connect(lambda msg: done(msg, False))
        self._export_worker = worker  # evita que Python lo libere mientras trabaja
        worker.start()

    def closeEvent(self, event):
        self._commit_history()
        for w in (self._face_worker, self._parse_worker, self._erase_worker):
            if w is not None:
                w.wait()
        worker = self._export_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait()
        self.renderer.shutdown()
        self.filmstrip.shutdown()
        super().closeEvent(event)

    # --- Arrastrar y soltar ------------------------------------------------

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile() and is_supported(urls[0].toLocalFile()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.open_image(str(Path(event.mimeData().urls()[0].toLocalFile())))
