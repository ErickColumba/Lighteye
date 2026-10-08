"""Diálogo de exportación y trabajo en segundo plano con barra de progreso."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QSlider,
    QLabel,
    QSpinBox,
    QWidget,
)

from app.core.exporter import FORMATS, ExportOptions, default_output_path, export_image
from app.core.settings import Settings


class ExportDialog(QDialog):
    """Formato, calidad y tamaño. Recuerda la última elección."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exportar")
        store = QSettings()

        self.format = QComboBox()
        for key, (name, _, _) in FORMATS.items():
            self.format.addItem(name, key)
        self.format.setCurrentIndex(max(0, self.format.findData(store.value("export/fmt", "jpeg"))))

        self.quality = QSlider(Qt.Orientation.Horizontal)
        self.quality.setRange(50, 100)
        self.quality.setValue(int(store.value("export/quality", 92)))
        self.quality_label = QLabel()
        quality_row = QWidget()
        row = QHBoxLayout(quality_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.quality, 1)
        row.addWidget(self.quality_label)

        self.resize_box = QCheckBox("Limitar el lado largo a")
        self.resize_box.setChecked(store.value("export/resize", "false") == "true")
        self.long_side = QSpinBox()
        self.long_side.setRange(100, 20000)
        self.long_side.setSuffix(" px")
        self.long_side.setValue(int(store.value("export/long_side", 2048)))
        size_row = QWidget()
        row = QHBoxLayout(size_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.resize_box)
        row.addWidget(self.long_side, 1)

        form = QFormLayout(self)
        form.addRow("Formato", self.format)
        form.addRow("Calidad JPG", quality_row)
        form.addRow("Tamaño", size_row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Exportar…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.quality.valueChanged.connect(lambda v: self.quality_label.setText(str(v)))
        self.format.currentIndexChanged.connect(self._update_enabled)
        self.resize_box.toggled.connect(self._update_enabled)
        self.quality_label.setText(str(self.quality.value()))
        self._update_enabled()

    def _update_enabled(self) -> None:
        is_jpeg = self.format.currentData() == "jpeg"
        self.quality.setEnabled(is_jpeg)
        self.quality_label.setEnabled(is_jpeg)
        self.long_side.setEnabled(self.resize_box.isChecked())

    def options(self) -> ExportOptions:
        return ExportOptions(
            fmt=self.format.currentData(),
            quality=self.quality.value(),
            long_side=self.long_side.value() if self.resize_box.isChecked() else None,
        )

    def accept(self) -> None:
        store = QSettings()
        store.setValue("export/fmt", self.format.currentData())
        store.setValue("export/quality", self.quality.value())
        store.setValue("export/resize", "true" if self.resize_box.isChecked() else "false")
        store.setValue("export/long_side", self.long_side.value())
        super().accept()


def ask_output_path(parent, src: Path, options: ExportOptions) -> Path | None:
    store = QSettings()
    folder = Path(store.value("export/dir", str(src.parent)))
    if not folder.is_dir():
        folder = src.parent
    suggested = folder / default_output_path(src, options).name
    name = FORMATS[options.fmt][0]
    path, _ = QFileDialog.getSaveFileName(parent, "Exportar como", str(suggested),
                                          f"{name} (*{options.extension})")
    if not path:
        return None
    path = Path(path)
    if path.suffix.lower() not in (options.extension, ".jpeg", ".tiff"):
        path = path.with_name(path.name + options.extension)
    store.setValue("export/dir", str(path.parent))
    return path


class ExportWorker(QThread):
    """Exporta en un hilo aparte. `cancel()` la detiene en el siguiente paso."""

    progress = Signal(int)  # 0–100
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, src: Path, image: np.ndarray, settings: Settings, out_path: Path,
                 options: ExportOptions, parent=None):
        super().__init__(parent)
        self.src, self.image, self.settings = src, image, settings.copy()
        self.out_path, self.options = out_path, options
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _report(self, fraction: float) -> None:
        if self._cancelled:
            raise InterruptedError
        self.progress.emit(round(fraction * 100))

    def run(self) -> None:
        try:
            path = export_image(self.src, self.image, self.settings, self.out_path,
                                self.options, self._report)
        except InterruptedError:
            self.failed.emit("")  # cancelada: sin mensaje de error
        except Exception as exc:  # noqa: BLE001 — se muestra al usuario
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.succeeded.emit(str(path))


def ask_output_folder(parent, start: Path) -> Path | None:
    store = QSettings()
    folder = QFileDialog.getExistingDirectory(parent, "Carpeta de destino",
                                              str(store.value("export/dir", str(start))))
    if not folder:
        return None
    store.setValue("export/dir", folder)
    return Path(folder)


class BatchExportWorker(QThread):
    """Exporta varias fotos seguidas, cada una con sus propios ajustes."""

    progress = Signal(int, str)  # 0–100, texto
    finished_all = Signal(list, bool)  # [(ruta, error)], cancelada

    def __init__(self, paths: list[Path], out_dir: Path, options: ExportOptions, parent=None):
        super().__init__(parent)
        self.paths, self.out_dir, self.options = paths, out_dir, options
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _report(self, fraction: float, index: int, path: Path) -> None:
        if self._cancelled:
            raise InterruptedError
        n = len(self.paths)
        self.progress.emit(round(fraction * 100),
                           f"Exportando {min(index + 1, n)} de {n}: {path.name}")

    def run(self) -> None:
        from app.core.exporter import export_many

        try:
            errors = export_many(self.paths, self.out_dir, self.options, self._report)
        except InterruptedError:
            self.finished_all.emit([], True)
        else:
            self.finished_all.emit([(str(p), msg) for p, msg in errors], False)
