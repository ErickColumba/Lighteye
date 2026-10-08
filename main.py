"""Punto de entrada de Lighteye."""

import sys

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
from PySide6.QtWidgets import QApplication

from app.core.memory import tune_allocator
from app.ui.main_window import MainWindow


def main() -> int:
    tune_allocator()
    if "--self-test" in sys.argv:
        from app.selftest import run

        return run()
    app = QApplication(sys.argv)
    app.setApplicationName("Lighteye")
    app.setOrganizationName("Lighteye")
    from pathlib import Path

    from PySide6.QtGui import QIcon

    from app import __version__

    app.setApplicationVersion(__version__)
    import app as app_package  # su ruta sirve igual en el repositorio y empaquetado

    icon = Path(app_package.__file__).resolve().parent / "resources" / "lighteye.png"
    app.setWindowIcon(QIcon(str(icon)))
    # En Linux (Wayland/KDE) asocia la ventana con su lanzador .desktop.
    app.setDesktopFileName("lighteye")

    # Textos estándar de Qt (Cancelar, Aceptar, diálogos de archivo) en español.
    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale("es"), "qtbase", "_", path):
        app.installTranslator(translator)

    window = MainWindow()
    window.show()

    # Permite abrir una foto directamente: python main.py foto.jpg
    if len(sys.argv) > 1:
        window.open_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
