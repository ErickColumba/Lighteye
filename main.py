"""Punto de entrada de Lighteye."""

import sys

from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator
from PySide6.QtWidgets import QApplication

from app.core.memory import tune_allocator
from app.ui.main_window import MainWindow


def main() -> int:
    tune_allocator()
    app = QApplication(sys.argv)
    app.setApplicationName("Lighteye")
    app.setOrganizationName("Lighteye")

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
