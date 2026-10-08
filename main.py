"""Punto de entrada de Lighteye."""

import sys

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Lighteye")
    app.setOrganizationName("Lighteye")

    window = MainWindow()
    window.show()

    # Permite abrir una foto directamente: python main.py foto.jpg
    if len(sys.argv) > 1:
        window.open_image(sys.argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
