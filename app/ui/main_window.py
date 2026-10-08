"""Ventana principal de Lighteye."""

from PySide6.QtWidgets import QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lighteye")
        self.resize(1400, 900)

    def open_image(self, path: str) -> None:
        pass
