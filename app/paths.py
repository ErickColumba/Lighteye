"""Dónde guarda Lighteye sus cosas, según el sistema y si está empaquetado.

- Linux:   ~/.config/lighteye  y  ~/.cache/lighteye  (respetando XDG_*).
- Windows: %APPDATA%\\Lighteye  y  %LOCALAPPDATA%\\Lighteye\\cache.
- Empaquetado (PyInstaller): los modelos de IA van en la carpeta `models`
  junto al ejecutable.
"""

import os
import sys
from pathlib import Path

APP_NAME = "Lighteye"
FROZEN = bool(getattr(sys, "frozen", False))
IS_WINDOWS = sys.platform.startswith("win")


def install_dir() -> Path:
    """Carpeta del programa: la del ejecutable si está empaquetado; si no, el repositorio."""
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    if os.environ.get("XDG_CONFIG_HOME"):
        return Path(os.environ["XDG_CONFIG_HOME"]) / "lighteye"
    if IS_WINDOWS and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / APP_NAME
    return Path.home() / ".config" / "lighteye"


def cache_dir() -> Path:
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "lighteye"
    if IS_WINDOWS and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / APP_NAME / "cache"
    return Path.home() / ".cache" / "lighteye"


def models_dir() -> Path:
    if os.environ.get("LIGHTEYE_MODELS"):
        return Path(os.environ["LIGHTEYE_MODELS"])
    return install_dir() / "models"
