"""Conversiones entre sRGB (gamma) y RGB lineal.

Toda la edición se hace en float32 lineal; solo se pasa a sRGB para mostrar
o exportar.
"""

import numpy as np

# Pesos de luminancia Rec. 709 (primarios sRGB), en espacio lineal.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(
        img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(
        img <= 0.0031308, img * 12.92, 1.055 * img ** (1 / 2.4) - 0.055
    ).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Luminancia lineal (H, W) de una imagen RGB lineal (H, W, 3)."""
    return img @ LUMA


def to_display_u8(img: np.ndarray) -> np.ndarray:
    """Imagen lineal float32 -> sRGB uint8 contigua, lista para QImage."""
    out = linear_to_srgb(img) * 255.0 + 0.5
    return np.ascontiguousarray(out.astype(np.uint8))
