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


# Las curvas 1D se aplican con tablas de 65536 entradas: cuantizar a 16 bits
# e indexar es mucho más rápido que evaluar potencias píxel a píxel.
LUT_SIZE = 65536


def lut_index(img: np.ndarray) -> np.ndarray:
    """Valores 0–1 (se recortan) -> índices uint16 para una LUT de LUT_SIZE."""
    idx = np.clip(img, 0.0, 1.0)
    idx *= LUT_SIZE - 1
    idx += 0.5
    return idx.astype(np.uint16)


def lut_domain() -> np.ndarray:
    """Valores de entrada (0–1) que corresponden a cada entrada de la LUT."""
    return np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)


_DISPLAY_LUT = (linear_to_srgb(lut_domain()) * 255.0 + 0.5).astype(np.uint8)
_TO_SRGB_LUT = linear_to_srgb(lut_domain())
_TO_LINEAR_LUT = srgb_to_linear(lut_domain())


# Tablas exactas para imágenes enteras: indexar es ~50 veces más rápido que
# evaluar la potencia de la curva sRGB en cada píxel.
_U8_TO_LINEAR = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
_U16_TO_LINEAR = srgb_to_linear(np.arange(65536, dtype=np.float32) / 65535.0)


def int_srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB uint8/uint16 -> lineal float32 (exacto, con tabla)."""
    if img.dtype == np.uint8:
        return _U8_TO_LINEAR[img]
    if img.dtype == np.uint16:
        return _U16_TO_LINEAR[img]
    raise TypeError(f"Tipo no soportado: {img.dtype}")


def to_srgb_fast(img: np.ndarray) -> np.ndarray:
    """linear_to_srgb con LUT (recorta a 0–1)."""
    return _TO_SRGB_LUT[lut_index(img)]


def to_linear_fast(img: np.ndarray) -> np.ndarray:
    """srgb_to_linear con LUT (recorta a 0–1)."""
    return _TO_LINEAR_LUT[lut_index(img)]


def to_display_u8(img: np.ndarray) -> np.ndarray:
    """Imagen lineal float32 -> sRGB uint8 contigua, lista para QImage."""
    return np.ascontiguousarray(_DISPLAY_LUT[lut_index(img)])
