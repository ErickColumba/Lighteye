"""Una función pura por ajuste.

Cada función recibe una imagen RGB lineal float32 (H, W, 3) y su valor, y
devuelve una imagen nueva sin modificar la de entrada. Los valores pueden
superar 1.0 (luces por encima del blanco) hasta el final del pipeline.
"""

from functools import lru_cache

import cv2
import numpy as np

from app.core.color import LUMA, lut_domain, lut_index

# Gris medio (18 %) en lineal: pivote para el contraste.
MID_GREY = 0.18
# Gamma perceptual aproximada usada para que las curvas actúen "como se ven".
PERCEPTUAL_GAMMA = 2.2


def white_balance(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Temperatura (−100 frío … +100 cálido) y tinte (−100 verde … +100 magenta).

    Ganancias por canal normalizadas para no cambiar la luminancia global.
    """
    t = temperature / 100.0
    g = tint / 100.0
    gains = np.array(
        [2.0 ** (0.5 * t), 2.0 ** (-0.35 * g), 2.0 ** (-0.5 * t)], dtype=np.float32
    )
    gains /= float(gains @ LUMA)
    return cv2.transform(img, np.diag(gains))


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """Exposición en pasos (EV): multiplicar en lineal por 2^ev."""
    return img * np.float32(2.0 ** ev)


@lru_cache(maxsize=8)
def _contrast_lut(amount: float) -> np.ndarray:
    k = 1.0 + amount / 100.0 * 0.8  # 0.2 … 1.8
    pivot = MID_GREY ** (1 / PERCEPTUAL_GAMMA)
    p = lut_domain() ** (1 / PERCEPTUAL_GAMMA)
    low = pivot * (p / pivot) ** k
    high = 1.0 - (1.0 - pivot) * ((1.0 - p) / (1.0 - pivot)) ** k
    return (np.where(p < pivot, low, high) ** PERCEPTUAL_GAMMA).astype(np.float32)


def contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Curva S (amount > 0) o aplanado (amount < 0) alrededor del gris medio.

    Se aplica en espacio perceptual con una curva de potencia a cada lado del
    pivote: es monótona, deja fijos 0, el gris medio y 1, y es la identidad
    cuando amount = 0. Los valores por encima de 1 no se tocan.
    """
    out = _contrast_lut(float(amount))[lut_index(img)]
    # Lo que pasa de 1.0 se suma tal cual: la curva vale 1 en 1, así que es continuo.
    out += np.maximum(img - 1.0, 0.0)
    return out


def saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """−100 = blanco y negro, +100 = doble de saturación. Conserva la luminancia."""
    # lum + (img − lum)·f es lineal en RGB: se aplica como una matriz 3×3.
    f = 1.0 + amount / 100.0
    matrix = (f * np.eye(3) + (1.0 - f) * LUMA[None, :]).astype(np.float32)
    out = cv2.transform(img, matrix)
    return np.maximum(out, 0.0, out=out)
