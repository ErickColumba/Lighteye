"""Una función pura por ajuste.

Cada función recibe una imagen RGB lineal float32 (H, W, 3) y su valor, y
devuelve una imagen nueva sin modificar la de entrada. Los valores pueden
superar 1.0 (luces por encima del blanco) hasta el final del pipeline.
"""

import numpy as np

from app.core.color import LUMA, luminance

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
    return img * gains


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """Exposición en pasos (EV): multiplicar en lineal por 2^ev."""
    return img * np.float32(2.0 ** ev)


def contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Curva S (amount > 0) o aplanado (amount < 0) alrededor del gris medio.

    Se aplica en espacio perceptual con una curva de potencia a cada lado del
    pivote: es monótona, deja fijos 0, el gris medio y 1, y es la identidad
    cuando amount = 0. Los valores por encima de 1 no se tocan.
    """
    k = 1.0 + amount / 100.0 * 0.8  # 0.2 … 1.8
    pivot = MID_GREY ** (1 / PERCEPTUAL_GAMMA)

    x = np.clip(img, 0.0, None)
    p = np.minimum(x, 1.0) ** (1 / PERCEPTUAL_GAMMA)
    low = pivot * (p / pivot) ** k
    high = 1.0 - (1.0 - pivot) * ((1.0 - p) / (1.0 - pivot)) ** k
    curved = np.where(p < pivot, low, high) ** PERCEPTUAL_GAMMA
    return np.where(x > 1.0, x, curved).astype(np.float32)


def saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """−100 = blanco y negro, +100 = doble de saturación. Conserva la luminancia."""
    factor = np.float32(1.0 + amount / 100.0)
    lum = luminance(img)[..., None]
    return np.clip(lum + (img - lum) * factor, 0.0, None)
