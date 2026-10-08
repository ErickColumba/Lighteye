"""Histograma de la imagen tal como se ve (sRGB de 8 bits)."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Histogram:
    r: np.ndarray  # (256,) recuentos por nivel
    g: np.ndarray
    b: np.ndarray
    luma: np.ndarray
    clipped_shadows: float  # fracción de píxeles negros puros (0, 0, 0)
    clipped_highlights: float  # fracción de píxeles con algún canal a 255


def compute_histogram(rgb_u8: np.ndarray) -> Histogram:
    def hist(channel):
        return cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()

    r, g, b = cv2.split(rgb_u8)
    luma = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    mx = cv2.max(cv2.max(r, g), b)
    total = float(max(1, mx.size))
    return Histogram(
        hist(r), hist(g), hist(b), hist(luma),
        clipped_shadows=float(np.count_nonzero(mx == 0)) / total,
        clipped_highlights=float(np.count_nonzero(mx == 255)) / total,
    )


def clipping_overlay(rgb_u8: np.ndarray) -> np.ndarray:
    """Copia de la imagen con las altas luces quemadas en rojo y las sombras
    empastadas en azul, para ver dónde se pierde detalle."""
    out = rgb_u8.copy()
    mx = rgb_u8.max(axis=2)
    out[mx == 255] = (255, 0, 0)
    out[mx == 0] = (0, 80, 255)
    return out
