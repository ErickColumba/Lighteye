"""Interpolación de curvas de tono a partir de puntos de control."""

from functools import lru_cache

import numpy as np


def monotone_cubic(points, x: np.ndarray) -> np.ndarray:
    """Interpolación cúbica monótona (Fritsch–Carlson / PCHIP).

    A diferencia de un spline normal, no se pasa de largo entre puntos: si
    los puntos suben, la curva también sube. Fuera del primer y último
    punto, la curva se mantiene plana. Resultado recortado a 0–1.
    """
    xs = np.array([pt[0] for pt in points], dtype=np.float64)
    ys = np.array([pt[1] for pt in points], dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if len(xs) == 2:
        return np.clip(np.interp(x, xs, ys), 0.0, 1.0)

    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty_like(xs)
    m[0], m[-1] = delta[0], delta[-1]
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    same_sign = delta[:-1] * delta[1:] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    m[1:-1] = np.where(same_sign, harmonic, 0.0)

    xc = np.clip(x, xs[0], xs[-1])
    i = np.clip(np.searchsorted(xs, xc, side="right") - 1, 0, len(xs) - 2)
    t = (xc - xs[i]) / h[i]
    t2, t3 = t * t, t * t * t
    y = ((2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h[i] * m[i]
         + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h[i] * m[i + 1])
    return np.clip(y, 0.0, 1.0)


@lru_cache(maxsize=32)
def sample(points: tuple, n: int) -> np.ndarray:
    """Curva evaluada en n valores equiespaciados de 0 a 1."""
    return monotone_cubic(points, np.linspace(0.0, 1.0, n)).astype(np.float32)
