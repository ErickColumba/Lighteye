"""Retoque de retrato con las máscaras del análisis facial (BiSeNet).

Cada efecto actúa solo dentro de su zona (piel, ojos, labios o pelo), con
bordes suaves, y su tamaño se adapta al de la cara en la foto.
"""

import cv2
import numpy as np

from app.ai import bisenet
from app.ai.faces import FACE_SIZE, FaceParse
from app.core.color import LUMA, linear_to_srgb, srgb_to_linear

RETOUCH_KEYS = ("skin_smooth", "eyes_brighten", "lips_saturation", "hair_shine")


def _mask(labels_roi: np.ndarray, classes, feather: float, grow: float = 0.0) -> np.ndarray:
    m = np.isin(labels_roi, classes).astype(np.float32)
    if grow > 0:
        k = max(1, int(grow))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    if feather > 0.3:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m[..., None]


def _saturate(img: np.ndarray, factor: float) -> np.ndarray:
    lum = cv2.transform(img, LUMA[None, :])[..., None]
    return np.maximum(lum + (img - lum) * factor, 0.0)


def apply_retouch(img: np.ndarray, parses: list[FaceParse], settings,
                  transform: np.ndarray | None = None) -> np.ndarray:
    """Aplica los retoques a `img` (lineal). `transform` (3×3): foto original → img."""
    skin = settings["skin_smooth"] / 100
    eyes = settings["eyes_brighten"] / 100
    lips = settings["lips_saturation"] / 100
    hair = settings["hair_shine"] / 100
    if not parses or not (skin or eyes or lips or hair):
        return img
    out = img.copy()
    h, w = img.shape[:2]
    t = np.eye(3) if transform is None else transform
    for p in parses:
        m = np.vstack([p.matrix, [0, 0, 1]]) @ np.linalg.inv(t)  # img → recorte
        inv = np.linalg.inv(m)
        face_px = FACE_SIZE / np.sqrt(abs(np.linalg.det(m[:2, :2])))  # ancho del recorte en img
        corners = inv[:2] @ np.array([[0, FACE_SIZE, 0, FACE_SIZE], [0, 0, FACE_SIZE, FACE_SIZE],
                                      [1, 1, 1, 1]])
        x0, y0 = max(0, int(corners[0].min())), max(0, int(corners[1].min()))
        x1, y1 = min(w, int(np.ceil(corners[0].max()))), min(h, int(np.ceil(corners[1].max())))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        # Etiquetas llevadas al rectángulo de la cara en img (255 = fuera del recorte).
        to_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ inv
        labels = cv2.warpAffine(p.labels, to_roi[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        feather = face_px * 0.006
        roi = out[y0:y1, x0:x1]

        if skin:
            mask = _mask(labels, bisenet.SKIN, feather * 2)
            srgb = linear_to_srgb(roi)
            smooth = cv2.bilateralFilter(srgb, d=0, sigmaColor=0.05 + 0.08 * skin,
                                         sigmaSpace=max(1.0, face_px * 0.012))
            smooth = smooth + (srgb - smooth) * 0.3  # conserva algo de textura (poros)
            roi = roi + (srgb_to_linear(smooth) - roi) * (mask * skin)
        if eyes:
            mask = _mask(labels, bisenet.EYES, feather, grow=face_px * 0.004)
            bright = _saturate(roi * np.float32(2 ** (0.7 * eyes)), 1 + 0.25 * eyes)
            roi = roi + (bright - roi) * mask
        if lips:
            mask = _mask(labels, bisenet.LIPS, feather)
            roi = roi + (_saturate(roi, 1 + 0.8 * lips) - roi) * mask
        if hair:
            mask = _mask(labels, bisenet.HAIR, feather * 2)
            lum = np.maximum(cv2.transform(roi, LUMA[None, :]), 1e-6)
            y = cv2.pow(lum, 1 / 2.2)
            detail = y - cv2.GaussianBlur(y, (0, 0), max(1.0, face_px * 0.015))
            new_y = np.maximum(y + detail * (0.6 * hair) + 0.02 * hair, 0)
            shiny = roi * (cv2.pow(new_y, 2.2) / lum)[..., None]
            roi = roi + (shiny - roi) * mask
        out[y0:y1, x0:x1] = roi
    return out
