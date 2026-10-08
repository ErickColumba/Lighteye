"""Restauración de rostros con GFPGAN o CodeFormer.

1. Se detectan las caras con YuNet (OpenCV) y sus 5 puntos (ojos, nariz,
   comisuras).
2. Cada cara se alinea a la plantilla de 512×512 con la que se entrenaron
   los modelos (la de FFHQ), se restaura y se pega de vuelta con un borde
   suave.

El resultado de cada cara (512×512 + su transformación) se guarda en caché:
pegarlo sobre la vista previa o sobre la foto completa es instantáneo, y el
slider de intensidad no vuelve a ejecutar la IA.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ai import runtime
from app.ai.registry import MODELS

FACE_SIZE = 512
# Posición de ojos, nariz y comisuras en la plantilla de 512×512 (FFHQ).
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.19360], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
DETECT_SIDE = 1600  # la detección se hace sobre una versión reducida
MIN_FACE_PX = 24  # caras más pequeñas no merecen la pena


@dataclass
class RestoredFace:
    face: np.ndarray  # (512, 512, 3) sRGB float32 restaurada
    matrix: np.ndarray  # (2, 3) afín: foto original → recorte de 512


def detect_faces(srgb: np.ndarray) -> list[np.ndarray]:
    """Puntos (5, 2) de cada cara, en píxeles de `srgb` (foto completa)."""
    h, w = srgb.shape[:2]
    scale = min(1.0, DETECT_SIDE / max(h, w))
    small = cv2.resize(srgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor((np.clip(small, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    detector = cv2.FaceDetectorYN.create(str(MODELS["yunet"].path), "", (bgr.shape[1], bgr.shape[0]),
                                         score_threshold=0.7)
    _, found = detector.detect(bgr)
    faces = []
    for row in found if found is not None else []:
        if min(row[2], row[3]) / scale < MIN_FACE_PX:
            continue
        points = row[4:14].reshape(5, 2) / scale
        # Ojo y comisura de la izquierda de la imagen primero, como la plantilla.
        if points[0, 0] > points[1, 0]:
            points[[0, 1]] = points[[1, 0]]
        if points[3, 0] > points[4, 0]:
            points[[3, 4]] = points[[4, 3]]
        faces.append(points.astype(np.float32))
    return faces


def _restore_crop(model, crop: np.ndarray, codeformer: bool, fidelity: float) -> np.ndarray:
    import torch

    t = runtime._to_tensor(crop * 2 - 1, model)  # estos modelos trabajan en −1…1
    with torch.inference_mode():
        out = model.model(t, weight=fidelity) if codeformer else model.model(t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return np.clip((out[0].float().permute(1, 2, 0).cpu().numpy() + 1) / 2, 0, 1)


def restore_faces(srgb: np.ndarray, use_codeformer: bool = False, fidelity: float = 0.7,
                  progress=None, device: str | None = None) -> list[RestoredFace]:
    """Detecta y restaura las caras de una foto (sRGB 0–1, resolución completa)."""
    points = detect_faces(srgb)
    key = "codeformer" if use_codeformer else "gfpgan"
    results = []
    try:
        model = runtime.load(key, device)
        for i, pts in enumerate(points):
            matrix, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE, method=cv2.LMEDS)
            if matrix is None:
                continue
            crop = cv2.warpAffine(srgb, matrix, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT101)
            face = _restore_crop(model, crop.astype(np.float32), use_codeformer, fidelity)
            results.append(RestoredFace(face, matrix.astype(np.float32)))
            if progress:
                progress((i + 1) / len(points))
    finally:
        runtime.release()
    return results


def _feather_mask() -> np.ndarray:
    mask = np.zeros((FACE_SIZE, FACE_SIZE), np.float32)
    border = FACE_SIZE // 16
    mask[border:-border, border:-border] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), FACE_SIZE / 24)


_MASK = None


def paste_faces(img: np.ndarray, faces: list[RestoredFace], strength: float,
                transform: np.ndarray | None = None) -> np.ndarray:
    """Pega las caras restauradas sobre `img` (lineal), mezcladas según
    `strength` (0–1). `transform` (3×3) lleva puntos de la foto original a
    `img`: incluye reducción, giros y recorte (ver geometry.geometry_matrix).

    Solo se procesa el rectángulo de cada cara, así es rápido a cualquier
    resolución.
    """
    global _MASK
    if not faces or strength <= 0:
        return img
    if _MASK is None:
        _MASK = _feather_mask()
    from app.core.color import srgb_to_linear

    out = img.copy()
    h, w = img.shape[:2]
    for f in faces:
        # img → foto original → recorte de 512.
        t = np.eye(3) if transform is None else transform
        m = np.vstack([f.matrix, [0, 0, 1]]) @ np.linalg.inv(t)
        inv = np.linalg.inv(m)[:2]
        corners = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]]).T
        pts = inv @ corners
        x0, y0 = max(0, int(pts[0].min())), max(0, int(pts[1].min()))
        x1, y1 = min(w, int(np.ceil(pts[0].max())) + 1), min(h, int(np.ceil(pts[1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        shift = np.array([[1, 0, -x0], [0, 1, -y0]], np.float64)
        roi_m = shift @ np.vstack([inv, [0, 0, 1]])
        size = (x1 - x0, y1 - y0)
        face = cv2.warpAffine(f.face, roi_m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        mask = cv2.warpAffine(_MASK, roi_m, size, flags=cv2.INTER_LINEAR)[..., None] * strength
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = roi + (srgb_to_linear(face) - roi) * mask
    return out


# --- Caché en disco -------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "lighteye" / "faces"


def cache_key(photo: Path, use_codeformer: bool, fidelity: float) -> str:
    st = Path(photo).stat()
    fidelity = round(fidelity * 20) / 20  # pasos de 0.05: menos cálculos al mover el slider
    model = f"codeformer{fidelity:.2f}" if use_codeformer else "gfpgan14"
    raw = f"{Path(photo).resolve()}|{st.st_mtime_ns}|{st.st_size}|{model}"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_cached(key: str) -> list[RestoredFace] | None:
    path = _cache_dir() / f"{key}.npz"
    if not path.is_file():
        return None
    try:
        data = np.load(path)
        n = int(data["count"])
        return [RestoredFace(data[f"face{i}"].astype(np.float32) / 255.0 if data[f"face{i}"].dtype == np.uint8
                             else data[f"face{i}"].astype(np.float32), data[f"matrix{i}"])
                for i in range(n)]
    except (OSError, ValueError, KeyError):
        return None


def save_cached(key: str, faces: list[RestoredFace]) -> None:
    folder = _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {"count": np.array(len(faces))}
    for i, f in enumerate(faces):
        arrays[f"face{i}"] = f.face.astype(np.float16)
        arrays[f"matrix{i}"] = f.matrix
    tmp = folder / f"{key}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(folder / f"{key}.npz")


def faces_for(photo: Path, linear_full: np.ndarray, use_codeformer: bool, fidelity: float,
              progress=None) -> list[RestoredFace]:
    """Caras restauradas de una foto: de la caché o calculándolas (lento)."""
    from app.core.color import to_srgb_fast

    fidelity = round(fidelity * 20) / 20  # igual que en cache_key
    key = cache_key(photo, use_codeformer, fidelity)
    cached = load_cached(key)
    if cached is not None:
        return cached
    faces = restore_faces(to_srgb_fast(linear_full), use_codeformer, fidelity, progress)
    try:
        save_cached(key, faces)
    except OSError:
        pass
    return faces


# --- Análisis facial (BiSeNet): máscaras de piel, ojos, labios, pelo -------------


@dataclass
class FaceParse:
    labels: np.ndarray  # (512, 512) uint8, clases de bisenet.CLASSES
    matrix: np.ndarray  # (2, 3) afín: foto original → recorte de 512


def parse_faces(srgb: np.ndarray, progress=None, device: str | None = None) -> list[FaceParse]:
    """Detecta las caras y etiqueta cada píxel (piel, ojos, pelo…)."""
    from app.ai import bisenet

    points = detect_faces(srgb)
    results = []
    if not points:
        return results
    device = device or runtime.pick_device("bisenet")
    net = bisenet.load(str(MODELS["bisenet"].path), device)
    try:
        for i, pts in enumerate(points):
            matrix, _ = cv2.estimateAffinePartial2D(pts, TEMPLATE, method=cv2.LMEDS)
            if matrix is None:
                continue
            crop = cv2.warpAffine(srgb, matrix, (FACE_SIZE, FACE_SIZE), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT101)
            results.append(FaceParse(bisenet.parse(net, crop), matrix.astype(np.float32)))
            if progress:
                progress((i + 1) / len(points))
    finally:
        del net
        runtime.release()
    return results


def parse_cache_key(photo: Path) -> str:
    st = Path(photo).stat()
    raw = f"{Path(photo).resolve()}|{st.st_mtime_ns}|{st.st_size}|bisenet"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_parses_cached(photo: Path) -> list[FaceParse] | None:
    """Análisis facial guardado en caché, o None si no se ha calculado."""
    path = _cache_dir() / f"parse-{parse_cache_key(photo)}.npz"
    if path.is_file():
        try:
            data = np.load(path)
            return [FaceParse(data[f"labels{i}"], data[f"matrix{i}"]) for i in range(int(data["count"]))]
        except (OSError, ValueError, KeyError):
            pass
    return None


def parses_for(photo: Path, linear_full: np.ndarray, progress=None) -> list[FaceParse]:
    """Análisis facial de una foto: de la caché o calculándolo."""
    from app.core.color import to_srgb_fast

    cached = load_parses_cached(photo)
    if cached is not None:
        return cached
    path = _cache_dir() / f"parse-{parse_cache_key(photo)}.npz"
    parses = parse_faces(to_srgb_fast(linear_full), progress)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"count": np.array(len(parses))}
        for i, p in enumerate(parses):
            arrays[f"labels{i}"], arrays[f"matrix{i}"] = p.labels, p.matrix
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    except OSError:
        pass
    return parses
