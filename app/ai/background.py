"""Quitar el fondo con BiRefNet: máscara del sujeto (alfa) y composición.

La máscara se calcula una vez por foto (a 1024×1024, que es la resolución del
modelo) y se guarda en caché; después se lleva a cualquier tamaño, recorte o
giro con una sola transformación afín.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.ai import runtime
from app.ai.registry import MODELS

MODEL_SIDE = 1024
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load(device: str):
    import torch  # noqa: F401
    from safetensors.torch import load_file

    from app.ai.vendor.birefnet.BiRefNet_config import BiRefNetConfig
    from app.ai.vendor.birefnet.birefnet import BiRefNet

    net = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
    net.load_state_dict(load_file(str(MODELS["birefnet"].path)), strict=True)
    net = net.to(device).eval()
    return net.half() if device == "cuda" else net


def compute_alpha(srgb: np.ndarray, device: str | None = None) -> np.ndarray:
    """Máscara del sujeto (MODEL_SIDE × MODEL_SIDE, float32 0–1) de una foto sRGB."""
    import torch

    device = device or runtime.pick_device("birefnet")
    x = cv2.resize(srgb.astype(np.float32), (MODEL_SIDE, MODEL_SIDE), interpolation=cv2.INTER_AREA)
    x = (x - _MEAN) / _STD
    net = _load(device)
    try:
        p = next(net.parameters())
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(p.device, p.dtype)
        with torch.inference_mode():
            pred = net(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
    finally:
        del net
        runtime.release()
    return pred.astype(np.float32)


def warp_alpha(alpha: np.ndarray, orig_w: int, orig_h: int, transform: np.ndarray | None,
               out_w: int, out_h: int) -> np.ndarray:
    """Lleva la máscara (cuadrada, sobre la foto original) a una imagen de
    out_w×out_h. `transform` (3×3): foto original → imagen de salida."""
    side = alpha.shape[0]
    to_orig = np.diag([orig_w / side, orig_h / side, 1.0])
    m = (np.eye(3) if transform is None else transform) @ to_orig
    return cv2.warpAffine(alpha, m[:2], (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def adjust_edge(alpha: np.ndarray, edge: float) -> np.ndarray:
    """Encoge (edge < 0) o agranda (edge > 0) el recorte; edge en −1…1."""
    if edge == 0:
        return alpha
    return np.clip((alpha - 0.5) * 1.5 + 0.5 + edge * 0.35, 0.0, 1.0)


def composite(rgb: np.ndarray, alpha: np.ndarray, background) -> np.ndarray:
    """Pone el sujeto sobre `background`: un color (r, g, b) en las mismas
    unidades que `rgb`, o una imagen del mismo tamaño."""
    a = alpha[..., None]
    bg = np.asarray(background, dtype=np.float32)
    return rgb * a + bg * (1.0 - a)


def checkerboard(h: int, w: int, cell: int = 12) -> np.ndarray:
    """Damero gris claro (uint8 RGB) para mostrar la transparencia."""
    yy, xx = np.indices((h, w))
    light = ((yy // cell + xx // cell) % 2).astype(bool)
    out = np.full((h, w, 3), 204, np.uint8)
    out[light] = 255
    return out


# --- Caché ------------------------------------------------------------------------


def _cache_dir() -> Path:
    from app.paths import cache_dir

    return cache_dir() / "background"


def cache_key(photo: Path) -> str:
    st = Path(photo).stat()
    raw = f"{Path(photo).resolve()}|{st.st_mtime_ns}|{st.st_size}|birefnet"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_cached(photo: Path) -> np.ndarray | None:
    path = _cache_dir() / f"{cache_key(photo)}.png"
    if path.is_file():
        data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if data is not None:
            return data.astype(np.float32) / 65535.0
    return None


def alpha_for(photo: Path, linear_full: np.ndarray) -> np.ndarray:
    """Máscara del sujeto de una foto: de la caché o calculándola."""
    from app.core.color import to_srgb_fast

    cached = load_cached(photo)
    if cached is not None:
        return cached
    alpha = compute_alpha(to_srgb_fast(linear_full))
    try:
        folder = _cache_dir()
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{cache_key(photo)}.tmp.png"
        cv2.imwrite(str(tmp), (alpha * 65535 + 0.5).astype(np.uint16))
        tmp.replace(folder / f"{cache_key(photo)}.png")
    except OSError:
        pass
    return alpha
