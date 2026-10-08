"""Ejecución de los modelos de IA (PyTorch + spandrel).

- Usa la GPU NVIDIA si hay; si no, el procesador (más lento, pero funciona).
- Procesa por mosaicos para no agotar la memoria de vídeo, y si aun así se
  agota (p. ej. otro programa usa la GPU) reintenta en el procesador.
- Los modelos se descargan de memoria al terminar cada tarea (`release`),
  para no acaparar la GPU.

Todo trabaja en sRGB float32 (H, W, 3) en 0–1, que es lo que esperan estos
modelos.
"""

import threading
import warnings
from functools import lru_cache

import cv2
import numpy as np

from app.ai.registry import MODELS

# Avisos internos de las librerías que no indican ningún problema:
# - PyTorch avisa de que torch.jit (con el que está guardado LaMa) quedará obsoleto.
# - OpenCV avisa al preparar el detector de caras YuNet con su nuevo motor.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.jit")
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

_lock = threading.Lock()
# Se activa para repetir en el procesador una tarea que agotó la memoria de vídeo.
FORCE_CPU = threading.Event()
_loaded: dict[str, object] = {}


@lru_cache(maxsize=1)
def torch_available() -> bool:
    """¿Están instalados PyTorch y spandrel? Se comprueba sin importarlos:
    cargar PyTorch cuesta ~1 s y cientos de MB, y solo hace falta al usar la IA."""
    from importlib.util import find_spec

    return find_spec("torch") is not None and find_spec("spandrel") is not None


def device_name() -> str:
    if not torch_available():
        return "no disponible (falta PyTorch)"
    if "torch" not in __import__("sys").modules:
        return "GPU NVIDIA si hay memoria libre; si no, el procesador"
    import torch

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "procesador (CPU)"


# Memoria de vídeo libre necesaria (aprox.) para usar la GPU con cada modelo.
VRAM_NEEDED_MB = {"realesrgan_x4": 1200, "realesrgan_x2": 1200, "gfpgan": 1500,
                  "codeformer": 1500, "bisenet": 600, "lama": 2500, "birefnet": 3500}


def pick_device(key: str) -> str:
    """GPU si hay una con memoria libre suficiente; si no, el procesador.

    Así Lighteye funciona aunque otro programa (p. ej. ComfyUI) tenga la
    GPU casi llena.
    """
    import torch

    if FORCE_CPU.is_set() or not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 2**20 >= VRAM_NEEDED_MB.get(key, 1500) else "cpu"


def model_available(key: str) -> bool:
    return torch_available() and MODELS[key].available()


def load(key: str, device: str | None = None):
    """Modelo `key` listo para usar (se reutiliza mientras no se libere)."""
    import spandrel
    import torch

    if key == "codeformer":
        import spandrel_extra_arches

        spandrel_extra_arches.install(ignore_duplicates=True)
    device = device or pick_device(key)
    cache_key = f"{key}@{device}"
    with _lock:
        if cache_key not in _loaded:
            info = MODELS[key]
            if not info.available():
                raise FileNotFoundError(
                    f"Falta el modelo {info.file}. Descárgalo con: python tools/download_models.py {key}")
            model = spandrel.ModelLoader().load_from_file(str(info.path))
            model.to(device).eval()
            if device == "cuda" and model.supports_half:
                model.half()
            _loaded[cache_key] = model
        return _loaded[cache_key]


def release() -> None:
    """Libera los modelos y la memoria de la GPU."""
    with _lock:
        _loaded.clear()
    if torch_available():
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _to_tensor(img: np.ndarray, model):
    import torch

    t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
    p = next(model.model.parameters())
    return t.to(device=p.device, dtype=p.dtype)


def _to_array(t) -> np.ndarray:
    return t[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def run_tiled(model, img: np.ndarray, tile: int = 512, overlap: int = 24, progress=None) -> np.ndarray:
    """Aplica un modelo de imagen a imagen por mosaicos con solapamiento."""
    import torch

    scale = model.scale
    h, w = img.shape[:2]
    out = np.zeros((h * scale, w * scale, 3), np.float32)
    ys = list(range(0, max(h - overlap, 1), tile - 2 * overlap)) if h > tile else [0]
    xs = list(range(0, max(w - overlap, 1), tile - 2 * overlap)) if w > tile else [0]
    total, done = len(ys) * len(xs), 0
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                y0, x0 = min(y, max(h - tile, 0)), min(x, max(w - tile, 0))
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                result = _to_array(model(_to_tensor(img[y0:y1, x0:x1], model)))
                # Se descarta el borde solapado (salvo en los bordes de la foto).
                cy0 = 0 if y0 == 0 else overlap
                cx0 = 0 if x0 == 0 else overlap
                cy1 = (y1 - y0) if y1 == h else (y1 - y0 - overlap)
                cx1 = (x1 - x0) if x1 == w else (x1 - x0 - overlap)
                out[(y0 + cy0) * scale:(y0 + cy1) * scale, (x0 + cx0) * scale:(x0 + cx1) * scale] = \
                    result[cy0 * scale:cy1 * scale, cx0 * scale:cx1 * scale]
                done += 1
                if progress:
                    progress(done / total)
    return out


def with_cpu_fallback(fn, *args, **kwargs):
    """Ejecuta fn en la GPU; si se queda sin memoria, libera y repite en CPU."""
    import torch

    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        release()
        return fn(*args, device="cpu", **kwargs)


def upscale(img: np.ndarray, factor: int, progress=None, device: str | None = None) -> np.ndarray:
    """Escala ×2 o ×4 con Real-ESRGAN (sRGB 0–1 → sRGB 0–1)."""
    key = {2: "realesrgan_x2", 4: "realesrgan_x4"}[factor]

    def go(device=None):
        return run_tiled(load(key, device), img, progress=progress)

    try:
        return with_cpu_fallback(go) if device is None else go(device)
    finally:
        release()
