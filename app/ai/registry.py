"""Modelos de IA que usa Lighteye: de dónde salen, cuánto pesan y su licencia.

Los archivos van en la carpeta `models/` de Lighteye (se incluyen en el
paquete instalable). En el repositorio no se guardan por su tamaño: se
descargan con `python tools/download_models.py`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def models_dir() -> Path:
    """Carpeta de modelos. Se puede cambiar con LIGHTEYE_MODELS."""
    return Path(os.environ.get("LIGHTEYE_MODELS") or ROOT / "models")


@dataclass(frozen=True)
class ModelInfo:
    key: str
    file: str
    url: str
    size_mb: int
    license: str
    purpose: str

    @property
    def path(self) -> Path:
        return models_dir() / self.file

    def available(self) -> bool:
        return self.path.is_file()


MODELS = {m.key: m for m in [
    ModelInfo("realesrgan_x4", "RealESRGAN_x4plus.pth",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
              67, "BSD-3-Clause", "Escalado ×4"),
    ModelInfo("realesrgan_x2", "RealESRGAN_x2plus.pth",
              "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
              67, "BSD-3-Clause", "Escalado ×2"),
    ModelInfo("gfpgan", "GFPGANv1.4.pth",
              "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
              348, "Apache-2.0", "Restaurar rostros"),
    ModelInfo("codeformer", "codeformer.pth",
              "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
              376, "S-Lab License 1.0 (solo uso no comercial)", "Restaurar rostros manteniendo identidad"),
    ModelInfo("bisenet", "parsing_bisenet.pth",
              "https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth",
              53, "MIT", "Máscaras de piel, cabello, ojos…"),
    ModelInfo("lama", "big-lama.pt",
              "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
              205, "Apache-2.0", "Borrar objetos"),
    ModelInfo("yunet", "face_detection_yunet_2023mar.onnx",
              "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
              "face_detection_yunet_2023mar.onnx",
              1, "MIT", "Detectar rostros"),
]}
