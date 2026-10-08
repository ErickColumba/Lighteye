"""Autocomprobación: `Lighteye --self-test`.

Verifica, sin abrir ventanas, que la instalación (sobre todo la versión
empaquetada) tiene todo lo necesario: el pipeline de edición, la exportación
y que cada modelo de IA se carga y produce un resultado. Devuelve 0 si todo
va bien. Se usa en la compilación automática (GitHub Actions).
"""

import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


def _check(name: str, fn, results: list) -> None:
    t = time.perf_counter()
    try:
        detail = fn() or ""
        results.append((True, name, f"{time.perf_counter() - t:.1f} s {detail}".strip()))
    except Exception:  # noqa: BLE001 — se informa de cualquier fallo
        results.append((False, name, traceback.format_exc(limit=3).strip().splitlines()[-1]))


def run() -> int:
    from app import __version__
    from app.paths import models_dir

    print(f"Lighteye {__version__} — autocomprobación")
    print(f"Modelos en: {models_dir()}")
    results: list = []
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, (96, 128, 3)).astype(np.float32)

    def pipeline():
        from app.core.pipeline import process
        from app.core.presets import list_presets, stack_preset
        from app.core.settings import Settings

        presets = list_presets()
        s = Settings()
        for p in presets[:5]:
            s = stack_preset(s, p)
        process(img, s)
        return f"({len(presets)} presets, LUT y curvas)"

    def export():
        import cv2

        from app.core.exporter import ExportOptions, export_image
        from app.core.settings import Settings

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.png"
            cv2.imencode(".png", (img * 255).astype(np.uint8))[1].tofile(src)
            for fmt in ("jpeg", "png16", "tiff16"):
                export_image(src, img, Settings({"exposure": 0.5}), Path(tmp) / f"o{fmt}",
                             ExportOptions(fmt))

    _check("Pipeline de edición", pipeline, results)
    _check("Exportar JPG/PNG/TIFF", export, results)

    from app.ai import runtime
    from app.ai.registry import MODELS

    if not runtime.torch_available():
        results.append((False, "PyTorch", "no está instalado"))
    else:
        def torch_info():
            import torch

            return f"({torch.__version__}, {runtime.device_name()})"

        _check("PyTorch", torch_info, results)
        small = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        for key in ("realesrgan_x2", "realesrgan_x4", "gfpgan", "codeformer", "lama"):
            def load_and_run(key=key):
                if not MODELS[key].available():
                    raise FileNotFoundError(MODELS[key].file)
                model = runtime.load(key, "cpu")
                if key == "lama":
                    import torch

                    t = runtime._to_tensor(small, model)
                    model(t, torch.zeros_like(t[:, :1]))
                elif key.startswith("realesrgan"):
                    runtime.run_tiled(model, small)
                runtime.release()

            _check(f"Modelo {key}", load_and_run, results)

        def bisenet():
            from app.ai import bisenet as b

            net = b.load(str(MODELS["bisenet"].path), "cpu")
            b.parse(net, rng.uniform(0, 1, (512, 512, 3)).astype(np.float32))

        def birefnet():
            from app.ai.background import _load

            _load("cpu")

        def yunet():
            from app.ai.faces import detect_faces

            detect_faces(img)

        _check("Modelo bisenet", bisenet, results)
        _check("Modelo birefnet", birefnet, results)
        _check("Modelo yunet", yunet, results)

    ok = all(r[0] for r in results)
    for passed, name, detail in results:
        print(f"  {'OK   ' if passed else 'FALLA'} {name:<24} {detail}")
    print("Todo correcto." if ok else "Hay fallos.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
