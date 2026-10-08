"""Mide cuánto tarda cada paso del pipeline sobre la vista previa.

Uso:  python tools/bench_pipeline.py [foto]
      python tools/bench_pipeline.py --drag    (simula arrastrar sliders)
Sin foto se usa una imagen sintética de 1600×1067 con todos los ajustes activos.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.color import to_display_u8  # noqa: E402
from app.core.loader import load_image, make_preview  # noqa: E402
from app.core.memory import tune_allocator  # noqa: E402
from app.core.pipeline import STEPS, process  # noqa: E402
from app.core.settings import Settings  # noqa: E402

EVERYTHING = {
    "temperature": 15, "tint": -5, "exposure": 0.4, "nr_luma": 30, "nr_color": 40,
    "highlights": -40, "shadows": 35, "whites": 10, "blacks": -10, "contrast": 20,
    "curves": {"rgb": [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]]},
    "hsl_s_blue": 30, "hsl_l_green": -20, "vibrance": 25, "saturation": 10,
    "clarity": 30, "dehaze": 20, "bw": 0, "split_shadow_sat": 20, "split_high_sat": 15,
    "vignette": -30, "sharpen": 40, "grain": 20,
}


def median_ms(fn, runs=7):
    fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2] * 1000


def main():
    tune_allocator()
    if len(sys.argv) > 1 and sys.argv[1] != "--drag":
        img = make_preview(load_image(sys.argv[1]).image)
    else:
        img = np.random.default_rng(0).uniform(0, 1, (1067, 1600, 3)).astype(np.float32)
    settings = Settings(EVERYTHING)
    print(f"Imagen {img.shape[1]}×{img.shape[0]}\n")
    total = 0.0
    for keys, fn in STEPS:
        if all(settings.is_default(k) for k in keys):
            continue
        args = [settings[k] for k in keys]
        ms = median_ms(lambda: fn(img, *args))
        total += ms
        print(f"  {fn.__name__:<18} {ms:6.1f} ms")
    print(f"  {'(suma)':<18} {total:6.1f} ms")
    print(f"\nPipeline completo + conversión a pantalla: "
          f"{median_ms(lambda: to_display_u8(process(img, settings))):.0f} ms")


if __name__ == "__main__" and "--drag" not in sys.argv:
    main()


def drag(slider: str, values, img, base, draft=False):
    """Tiempo medio por movimiento al arrastrar un slider (con caché)."""
    import cv2

    from app.core.pipeline import PipelineCache

    if draft:
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cache = PipelineCache()
    settings = dict(base)
    to_display_u8(process(img, Settings(settings), cache=cache))
    times = []
    for v in values:
        settings[slider] = v
        t = time.perf_counter()
        rgb = to_display_u8(process(img, Settings(settings), cache=cache,
                                    detail_scale=0.5 if draft else 1.0))
        if draft:
            rgb = cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2] * 1000


def drags():
    tune_allocator()
    img = np.random.default_rng(0).uniform(0, 1, (1067, 1600, 3)).astype(np.float32)
    print("\nArrastrar un slider con todo activo (ms por movimiento):")
    print(f"  {'slider':<14} {'completo':>9} {'borrador':>9}")
    for slider, values in [("exposure", np.linspace(0.4, 1.2, 9)), ("contrast", range(20, 40, 2)),
                           ("vibrance", range(25, 45, 2)), ("vignette", range(-30, -50, -2)),
                           ("grain", range(20, 40, 2))]:
        print(f"  {slider:<14} {drag(slider, values, img, EVERYTHING):9.0f} "
              f"{drag(slider, values, img, EVERYTHING, draft=True):9.0f}")


if __name__ == "__main__" and "--drag" in sys.argv:
    drags()
