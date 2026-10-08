"""Genera los presets incluidos con Lighteye (app/presets/*.json).

Están pensados para retratos, a partir del análisis de una serie de retratos
de prueba (luz suave, fondos de estudio claros u oscuros y exteriores):
- Las fotos salían algo planas: negros sin llegar a negro (~21/255) y blancos
  sin llegar a blanco (~216/255). Casi todos los presets amplían ese rango
  (Negros −, Blancos +).
- Saturación contenida y piel en torno a 22° de tono (naranja): se usa
  Intensidad en vez de Saturación, que protege la piel, y HSL naranja con
  cuidado.

Uso:  python tools/make_default_presets.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.presets import BUILTIN_DIR, save_preset  # noqa: E402
from app.core.settings import Settings  # noqa: E402

PRESETS = {
    # Corrección suave para casi cualquier retrato.
    "Retrato · Natural": {
        "whites": 35, "blacks": -22, "contrast": 12, "highlights": -12, "shadows": 12,
        "vibrance": 22, "clarity": -5, "sharpen": 15,
    },
    # Piel más uniforme y luminosa, contraste local reducido.
    "Retrato · Piel suave": {
        "exposure": 0.12, "whites": 25, "blacks": -10, "contrast": -5, "highlights": -25,
        "shadows": 22, "clarity": -45, "nr_luma": 25, "temperature": 6, "vibrance": 10,
        "hsl_l_orange": 14, "hsl_s_orange": -10, "vignette": -15,
    },
    # Realza ojos verdes/aguamarina y el pelo plateado (ligeramente frío).
    "Retrato · Ojos y plata": {
        "whites": 20, "blacks": -20, "contrast": 12, "clarity": 12, "temperature": -8,
        "hsl_s_green": 30, "hsl_l_green": 10, "hsl_s_aqua": 25, "hsl_s_orange": -5,
        "vibrance": 10, "sharpen": 25,
    },
    # Fondos blancos o claros: luminoso y limpio.
    "Estudio · Clave alta": {
        "exposure": 0.45, "whites": 40, "highlights": -20, "shadows": 30, "blacks": -8,
        "contrast": -8, "saturation": -12, "vibrance": 12, "clarity": -12, "temperature": 3,
    },
    # Fondos oscuros: más profundidad y viñeta.
    "Estudio · Clave baja": {
        "exposure": -0.3, "blacks": -40, "shadows": -15, "highlights": 10, "whites": 30,
        "contrast": 30, "clarity": 18, "vignette": -40, "saturation": -12, "temperature": -6,
    },
    # Blanco y negro: los rojos aclaran la piel, los azules oscurecen el fondo.
    "B/N · Retrato clásico": {
        "bw": 1, "bw_red": 35, "bw_green": -10, "bw_blue": -20, "contrast": 25,
        "whites": 20, "blacks": -25, "clarity": 15, "vignette": -20, "grain": 12, "sharpen": 15,
    },
    # Exteriores (parque, playa, calle): luz cálida de tarde.
    "Exterior · Dorado": {
        "temperature": 25, "tint": 5, "vibrance": 22, "highlights": -20, "shadows": 15,
        "whites": 10, "blacks": -15, "dehaze": 8, "split_high_hue": 42, "split_high_sat": 25,
        "split_shadow_hue": 210, "split_shadow_sat": 8, "vignette": -15,
    },
    # Look de revista: frío, apagado y con sombras azuladas.
    "Editorial · Frío": {
        "temperature": -14, "contrast": 18, "highlights": -20, "whites": 15, "blacks": -20,
        "saturation": -18, "vibrance": 8, "clarity": 10, "split_shadow_hue": 200,
        "split_shadow_sat": 22, "split_high_hue": 35, "split_high_sat": 10, "split_balance": -10,
    },
    # Película mate: negros lavados (curva) y grano.
    "Película · Mate": {
        "curves": {"rgb": [[0, 0.07], [0.25, 0.26], [0.75, 0.77], [1, 0.96]]},
        "saturation": -15, "contrast": 5, "temperature": 4, "split_shadow_hue": 215,
        "split_shadow_sat": 15, "split_high_hue": 40, "split_high_sat": 15,
        "grain": 18, "grain_size": 30, "vignette": -10,
    },
    # Cine: LUT teal & orange incluido, al 55 %.
    "Cine · Teal & orange": {
        "lut_path": "lighteye:teal_orange.cube", "lut_amount": 55, "contrast": 10,
        "highlights": -15, "blacks": -15, "whites": 10, "vignette": -20, "grain": 6,
    },
}


def main():
    BUILTIN_DIR.mkdir(parents=True, exist_ok=True)
    for old in BUILTIN_DIR.glob("*.json"):
        old.unlink()
    for name, values in PRESETS.items():
        settings = Settings(values)
        unknown = set(values) - set(settings.non_default())
        if unknown:
            raise SystemExit(f"{name}: ajustes no válidos o sin efecto: {unknown}")
        print("escrito", save_preset(name, settings, BUILTIN_DIR).path.name)


if __name__ == "__main__":
    main()
