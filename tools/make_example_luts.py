"""Genera los LUT de ejemplo que vienen con Lighteye (app/luts/*.cube).

Uso:  python tools/make_example_luts.py
"""

from pathlib import Path

import numpy as np

N = 17  # tamaño estándar para LUT ligeros; los efectos son suaves
OUT = Path(__file__).resolve().parent.parent / "app" / "luts"
LUMA = np.array([0.2126, 0.7152, 0.0722])


def luma(c):
    return c @ LUMA


def teal_orange(c):
    l = luma(c)
    shadows = np.array([-0.14, 0.05, 0.16]) * (1 - l) ** 1.5  # sombras verde azuladas
    lights = np.array([0.18, 0.06, -0.14]) * l ** 1.5  # luces anaranjadas
    s = l + (c - l) * 1.25
    return l + (s - l) + shadows * (1 - l) + lights * l + (shadows + lights) * 0.5


def calido(c):
    return c * np.array([1.10, 1.02, 0.82]) + np.array([0.03, 0.015, 0.0])


def pelicula_desvaida(c):
    l = luma(c)
    c = l + (c - l) * 0.75  # menos saturación
    c = 0.06 + c * 0.88  # negros lavados y blancos suaves
    return c + np.array([0.02, 0.0, -0.02]) * (1 - l)


def blanco_negro_contraste(c):
    l = c @ np.array([0.35, 0.55, 0.10])
    l = l * l * (3 - 2 * l)  # curva S
    return np.array([l, l, l])


def noche_azul(c):
    l = luma(c)
    return np.array([l * 0.7, l * 0.85, l * 1.15]) * 0.85 + (c - l) * 0.3


LUTS = [
    ("teal_orange.cube", "Cine (teal & orange)", teal_orange),
    ("calido.cube", "Cálido", calido),
    ("pelicula_desvaida.cube", "Película desvaída", pelicula_desvaida),
    ("blanco_negro_contraste.cube", "Blanco y negro contrastado", blanco_negro_contraste),
    ("noche_azul.cube", "Noche azul", noche_azul),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, N)
    for name, title, fn in LUTS:
        lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {N}"]
        for b in grid:
            for g in grid:
                for r in grid:  # en .cube el rojo varía más rápido
                    lines.append("%.5f %.5f %.5f" % tuple(np.clip(fn(np.array([r, g, b])), 0, 1)))
        (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("escrito", OUT / name)


if __name__ == "__main__":
    main()
