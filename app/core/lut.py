"""Lectura y aplicación de LUT en formato .cube (Adobe / Resolve).

Admite LUT 3D (LUT_3D_SIZE) con interpolación trilineal y LUT 1D
(LUT_1D_SIZE). Se aplican en espacio sRGB, que es lo que esperan casi todas
las LUT creativas.

Lighteye trae algunos LUT de ejemplo en app/luts/. En los ajustes se guardan
como "lighteye:nombre.cube" (no como ruta absoluta), para que sigan
funcionando aunque la aplicación se mueva de carpeta.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CubeLUT:
    title: str
    size: int
    is_3d: bool
    table: np.ndarray  # 3D: (N, N, N, 3) indexada [b, g, r]; 1D: (N, 3)
    domain_min: np.ndarray
    domain_max: np.ndarray


def parse_cube(text: str) -> CubeLUT:
    title = ""
    size = None
    is_3d = True
    dmin = np.zeros(3, np.float32)
    dmax = np.ones(3, np.float32)
    rows = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, rest = line.partition(" ")
        key = key.upper()
        if key == "TITLE":
            title = rest.strip().strip('"')
        elif key == "LUT_3D_SIZE":
            size, is_3d = int(rest), True
        elif key == "LUT_1D_SIZE":
            size, is_3d = int(rest), False
        elif key == "DOMAIN_MIN":
            dmin = np.array(rest.split(), np.float32)
        elif key == "DOMAIN_MAX":
            dmax = np.array(rest.split(), np.float32)
        elif key in ("LUT_3D_INPUT_RANGE", "LUT_1D_INPUT_RANGE"):
            lo, hi = map(float, rest.split())
            dmin, dmax = np.full(3, lo, np.float32), np.full(3, hi, np.float32)
        elif key[0].isdigit() or key[0] in "-.":
            rows.append(line.split())
        # Otras palabras clave (LUT_IN_VIDEO_RANGE, etc.) se ignoran.

    if size is None or size < 2:
        raise ValueError("El archivo no indica LUT_3D_SIZE ni LUT_1D_SIZE")
    expected = size**3 if is_3d else size
    if len(rows) != expected:
        raise ValueError(f"Se esperaban {expected} filas de datos y hay {len(rows)}")
    try:
        data = np.array(rows, dtype=np.float32)
    except ValueError as exc:
        raise ValueError("Datos numéricos inválidos en el LUT") from exc
    if data.shape[1] != 3:
        raise ValueError("Cada fila del LUT debe tener 3 valores")

    # En .cube el rojo varía más rápido, luego el verde y luego el azul.
    table = data.reshape(size, size, size, 3) if is_3d else data
    return CubeLUT(title, size, is_3d, table, dmin, dmax)


BUILTIN_PREFIX = "lighteye:"
BUILTIN_DIR = Path(__file__).resolve().parent.parent / "luts"


def resolve_path(path: str | Path) -> Path:
    """Ruta real de un LUT (traduce los "lighteye:…" incluidos)."""
    text = str(path)
    if text.startswith(BUILTIN_PREFIX):
        return BUILTIN_DIR / Path(text[len(BUILTIN_PREFIX):]).name
    return Path(text)


def _title(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for _ in range(20):
                line = f.readline()
                if line.upper().startswith("TITLE"):
                    return line.partition(" ")[2].strip().strip('"') or path.stem
    except OSError:
        pass
    return path.stem


def builtin_luts() -> list[tuple[str, str]]:
    """[(título, "lighteye:archivo.cube")] de los LUT incluidos, por título."""
    if not BUILTIN_DIR.is_dir():
        return []
    items = [(_title(f), BUILTIN_PREFIX + f.name) for f in BUILTIN_DIR.glob("*.cube")]
    return sorted(items, key=lambda item: item[0].lower())


def display_name(path: str | Path) -> str:
    """Nombre para mostrar: el título de un LUT incluido o el nombre del archivo."""
    if str(path).startswith(BUILTIN_PREFIX):
        return _title(resolve_path(path))
    return Path(path).name


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float) -> CubeLUT:
    return parse_cube(Path(path).read_text(encoding="utf-8", errors="replace"))


def load_cube(path: str | Path) -> CubeLUT:
    path = resolve_path(path)
    if not path.is_file():
        raise ValueError(f"No se encontró el LUT: {path.name}")
    return _load_cached(str(path), path.stat().st_mtime)


def apply_cube(lut: CubeLUT, rgb: np.ndarray) -> np.ndarray:
    """Aplica el LUT a una imagen sRGB (H, W, 3) float32 en 0–1."""
    x = (rgb - lut.domain_min) / (lut.domain_max - lut.domain_min)
    x = np.clip(x, 0.0, 1.0) * np.float32(lut.size - 1)

    if not lut.is_3d:
        grid = np.arange(lut.size, dtype=np.float32)
        out = np.empty_like(rgb)
        for c in range(3):
            out[..., c] = np.interp(x[..., c], grid, lut.table[:, c])
        return out

    # Trilineal canal por canal, en float32 y con np.take sobre tablas 1D
    # contiguas: es mucho más rápido que indexar la tabla (N³, 3) completa.
    n = lut.size
    i0 = np.minimum(x.astype(np.int32), n - 2)
    f = x - i0.astype(np.float32)
    r, g, b = (np.ascontiguousarray(i0[..., c]) for c in range(3))
    fr, fg, fb = (np.ascontiguousarray(f[..., c]) for c in range(3))
    base = (b * n + g) * n + r
    offsets = [(db * n + dg) * n + dr for db in (0, 1) for dg in (0, 1) for dr in (0, 1)]
    idx = [base + o for o in offsets]

    out = np.empty(rgb.shape, np.float32)
    flat = lut.table.reshape(-1, 3)
    for c in range(3):
        t = np.ascontiguousarray(flat[:, c])
        v = [np.take(t, i) for i in idx]  # esquinas en el orden (b, g, r)
        c00 = v[0] + (v[1] - v[0]) * fr
        c01 = v[2] + (v[3] - v[2]) * fr
        c10 = v[4] + (v[5] - v[4]) * fr
        c11 = v[6] + (v[7] - v[6]) * fr
        c0 = c00 + (c01 - c00) * fg
        c1 = c10 + (c11 - c10) * fg
        out[..., c] = c0 + (c1 - c0) * fb
    return out
