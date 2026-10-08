"""Ajuste del asignador de memoria de glibc para NumPy.

Por defecto, glibc pide al sistema cada array grande (> ~128 KB) con mmap y
lo devuelve al liberarlo; cada operación de NumPy sobre la vista previa
paga entonces miles de fallos de página. Subir los umbrales hace que esa
memoria se reutilice y acelera el pipeline unas 3 veces.
"""

import ctypes
import ctypes.util

M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3
# Máximo que admite glibc en 64 bits para M_MMAP_THRESHOLD; cubre los arrays
# de la vista previa (1600 px × 3 canales float32 ≈ 20 MB).
MMAP_THRESHOLD = 32 * 1024 * 1024
TRIM_THRESHOLD = 512 * 1024 * 1024


def tune_allocator() -> bool:
    """Devuelve True si se pudo aplicar (solo en Linux con glibc)."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        return False
    ok = mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD)
    ok &= mallopt(M_TRIM_THRESHOLD, TRIM_THRESHOLD)
    return bool(ok)
