"""Descarga los modelos de IA de Lighteye en la carpeta models/.

Uso:  python tools/download_models.py            (todos los que falten)
      python tools/download_models.py gfpgan     (solo los indicados)
      python tools/download_models.py --list     (ver la lista)
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ai.registry import MODELS, models_dir  # noqa: E402


def download(info) -> None:
    dest = info.path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"Descargando {info.file} (~{info.size_mb} MB) — {info.license}")
    with urllib.request.urlopen(info.url, timeout=60) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done * 100 // total:3d} %", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> None:
    args = sys.argv[1:]
    if "--list" in args:
        for m in MODELS.values():
            state = "✓" if m.available() else "falta"
            print(f"{m.key:<14} {m.size_mb:>5} MB  {state:<6} {m.purpose} — {m.license}")
        return
    wanted = [MODELS[k] for k in args] if args else list(MODELS.values())
    print(f"Carpeta: {models_dir()}")
    for info in wanted:
        if info.available():
            print(f"Ya está: {info.file}")
        else:
            download(info)


if __name__ == "__main__":
    main()
