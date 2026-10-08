"""Compila Lighteye en una carpeta lista para usar y la empaqueta.

Uso:  python packaging/build.py --variant gpu|cpu [--archive]

1. PyInstaller genera dist/Lighteye/ (ejecutable + librerías).
2. Se copian los modelos de IA a dist/Lighteye/models/ (se descargan si faltan).
3. Con --archive se crea dist/Lighteye-<versión>-<sistema>-<variante>.7z, partido
   en trozos de menos de 2 GB (.7z.001, .002…) si hace falta: GitHub no admite
   archivos más grandes.
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_DIR = DIST / "Lighteye"
PART_SIZE = "1900m"  # menos de 2 GB por trozo


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def system_name() -> str:
    return {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(platform.system(), "otro")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["gpu", "cpu"], required=True,
                        help="gpu = PyTorch con CUDA (NVIDIA); cpu = solo procesador (más ligero)")
    parser.add_argument("--archive", action="store_true", help="crear el .7z (partido si hace falta)")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from app import __version__

    # 1. Compilar
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(DIST), "--workpath", str(ROOT / "build"),
         str(ROOT / "packaging" / "lighteye.spec")])

    # 2. Modelos de IA junto al ejecutable
    run([sys.executable, str(ROOT / "tools" / "download_models.py")])
    target = APP_DIR / "models"
    target.mkdir(exist_ok=True)
    for model in (ROOT / "models").iterdir():
        if model.is_file() and not model.name.endswith(".part"):
            shutil.copy2(model, target / model.name)

    # Licencias y notas
    shutil.copy2(ROOT / "packaging" / "LEEME.txt", APP_DIR / "LEEME.txt")
    (APP_DIR / "VARIANTE.txt").write_text(
        f"Lighteye {__version__} · {system_name()} · "
        + ("GPU NVIDIA (CUDA) + procesador" if args.variant == "gpu" else "solo procesador") + "\n",
        encoding="utf-8")
    if system_name() == "linux":
        for name in ("install.sh", "uninstall.sh"):
            shutil.copy2(ROOT / "packaging" / "linux" / name, APP_DIR / name)
            (APP_DIR / name).chmod(0o755)
        shutil.copy2(ROOT / "packaging" / "lighteye.png", APP_DIR / "lighteye.png")
    elif system_name() == "windows":
        shutil.copy2(ROOT / "packaging" / "windows" / "Crear accesos directos.bat", APP_DIR)

    size = sum(f.stat().st_size for f in APP_DIR.rglob("*") if f.is_file())
    print(f"Carpeta lista: {APP_DIR} ({size / 1e9:.1f} GB)")

    # 3. Archivo comprimido (partido en trozos < 2 GB)
    if args.archive:
        name = f"Lighteye-{__version__}-{system_name()}-{args.variant}.7z"
        for old in DIST.glob(name + "*"):
            old.unlink()
        seven = shutil.which("7z") or shutil.which("7za") or r"C:\Program Files\7-Zip\7z.exe"
        run([seven, "a", "-mx=5", "-mmt=on", f"-v{PART_SIZE}", str(DIST / name), str(APP_DIR)])
        parts = sorted(DIST.glob(name + "*"))
        print("Archivos:", *[f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" for p in parts], sep="\n  ")


if __name__ == "__main__":
    main()
