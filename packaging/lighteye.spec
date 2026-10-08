# Receta de PyInstaller para Lighteye (Linux y Windows).
# Se usa desde packaging/build.py; no hace falta llamarla a mano.
# ruff: noqa

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
WINDOWS = sys.platform.startswith("win")

datas = [
    (str(ROOT / "app" / "luts"), "app/luts"),
    (str(ROOT / "app" / "presets"), "app/presets"),
    (str(ROOT / "app" / "resources"), "app/resources"),
    (str(ROOT / "app" / "ai" / "vendor" / "birefnet" / "LICENSE"), "app/ai/vendor/birefnet"),
]
datas += collect_data_files("timm")

hiddenimports = []
binaries = []
# torchvision tiene operadores compilados (_C) que PyInstaller no detecta solo;
# sin ellos fallan Real-ESRGAN, GFPGAN, CodeFormer, LaMa y BiRefNet.
tv_datas, tv_binaries, tv_hidden = collect_all("torchvision")
datas += tv_datas
binaries += tv_binaries
hiddenimports += tv_hidden
# Las arquitecturas de spandrel y BiRefNet se importan al usar la IA.
for package in ("spandrel", "spandrel_extra_arches", "app"):
    hiddenimports += collect_submodules(package)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    # triton: compilador de PyTorch (torch.compile) que Lighteye no usa; ~0.9 GB.
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "transformers", "tensorboard", "triton"],
    noarchive=False,
    # TorchScript (@torch.jit.script) necesita leer el código fuente .py de estas
    # librerías al cargar CodeFormer y BiRefNet: se incluyen también como .py.
    module_collection_mode={
        "kornia": "pyz+py",
        "spandrel": "pyz+py",
        "spandrel_extra_arches": "pyz+py",
        "timm": "pyz+py",
        "app.ai.vendor": "pyz+py",
    },
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Lighteye",
    console=False,  # sin ventana de terminal
    icon=str(ROOT / "packaging" / "lighteye.ico") if WINDOWS else None,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Lighteye", upx=False)
