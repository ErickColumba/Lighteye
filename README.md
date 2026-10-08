# Lighteye

Editor de fotos ligero, no destructivo y nativo en Linux.

- Diseño y funciones: [docs/funciones.md](docs/funciones.md)
- Plan de construcción por etapas: [docs/pasos.md](docs/pasos.md)

## Instalación (CachyOS / fish)

```fish
python3.13 -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

## Uso

```fish
python main.py              # abre la aplicación
python main.py foto.jpg     # abre directamente una foto
python -m pytest            # ejecuta las pruebas
python tools/bench_pipeline.py          # tiempo de cada ajuste
python tools/bench_pipeline.py --drag   # fluidez al arrastrar sliders
```

## Modelos de IA

Van en `models/` (no están en el repositorio por su tamaño):

```fish
python tools/download_models.py          # descarga los que falten (~1,6 GB)
python tools/download_models.py --list   # ver estado y licencias
```

## Compilar e instalar

```fish
python packaging/build.py --variant gpu   # o cpu; crea dist/Lighteye/
python packaging/build.py --variant gpu --archive   # además, el .7z en trozos < 2 GB
dist/Lighteye/Lighteye --self-test        # comprueba que el paquete está completo
dist/Lighteye/install.sh                  # lo añade al menú de aplicaciones (Linux)
```

Las versiones para **Linux y Windows** (variantes `gpu` y `cpu`) se compilan en
GitHub Actions (`.github/workflows/compilar.yml`): pestaña *Actions → Compilar
Lighteye → Run workflow*, o al subir una etiqueta `v*`. Cada paquete se
comprueba con `--self-test` y se publica en *Releases*.
