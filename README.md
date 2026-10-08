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
