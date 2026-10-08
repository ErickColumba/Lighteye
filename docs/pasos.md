# Lighteye — Pasos para construir la aplicación

Guía en orden, pensada para CachyOS con shell fish. Cada etapa termina con algo que ya funciona; no pases a la siguiente sin cumplir su "Listo cuando".

---

## Etapa 0 — Preparar el entorno

```fish
sudo pacman -S --needed git
mkdir -p ~/proyectos/lighteye
cd ~/proyectos/lighteye
git init
python3.13 -m venv .venv          # si no tienes python3.13: paru -S python313
source .venv/bin/activate.fish
pip install pyside6 numpy opencv-python rawpy pillow piexif
pip freeze > requirements.txt
```

Se usa Python 3.13 porque algunas librerías (rawpy, opencv) pueden no tener aún versión para el 3.14 del sistema.

**Listo cuando:** `python -c "import PySide6, cv2, rawpy"` no da error.

---

## Etapa 1 — Estructura del proyecto

```
lighteye/
├── main.py                 # punto de entrada
├── app/
│   ├── ui/
│   │   ├── main_window.py  # ventana principal
│   │   ├── viewer.py       # visor de imagen con zoom
│   │   ├── panels.py       # sliders de ajustes
│   │   └── histogram.py
│   ├── core/
│   │   ├── loader.py       # abrir JPG/PNG/TIFF/RAW
│   │   ├── pipeline.py     # aplica los ajustes en orden
│   │   ├── adjustments.py  # una función por ajuste
│   │   ├── settings.py     # estructura de ajustes + sidecar JSON
│   │   └── exporter.py
│   └── presets/
└── tests/
```

**Listo cuando:** la estructura existe y `main.py` abre una ventana vacía de PySide6.

---

## Etapa 2 — Abrir y mostrar una foto

1. `loader.py`: cargar JPG/PNG con OpenCV y RAW con rawpy; convertir siempre a **float32 en rango 0–1**.
2. `viewer.py`: mostrar la imagen ajustada a la ventana, con zoom (rueda del ratón) y desplazamiento.
3. Crear una **versión de vista previa** reducida al tamaño de pantalla (por ejemplo, lado largo de 1600 px). Todos los ajustes se calculan sobre ella para que sea fluido.
4. Menú Archivo → Abrir.

**Listo cuando:** abres un JPG y un RAW de tu cámara y se ven correctamente.

---

## Etapa 3 — Motor de ajustes (pipeline)

1. `settings.py`: una clase/diccionario con todos los valores por defecto (exposición 0, contraste 0, etc.).
2. `adjustments.py`: una función pura por ajuste, cada una recibe `(imagen, valor)` y devuelve la imagen.
3. `pipeline.py`: aplica las funciones en el orden definido en el documento de funciones (balance de blancos → exposición → tonos → color → detalle → efectos).
4. Implementar primero solo: **exposición, contraste, temperatura, saturación**.
5. Escribir pruebas en `tests/` (por ejemplo, exposición 0 no cambia la imagen; +1 EV duplica valores lineales).

**Listo cuando:** las pruebas pasan con `python -m pytest`.

---

## Etapa 4 — Panel de ajustes en la interfaz

1. `panels.py`: un slider por ajuste, agrupados en secciones plegables (Luz, Color, Detalle, Efectos).
2. Al mover un slider: actualizar `settings` → recalcular la vista previa → refrescar el visor.
3. Ejecutar el procesamiento en un **hilo aparte** (`QThread` o `QThreadPool`) y cancelar el cálculo anterior si llega uno nuevo, para que la interfaz no se congele.
4. Doble clic en un slider = volver a su valor por defecto.

**Listo cuando:** mover cualquier slider actualiza la foto sin trabarse.

---

## Etapa 5 — Completar los ajustes del MVP

Añadir de uno en uno, con su prueba y su slider:

1. Altas luces, sombras, blancos, negros
2. Curvas (widget de curva con puntos arrastrables)
3. Intensidad (vibrance) y HSL por color
4. Nitidez, reducción de ruido, claridad
5. Neblina (dehaze)
6. Viñeta, grano, blanco y negro
7. Carga de LUT `.cube`

Haz un commit de git después de cada ajuste que funcione.

**Listo cuando:** todos los ajustes del MVP funcionan sobre la vista previa.

---

## Etapa 6 — Herramientas básicas

1. **Histograma** en tiempo real (calculado sobre la vista previa).
2. **Antes / Después** (tecla `\`).
3. **Historial** de deshacer/rehacer: guardar copias de `settings` (no de imágenes).
4. **Recorte y rotación** con proporciones fijas y enderezado.

**Listo cuando:** puedes editar una foto completa y deshacer cualquier paso.

---

## Etapa 7 — Guardar y exportar

1. **Sidecar JSON**: al cambiar ajustes, guardar `foto.jpg.json` con los valores; al abrir la foto, cargarlo si existe.
2. **Exportar**: procesar la imagen a **resolución completa** con el mismo pipeline, convertir a sRGB y guardar en JPG (calidad), PNG o TIFF 16 bits.
3. Conservar los metadatos EXIF.
4. Mostrar una barra de progreso durante la exportación.

**Listo cuando:** exportas una foto RAW y el resultado coincide con la vista previa.

---

## Etapa 8 — Presets y lotes

1. Guardar los ajustes actuales como preset en `~/.config/lighteye/presets/`.
2. Lista de presets con miniatura de vista previa.
3. Copiar / pegar ajustes entre fotos.
4. Navegador de carpeta con miniaturas y exportación por lotes.

**Listo cuando:** aplicas un preset a 20 fotos y las exportas de una vez.

---

## Etapa 9 — Rendimiento

1. Medir qué ajuste es más lento (`time.perf_counter`).
2. Precalcular **LUTs** para ajustes tonales (curvas, contraste) en vez de calcular píxel a píxel.
3. Evitar copias innecesarias de arrays de NumPy.
4. Opcional: mover las operaciones más pesadas a la GPU con CuPy si tienes CUDA.

**Listo cuando:** la vista previa responde en menos de ~100 ms por cambio de slider.

---

## Etapa 10 — Empaquetar para Linux

1. Crear un archivo `.desktop` e icono para que aparezca en el menú de KDE.
2. Empaquetar con **PyInstaller** o crear un **AppImage**.
3. Opcional: escribir un `PKGBUILD` para instalarlo con `makepkg -si` o publicarlo en AUR.

**Listo cuando:** la app se abre desde el menú de aplicaciones sin activar el entorno virtual.

---

## Etapas siguientes (después del MVP)

- **Fase 2:** máscaras de pincel, degradado lineal y radial, máscara por luminancia/color, pincel corrector.
- **Fase 3:** funciones con IA usando la GPU (selección de sujeto y cielo, reemplazo de cielo, borrar objetos, escalado con Real-ESRGAN).

---

## Consejos

- Trabaja siempre en **float32 lineal** y convierte a sRGB solo al mostrar o exportar; evita bandas y colores raros.
- Una función por ajuste, sin estado: facilita probar y reordenar.
- La interfaz nunca procesa la imagen directamente; solo cambia `settings` y pide un nuevo cálculo.
- Usa fotos de prueba variadas (contraluz, noche, retrato, RAW de tu cámara).
