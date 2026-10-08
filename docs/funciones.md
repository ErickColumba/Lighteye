# Lighteye — Funciones para un editor de fotos ligero

Objetivo: un editor no destructivo, rápido y nativo en Linux, con lo esencial de un editor tipo Luminar, sin las funciones pesadas de IA en la primera versión.

---

## 1. Núcleo (MVP)

### 1.1 Archivos y biblioteca
| Función | Descripción | Implementación sugerida |
|---|---|---|
| Abrir imagen | JPG, PNG, TIFF, WebP | Pillow / OpenCV |
| Abrir RAW | CR2, CR3, NEF, ARW, DNG, RAF | `rawpy` (LibRaw) |
| Navegador de carpeta | Miniaturas de una carpeta | Caché de miniaturas en `~/.cache/lighteye` |
| Edición no destructiva | El original nunca se modifica | Guardar ajustes en un archivo `.json` junto a la foto (sidecar) |
| Exportar | JPG (calidad), PNG, TIFF 16 bits, cambio de tamaño | Pillow / OpenCV |
| Metadatos | Leer y conservar EXIF al exportar | `piexif` o `exiftool` |

### 1.2 Ajustes de luz
| Función | Rango | Nota técnica |
|---|---|---|
| Exposición | −4 a +4 EV | Multiplicar en espacio lineal: `img * 2^ev` |
| Contraste | −100 a +100 | Curva S alrededor del gris medio |
| Altas luces | −100 a +100 | Máscara por luminancia alta + ajuste tonal |
| Sombras | −100 a +100 | Máscara por luminancia baja |
| Blancos / Negros | −100 a +100 | Mover punto blanco y punto negro |
| Curvas | RGB + canales R, G, B | Interpolación spline → LUT de 256/65536 valores |

### 1.3 Color
| Función | Nota técnica |
|---|---|
| Temperatura / Tinte | Ganancias por canal (balance de blancos); en RAW, usar el WB de LibRaw |
| Saturación | Escalar canal S en HSV/HSL |
| Intensidad (Vibrance) | Saturar más los colores poco saturados y proteger tonos de piel |
| HSL por color | Tono/Saturación/Luminancia para 8 rangos (rojo, naranja, amarillo, verde, aguamarina, azul, morado, magenta) |
| Perfil / LUT | Cargar archivos `.cube` (aplicar con interpolación trilineal) |

### 1.4 Detalle
| Función | Nota técnica |
|---|---|
| Nitidez | Máscara de enfoque (unsharp mask): `img + k * (img − blur(img))` |
| Reducción de ruido | Luminancia y color por separado; `cv2.fastNlMeansDenoisingColored` o bilateral |
| Claridad / Estructura | Contraste local: unsharp mask con radio grande (20–50 px) |
| Neblina (Dehaze) | Dark Channel Prior simplificado |

### 1.5 Geometría
| Función | Nota técnica |
|---|---|
| Recortar | Con proporciones fijas (1:1, 4:5, 3:2, 16:9, libre) |
| Rotar / Enderezar | Rotación libre con recorte automático |
| Voltear | Horizontal / vertical |
| Corrección de lente básica | Viñeteo y distorsión con perfiles de `lensfun` |

### 1.6 Efectos básicos
| Función | Nota técnica |
|---|---|
| Viñeta | Máscara radial con cantidad, tamaño y suavidad |
| Grano | Ruido gaussiano escalado por luminancia |
| Blanco y negro | Mezcla de canales configurable |
| Tono dividido | Color distinto para sombras y luces |

### 1.7 Interfaz y flujo de trabajo
| Función | Descripción |
|---|---|
| Vista previa en tiempo real | Procesar una versión reducida al tamaño de pantalla; la resolución completa solo al exportar |
| Antes / Después | Tecla rápida o divisor deslizante |
| Historial | Deshacer / rehacer (Ctrl+Z / Ctrl+Shift+Z) |
| Histograma | RGB + luminancia, con aviso de recorte |
| Presets | Guardar y aplicar conjuntos de ajustes (`.json`) |
| Copiar / pegar ajustes | Aplicar la misma edición a varias fotos |
| Exportación por lotes | Varias fotos con el mismo preset |

---

## 2. Fase 2 — Ajustes locales

| Función | Nota técnica |
|---|---|
| Máscara de pincel | Pintar zona con tamaño, dureza y opacidad |
| Degradado lineal | Para cielos y horizontes |
| Degradado radial | Para resaltar al sujeto |
| Máscara por luminancia / color | Seleccionar por rango tonal o de color |
| Ajustes por máscara | Cada máscara con su propia exposición, color, nitidez, etc. |
| Pincel corrector | Eliminar manchas: `cv2.inpaint` |

---

## 3. Fase 3 — IA opcional (usando la GPU NVIDIA)

Se activan solo si hay GPU con CUDA; la app debe funcionar sin ellas.

| Función | Modelo / técnica sugerida |
|---|---|
| Mejora automática | Algoritmo de autoexposición + autocontraste basado en histograma (sin IA) |
| Selección de sujeto | Segmentación con modelo ligero (por ejemplo rembg / U²-Net) |
| Selección de cielo | Segmentación semántica ligera |
| Reemplazo de cielo | Máscara de cielo + mezcla con imagen nueva y ajuste de color |
| Eliminación de objetos | Inpainting con modelo tipo LaMa |
| Retoque de piel | Detección de rostro + suavizado selectivo (bilateral) |
| Escalado (upscale) | Real-ESRGAN |

---

## 4. Orden del pipeline de procesamiento

El orden importa para que el resultado sea predecible:

1. Decodificar RAW / cargar imagen → convertir a **float32 lineal**
2. Balance de blancos
3. Exposición
4. Corrección de lente (viñeteo, distorsión)
5. Reducción de ruido
6. Altas luces / sombras / blancos / negros
7. Contraste y curvas
8. Color: HSL, saturación, intensidad, LUT
9. Claridad / estructura / neblina
10. Ajustes locales (máscaras)
11. Efectos (viñeta, grano, B/N)
12. Recorte y rotación
13. Nitidez de salida
14. Convertir a sRGB 8/16 bits y exportar

---

## 5. Stack técnico sugerido (Linux)

| Componente | Opción ligera |
|---|---|
| Lenguaje | Python 3.13 (o Rust si se busca máximo rendimiento) |
| Interfaz | PySide6 (Qt, se integra con KDE Plasma) |
| Procesamiento | NumPy + OpenCV |
| RAW | rawpy |
| Aceleración GPU (opcional) | CuPy o PyTorch con CUDA |
| Formato de ajustes | JSON sidecar |
| Empaquetado | AppImage o PKGBUILD para AUR |

---

## 6. Fuera de alcance (para mantenerlo ligero)

- Catálogo con base de datos y álbumes complejos
- Capas tipo Photoshop
- Sincronización en la nube
- Marketplace de presets
- Generación de imágenes con IA
