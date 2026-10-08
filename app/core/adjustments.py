"""Una función pura por ajuste.

Cada función recibe una imagen RGB lineal float32 (H, W, 3) y su valor, y
devuelve una imagen nueva sin modificar la de entrada. Los valores pueden
superar 1.0 (luces por encima del blanco) hasta el final del pipeline.
"""

from functools import lru_cache

import cv2
import numpy as np

from app.core import curves as curve_math
from app.core.color import (
    LUMA,
    LUT_SIZE,
    linear_to_srgb,
    lut_domain,
    lut_index,
    srgb_to_linear,
    to_linear_fast,
    to_srgb_fast,
)
from app.core.loader import PREVIEW_LONG_SIDE
from app.core.settings import HSL_COLORS

# Gris medio (18 %) en lineal: pivote para el contraste.
MID_GREY = 0.18
# Gamma perceptual aproximada usada para que las curvas actúen "como se ven".
PERCEPTUAL_GAMMA = 2.2


def white_balance(img: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    """Temperatura (−100 frío … +100 cálido) y tinte (−100 verde … +100 magenta).

    Ganancias por canal normalizadas para no cambiar la luminancia global.
    """
    t = temperature / 100.0
    g = tint / 100.0
    gains = np.array(
        [2.0 ** (0.5 * t), 2.0 ** (-0.35 * g), 2.0 ** (-0.5 * t)], dtype=np.float32
    )
    gains /= float(gains @ LUMA)
    return cv2.transform(img, np.diag(gains))


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    """Exposición en pasos (EV): multiplicar en lineal por 2^ev."""
    return img * np.float32(2.0 ** ev)


# Rango de luminancia lineal que cubren los ajustes tonales: hasta 2 pasos
# por encima del blanco, para poder recuperar altas luces (sobre todo en RAW).
TONE_HEADROOM = 4.0


def _tone_curve(p: np.ndarray, h: float, s: float, w: float, b: float) -> np.ndarray:
    """Curva tonal en espacio perceptual. h, s, w, b van de −1 a 1.

    Cada término es una "joroba" que vale 0 en los extremos de su zona, así
    que con todo en 0 la curva es la identidad. Los factores están elegidos
    para que la curva sea siempre creciente (sin inversiones de tono).
    """
    inside = np.clip(p, 0.0, 1.0)
    q = p.copy()
    q += s * 0.6 * inside * (1.0 - inside) ** 3          # sombras: zona ~25 %
    if h > 0:
        q += h * 0.6 * inside ** 3 * (1.0 - inside)      # altas luces: zona ~75 %
    q += np.where(p <= 1.0, w * 0.1 * p ** 4, w * (0.1 + 0.4 * (p - 1.0)))  # blancos
    q += b * 0.1 * (1.0 - inside) ** 4                   # negros
    if h < 0:
        # Recuperación: comprime suavemente todo lo que está por encima de k
        # (Reinhard extendido: el máximo del rango pasa a valer 1).
        k = 0.5
        p_max = TONE_HEADROOM ** (1 / PERCEPTUAL_GAMMA)
        white = (p_max - k) / (1.0 - k)
        x = np.maximum(q - k, 0.0) / (1.0 - k)
        rolled = k + (1.0 - k) * x * (1.0 + x / white**2) / (1.0 + x)
        q = np.where(q > k, q + (rolled - q) * -h, q)
    return q


@lru_cache(maxsize=8)
def _tone_gain_lut(h: float, s: float, w: float, b: float) -> np.ndarray:
    """Ganancia (salida / entrada) para cada luminancia lineal de 0 a TONE_HEADROOM."""
    lum = np.linspace(0.0, TONE_HEADROOM, LUT_SIZE)
    p = lum ** (1 / PERCEPTUAL_GAMMA)
    out = np.maximum(_tone_curve(p, h, s, w, b), 0.0) ** PERCEPTUAL_GAMMA
    gain = out / np.maximum(lum, lum[1])
    gain[0] = gain[1]
    return gain.astype(np.float32)


def tones(img: np.ndarray, highlights: float, shadows: float, whites: float,
          blacks: float) -> np.ndarray:
    """Altas luces, sombras, blancos y negros (−100 … +100).

    Se calcula sobre la luminancia y se aplica como ganancia a los tres
    canales, así el tono y la saturación relativa de cada color se conservan.
    """
    gain = _tone_gain_lut(highlights / 100, shadows / 100, whites / 100, blacks / 100)
    lum = cv2.transform(img, LUMA[None, :])
    idx = np.clip(lum * ((LUT_SIZE - 1) / TONE_HEADROOM) + 0.5, 0, LUT_SIZE - 1)
    return img * gain[idx.astype(np.uint16)][..., None]


@lru_cache(maxsize=8)
def _contrast_lut(amount: float) -> np.ndarray:
    k = 1.0 + amount / 100.0 * 0.8  # 0.2 … 1.8
    pivot = MID_GREY ** (1 / PERCEPTUAL_GAMMA)
    p = lut_domain() ** (1 / PERCEPTUAL_GAMMA)
    low = pivot * (p / pivot) ** k
    high = 1.0 - (1.0 - pivot) * ((1.0 - p) / (1.0 - pivot)) ** k
    return (np.where(p < pivot, low, high) ** PERCEPTUAL_GAMMA).astype(np.float32)


def contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """Curva S (amount > 0) o aplanado (amount < 0) alrededor del gris medio.

    Se aplica en espacio perceptual con una curva de potencia a cada lado del
    pivote: es monótona, deja fijos 0, el gris medio y 1, y es la identidad
    cuando amount = 0. Los valores por encima de 1 no se tocan.
    """
    out = _contrast_lut(float(amount))[lut_index(img)]
    # Lo que pasa de 1.0 se suma tal cual: la curva vale 1 en 1, así que es continuo.
    out += np.maximum(img - 1.0, 0.0)
    return out


@lru_cache(maxsize=8)
def _curve_luts(master: tuple, red: tuple, green: tuple, blue: tuple) -> list[np.ndarray]:
    """Una LUT lineal → lineal por canal: curva del canal ∘ curva maestra."""
    p = linear_to_srgb(lut_domain())
    idx = lambda v: np.clip(v * (LUT_SIZE - 1) + 0.5, 0, LUT_SIZE - 1).astype(np.int64)
    after_master = curve_math.sample(master, LUT_SIZE)[idx(p)]
    return [srgb_to_linear(curve_math.sample(ch, LUT_SIZE)[idx(after_master)])
            for ch in (red, green, blue)]


def curves(img: np.ndarray, curves: dict) -> np.ndarray:
    """Curvas RGB (maestra) y por canal, definidas en espacio sRGB (como se ven)."""
    luts = _curve_luts(curves["rgb"], curves["r"], curves["g"], curves["b"])
    idx = lut_index(img)
    out = np.empty_like(img)
    for c in range(3):
        out[..., c] = luts[c][idx[..., c]]
    # Por encima del blanco se conserva el exceso, como en el contraste.
    out += np.maximum(img - 1.0, 0.0)
    return out


def saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """−100 = blanco y negro, +100 = doble de saturación. Conserva la luminancia."""
    # lum + (img − lum)·f es lineal en RGB: se aplica como una matriz 3×3.
    f = 1.0 + amount / 100.0
    matrix = (f * np.eye(3) + (1.0 - f) * LUMA[None, :]).astype(np.float32)
    out = cv2.transform(img, matrix)
    return np.maximum(out, 0.0, out=out)


def vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Satura más los colores apagados y protege los tonos de piel."""
    r, g, b = cv2.split(img)  # canales contiguos: mucho más rápido que img[..., c]
    mx = cv2.max(cv2.max(r, g), b)
    mn = cv2.min(cv2.min(r, g), b)
    sat = (mx - mn) / (mx + 1e-6)

    # Tonos de piel: rojo dominante, azul mínimo y tono entre ~10° y ~50°.
    skin_hue = 60.0 * (g - b) / (r - b + 1e-6)
    skin = np.clip(1.0 - np.abs(skin_hue - 28.0) * (1 / 22.0), 0.0, 1.0)
    skin *= (r >= g) & (g >= b) & (sat > 0.1)

    weight = (1.0 - sat) * (1.0 - 0.7 * skin)
    factor = 1.0 + np.float32(amount / 100.0) * weight
    lum = cv2.transform(img, LUMA[None, :])
    out = cv2.merge([lum + (c - lum) * factor for c in (r, g, b)])
    return np.maximum(out, 0.0, out=out)


HUE_BINS = 1440  # resolución de 0.25° para las tablas por tono
MAX_HUE_SHIFT = 30.0  # grados con el slider a ±100


@lru_cache(maxsize=4)
def _hsl_luts(values: tuple) -> np.ndarray:
    """(3, HUE_BINS): desplazamiento de tono, saturación y luminancia por tono.

    Entre los centros de cada color se interpola linealmente, de forma
    circular (después del magenta vuelve el rojo).
    """
    centers = np.array([c for _, _, c in HSL_COLORS] + [360.0])
    hue = np.arange(HUE_BINS) * (360.0 / HUE_BINS)
    n = len(HSL_COLORS)
    luts = []
    for kind in range(3):
        v = np.array(values[kind * n:(kind + 1) * n], dtype=np.float64) / 100.0
        luts.append(np.interp(hue, centers, np.append(v, v[0])))
    return np.array(luts, dtype=np.float32)


def hsl(img: np.ndarray, *values: float) -> np.ndarray:
    """Tono, saturación y luminancia por rango de color (24 valores, −100 … +100).

    Orden de los valores: los 8 tonos, las 8 saturaciones y las 8 luminancias
    (ver settings.HSL_KEYS). Se trabaja en sRGB, que es como se perciben los
    colores; lo que pasa de 1.0 se conserva aparte.
    """
    dh, ds, dl = _hsl_luts(tuple(values))
    hls = cv2.cvtColor(to_srgb_fast(img), cv2.COLOR_RGB2HLS)
    h, light, s = hls[..., 0], hls[..., 1], hls[..., 2]
    idx = (h * (HUE_BINS / 360.0)).astype(np.int32) % HUE_BINS

    # Los grises no tienen tono: el cambio de luminancia se pondera por la saturación.
    hls[..., 0] = (h + dh[idx] * MAX_HUE_SHIFT) % 360.0
    hls[..., 2] = np.clip(s * (1.0 + ds[idx]), 0.0, 1.0)
    hls[..., 1] = np.clip(light * (1.0 + 0.5 * dl[idx] * s), 0.0, 1.0)

    out = to_linear_fast(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB))
    out += np.maximum(img - 1.0, 0.0)
    return out


# --- Detalle ----------------------------------------------------------------
#
# Los radios se expresan en píxeles de la vista previa (lado largo de 1600 px)
# y se escalan con el tamaño real, para que la exportación a resolución
# completa se vea igual que la vista previa.


def _px(img: np.ndarray, base: float) -> float:
    # Las imágenes menores que la vista previa no se reducen: usan el radio base.
    return base * max(1.0, max(img.shape[:2]) / PREVIEW_LONG_SIDE)


def fast_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano; para radios grandes se calcula en una versión
    reducida y se vuelve a ampliar (mucho más rápido y casi idéntico)."""
    if sigma <= 6:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    f = sigma / 3.0
    small = cv2.resize(x, (max(1, round(w / f)), max(1, round(h / f))),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _perceptual_luma(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(luminancia lineal, luminancia perceptual) de la imagen."""
    lum = np.maximum(cv2.transform(img, LUMA[None, :]), 0.0)
    return lum, lum ** (1 / PERCEPTUAL_GAMMA)


def _apply_luma(img: np.ndarray, lum: np.ndarray, new_p: np.ndarray) -> np.ndarray:
    """Aplica una nueva luminancia perceptual como ganancia sobre RGB."""
    new_lum = np.maximum(new_p, 0.0) ** PERCEPTUAL_GAMMA
    gain = np.minimum(new_lum / np.maximum(lum, 1e-6), 8.0)
    return img * gain[..., None]


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    """Máscara de enfoque sobre la luminancia: Y + k·(Y − blur(Y))."""
    lum, y = _perceptual_luma(img)
    detail = y - cv2.GaussianBlur(y, (0, 0), _px(img, 1.0))
    return _apply_luma(img, lum, y + amount / 100.0 * 1.5 * detail)


def clarity(img: np.ndarray, amount: float) -> np.ndarray:
    """Contraste local con radio grande, centrado en los medios tonos."""
    lum, y = _perceptual_luma(img)
    detail = y - fast_blur(y, _px(img, 25.0))
    # Atenúa las diferencias grandes (bordes fuertes) para evitar halos.
    detail /= 1.0 + 6.0 * np.abs(detail)
    midtones = np.clip(4.0 * y * (1.0 - y), 0.0, 1.0)
    return _apply_luma(img, lum, y + amount / 100.0 * 0.8 * detail * midtones)


def noise_reduction(img: np.ndarray, luma: float, color: float) -> np.ndarray:
    """Reducción de ruido de luminancia y de color, por separado.

    Luminancia: se separa el detalle fino (Y − blur) y se atenúan las
    variaciones pequeñas (ruido) conservando las grandes (bordes).
    Color: se desenfoca solo la crominancia, a la que el ojo es poco sensible.
    El cambio se limita a una fracción de la luminancia: el ruido de color es
    pequeño, y así los bordes de color reales no se "derraman".
    """
    out = img
    if color > 0:
        c = color / 100.0
        lum = cv2.transform(out, LUMA[None, :])[..., None]
        chroma = out - lum
        limit = (0.03 + 0.07 * c) * lum
        change = np.clip(fast_blur(chroma, _px(img, 0.8 + 4.0 * c)) - chroma, -limit, limit)
        # El recorte por canal puede alterar la luminancia: se le quita.
        change -= cv2.transform(change, LUMA[None, :])[..., None]
        out = np.maximum(out + change, 0.0)
    if luma > 0:
        a = luma / 100.0
        lum, y = _perceptual_luma(out)
        base = cv2.GaussianBlur(y, (0, 0), _px(img, 0.7 + 1.5 * a))
        detail = y - base
        threshold = 0.004 + 0.03 * a
        detail *= 1.0 - np.exp(-((detail / threshold) ** 2))
        out = _apply_luma(out, lum, base + detail)
    return out


HAZE_MAP_SIDE = 400  # el mapa de neblina se calcula a esta resolución


def dehaze(img: np.ndarray, amount: float) -> np.ndarray:
    """Quita (amount > 0) o añade (amount < 0) neblina.

    Dark Channel Prior simplificado: en una zona sin neblina casi siempre hay
    algún canal oscuro; si no lo hay, es por la neblina. El mapa de
    transmisión se calcula en pequeño y suavizado, así es rápido y el
    resultado coincide entre la vista previa y la exportación.
    """
    h, w = img.shape[:2]
    f = HAZE_MAP_SIDE / max(h, w)
    small = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                       interpolation=cv2.INTER_AREA) if f < 1 else img
    kernel = np.ones((7, 7), np.uint8)

    # Luz de la neblina (A): color medio del 0.1 % de píxeles más "neblinosos".
    dark = cv2.erode(small.min(axis=2), kernel)
    n = max(1, dark.size // 1000)
    brightest = np.argpartition(dark.ravel(), -n)[-n:]
    airlight = np.maximum(small.reshape(-1, 3)[brightest].mean(axis=0), 0.05).astype(np.float32)

    if amount < 0:
        return img + (airlight - img) * np.float32(-amount / 100.0 * 0.5)

    strength = 0.9 * amount / 100.0
    t = 1.0 - strength * cv2.erode((small / airlight).min(axis=2), kernel)
    t = cv2.GaussianBlur(t, (0, 0), 4.0)
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    t = np.maximum(t, 0.15)[..., None]
    return np.maximum((img - airlight) / t + airlight, 0.0)
