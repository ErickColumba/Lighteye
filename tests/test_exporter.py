import cv2
import numpy as np
import piexif
import pytest

from app.core.color import linear_to_srgb
from app.core.exporter import ExportOptions, default_output_path, export_image
from app.core.loader import load_image, make_preview
from app.core.pipeline import process
from app.core.settings import Settings


@pytest.fixture
def photo(tmp_path):
    """JPG de 1200×800 con EXIF (cámara, fecha y orientación 1)."""
    y, x = np.mgrid[0:800, 0:1200].astype(np.float32)
    bgr = np.stack([x / 1200, y / 800, (x + y) / 2000], axis=2) * 255
    path = tmp_path / "foto.jpg"
    cv2.imencode(".jpg", bgr.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 98])[1].tofile(path)
    exif = piexif.dump({
        "0th": {piexif.ImageIFD.Make: b"Lighteye Cam", piexif.ImageIFD.Orientation: 1},
        "Exif": {piexif.ExifIFD.DateTimeOriginal: b"2026:10:07 12:00:00"},
    })
    piexif.insert(exif, str(path))
    return path


SETTINGS = Settings({"exposure": 0.5, "contrast": 20, "saturation": 15, "rotate": 1,
                     "crop": [0.1, 0.1, 0.8, 0.8], "vignette": -20})


@pytest.mark.parametrize("fmt,dtype", [("jpeg", np.uint8), ("png8", np.uint8),
                                       ("png16", np.uint16), ("tiff16", np.uint16)])
def test_export_formats_and_bit_depth(tmp_path, photo, fmt, dtype):
    loaded = load_image(photo)
    options = ExportOptions(fmt)
    out = export_image(photo, loaded.image, SETTINGS, tmp_path / f"out{options.extension}", options)
    img = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.dtype == dtype
    # Girada 90° y recortada al 80 %: de 1200×800 a 640×960.
    assert img.shape[:2] == (960, 640)
    assert not (tmp_path / f"out{options.extension}.part").exists()


def test_export_matches_preview(tmp_path, photo):
    """Lo exportado, reducido, debe verse como la vista previa."""
    loaded = load_image(photo)
    s = Settings({"exposure": 0.3, "highlights": -30, "clarity": 25, "vibrance": 20})
    out = export_image(photo, loaded.image, s, tmp_path / "o.png", ExportOptions("png16"))
    exported = cv2.cvtColor(cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_UNCHANGED),
                            cv2.COLOR_BGR2RGB).astype(np.float32) / 65535
    preview = linear_to_srgb(process(make_preview(loaded.image, 600), s))
    exported_small = cv2.resize(exported, preview.shape[1::-1], interpolation=cv2.INTER_AREA)
    assert np.abs(exported_small - preview).mean() < 0.01


def test_exif_is_kept_and_fixed(tmp_path, photo):
    loaded = load_image(photo)
    for fmt in ("jpeg", "png8"):
        opts = ExportOptions(fmt)
        out = export_image(photo, loaded.image, SETTINGS, tmp_path / f"e{opts.extension}", opts)
        exif = piexif.load(str(out)) if fmt == "jpeg" else _png_exif(out)
        assert exif["0th"][piexif.ImageIFD.Make] == b"Lighteye Cam"
        assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:07 12:00:00"
        assert exif["0th"][piexif.ImageIFD.Orientation] == 1
        assert exif["Exif"][piexif.ExifIFD.PixelXDimension] == 640


def _png_exif(path):
    data = path.read_bytes()
    i = data.index(b"eXIf")
    length = int.from_bytes(data[i - 4:i], "big")
    return piexif.load(b"Exif\x00\x00" + data[i + 4:i + 4 + length])


def test_resize_and_progress(tmp_path, photo):
    loaded = load_image(photo)
    steps = []
    out = export_image(photo, loaded.image, Settings({"exposure": 1}), tmp_path / "s.jpg",
                       ExportOptions("jpeg", long_side=300), progress=steps.append)
    img = cv2.imdecode(np.fromfile(out, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 300
    assert steps[-1] == 1.0 and steps == sorted(steps)


def test_cancel_leaves_no_file(tmp_path, photo):
    loaded = load_image(photo)

    def cancel(_):
        raise InterruptedError

    with pytest.raises(InterruptedError):
        export_image(photo, loaded.image, SETTINGS, tmp_path / "c.jpg", ExportOptions(), cancel)
    assert list(tmp_path.glob("c.jpg*")) == []


def test_default_output_path(photo):
    assert default_output_path(photo, ExportOptions("tiff16")).name == "foto_lighteye.tif"
