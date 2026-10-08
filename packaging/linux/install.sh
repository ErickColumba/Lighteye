#!/usr/bin/env bash
# Instala Lighteye para el usuario actual (sin sudo):
#   programa  → ~/.local/share/lighteye
#   menú      → ~/.local/share/applications/lighteye.desktop
#   icono     → ~/.local/share/icons/hicolor/256x256/apps/lighteye.png
#   comando   → ~/.local/bin/lighteye
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/lighteye"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
BIN="$HOME/.local/bin"

if [ "$SRC" != "$DEST" ]; then
    echo "Copiando Lighteye a $DEST …"
    rm -rf "$DEST.new"
    cp -a "$SRC" "$DEST.new"
    rm -rf "$DEST"
    mv "$DEST.new" "$DEST"
fi

mkdir -p "$APPS" "$ICONS" "$BIN"
cp "$DEST/lighteye.png" "$ICONS/lighteye.png"
ln -sf "$DEST/Lighteye" "$BIN/lighteye"

cat > "$APPS/lighteye.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Lighteye
GenericName=Editor de fotos
Comment=Editor de fotos no destructivo con herramientas de IA
Exec="$DEST/Lighteye" %f
Icon=lighteye
Terminal=false
Categories=Graphics;2DGraphics;Photography;RasterGraphics;
MimeType=image/jpeg;image/png;image/tiff;image/webp;image/x-adobe-dng;image/x-canon-cr2;image/x-canon-cr3;image/x-nikon-nef;image/x-sony-arw;image/x-fuji-raf;image/x-olympus-orf;image/x-panasonic-rw2;
StartupWMClass=lighteye
Keywords=foto;editor;raw;ia;retoque;
DESKTOP

command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q "${ICONS%/256x256/apps}" 2>/dev/null || true
command -v kbuildsycoca6 >/dev/null && kbuildsycoca6 --noincremental >/dev/null 2>&1 || true

echo "Listo: Lighteye ya aparece en el menú de aplicaciones (y como comando: lighteye)."
