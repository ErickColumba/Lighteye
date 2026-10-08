#!/usr/bin/env bash
# Desinstala Lighteye del usuario actual. No borra tus fotos ni sus ajustes
# (los archivos .json junto a cada foto), ni tus presets (~/.config/lighteye).
set -euo pipefail
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -rf "$DATA/lighteye" "$DATA/applications/lighteye.desktop" \
       "$DATA/icons/hicolor/256x256/apps/lighteye.png" "$HOME/.local/bin/lighteye"
command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v kbuildsycoca6 >/dev/null && kbuildsycoca6 --noincremental >/dev/null 2>&1 || true
echo "Lighteye desinstalado. La caché (~/.cache/lighteye) se puede borrar a mano si quieres liberar espacio."
