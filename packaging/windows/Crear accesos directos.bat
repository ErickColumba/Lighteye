@echo off
rem Crea accesos directos a Lighteye en el menú Inicio y en el Escritorio.
set "APP=%~dp0Lighteye.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=New-Object -ComObject WScript.Shell;" ^
  "foreach($d in @([Environment]::GetFolderPath('Programs'),[Environment]::GetFolderPath('Desktop'))){" ^
  "$l=$s.CreateShortcut((Join-Path $d 'Lighteye.lnk'));$l.TargetPath='%APP%';" ^
  "$l.WorkingDirectory='%~dp0';$l.IconLocation='%APP%,0';$l.Description='Editor de fotos';$l.Save()}"
echo Listo: Lighteye aparece en el menu Inicio y en el Escritorio.
pause
