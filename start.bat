@echo off
setlocal
cd /d "%~dp0"
rem Python-Umgebung liegt unter kurzem Pfad (lange Pfade machen pip Probleme)
set "VENV=%LOCALAPPDATA%\WowMocap\venv"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VPY%" (
  "%VPY%" -c "import cv2, mediapipe, numpy, PIL" >nul 2>&1 && goto run
)

echo Richte die Python-Umgebung ein (nur beim ersten Start, dauert ein paar Minuten) ...
set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY%" (
  "%PY%" -m venv --clear "%VENV%" || goto fail
) else (
  py -3.12 -m venv --clear "%VENV%" || goto nopython
)
"%VPY%" -m pip install --disable-pip-version-check -r requirements.txt || goto fail

:run
echo WoW Motion Capture laeuft. Dieses Fenster offen lassen (zeigt Fehlermeldungen).
"%VPY%" wow_mocap.py
if errorlevel 1 pause
exit /b

:nopython
echo.
echo Python 3.12 wurde nicht gefunden. Installieren mit:
echo   winget install --id Python.Python.3.12 --source winget --scope user
pause
exit /b 1

:fail
echo.
echo Einrichtung fehlgeschlagen - siehe Meldungen oben.
pause
exit /b 1
