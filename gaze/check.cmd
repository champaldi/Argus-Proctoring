@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
cd /d "%~dp0.."
set "GAZE_PYTHON="
set "GAZE_HOST="
if exist "parts\integration-ui\.venv\Scripts\python.exe" set "GAZE_PYTHON=%CD%\parts\integration-ui\.venv\Scripts\python.exe"
if not defined GAZE_PYTHON if exist ".venv\Scripts\python.exe" set "GAZE_PYTHON=%CD%\.venv\Scripts\python.exe"
if exist "events.py" set "GAZE_HOST=%CD%"
if not defined GAZE_HOST if exist "parts\integration-ui\events.py" set "GAZE_HOST=%CD%\parts\integration-ui"
if not defined GAZE_PYTHON (
    echo Не найдено окружение Python. Установите зависимости из gaze\requirements.txt.
    pause
    exit /b 1
)
if not defined GAZE_HOST (
    echo Нет общего events.py. Поместите gaze рядом с events.py общей сборки.
    pause
    exit /b 1
)
"%GAZE_PYTHON%" -m gaze.check --host-path "%GAZE_HOST%" %*
set "GAZE_RESULT=%ERRORLEVEL%"
pause
exit /b %GAZE_RESULT%
