@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
cd /d "%~dp0.."
set "GAZE_PYTHON="
if exist "parts\integration-ui\.venv\Scripts\python.exe" set "GAZE_PYTHON=%CD%\parts\integration-ui\.venv\Scripts\python.exe"
if not defined GAZE_PYTHON if exist ".venv\Scripts\python.exe" set "GAZE_PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined GAZE_PYTHON (
    echo Не найдено окружение Python. Установите зависимости из gaze\requirements.txt.
    pause
    exit /b 1
)
echo Закройте другие приложения, использующие камеру.
echo В окне камеры: C — экран, K — клавиатура, R — сброс таймеров, Q — выход.
"%GAZE_PYTHON%" -m gaze.demo %*
set "GAZE_RESULT=%ERRORLEVEL%"
if not "%GAZE_RESULT%"=="0" pause
exit /b %GAZE_RESULT%
