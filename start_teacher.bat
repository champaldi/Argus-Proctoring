@echo off
rem Панель преподавателя с живым режимом: уведомления, журнал, страница в браузере.
chcp 65001 >nul
cd /d "%~dp0"
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
echo Панель преподавателя запускается. Адрес для студентов будет в блоке "Идёт экзамен".
"%PY%" -m teacher --live
echo.
pause
