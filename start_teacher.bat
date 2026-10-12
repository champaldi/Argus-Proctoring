@echo off
rem Панель преподавателя с живым режимом: уведомления, журнал, страница в браузере.
chcp 65001 >nul
cd /d "%~dp0"
rem Python: активное окружение, затем .venv или venv проекта, иначе python из PATH.
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined VIRTUAL_ENV set "PY=%VIRTUAL_ENV%\Scripts\python.exe"
if defined PROCTOR_PYTHON set "PY=%PROCTOR_PYTHON%"
"%PY%" -c "import PySide6" 2>nul || (
    echo Не найден PySide6 у %PY%.
    echo Запустите этот файл из терминала, где обычно работает python main.py,
    echo или укажите Python: set PROCTOR_PYTHON=C:\путь\к\python.exe
    pause
    exit /b 1
)
echo Python: %PY%
echo Панель преподавателя запускается. Адрес для студентов будет в блоке "Идёт экзамен".
"%PY%" -m teacher --live
echo.
pause
