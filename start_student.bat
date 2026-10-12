@echo off
rem Запуск теста студента. Адрес преподавателя хранится в data\teacher_url.txt,
rem его не нужно вводить каждый раз.
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
if not exist "data" mkdir "data"

set "TEACHER="
if exist "data\teacher_url.txt" set /p TEACHER=<"data\teacher_url.txt"
if "%TEACHER%"=="" (
    echo Адрес преподавателя не задан: тест пойдёт без живых уведомлений.
) else (
    echo Адрес преподавателя: %TEACHER%
)
echo.
echo Enter - оставить как есть
echo Новый адрес, например 192.168.1.16 или 100.97.174.108 - заменить
echo Минус "-" - работать без преподавателя
set "NEW="
set /p NEW=^> 

if "%NEW%"=="-" (
    if exist "data\teacher_url.txt" del "data\teacher_url.txt"
    set "TEACHER="
    goto run
)
if not "%NEW%"=="" set "TEACHER=%NEW%"
if not "%TEACHER%"=="" (>"data\teacher_url.txt" echo %TEACHER%)

:run
rem Адрес передаётся явно, поэтому он работает и с другой папкой данных (PROCTOR_DATA_DIR).
set "PROCTOR_TEACHER_URL=%TEACHER%"
"%PY%" main.py
echo.
pause
