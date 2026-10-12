@echo off
rem Запуск теста для записи ролика: окно теста видно в записи экрана,
rem справа превью камеры, отдельная чистая база data_demo.
chcp 65001 >nul
cd /d "%~dp0"
set "PROCTOR_ALLOW_SCREEN_CAPTURE=1"
set "PROCTOR_STUDENT_VIEW=0"
set "PROCTOR_DATA_DIR=data_demo"
echo Режим записи: окно теста НЕ скрыто от записи экрана. Только для ролика.
call "%~dp0start_student.bat"
