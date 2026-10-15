@echo off
rem Панель преподавателя для записи ролика: отдельная чистая база data_demo.
chcp 65001 >nul
cd /d "%~dp0"
set "PROCTOR_DATA_DIR=data_demo"
call "%~dp0start_teacher.bat"
