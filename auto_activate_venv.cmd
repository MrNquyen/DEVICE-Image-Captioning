@echo off
REM Nếu tồn tại venv\Scripts\activate.bat thì chạy nó
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)