@echo off
setlocal
cd /d "%~dp0"
py -3.12 verify_package.py
exit /b %errorlevel%
