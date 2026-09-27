@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0RUN_K_STABILITY.ps1" %*
exit /b %errorlevel%
