@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: RUN_INVENTORY.bat "path-to-history.csv" ["path-to-day-files-directory"]
    exit /b 2
)
if not exist ".venv\Scripts\python.exe" (
    echo Run SETUP_V1.bat first.
    exit /b 2
)
if "%~2"=="" (
    ".venv\Scripts\python.exe" run_v1.py --mode inventory --data "%~1" --out outputs
) else (
    ".venv\Scripts\python.exe" run_v1.py --mode inventory --data "%~1" "%~2" --out outputs
)
exit /b %errorlevel%
