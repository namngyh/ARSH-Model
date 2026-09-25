@echo off
setlocal
set "V06_DIR=%~dp0"
if not defined ARSH_REPO (
    if exist "%V06_DIR%..\ARSH-Model\requirements.txt" (
        set "ARSH_REPO=%V06_DIR%..\ARSH-Model"
    ) else if exist "%V06_DIR%..\..\ARSH-Model\requirements.txt" (
        set "ARSH_REPO=%V06_DIR%..\..\ARSH-Model"
    ) else (
        set "ARSH_REPO=%V06_DIR%..\ARSH-Model"
    )
)
if not exist "%ARSH_REPO%\requirements.txt" (
    echo Cannot find ARSH-Model requirements at "%ARSH_REPO%".
    echo Put the repository next to this folder, or set ARSH_REPO to its directory.
    exit /b 1
)
if not exist "%V06_DIR%.venv\Scripts\python.exe" (
    where py >nul 2>nul
    if errorlevel 1 (
        python -m venv "%V06_DIR%.venv"
    ) else (
        py -3.12 -m venv "%V06_DIR%.venv"
    )
    if errorlevel 1 (
        echo Python 3.12 is required. Install it and run this file again.
        exit /b 1
    )
)
"%V06_DIR%.venv\Scripts\python.exe" -m pip install -r "%ARSH_REPO%\requirements.txt"
if errorlevel 1 (
    echo Package installation failed. See the error above.
    exit /b 1
)
echo Python environment is ready. Run RUN_V06_EOD.bat at the end of a data day.
endlocal
