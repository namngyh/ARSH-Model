@echo off
setlocal
set "V06_DIR=%~dp0"
if not defined ARSH_REPO (
    if exist "%V06_DIR%..\runtime.py" (
        set "ARSH_REPO=%V06_DIR%.."
    ) else if exist "%V06_DIR%..\ARSH-Model\runtime.py" (
        set "ARSH_REPO=%V06_DIR%..\ARSH-Model"
    ) else if exist "%V06_DIR%..\..\ARSH-Model\runtime.py" (
        set "ARSH_REPO=%V06_DIR%..\..\ARSH-Model"
    ) else if exist "%V06_DIR%..\ARSH_v0.5_CUDA_worker\runtime.py" (
        set "ARSH_REPO=%V06_DIR%..\ARSH_v0.5_CUDA_worker"
    ) else (
        set "ARSH_REPO=%V06_DIR%..\ARSH-Model"
    )
)
if not defined ARSH_DATA set "ARSH_DATA=%ARSH_REPO%\data\ohlc_export.csv"
if not defined ARSH_PYTHON (
    if exist "%V06_DIR%.venv\Scripts\python.exe" (
        set "ARSH_PYTHON=%V06_DIR%.venv\Scripts\python.exe"
    ) else if exist "%ARSH_REPO%\.venv-cuda\Scripts\python.exe" (
        set "ARSH_PYTHON=%ARSH_REPO%\.venv-cuda\Scripts\python.exe"
    ) else (
        set "ARSH_PYTHON=python.exe"
    )
)
if not exist "%ARSH_REPO%\runtime.py" (
    echo Cannot find ARSH-Model at "%ARSH_REPO%".
    echo Put the repository next to this folder, or set ARSH_REPO to its directory.
    exit /b 1
)
if not exist "%ARSH_DATA%" (
    echo Cannot find data CSV at "%ARSH_DATA%".
    echo Set ARSH_DATA to the v0.5-format CSV path.
    exit /b 1
)
if "%~1"=="" (
    "%ARSH_PYTHON%" "%V06_DIR%v06_replay_eod.py" --repo "%ARSH_REPO%" --data "%ARSH_DATA%" --out "%V06_DIR%outputs\eod"
) else (
    "%ARSH_PYTHON%" "%V06_DIR%v06_replay_eod.py" --repo "%ARSH_REPO%" --data "%ARSH_DATA%" --date "%~1" --out "%V06_DIR%outputs\eod"
)
if errorlevel 1 (
    echo v0.6 replay failed. See the error above.
    exit /b 1
)
echo v0.6 end-of-day report finished.
endlocal
