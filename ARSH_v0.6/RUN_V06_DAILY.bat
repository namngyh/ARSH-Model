@echo off
setlocal
rem Lay cac ngay moi tu PostgreSQL (PG_DSN) roi lap bao cao cuoi ngay cho ngay chua xu ly.
set "V06_DIR=%~dp0"
if not defined ARSH_REPO (
    if exist "%V06_DIR%..\runtime.py" (
        set "ARSH_REPO=%V06_DIR%.."
    ) else if exist "%V06_DIR%..\ARSH-Model\runtime.py" (
        set "ARSH_REPO=%V06_DIR%..\ARSH-Model"
    ) else (
        set "ARSH_REPO=%V06_DIR%..\ARSH_v0.5_CUDA_worker"
    )
)
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
    echo Cannot find ARSH v0.5 code at "%ARSH_REPO%". Set ARSH_REPO.
    exit /b 1
)
if not defined PG_DSN (
    echo PG_DSN is not set; cannot read the database.
    exit /b 1
)
rem Model v0.6 da khoa (V06_TRAINING_PLAN.md) thi dung no, bao cao rieng o outputs\eod_v06.
set "V06_MODEL=%V06_DIR%models\v06_k7_cut20260801"
set "MODEL_ARGS="
if exist "%V06_MODEL%\artifact_manifest.json" set MODEL_ARGS=--shard "%V06_MODEL%" --out "%V06_DIR%outputs\eod_v06" --start 2026-08-01
"%ARSH_PYTHON%" "%V06_DIR%v06_daily.py" --repo "%ARSH_REPO%" %MODEL_ARGS% %*
set "RESULT=%errorlevel%"
if not "%RESULT%"=="0" (
    echo v0.6 daily run ended with code %RESULT%. See the log above.
    exit /b %RESULT%
)
endlocal
