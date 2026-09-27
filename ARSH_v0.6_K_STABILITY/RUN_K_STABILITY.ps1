param(
    [string]$PythonPath = $env:ARSH_PYTHON,
    [string]$DataPath = $env:ARSH_TRAINING_DATA,
    [ValidateSet('cuda', 'cpu')][string]$Backend = 'cuda',
    [ValidateSet('all', 'selection', 'final')][string]$Phase = 'all',
    [string]$OutputDir = $env:ARSH_EXPERIMENT_OUT,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectDir = $PSScriptRoot
if (-not $PythonPath) {
    $candidates = @(
        (Join-Path $projectDir '.venv-cuda\Scripts\python.exe'),
        (Join-Path $projectDir '..\ARSH-Model\.venv-cuda\Scripts\python.exe'),
        (Join-Path $projectDir '..\ARSH_v0.5_CUDA_worker\.venv-cuda\Scripts\python.exe'),
        (Join-Path $projectDir '..\ARSH v0.6\.venv\Scripts\python.exe')
    )
    if ($env:ARSH_REPO) {
        $candidates = @((Join-Path $env:ARSH_REPO '.venv-cuda\Scripts\python.exe')) + $candidates
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $PythonPath = $candidate; break }
    }
    if (-not $PythonPath) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $PythonPath = $pythonCommand.Source }
    }
}
if (-not $PythonPath -or -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Set ARSH_PYTHON to the Python executable of the existing CUDA environment.'
}
& $PythonPath (Join-Path $projectDir 'verify_package.py')
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }
& $PythonPath (Join-Path $projectDir 'run_experiment.py') plan
if ($LASTEXITCODE -ne 0) { throw 'Plan check failed. See NGU_CANH_BAN_GIAO.md.' }
if ($PlanOnly) { return }
if (-not $OutputDir) { $OutputDir = Join-Path $projectDir 'outputs\full' }
$runArgs = @('run', '--backend', $Backend, '--phase', $Phase, '--out', $OutputDir)
if ($DataPath) {
    if (-not (Test-Path -LiteralPath $DataPath -PathType Leaf)) { throw 'Training CSV does not exist.' }
    $runArgs += @('--data', $DataPath)
} else {
    if (-not $env:PG_DSN) {
        throw 'Provide -DataPath with the complete training CSV, or use the existing PG_DSN on the database machine.'
    }
    $runArgs += '--fetch-db'
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
& $PythonPath -u (Join-Path $projectDir 'run_experiment.py') @runArgs 2>&1 |
    Tee-Object -FilePath (Join-Path $OutputDir 'console.log') -Append
if ($LASTEXITCODE -ne 0) { throw 'Experiment stopped. Completed checkpoints are retained. Read console.log.' }
