param(
    [string]$TorchIndexUrl = ''
)
$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$venv = Join-Path $project '.venv-cuda'

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'Không tìm thấy NVIDIA driver/nvidia-smi. Cài driver CUDA phù hợp trước.'
}
& nvidia-smi
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Không tìm thấy Python launcher. Cài Python 3.12 x64 trước.'
}
if (-not (Test-Path -LiteralPath $venv)) {
    & py -3.12 -m venv $venv
}
$python = Join-Path $venv 'Scripts\python.exe'
& $python (Join-Path $project 'verify_cuda_bundle.py')
if ($LASTEXITCODE -ne 0) { throw 'CUDA bundle integrity check failed' }
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $project 'requirements.txt')
if ($TorchIndexUrl) {
    & $python -m pip install torch --index-url $TorchIndexUrl
} else {
    & $python -m pip install torch
}
& $python -c "import torch; assert torch.cuda.is_available(), 'PyTorch không nhận CUDA'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
& $python (Join-Path $project 'verify_cuda.py') --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'CPU/CUDA parity gate failed' }
Write-Host 'CUDA worker đã sẵn sàng và vượt qua parity gate.' -ForegroundColor Green
