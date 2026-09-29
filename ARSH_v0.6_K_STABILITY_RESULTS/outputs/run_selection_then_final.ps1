# Chay buoc 2 (selection) roi, chi khi selection_decision.json da duoc ghi, chay buoc 3 (final).
# Dung dung launcher cua goi (RUN_K_STABILITY.ps1), cung nguon/config/backend/output; chay bang
# PowerShell 7 vi Windows PowerShell 5.1 dung launcher khi Python ghi canh bao ra stderr.
$ErrorActionPreference = 'Continue'
$env:ARSH_PYTHON = 'C:\Users\Admin\Downloads\ARSH_v0.5_CUDA_worker\.venv-cuda\Scripts\python.exe'
$pkg = 'C:\Users\Admin\Downloads\ARSH_v0.6_K_STABILITY'
$log = Join-Path $pkg 'outputs\chain.log'
function Note($m) { "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m | Add-Content -Path $log -Encoding utf8 }

Note 'Bat dau buoc 2 (selection)'
& (Join-Path $pkg 'RUN_K_STABILITY.ps1') -Phase selection *>&1 | Out-File (Join-Path $pkg 'outputs\selection_launcher.log') -Encoding utf8
$decision = Join-Path $pkg 'outputs\full\selection_decision.json'
if (-not (Test-Path $decision)) { Note 'Buoc 2 dung ma chua co selection_decision.json - KHONG chay buoc 3'; exit 1 }
Note 'Buoc 2 xong - bat dau buoc 3 (final)'
& (Join-Path $pkg 'RUN_K_STABILITY.ps1') -Phase final *>&1 | Out-File (Join-Path $pkg 'outputs\final_launcher.log') -Encoding utf8
if (Test-Path (Join-Path $pkg 'outputs\full\experiment_status.json')) { Note 'Buoc 3 xong' } else { Note 'Buoc 3 dung truoc khi ghi experiment_status.json - xem console.log' ; exit 1 }
