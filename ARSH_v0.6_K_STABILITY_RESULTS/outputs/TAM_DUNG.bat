@echo off
rem Tam dung thi nghiem K Stability. Checkpoint luu moi 25 vong EM nen mat toi da ~40 giay tinh toan.
rem Bo qua chinh tien trinh PowerShell nay ($PID): dong lenh cua no cung chua cac chu can tim.
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -match 'run_selection_then_final|RUN_K_STABILITY|run_experiment.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('Da dung ' + $_.Name + ' PID ' + $_.ProcessId) }"
echo [%date% %time%] Tam dung boi nguoi dung>> "%~dp0chain.log"
echo Da tam dung. Bam CHAY_TIEP.bat de chay tiep tu checkpoint.
pause
