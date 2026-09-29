@echo off
rem Chay tiep thi nghiem tu checkpoint. Neu buoc 2 da xong, cac fit da xong duoc doc lai tu checkpoint roi sang buoc 3.
rem Chi xet tien trinh python.exe: lenh kiem tra nay cung chua chu run_experiment.py nen khong duoc tu dem chinh no.
powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'run_experiment.py' }) { Write-Host 'Thi nghiem DANG CHAY roi - khong khoi dong them.'; exit 1 }; Start-Process pwsh -ArgumentList '-NoProfile','-WindowStyle','Hidden','-File','%~dp0run_selection_then_final.ps1' -WindowStyle Hidden; Write-Host 'Da chay tiep (chay an, doc lap).'"
pause
