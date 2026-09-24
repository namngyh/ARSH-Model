# Bắt đầu trên máy NVIDIA CUDA

Nếu có AI hỗ trợ trên máy đích, yêu cầu AI đọc toàn bộ `AI_HANDOFF_CUDA.md`
trước khi thao tác.

Bạn chỉ cần chuyển **một file** `ARSH_v0.5_CUDA_worker.zip` sang máy GPU. ZIP đã
có sẵn đúng `data/ohlc_export.csv`, code và `v05_policy_job_plan.json`; không cần
tìm hoặc tải dữ liệu riêng.

1. Giải nén toàn bộ ZIP vào một thư mục mới.
2. Mở PowerShell tại thư mục vừa giải nén.
3. Kiểm tra GPU bằng `nvidia-smi`.
4. Cài môi trường và chạy cổng đối chiếu CPU/CUDA:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_cuda.ps1
```

5. Liệt kê 10 shard CUDA của hai policy có thể batch (`daily` và `session`).
   Năm shard `continuous_carry` được chạy riêng trên Colab CPU:

```powershell
& .\.venv-cuda\Scripts\python.exe -c "import json; print(*[j['job_id'] for j in json.load(open('v05_policy_job_plan.json'))['jobs'] if j['execution_target']=='cuda'], sep='\n')"
```

6. Chạy từng shard; ví dụ:

```powershell
& .\.venv-cuda\Scripts\python.exe .\run_planned_job.py `
  --plan .\v05_policy_job_plan.json `
  --job policy_continuous__f00_01
```

Có thể chạy lại đúng lệnh sau khi dừng vì checkpoint có `--resume`. Không chạy
job có `execution_target=cpu` bằng máy CUDA; notebook Colab phụ trách các job đó. Không cho hai
tiến trình ghi vào cùng một thư mục output. Kết quả nằm trong
`runs_v05_revised/`; chép nguyên các thư mục shard đã hoàn thành về Google Drive
để notebook Colab CPU ghép và lập báo cáo.

Không sửa `arsh_v05.py`, CSV hoặc job plan sau khi giải nén. Mọi sai khác sẽ bị
SHA-256 gate từ chối, tránh vô tình trộn hai bộ dữ liệu hoặc hai phiên bản code.
