# Bàn giao ARSH v0.5 cho AI trên máy CUDA

## Chỉ thị đầu tiên cho AI nhận gói

Hãy đọc toàn bộ file này trước khi chạy lệnh. Nhiệm vụ hiện tại là **thực thi
đúng các CUDA shard đã khóa trong job plan, theo dõi checkpoint và bàn giao lại
output nguyên vẹn**. Không tự thiết kế lại mô hình, không đổi dữ liệu, không đổi
tham số nghiên cứu và không tạo kết luận khoa học từ kết quả chưa được ghép.

## 1. Bối cảnh dự án

ARSH là viết tắt của **Adaptive Regime-Switching — Hieu**. Phiên bản v0.5 là
phiên bản nghiên cứu offline dùng dữ liệu phút của mã tổng hợp VN30F1M.

Vai trò của v0.5:

- khóa lớp dữ liệu và quy tắc tạo log-return intraday;
- so sánh các horizon 1, 2, 3, 5, 10, 15, 20, 30, 45, 60 và 90 phút;
- so sánh Gaussian, Student-t, GMM, Gaussian HMM và hai Student-t HMM;
- kiểm tra K, convergence, occupancy, độ bền qua fold và các sensitivity;
- chỉ chuyển sang v0.6 sau khi mọi completion gate đạt yêu cầu.

Ngoài phạm vi v0.5: tín hiệu mua/bán, tối ưu lợi nhuận, paper trading, dữ liệu
trực tiếp, online parameter learning, HireVAE và tuyên bố mô hình production.

## 2. Trạng thái hiện tại

Đây là code v0.5 **revised**. Kết quả 10 fold cũ là legacy vì được tạo trước khi
sửa sequence semantics; không được trộn hoặc resume bằng code trong gói này.

Giai đoạn đang chạy là `policy`: so sánh ba cách xử lý ranh giới chuỗi:

- `continuous_carry`: chạy trên Colab CPU;
- `daily_sequence`: chạy trên NVIDIA CUDA;
- `session_sequence`: chạy trên NVIDIA CUDA.

Job plan hiện tại có 5 shard CPU và 10 shard CUDA, mỗi shard gồm 2 fold. Máy này
chỉ được chạy job có `execution_target` bằng `cuda`. Không chạy
`continuous_carry` trên GPU vì một chuỗi dài có phụ thuộc tuần tự và không hưởng
lợi từ batching CUDA như daily/session.

Sau khi cả ba policy hoàn tất, Colab CPU sẽ ghép checkpoint và chọn policy chỉ
từ validation. Post-policy plan chưa được phép tự suy đoán hoặc tạo trước bước
chọn này.

## 3. Dữ liệu và tính toàn vẹn

ZIP đã chứa dữ liệu tại `data/ohlc_export.csv`; không yêu cầu máy đích có dữ liệu
từ trước.

- Kích thước dữ liệu gốc: 41.501.327 byte.
- SHA-256 bắt buộc:
  `bf84b23d6fa48b9fd90c477ef6788cc0aca19e86a0480d91600fa77363e71096`.
- Job plan: `v05_policy_job_plan.json`.
- Manifest gói: `CUDA_BUNDLE_SHA256.json`.

`verify_cuda_bundle.py` và `run_planned_job.py` phải từ chối chạy nếu data/code
không khớp. Không chỉnh sửa `arsh_v05.py`, CSV, manifest hoặc job plan.

## 4. Môi trường đích

Giả định máy Windows có GPU NVIDIA CUDA. Cần:

- NVIDIA driver hoạt động và lệnh `nvidia-smi` chạy được;
- Python 3.12 x64 với Python launcher `py`;
- mạng để cài dependencies và PyTorch CUDA, trừ khi wheel đã được chuẩn bị sẵn;
- đủ dung lượng cho checkpoint và output.

Từ PowerShell tại thư mục vừa giải nén, chạy:

```powershell
nvidia-smi
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_cuda.ps1
```

`setup_cuda.ps1` sẽ:

1. xác minh manifest và dữ liệu;
2. tạo `.venv-cuda`;
3. cài thư viện;
4. kiểm tra PyTorch nhận GPU;
5. chạy `verify_cuda.py` để đối chiếu CPU/CUDA cho Gaussian HMM và hai Student-t
   HMM.

Không chạy thí nghiệm dài nếu parity gate thất bại. Nếu bản PyTorch mặc định
không hợp driver, chỉ dùng CUDA wheel index chính thức của PyTorch qua tham số
`-TorchIndexUrl`.

## 5. Liệt kê và chạy đúng CUDA job

Liệt kê 10 job được giao cho máy CUDA:

```powershell
& .\.venv-cuda\Scripts\python.exe -c "import json; print(*[j['job_id'] for j in json.load(open('v05_policy_job_plan.json'))['jobs'] if j['execution_target']=='cuda'], sep='\n')"
```

Chạy một job, ví dụ:

```powershell
& .\.venv-cuda\Scripts\python.exe .\run_planned_job.py `
  --plan .\v05_policy_job_plan.json `
  --job policy_daily__f00_01
```

Lặp lại với từng job ID trong danh sách. Có thể chạy nhiều job song song chỉ khi
đã kiểm tra đủ VRAM/RAM, nhưng tuyệt đối không để hai tiến trình ghi vào cùng một
output folder. Cách an toàn ban đầu là chạy tuần tự một job.

Nếu phiên chạy bị ngắt, chạy lại **đúng cùng lệnh và cùng job ID**. Runner đã dùng
`--resume` và sẽ tiếp tục từ checkpoint phù hợp. Không xóa `_checkpoints`.

## 6. Tiêu chí một shard hoàn tất

Trong output của shard phải có:

- `progress.json` với `"complete": true`;
- `_checkpoints/run_identity.joblib`;
- đủ checkpoint `fold_XX.joblib` cho hai fold được giao;
- data hash, code hash, backend, policy và cấu hình khớp job plan;
- không có traceback chưa xử lý ở cuối log.

Không coi smoke test, output dở dang hoặc chỉ một fold là kết quả hoàn chỉnh của
toàn bộ policy.

## 7. Kết quả cần gửi lại

Sau khi đủ 10 CUDA shard, chép nguyên hai thư mục sau, giữ nguyên cây thư mục:

```text
runs_v05_revised/policy_daily/
runs_v05_revised/policy_session/
```

Đưa chúng vào:

```text
My Drive/ARSH_Revised/runs_v05_revised/
```

Colab CPU sẽ có sẵn `policy_continuous`, sau đó dùng `merge_shards.py`,
`finalize_experiment.py` và `select_policy.py`. Máy CUDA không tự ghép kết quả
policy và không tự chọn policy.

## 8. Khi gặp lỗi

AI bên máy đích cần giữ nguyên checkpoint và báo lại:

- toàn bộ output của `nvidia-smi`;
- output của
  `.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"`;
- lệnh/job ID đang chạy;
- traceback đầy đủ;
- nội dung `progress.json` nếu đã được tạo;
- dung lượng RAM/VRAM tại thời điểm lỗi.

Không chữa lỗi bằng cách giảm K, giảm số vòng, đổi seed, bỏ fold, đổi policy hoặc
sửa data. Những thay đổi đó làm mất tính so sánh của protocol và phải được quyết
định lại ở máy điều phối.

## 9. Tài liệu tham chiếu trong gói

- `START_HERE_CUDA.md`: hướng dẫn ngắn cho người vận hành.
- `CUDA_RUNBOOK.md`: quy trình CUDA.
- `COLAB_CPU_RUNBOOK.md`: trách nhiệm của máy CPU.
- `PROJECT_CONTEXT_FOR_AI.md`: ngữ cảnh nghiên cứu đầy đủ.
- `ARSH_PROJECT_REQUIREMENTS.md`: yêu cầu và giới hạn đã khóa.
- `README.md`: tổng quan code và protocol.

Mục tiêu bàn giao của máy CUDA ở giai đoạn này chỉ là: **10/10 CUDA shard policy
hoàn tất, toàn vẹn và được chép về đúng cấu trúc**.

