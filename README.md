# ARSH v0.5

**Adaptive Regime-Switching — Hieu**

V0.5 khóa lớp dữ liệu, cách tạo lợi suất intraday, horizon, họ phân phối và phạm vi số state trước khi v0.6 tập trung vào ý nghĩa và độ ổn định của regime.

## Phạm vi chạy chính

- Horizon riêng: 1, 2, 3, 5, 10, 15, 20, 30, 45, 60 và 90 phút.
- Log-return close-to-close không chồng lấn trong phiên liên tục.
- Không tạo return thông thường xuyên nghỉ trưa, ATC, qua đêm hoặc cửa sổ thiếu phút.
- K=2–7; soft occupancy chính 1%, sensitivity 0,5%/2%/5%.
- Gaussian, Student-t, GMM, Gaussian HMM và hai Student-t HMM.
- Walk-forward 3 năm train, 6 tháng validation, 6 tháng test, bước 6 tháng.
- Screening dùng 3 seed/100 vòng; refit dùng 5 seed/300 vòng.
- Chưa có lợi suất ngày, dữ liệu trực tiếp, online learning, tín hiệu hoặc paper trading.

## Cài đặt trên máy mới

Giải nén gói, mở PowerShell tại thư mục dự án rồi chạy:

```powershell
.\setup.ps1
```

Gói portable đã kèm `data/ohlc_export.csv`. Nếu tách riêng source code, chép file này vào `data`, sau đó chạy liên tục:

```powershell
.\run_v05.ps1
```

Chương trình lưu từng seed HMM và từng tác vụ đã hoàn tất bằng thao tác ghi nguyên tử. `run_v05.ps1` luôn dùng `--resume`: thư mục mới sẽ chạy từ đầu, thư mục có checkpoint hợp lệ sẽ chạy tiếp. Nhấn `Ctrl+C` một lần trong terminal để dừng an toàn.

## Smoke test

```powershell
& '.\.venv\Scripts\python.exe' '.\arsh_v05.py' `
  --output '.\outputs_smoke' `
  --horizons 5 10 --k-values 2 --max-folds 1 `
  --selection-iterations 3 --selection-restarts 1 `
  --iterations 3 --restarts 1 --top-horizons 1
```

## Kiểm thử

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s '.' -p 'test_*.py' -v
```

## Kiến trúc chạy bản hoàn thiện

- `outputs_main_legacy/` giữ nguyên main run CPU cũ làm đối chứng; không resume bằng code mới.
- Colab CPU phụ trách audit, merge, bootstrap, state alignment và báo cáo.
- Máy NVIDIA CUDA fit Gaussian/Student-t HMM theo các fold shard độc lập.
- `make_cuda_bundle.ps1` tạo một ZIP có đúng dữ liệu nguồn, policy job plan và manifest SHA-256; máy CUDA không cần có dữ liệu từ trước.
- `verify_cuda.py` là cổng bắt buộc: CPU/CUDA phải tương đương số học trước full run.
- Chạy ba policy trước, chọn policy bằng validation, rồi mới tạo main/sensitivity plan.

Đọc `CUDA_RUNBOOK.md` và `COLAB_CPU_RUNBOOK.md` trước khi phân phối job. Không cho hai worker ghi chung một output folder.

## Trạng thái nghiên cứu

Kết quả trong `outputs_main_legacy` là research benchmark của code trước khi sửa sequence semantics. Không dùng kết quả smoke test để kết luận. Dữ liệu nguồn chỉ có mã tổng hợp `VN30F1M`; audit có thể phát hiện dấu hiệu rollover và thay đổi cấu trúc timestamp nhưng không thể xác minh tuyệt đối quy tắc của nhà cung cấp nếu thiếu dữ liệu từng hợp đồng hoặc tick data.

Sau khi ba policy experiment hoàn tất và đã có `policy_decision.json`, chạy:

```powershell
.\run_sensitivities.ps1
```

Script này tạo post-policy CUDA plan gồm main revised, K=2–9 boundary, intraday adjustment, overlapping, train 1/2 năm, expanding, zero, outlier, rollover-adjacent và cửa sổ thiếu đúng một phút. So sánh Gaussian/Student-t/skew-t và audit thay đổi timestamp chạy trên CPU; ca hội tụ khó được audit 500 vòng trên CUDA.

Trước khi chỉnh code hoặc chạy trên máy khác, đọc [PROJECT_CONTEXT_FOR_AI.md](PROJECT_CONTEXT_FOR_AI.md) và [ARSH_PROJECT_REQUIREMENTS.md](ARSH_PROJECT_REQUIREMENTS.md).
