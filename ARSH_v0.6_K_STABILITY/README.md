# ARSH v0.6 — thử K=5,6,8, đối chứng K=7, nhiều seed và xác nhận nhãn

**Đọc `NGU_CANH_BAN_GIAO.md` trước.** File đó ghi việc đã đạt, phần còn thiếu, yêu cầu người dùng và điều kiện hoàn thành. Gói này không thay model cuối ngày đang chạy.

## Chạy trên máy kia

1. Giải nén ZIP thành một thư mục mới, không giải nén chồng lên repo hoặc thư mục model cũ.
2. Dùng môi trường Python 3.12/CUDA đang chạy được trên máy kia. Đặt `ARSH_PYTHON` tới đúng executable, không phải tên thư mục. Nếu thiếu thư viện lõi, cài `requirements.txt` trong môi trường đó; file này không thay PyTorch CUDA. Sau khi đổi thư viện, phải chạy lại kiểm tra.
3. Chạy kiểm tra mã một lần. Bộ kiểm tra tích hợp dùng dữ liệu giả lập và tiêu chí hội tụ nới lỏng để kiểm tra đường chạy; không dùng kết quả đó làm bằng chứng nghiên cứu.
4. Cung cấp CSV huấn luyện đầy đủ hoặc dùng biến môi trường `PG_DSN` đã có trên máy kia. Không ghi thông tin đăng nhập vào file bàn giao.
5. Chạy `RUN_K_STABILITY.bat`. Chương trình tự kiểm tra tương đương CPU/CUDA trước khi fit thật. Chỉ một worker ghi một output directory.

Ví dụ PowerShell, thay đường dẫn bằng đường dẫn thực tế trên máy kia:

```powershell
$env:ARSH_PYTHON = 'C:\duong-dan\ARSH-Model\.venv-cuda\Scripts\python.exe'
& $env:ARSH_PYTHON .\verify_package.py
& $env:ARSH_PYTHON -m unittest test_experiment -v
.\RUN_K_STABILITY.bat -PlanOnly
```

Nếu máy đã có `PG_DSN`, lệnh sau đọc dữ liệu trước tháng 8 từ DB và lưu bản xuất cố định:

```powershell
.\RUN_K_STABILITY.bat
```

Hoặc dùng CSV huấn luyện gốc `training_data_2023-02-01_2026-07-31.csv` của lần fit v0.6:

```powershell
.\RUN_K_STABILITY.bat -DataPath 'C:\duong-dan\training_data_2023-02-01_2026-07-31.csv'
```

Có thể chạy hai giai đoạn riêng để xem kết quả selection trước:

```powershell
.\RUN_K_STABILITY.bat -Phase selection
.\RUN_K_STABILITY.bat -Phase final
```

Hai lệnh phải dùng cùng nguồn/config/backend/output. Bấm Ctrl+C để dừng; chạy lại đúng lệnh sẽ dùng checkpoint hoàn tất và tiếp tục checkpoint trong seed. Mất tối đa phần sau checkpoint gần nhất, lưu mỗi 25 cập nhật EM. Không có cơ chế tự chuyển từ CUDA sang CPU khi CUDA lỗi.

## Cấu hình mặc định

- K=5,6,7,8; `student_t_shared`; horizon 1 phút; `daily_sequence`.
- 10 seed selection mỗi K và 10 seed final mỗi K: **80 lần fit**. Không chọn seed dựa trên điểm validation; trong mỗi K chọn seed hội tụ có likelihood train cao nhất.
- Tối đa 1.500 cập nhật EM, tối thiểu 50; cần 5 cải thiện nhỏ liên tiếp, thỏa cả delta LL ≤0,01 và delta/quan sát ≤1e-7. Giảm LL quá tolerance được ghi lỗi, không coi là hội tụ. Fit hết ngân sách mà chưa hội tụ không được dùng làm ứng viên.
- Giữ K=7 gốc để đối chứng trên replay đã xem; fit thêm K=7 cùng ngân sách để so K công bằng trên validation. Model K=7 gốc đã học cả đoạn validation 02–07/2026, nên không đưa model đó vào bảng chọn K của đoạn này.
- Selection train: 01/02/2023–31/01/2026; validation: 01/02–31/07/2026. Final train: 01/08/2023–31/07/2026. CSV thiếu đến 31/07 sẽ bị từ chối, không âm thầm rút ngắn cửa sổ.
- K đủ điều kiện: seed được chọn hội tụ và mọi state có occupancy mềm validation ≥1%. Các K trong biên 0,002 so với điểm tốt nhất được xem là hòa; ưu tiên K nhỏ hơn. Đây là ứng viên nghiên cứu, không thay model sản xuất.
- Nhãn: giữ raw, thử `confirm_2` và `confirm_3`. Nếu nhãn đã xác nhận không còn được posterior ủng hộ đủ, đầu ra là `UNCERTAIN`; bộ nhớ nhãn vẫn lưu để truy vết. Xác suất và log density không đổi.

## Kết quả đọc ở đâu

- `outputs/full/selection_results.csv`: so sánh K trên validation trước tháng 8.
- `selection_seed_metrics.csv`, `final_seed_metrics.csv`: lý do dừng, số vòng, thời gian và hội tụ từng seed.
- `selection_audit/k*/seed_alignment.csv`: ghép state giữa các seed, sai khác posterior và nhãn sau ghép. Không ghép một-một giữa K khác nhau.
- `selection_decision.json`: K ứng viên chọn bằng validation.
- `final_models/k*/`: artifact riêng theo K, baseline Student-t và manifest khóa.
- `final_results.csv`: audit **trong mẫu** của fit cuối, không phải kết quả validation hoặc holdout.
- `seen_replay/k*/`, `frozen_k7_reference/`: so sánh nhãn trên tháng 8–9 đã xem, chỉ mô tả.
- `label_metrics.json`: mức đổi nhãn, tỷ lệ có nhãn xác nhận, mức bất định và thời gian chờ. `display` trong metrics đo bộ nhớ xác nhận; không phải thời lượng regime đã được xác thực.
- `experiment_status.json`: phần hoàn tất và K xuất được; không có K đủ điều kiện vẫn là kết quả cần báo trung thực.

## Kiểm tra tương lai

Sau khi khóa model, đánh giá dữ liệu mới **sau ngày khóa** và không trước 28/09/2026. Nếu fit kết thúc muộn hơn, chương trình tự đẩy ngày bắt đầu đến sau ngày khóa. Các file CSV phải là dữ liệu ngày đã hoàn tất; chế độ CSV kiểm tra ATC và giả định nguồn xuất chỉ gồm nến đã final.

```powershell
& $env:ARSH_PYTHON .\run_experiment.py evaluate-future `
  --model .\outputs\full\final_models\k5 `
  --data 'C:\duong-dan\du-lieu-moi-da-hoan-tat.csv' `
  --out .\outputs\future\k5
```

Chạy tương tự cho các K khác. Lệnh này không fit lại hoặc chọn K. Không gọi tháng 8–9 đã xem là holdout mới. Gói không thiết lập Task Scheduler, không sửa báo cáo đang chạy và không gửi dữ liệu ra ngoài.
