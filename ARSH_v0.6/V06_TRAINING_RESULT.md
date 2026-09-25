# Kết quả huấn luyện và kiểm tra ngoài mẫu — ARSH v0.6

Theo `V06_TRAINING_PLAN.md` (khóa 25/09/2026, SHA-256 `41b05938…`). File kết quả tách riêng để không đổi hash của kế hoạch đã khóa. Số liệu gốc: `models/v06_k7_cut20260801/selection.csv`, `run_config.json`, `artifact_manifest.json`, `holdout_evaluation.json`.

## Model đã khóa

| Mục | Giá trị |
|---|---|
| Phiên bản | `ARSH-v0.6-k7-h1-daily-cut20260801` |
| SHA-256 artifact | `9071bb38709abb3ba91012b1d4a7e18d3285e0e1de7bf2c37e8a186091cf46e1` |
| Khóa lúc | 26/09/2026 02:02:44 |
| Họ HMM | `student_t_shared` (Student-t, bậc tự do chung), K=7, lợi suất 1 phút, chia chuỗi theo ngày |
| Cửa sổ huấn luyện cuối | 01/08/2023 → **31/07/2026** (ngày cuối thực tế), 177.695 quan sát, hội tụ |
| Ngày loại vì chưa hoàn tất | 06/12/2023, 15/04/2024, 22/05/2024, 23/05/2024 (thiếu nến ATC 14:45) |
| Mã v0.5 dùng lại | `arsh_v05.py` SHA-256 `b5e19665…` — không sửa, trùng hash kế hoạch v0.5 |

## Chọn họ HMM (validation 01/02–31/07/2026)

| Họ | Log density validation | Kém họ tốt nhất | Hiện diện mềm nhỏ nhất | Trong ngưỡng 0,002 |
|---|---:|---:|---:|---|
| gaussian_hmm | 6,16219 | 0,00619 | 3,4% | không |
| **student_t_shared** | **6,16839** | 0 | 5,2% | có — **chọn** |
| student_t_state | 6,16797 | 0,00042 | 4,6% | có |

`student_t_shared` vừa cao nhất vừa đơn giản nhất trong nhóm hòa, nên quy tắc phá hòa không làm đổi kết quả. Ở bước chọn (100 vòng EM) cả ba họ chưa đạt tiêu chí hội tụ chặt, giống giai đoạn sàng lọc của v0.5; model cuối (300 vòng) đã hội tụ.

## Kiểm tra ngoài mẫu: **ĐẠT**

Giai đoạn: 03/08/2026 (phiên đầu tiên sau 01/08) → 25/09/2026, 37 phiên, 8.796 quan sát. Hai phiên thiếu nến: 18/08 (09:01, 09:04, 09:06), 25/09 (13:35, 13:37). Script đánh giá có hash trùng bản ghi trong `EVALUATION_LOCK.txt` (lưu trước khi có model).

| Tiêu chí | Ngưỡng | Kết quả |
|---|---|---|
| 1. Log density hơn Student-t độc lập | > 0 | **+0,0324** mỗi quan sát (tháng 8: +0,0296; tháng 9: +0,0357) — đạt |
| 2. Hiện diện mềm nhỏ nhất | ≥ 1% | **1,20%** (`v06::S6`) — đạt, sát ngưỡng |

Mô tả (không phải tiêu chí):

| | Model v0.6 | Artifact v0.5, cùng quan sát |
|---|---:|---:|
| Log density trung bình | 6,1486 | 6,1393 |
| Hiện diện mềm nhỏ nhất | 1,20% | 0,70% (R007), 0,72% (R006) |
| Độ tin cậy trung bình | 0,615 | 0,593 |
| Độ dài chuỗi trạng thái trung bình | 2,27 quan sát | 1,77 quan sát |
| Tỷ lệ chuỗi chỉ 1 quan sát | 59% | 57% |

## Giới hạn cần đọc cùng kết quả

- **Một giai đoạn 37 phiên**, không có khoảng tin cậy (kế hoạch không đặt bootstrap). "Đạt" nghĩa là qua hai ngưỡng đã chốt, không phải đã chứng minh ổn định lâu dài.
- **`v06::S6` chỉ chiếm 1,2%**, sát ngưỡng loại. Đây là trạng thái có độ lệch chuẩn lớn nhất; cần theo dõi xem nó có lặp lại hay biến mất trên dữ liệu mới (liên quan D03).
- **Trạng thái đổi rất nhanh:** gần 60% chuỗi trạng thái chỉ kéo dài 1 phút, ở cả hai model. Nhãn trạng thái từng phút vì vậy nhiễu; nên đọc 7 xác suất thay vì chỉ nhãn.
- Từ nay giai đoạn 03/08–25/09/2026 **đã được xem**. Nếu chỉnh model dựa trên kết quả này, cần một giai đoạn tương lai khác để kiểm tra độc lập.
