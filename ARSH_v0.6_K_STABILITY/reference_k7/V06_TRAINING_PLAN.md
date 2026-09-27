# Kế hoạch huấn luyện K=7 theo mốc 01/08/2026 — ARSH v0.6

**Khóa ngày 25/09/2026, trước khi huấn luyện và trước khi xem bất kỳ kết quả nào của model mới trên tháng 8–9.** Các con số trong file này là quy tắc đã chốt, không được sửa sau khi thấy kết quả. `v06_train.py` đọc cấu hình từ `v06_training_config.json` và ghi SHA-256 của cả hai file này vào manifest của model. Nếu phải đổi quy tắc, tạo kế hoạch phiên bản mới và ghi lý do, không sửa file này.

Lưu ý trung thực: ngày 25/09/2026 đã xem số liệu tổng hợp tháng 8–9 của **artifact v0.5** (độ tin cậy trung bình cuối ngày 0,59; tần suất trạng thái cuối ngày). Không quy tắc nào dưới đây dựa trên các số đó.

## 1. Dữ liệu

- Nguồn: PostgreSQL `bars_1m`, qua `v06_fetch_db.py` (quy ước đã xác minh ở D06). Chỉ dùng ngày **hoàn tất** (có nến ATC 14:45, mọi nến `is_final`).
- Chỉ dùng nến có thời điểm **trước 01/08/2026**. Ngày cuối thực tế được ghi vào manifest; dự kiến 31/07/2026 nếu ngày này hoàn tất.
- Cách tạo lợi suất giữ đúng v0.5: `build_returns`, lợi suất log 1 phút, không chồng lấn, không cho phép thiếu phút bên trong, không lọc outlier, không bỏ lợi suất 0, không điều chỉnh intraday.

## 2. Những gì cố định, không chọn lại

| Mục | Giá trị | Căn cứ (chỉ dữ liệu trước 01/08/2026) |
|---|---|---|
| K | 7 | Quyết định dự án, `K_DECISION.md` |
| Khoảng tính lợi suất | 1 phút | v0.5 validation: thắng 10/10 fold ở cả hai quy tắc chia chuỗi |
| Quy tắc chia chuỗi | `daily_sequence` | v0.5 validation: thắng `session_sequence` 20/20 cặp fold × horizon |

## 3. Chọn họ HMM trên validation

- **Đoạn chọn:** huấn luyện 01/02/2023 ≤ t < 01/02/2026 (3 năm); validation 01/02/2026 ≤ t < 01/08/2026 (6 tháng, như v0.5).
- **Bộ chuẩn hóa:** `StandardScaler` học trên đoạn huấn luyện chọn.
- **Ứng viên:** `gaussian_hmm`, `student_t_shared`, `student_t_state`; K=7; mỗi họ 3 seed (62, 63, 64), 100 vòng EM, tol 0,01; lấy seed hội tụ có log-likelihood huấn luyện cao nhất (như `fit_hmm_restarts` của v0.5).
- **Điểm:** log density dự báo trung bình trên mỗi quan sát validation, bộ lọc nhân quả, đặt lại đầu mỗi ngày, đã trừ Jacobian của bộ chuẩn hóa (cùng thước đo `validation_log_density` của v0.5).
- **Hợp lệ:** tỷ lệ hiện diện mềm nhỏ nhất trên validation ≥ 1%. Họ không hợp lệ bị loại.
- **Quy tắc chọn:** gọi `best` là điểm cao nhất trong các họ hợp lệ. Mọi họ hợp lệ có điểm ≥ `best − 0,002` là "hòa"; trong nhóm hòa chọn họ **ít tham số nhất** theo thứ tự `gaussian_hmm` < `student_t_shared` < `student_t_state`.
- Nếu không họ nào hợp lệ: dừng, không tạo model, báo lại người dùng.

## 4. Huấn luyện model cuối

- **Cửa sổ:** 3 năm gần nhất kết thúc ở ngày dữ liệu hoàn tất cuối cùng trước 01/08/2026: 01/08/2023 ≤ t < 01/08/2026.
- Họ đã chọn ở mục 3, K=7; bộ chuẩn hóa học lại trên cửa sổ này; 5 seed (142–146), 300 vòng EM, tol 0,01 (như refit v0.5); lấy seed hội tụ có log-likelihood cao nhất.
- ID trạng thái hiển thị: `v06::S0`…`v06::S6`, sắp theo độ lệch chuẩn tăng dần. Không so với ID của v0.5 và không gán nhãn tăng/giảm.
- Lưu: `runtime_research_candidate.joblib` (model + bộ chuẩn hóa + quy tắc chia chuỗi), `selection.csv`, `fit_diagnostics.json`, `run_config.json`, `folds.csv`, `artifact_manifest.json` (SHA-256 dữ liệu, mã, kế hoạch, cấu hình, model).

## 5. Khóa và kiểm tra ngoài mẫu

- Sau khi `artifact_manifest.json` được ghi, model bị khóa. Không huấn luyện lại, không đổi bộ chuẩn hóa hay cấu hình trong suốt giai đoạn kiểm tra.
- **Giai đoạn kiểm tra:** 01/08/2026 → ngày dữ liệu hoàn tất cuối cùng tại lúc model khóa. Ghi ngày cuối thực tế.
- Chạy `v06_replay_eod.py` từng ngày theo thứ tự thời gian với model mới. Báo cáo cho giai đoạn này:
  1. **Chính:** log density dự báo trung bình trên mỗi quan sát, so với Student-t độc lập học trên cùng cửa sổ 3 năm. **Đạt** nếu chênh lệch > 0.
  2. Tỷ lệ hiện diện mềm nhỏ nhất ≥ 1% (mọi trạng thái vẫn xuất hiện).
  3. Mô tả, không phải tiêu chí đạt/không đạt: độ tin cậy trung bình, độ dài chuỗi trạng thái, tỷ lệ chuỗi một quan sát, cùng các số đó của artifact v0.5 trên cùng các ngày.
- Nếu model không đạt: ghi kết quả vào sổ, **không** chỉnh model rồi đánh giá lại trên cùng giai đoạn.
