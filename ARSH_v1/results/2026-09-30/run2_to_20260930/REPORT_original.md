# ARSH v1 — 5 kiểm định lịch sử (original)

- Model SHA-256: 9071bb38709abb3ba91012b1d4a7e18d3285e0e1de7bf2c37e8a186091cf46e1
- Nến nguồn: 219,978; lợi suất 1 phút hợp lệ: 217,195.
- Khoảng quan sát: 2023-02-01 09:01:00 đến 2026-09-30 14:29:00.
- Các số trên dữ liệu trước khi model được khóa chỉ mô tả lịch sử; không chứng minh hiệu quả tương lai.
- CSV đầu ra có event_bar_end nhưng available_at để trống vì CSV lịch sử không chứng minh thời điểm nến thực sự sẵn dùng.

## 1. Entropy posterior

Entropy chuẩn hóa trung bình: **0.4493** (0 = tập trung vào một state, 1 = đều trên 7 state).
Tỷ lệ quan sát có entropy chuẩn hóa > 0,8: 0.25%.
Entropy thấp chỉ cho thấy model tự tin, không chứng minh state đúng.

## 2. Chồng lấn và JSD của emission

| Cặp | Overlap | JSD (nat) |
|---|---:|---:|
| S4–S5 | 0.7022 | 0.0670 |
| S0–S1 | 0.6892 | 0.0671 |
| S2–S4 | 0.6613 | 0.0911 |
| S0–S2 | 0.6585 | 0.0807 |
| S3–S5 | 0.6502 | 0.0926 |
| S5–S6 | 0.6479 | 0.0988 |
| S1–S3 | 0.6428 | 0.0985 |
| S2–S5 | 0.5599 | 0.1486 |
| S1–S5 | 0.5585 | 0.1495 |
| S0–S5 | 0.5239 | 0.1715 |

Bảng đủ 21 cặp và hai ma trận nằm trong diagnostics JSON.

## 3–4. Chuyển trạng thái và thời lượng

| State | P(ở lại) lý thuyết | Thời lượng kỳ vọng (quan sát) | Số chuỗi argmax | Trung bình chuỗi argmax |
|---|---:|---:|---:|---:|
| S0 | 0.9839 | 61.95 | 4912 | 13.78 |
| S1 | 0.3963 | 1.66 | 26643 | 1.61 |
| S2 | 0.3767 | 1.60 | 26821 | 1.56 |
| S3 | 0.3378 | 1.51 | 4335 | 1.32 |
| S4 | 0.4516 | 1.82 | 3543 | 1.44 |
| S5 | 0.9866 | 74.55 | 3564 | 13.07 |
| S6 | 0.9790 | 47.71 | 828 | 8.55 |

Thời lượng HMM là từ ma trận tham số; chuỗi argmax là nhãn hiển thị thực tế và có thể ngắn hơn nhiều.

## 5. Đặc điểm kinh tế

| State | Soft share | Return cùng phút (%) | Độ lệch chuẩn (%) | Return phút kế tiếp (%) |
|---|---:|---:|---:|---:|
| S0 | 28.72% | 0.00003 | 0.02405 | -0.00003 |
| S1 | 18.36% | -0.01892 | 0.02862 | -0.00052 |
| S2 | 18.08% | 0.02090 | 0.02885 | 0.00037 |
| S3 | 5.40% | -0.04535 | 0.05024 | -0.00049 |
| S4 | 5.33% | 0.04149 | 0.05585 | 0.00090 |
| S5 | 20.27% | 0.00015 | 0.07719 | 0.00008 |
| S6 | 3.83% | -0.00294 | 0.18181 | -0.00164 |

Phút kế tiếp chỉ được ghép trong cùng chuỗi liên tục. Đây là mô tả thống kê, không phải hiệu quả giao dịch.

## Tách giai đoạn lịch sử

| Giai đoạn | Số quan sát | Số ngày | Entropy chuẩn hóa TB |
|---|---:|---:|---:|
| through_2026_07_model_development | 208,871 | 874 | 0.4458 |
| aug_sep_seen_before_strict_lock | 8,086 | 34 | 0.5360 |
| from_2026_09_30_post_lock_calendar | 238 | 1 | 0.5784 |
