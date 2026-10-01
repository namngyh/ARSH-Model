# ARSH v1 — 5 kiểm định lịch sử (strict)

- Model SHA-256: 4c67a274092bdcde4cd7c00af89b7308268ac8360285c3346450b30e3c86d69f
- Nến nguồn: 219,978; lợi suất 1 phút hợp lệ: 217,195.
- Khoảng quan sát: 2023-02-01 09:01:00 đến 2026-09-30 14:29:00.
- Các số trên dữ liệu trước khi model được khóa chỉ mô tả lịch sử; không chứng minh hiệu quả tương lai.
- CSV đầu ra có event_bar_end nhưng available_at để trống vì CSV lịch sử không chứng minh thời điểm nến thực sự sẵn dùng.

## 1. Entropy posterior

Entropy chuẩn hóa trung bình: **0.4494** (0 = tập trung vào một state, 1 = đều trên 7 state).
Tỷ lệ quan sát có entropy chuẩn hóa > 0,8: 0.30%.
Entropy thấp chỉ cho thấy model tự tin, không chứng minh state đúng.

## 2. Chồng lấn và JSD của emission

| Cặp | Overlap | JSD (nat) |
|---|---:|---:|
| S4–S5 | 0.7897 | 0.0331 |
| S0–S1 | 0.7103 | 0.0587 |
| S2–S4 | 0.6619 | 0.0918 |
| S5–S6 | 0.6461 | 0.0999 |
| S0–S2 | 0.6438 | 0.0885 |
| S1–S3 | 0.6362 | 0.1005 |
| S3–S5 | 0.6333 | 0.1026 |
| S2–S5 | 0.5669 | 0.1446 |
| S1–S5 | 0.5601 | 0.1489 |
| S4–S6 | 0.5565 | 0.1511 |

Bảng đủ 21 cặp và hai ma trận nằm trong diagnostics JSON.

## 3–4. Chuyển trạng thái và thời lượng

| State | P(ở lại) lý thuyết | Thời lượng kỳ vọng (quan sát) | Số chuỗi argmax | Trung bình chuỗi argmax |
|---|---:|---:|---:|---:|
| S0 | 0.9840 | 62.43 | 4924 | 14.01 |
| S1 | 0.4069 | 1.69 | 25931 | 1.66 |
| S2 | 0.3673 | 1.58 | 26767 | 1.56 |
| S3 | 0.3212 | 1.47 | 4876 | 1.31 |
| S4 | 0.5559 | 2.25 | 2856 | 1.57 |
| S5 | 0.9867 | 74.95 | 3543 | 12.91 |
| S6 | 0.9785 | 46.57 | 819 | 8.41 |

Thời lượng HMM là từ ma trận tham số; chuỗi argmax là nhãn hiển thị thực tế và có thể ngắn hơn nhiều.

## 5. Đặc điểm kinh tế

| State | Soft share | Return cùng phút (%) | Độ lệch chuẩn (%) | Return phút kế tiếp (%) |
|---|---:|---:|---:|---:|
| S0 | 29.28% | 0.00002 | 0.02409 | -0.00003 |
| S1 | 18.07% | -0.01766 | 0.02867 | -0.00050 |
| S2 | 18.21% | 0.02226 | 0.02986 | 0.00042 |
| S3 | 5.39% | -0.04569 | 0.04742 | -0.00046 |
| S4 | 5.49% | 0.03192 | 0.06444 | 0.00074 |
| S5 | 19.82% | -0.00022 | 0.07754 | 0.00004 |
| S6 | 3.73% | -0.00320 | 0.18327 | -0.00168 |

Phút kế tiếp chỉ được ghép trong cùng chuỗi liên tục. Đây là mô tả thống kê, không phải hiệu quả giao dịch.

## Tách giai đoạn lịch sử

| Giai đoạn | Số quan sát | Số ngày | Entropy chuẩn hóa TB |
|---|---:|---:|---:|
| through_2026_07_model_development | 208,871 | 874 | 0.4457 |
| aug_sep_seen_before_strict_lock | 8,086 | 34 | 0.5409 |
| from_2026_09_30_post_lock_calendar | 238 | 1 | 0.5792 |

## Online learning chạy song song

Trạng thái: experimental_shadow_only; số cập nhật: 8324.
Chênh lệch log density trước cập nhật (shadow − cố định): 0.002706.
Bộ cập nhật này là nguyên mẫu nghiên cứu; không thay artifact đang báo cáo.
