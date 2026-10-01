# ARSH v1 — 5 kiểm định lịch sử (strict)

- Model SHA-256: 4c67a274092bdcde4cd7c00af89b7308268ac8360285c3346450b30e3c86d69f
- Nến nguồn: 216,845; lợi suất 1 phút hợp lệ: 214,101.
- Khoảng quan sát: 2023-02-01 09:01:00 đến 2026-09-04 14:29:00.
- Các số trên dữ liệu trước khi model được khóa chỉ mô tả lịch sử; không chứng minh hiệu quả tương lai.
- CSV đầu ra có event_bar_end nhưng available_at để trống vì CSV lịch sử không chứng minh thời điểm nến thực sự sẵn dùng.

## 1. Entropy posterior

Entropy chuẩn hóa trung bình: **0.4480** (0 = tập trung vào một state, 1 = đều trên 7 state).
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
| S0 | 0.9840 | 62.43 | 4844 | 14.15 |
| S1 | 0.4069 | 1.69 | 25359 | 1.66 |
| S2 | 0.3673 | 1.58 | 26176 | 1.56 |
| S3 | 0.3212 | 1.47 | 4766 | 1.31 |
| S4 | 0.5559 | 2.25 | 2803 | 1.57 |
| S5 | 0.9867 | 74.95 | 3484 | 12.98 |
| S6 | 0.9785 | 46.57 | 818 | 8.42 |

Thời lượng HMM là từ ma trận tham số; chuỗi argmax là nhãn hiển thị thực tế và có thể ngắn hơn nhiều.

## 5. Đặc điểm kinh tế

| State | Soft share | Return cùng phút (%) | Độ lệch chuẩn (%) | Return phút kế tiếp (%) |
|---|---:|---:|---:|---:|
| S0 | 29.48% | 0.00002 | 0.02407 | -0.00002 |
| S1 | 17.95% | -0.01764 | 0.02866 | -0.00048 |
| S2 | 18.09% | 0.02226 | 0.02987 | 0.00043 |
| S3 | 5.36% | -0.04569 | 0.04744 | -0.00045 |
| S4 | 5.47% | 0.03197 | 0.06440 | 0.00079 |
| S5 | 19.87% | -0.00016 | 0.07762 | 0.00008 |
| S6 | 3.78% | -0.00320 | 0.18338 | -0.00169 |

Phút kế tiếp chỉ được ghép trong cùng chuỗi liên tục. Đây là mô tả thống kê, không phải hiệu quả giao dịch.

## Tách giai đoạn lịch sử

| Giai đoạn | Số quan sát | Số ngày | Entropy chuẩn hóa TB |
|---|---:|---:|---:|
| through_2026_07_model_development | 208,871 | 874 | 0.4457 |
| aug_sep_seen_before_strict_lock | 5,230 | 22 | 0.5369 |

**Chưa có quan sát từ 30/09/2026 trở đi trong đầu vào này.**

## Online learning chạy song song

Trạng thái: experimental_shadow_only; số cập nhật: 5230.
Chênh lệch log density trước cập nhật (shadow − cố định): 0.003407.
Bộ cập nhật này là nguyên mẫu nghiên cứu; không thay artifact đang báo cáo.
