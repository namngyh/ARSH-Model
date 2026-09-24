# CUDA worker – ghi chú sự cố (máy RTX 4060, driver 596.21, torch 2.14.0+cu130)

Ghi chú này đi kèm hai thư mục `runs_v05_revised/policy_daily/` và `runs_v05_revised/policy_session/`.
Không có tham số nghiên cứu, dữ liệu, code hay job plan nào bị thay đổi.

## Ba lần lỗi `CUDA error: an illegal memory access was encountered` (sticky)

| # | Thời điểm | Shard | Fit đang chạy khi lỗi | Số job song song |
|---|---|---|---|---|
| 1 | 2026-09-17 15:10 | `policy_daily__f02_03` (fold 2) | confirm h=1m `student_t_state` K=5 seed 64 | 5 |
| 2 | 2026-09-21 07:23 | `policy_session__f06_07` (fold 7) | confirm h=1m `student_t_state` K=3 seed 63 | 3 |
| 3 | 2026-09-21 20:23 | `policy_session__f00_01` (fold 0) | refit h=1m `student_t_state` K=7 seed 145 | 2 |
| 4 | 2026-09-22 17:53 | `policy_session__f00_01` (fold 1) | refit h=2m `gaussian_hmm` K=7 seed 146 | 2 |

Sau lỗi, CUDA context của tiến trình đó chết hẳn nên các seed kế tiếp fail tức thì
("Sticky error detected"). `fit_hmm_restarts` trong `arsh_v05.py` ghi cả kết quả thất bại
vào checkpoint (`(None, diagnostic)`), và khi resume sẽ coi seed đó là đã chạy → không fit lại.

## Xử lý (được người vận hành duyệt ngày 2026-09-22)

Đã xóa **đúng 9 file checkpoint thất bại** (8 file ngày 22/9 + 1 file sau lần lỗi thứ 4)
sau khi kiểm tra từng file là bản ghi
`(None, "AcceleratorError: CUDA error: an illegal memory access…")`, rồi resume để các seed
này được fit lại từ đầu với đúng seed/tham số:

- `policy_daily/policy_daily__f02_03/_checkpoints/fold_02_tasks/fits/`
  - `confirm_selection__h1__student_t_state__k5__seed64.joblib`
  - `confirm_selection__h1__student_t_state__k6__seed62.joblib`
  - `confirm_selection__h1__student_t_state__k6__seed63.joblib`
  - `confirm_selection__h1__student_t_state__k6__seed64.joblib`
- `policy_session/policy_session__f06_07/_checkpoints/fold_07_tasks/fits/`
  - `confirm_selection__h1__student_t_state__k3__seed63.joblib`
  - `confirm_selection__h1__student_t_state__k3__seed64.joblib`
- `policy_session/policy_session__f00_01/_checkpoints/fold_00_tasks/fits/`
  - `refit__h1__student_t_state__k7__seed145.joblib`
  - `refit__h1__student_t_state__k7__seed146.joblib`
- `policy_session/policy_session__f00_01/_checkpoints/fold_01_tasks/fits/`
  - `refit__h2__gaussian_hmm__k7__seed146.joblib` (lần lỗi thứ 4)

Không có file checkpoint thành công nào bị xóa hay sửa. Có thể xác minh: mọi fit trong
`fit_diagnostics.json` của shard hoàn tất đều không có trường `error`.

## Quan sát về driver

Windows System Event Log ghi lỗi `nvlddmkm` ID 13 (Level Error) **chỉ trong lúc ARSH chạy**,
tần suất tỷ lệ với số tiến trình CUDA đồng thời:
≈2.100/giờ (4–5 job), ≈550/giờ (3 job), ≈70–400/giờ (2 job), **0 khi chỉ 1 job** (đo 2 giờ liên tục),
0 khi tạm dừng. **Cả bốn lần illegal-memory-access đều xảy ra khi có ≥2 job CUDA song song.**
Từ 2026-09-22 17:56 chuyển sang chạy **đúng một job tại một thời điểm**; từ đó đến khi kết thúc
không còn lỗi nào. Máy điều phối nên cân nhắc đây là bất ổn driver/phần cứng dưới tải đồng thời,
không phải lỗi mô hình; nếu chạy lại trên máy tương tự nên giới hạn 1 tiến trình CUDA.

## Gián đoạn khác

- 2026-09-16 03:46: máy tự khởi động lại (Windows) → mất 1 fit dở, resume bình thường.
- 2026-09-16 ~21:40: tiến trình bị đóng ngoài ý muốn → mất 1 fit dở, resume bình thường.
- Nhiều lần tạm dừng theo yêu cầu người vận hành; mỗi lần chỉ mất các fit đang dở, checkpoint nguyên vẹn.
