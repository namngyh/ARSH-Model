# Quyết định K=7 cho ARSH v0.6 (24/09/2026)

**Quyết định thiết kế: cố định K=7 trong v0.6.** Mọi mô hình ARSH v0.6 dùng 7 trạng thái và xuất 7 xác suất trạng thái. Không tự thay K theo từng lần chia dữ liệu kiểm tra (`fold`), ngày hoặc lần chạy lại dữ liệu. Họ HMM, khoảng tính lợi suất, quy tắc chia chuỗi và tham số mô hình cụ thể vẫn cần được đánh giá riêng; quyết định K này không tuyên bố K=7 là tối ưu toàn cục hay cấu hình giao dịch đã được xác nhận.

Tên gọi thống nhất trong v0.6: **trạng thái thị trường** (`state`), **xác suất trạng thái** (`posterior`), **khoảng tính lợi suất** (`horizon`), **quy tắc chia chuỗi** (`policy`) và **chạy lại dữ liệu theo thời gian** (`replay`). Xem `THUAT_NGU_V06.md` khi gặp thuật ngữ trong mã hoặc bảng kết quả.

## Bằng chứng hiện có

Đọc 10 fold đã hoàn tất cho mỗi policy, ở từng horizon. Với mỗi K và fold, lấy họ HMM có điểm validation log density cao nhất trong các ứng viên vượt ngưỡng occupancy mềm 1%. Không sử dụng test để chọn K. Bảng đầy đủ và cách tính nằm trong `outputs/starting_point/k_validation_comparison.csv` và `audit_k_shortlist.py`.

| Policy, horizon | K=4 | K=5 | K=6 | K=7 |
|---|---:|---:|---:|---:|
| Theo ngày, 1 phút | 0.15268 | 0.15780 | 0.16168 | **0.16535** |
| Theo phiên, 1 phút | 0.14912 | 0.15596 | 0.15878 | **0.16291** |
| Theo ngày, 2 phút | 0.13368 | 0.14101 | 0.14307 | **0.14678** |
| Theo phiên, 2 phút | 0.12936 | 0.13637 | 0.13799 | **0.14061** |

Các số là **trung bình gain validation log density trên mỗi quan sát so với Student-t độc lập**, nên cao hơn là tốt hơn cho tiêu chí dự báo này. Ở horizon 1 phút, K=7 hơn K=6 trung bình `0.00367` (theo ngày) và `0.00414` (theo phiên); K=7 thắng K=6 ở 8/10 fold của mỗi policy. K=7 hơn K=5 lần lượt `0.00755` và `0.00696`, thắng 10/10 fold. K=2–4 có khoảng cách lớn hơn so với K=7. Horizon 2 phút vẫn cho xu hướng tương tự.

K=7 vượt điều kiện occupancy mềm 1% ở cả 10 fold của mỗi policy tại horizon 1 phút, nhưng tỷ lệ nhỏ nhất trung vị của state trong mô hình được chọn tại K=7 chỉ khoảng 2.9%. Vì vậy cần xem các state nhỏ có lặp lại ổn định và có ý nghĩa không. K=7 thắng trong miền đã thử 2–7 không chứng minh rằng 7 là tối ưu ngoài miền đó.

## Theo dõi quyết định trong v0.6

1. Đánh giá 7 state qua seed/fold và replay: occupancy, thời gian duy trì, các đoạn một bar, chuyển đổi qua lại quá nhanh và khả năng ghép state tương ứng giữa các lần fit. Nếu state nhỏ hoặc thiếu ổn định, ghi nhận đó là hạn chế của lựa chọn K=7.
2. Có thể dùng K=4–6 làm đối chứng và K=8–9 làm phép thử biên để hiểu độ nhạy. Những thí nghiệm này không tự động đổi K của v0.6; nếu bằng chứng buộc phải đổi, ghi một quyết định phiên bản mới thay vì âm thầm thay đổi output và ý nghĩa state.
3. Xác nhận model K=7 bằng replay nhân quả trên dữ liệu cập nhật hoặc một giai đoạn chưa dùng để thu hẹp lựa chọn. Không dùng lại tập test lịch sử đã xem như một holdout mới. Khi phát hành model cụ thể, cố định thêm policy, horizon, họ HMM, quy tắc reset và phiên bản dữ liệu/model; chỉ lúc đó ID và ý nghĩa từng state mới có mốc tham chiếu cố định.

K=7 đã được khóa **cho phạm vi v0.6 theo quyết định của dự án**, nên không cần đợi chạy K=8–9 để bắt đầu xây replay. Các fold lịch sử dùng chung chuỗi thời gian, nên tỷ lệ thắng 8/10 hay 10/10 là bằng chứng định hướng, không phải 10 phép thử độc lập để khẳng định ý nghĩa thống kê.

Các phần chưa tối ưu nhưng đang được dùng được theo dõi tại `DEFERRED_OPTIMIZATION_LOG.md`.
