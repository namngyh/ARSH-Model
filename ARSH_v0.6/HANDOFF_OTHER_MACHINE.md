# Bàn giao ARSH v0.6 cho AI ở máy chạy

**Trạng thái ngày 24/09/2026:** bản chạy thử CSV cuối ngày đã hoạt động với một mô hình K=7 lưu từ v0.5; kiểm tra lại ngày 05/05/2026 khớp 238/238 quan sát với kết quả v0.5 đã lưu. Người dùng muốn **nhận báo cáo cuối ngày** và đã quyết định **chỉ xây mô hình bằng dữ liệu trước 01/08/2026; giữ tháng 8–9/2026 để kiểm tra**. Bản huấn luyện theo mốc này chưa chạy; `.bat` hiện vẫn nạp artifact v0.5 (được huấn luyện bằng dữ liệu trước 06/11/2025). Chưa có kết nối nguồn trực tiếp. Không mô tả dự án là đã chạy trực tiếp hay đã có học tham số liên tục.

## Đọc trước khi triển khai

1. `README.md`: phạm vi và thứ tự công việc.
2. `THUAT_NGU_V06.md`: cách gọi thống nhất; dùng “nến” cho `bar`, “khoảng tính lợi suất” cho `horizon`, “xác suất trạng thái” cho `posterior`.
3. `K_DECISION.md`: quyết định K=7 và bằng chứng hiện có.
4. `V06_DATA_REPLAY_CONTRACT.md`: dữ liệu đầu vào, xử lý theo thời gian và báo cáo cuối ngày.
5. `DEFERRED_OPTIMIZATION_LOG.md`: những lựa chọn đã dùng dù chưa tối ưu; cập nhật khi chọn thêm mô hình hoặc nguồn tạm thời.
6. `SETUP_V06.bat`, `RUN_V06_EOD.bat`, `v06_replay_eod.py`: bản chạy thử cuối ngày với CSV định dạng v0.5 và model K=7 đã lưu. Đọc phần “Chạy thử báo cáo cuối ngày” trong README trước khi dùng.
7. `ARSH_V06_LOGIC_V1.pdf`: tài liệu giải thích v0.6 cho người dùng và lộ trình lên v1; đối chiếu tình trạng thực tế trước khi cập nhật tài liệu này.

Mã, dữ liệu và kết quả v0.5 đã lưu trong gói `ARSH_v0.5_PAUSED_2026-09-24` ở máy soạn tài liệu. Nếu máy chạy không có gói này, repo nguồn là `https://github.com/namngyh/ARSH-Model.git` (commit đã kiểm kê: `d32f4df`; cần tải cả Git LFS để có artifact). Đừng coi tệp `runtime_research_candidate.joblib` bất kỳ là cấu hình v0.6 đã chốt; kiểm tra K, họ HMM, khoảng tính lợi suất, quy tắc chia chuỗi, bộ chuẩn hóa, dữ liệu huấn luyện và phiên bản trước khi dùng.

Các script kiểm kê trong thư mục này mặc định tìm `../ARSH_v0.5_PAUSED_2026-09-24/runs_v05_revised`. Nếu máy chạy đặt repo ở chỗ khác, điều chỉnh đường dẫn đầu vào rồi ghi lại thay đổi; các bảng kết quả kiểm kê có sẵn trong `outputs/starting_point/` để đối chiếu.

File `.bat` chạy chương trình Python **trên chính máy Windows**, không cần nhờ AI tính lại mỗi ngày. Nó chỉ hiểu CSV theo định dạng v0.5; nếu nguồn của máy này là API hoặc CSV khác, viết bộ kết nối nguồn và xác nhận quy ước thời gian trước khi dùng báo cáo như dữ liệu trực tiếp. Bản chạy thử chưa tự lập lịch; nếu cần tự chạy cuối ngày, máy chạy có thể cấu hình Windows Task Scheduler sau khi nguồn xác nhận thời điểm ngày đã hoàn tất.

## Những câu hỏi AI ở máy chạy phải tự kiểm tra

1. **Nguồn dữ liệu:** dữ liệu VN30F1M đến theo API từng nến hay file cuối ngày? Trường nào là giá, khối lượng, mã hợp đồng, thời điểm nguồn và thời điểm dữ liệu sẵn dùng? Thời gian ghi là đầu hay cuối nến, theo múi giờ nào? Nến cuối đã hoàn thành chưa? Nếu tài liệu nguồn không giải đáp được, chỉ hỏi người dùng phần còn thiếu.
2. **Tính liên tục:** quy tắc phiên, nến thiếu/trùng/sửa lại, nghỉ giữa phiên, ngày nghỉ và chuyển hợp đồng được nguồn thể hiện ra sao? Không tạo lợi suất qua đoạn đứt quãng không hợp lệ.
3. **Mô hình K=7:** bản chạy thử đã chọn artifact ở shard `policy_daily__f08_09`, khoảng tính lợi suất 1 phút, HMM Student-t với bậc tự do theo trạng thái. Xác nhận artifact tải được cùng mã và thư viện tương thích trên máy này, hash khớp manifest và kết quả thử đã lưu. Nếu thay ứng viên, ghi hash/lý do vào sổ việc chưa tối ưu; không chọn dựa trên tập test lịch sử đã xem.
   **Bước mới trước khi kiểm tra tháng 8–9:** xác minh nguồn có dữ liệu hợp lệ đến hết 31/07/2026; CSV đang bàn giao mới đến 17/07/2026, nên ghi rõ ngày cuối thực tế nếu không lấy đủ phần còn lại của tháng 7. Chỉ dùng dữ liệu **trước 01/08/2026** cho huấn luyện, bộ chuẩn hóa, chọn cấu hình, kiểm tra nội bộ và các quyết định về họ HMM, khoảng tính lợi suất, quy tắc chia chuỗi. Giữ nguyên dữ liệu **01/08–30/09/2026** làm tập kiểm tra chưa dùng; khóa cấu hình và mã trước khi xem kết quả của tập này. Sau đó huấn luyện K=7, lưu seed, tham số, quy tắc tạo lợi suất, mốc dữ liệu, dữ liệu/mã/model hash và báo cáo hội tụ. Gắn phiên bản mô hình mới, căn lại ID trạng thái hoặc chỉ dùng ID kèm phiên bản; không gán nhãn “tăng/giảm” từ số thứ tự state. Kết nối báo cáo với artifact mới rồi chạy kiểm tra theo đúng thứ tự thời gian, không cập nhật tham số giữa tháng 8 và tháng 9. Nếu điều chỉnh theo kết quả tháng 8–9, giai đoạn này không còn là tập kiểm tra độc lập cho mô hình đã chỉnh. Tháng 9 chưa kết thúc tại ngày lập tài liệu; chỉ kết luận đủ hai tháng sau khi nguồn xác nhận dữ liệu tháng 9 hoàn tất.
4. **Báo cáo cuối ngày:** nguồn đánh dấu ngày dữ liệu đã hoàn tất như thế nào? Báo cáo phải có 7 xác suất ở quan sát cuối cùng, nhãn trạng thái, độ tin cậy, số quan sát, cảnh báo dữ liệu và phiên bản mô hình/dữ liệu. Giữ bản ghi từng quan sát để truy vết. Nếu ngày chưa hoàn tất, báo cáo ghi rõ tình trạng thiếu thay vì coi là ngày đủ dữ liệu.
5. **Chạy tiếp sau mỗi ngày:** chỉ xử lý nến mới; lưu điểm đã xử lý, xác suất trạng thái cuối và phiên bản. Với quy tắc chia theo ngày, đặt lại xác suất đầu chuỗi ở ngày mới. Lưu dữ liệu ngày mới riêng để đánh giá; **không tự thêm nó vào tập huấn luyện và không tự huấn luyện lại tham số**. Sau một thời gian vận hành, đề xuất lịch/cửa sổ huấn luyện lại dựa trên chất lượng thực tế để người dùng quyết định. Bản sửa dữ liệu cũ phải có phiên bản và chạy lại đoạn bị ảnh hưởng.
6. **Kiểm tra tối thiểu:** thử tải model K=7, chạy lại một đoạn dữ liệu theo thời gian, xác nhận 7 xác suất không âm có tổng xấp xỉ 1, không dùng nến tương lai, reset đúng và có báo cáo cuối ngày tái tạo được. Đây là kiểm tra pipeline, không phải yêu cầu chạy lại toàn bộ thí nghiệm v0.5.

## Khi nào phải hỏi lại người dùng

Chỉ hỏi khi máy chạy không có quyền/đường dẫn tới nguồn dữ liệu, tài liệu nguồn không xác định được quy ước dấu thời gian, hoặc yêu cầu về nội dung/thời điểm báo cáo cuối ngày khác với quyết định ở đây. Những chi tiết có thể đọc từ nguồn, mã và artifact thì AI tự xác minh rồi ghi kết quả, không chuyển thành câu hỏi cho người dùng.
