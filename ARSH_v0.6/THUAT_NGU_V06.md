# Thuật ngữ thống nhất cho ARSH v0.6

Trong trao đổi và tài liệu v0.6, dùng **tên tiếng Việt** ở cột đầu. Tên tiếng Anh trong ngoặc hoặc tên trường mã nguồn được ghi khi xuất hiện lần đầu để dễ đối chiếu với code và dữ liệu.

| Tên dùng thống nhất | Tên trong mã / tiếng Anh | Nghĩa trong v0.6 |
|---|---|---|
| **Nến 1 phút** | `bar` | Một dòng giá của một phút đã hoàn thành: giá mở, cao nhất, thấp nhất, đóng cửa và khối lượng (OHLCV). |
| **Khoảng tính lợi suất** | `horizon_min`, horizon | Độ dài thời gian dùng để tính một lợi suất đầu vào ARSH. Ví dụ, từ các nến 1 phút có thể tính lợi suất 1 phút hoặc 2 phút. Ở dự án này, horizon **không phải** thời gian dự báo giá tương lai. |
| **Lợi suất log** | `log_return` | Mức thay đổi giá đóng cửa được tính bằng `ln(giá cuối / giá đầu)` trên khoảng tính lợi suất hợp lệ. Đây là một quan sát đưa vào mô hình. |
| **Mô hình Markov ẩn** | HMM | Mô hình có các trạng thái không quan sát trực tiếp; mỗi trạng thái có phân phối lợi suất và xác suất chuyển sang trạng thái khác. |
| **Trạng thái thị trường** | `state`, regime | Một trong 7 trạng thái ẩn của ARSH v0.6. Mỗi trạng thái có phân phối lợi suất riêng trong mô hình đã huấn luyện. |
| **Xác suất trạng thái** | `posterior` | Bảy xác suất cho biết mức phù hợp của từng trạng thái sau khi nhận lợi suất mới; tổng xấp xỉ 100%. Nhãn trạng thái hiển thị là trạng thái có xác suất cao nhất. |
| **Tỷ lệ hiện diện mềm** | soft occupancy | Trung bình xác suất của một trạng thái qua nhiều quan sát; dùng để xem một trạng thái có quá hiếm không. |
| **Lần chia dữ liệu theo thời gian** | `fold` | Một lần chia dữ liệu cũ thành các đoạn huấn luyện, lựa chọn và kiểm tra theo thứ tự thời gian. |
| **Quy tắc chia chuỗi** | `policy` | Cách quyết định khi nào dòng quan sát tiếp tục và khi nào bắt đầu chuỗi mới, ví dụ theo ngày hoặc theo phiên. |
| **Đặt lại đầu chuỗi** | `reset` | Bắt đầu tính xác suất từ phân phối khởi đầu của mô hình, ví dụ ở ngày mới nếu dùng quy tắc chia theo ngày; không có nghĩa huấn luyện lại mô hình. |
| **Chạy lại dữ liệu theo thời gian** | `replay` | Đọc các nến lịch sử lần lượt theo thời điểm chúng có thể được sử dụng, như khi dữ liệu đến trực tiếp. |
| **Bộ lọc nhân quả** | causal filter | Cách cập nhật xác suất trạng thái chỉ từ quan sát hiện tại và quá khứ, không dùng dữ liệu tương lai. |
| **Suy luận khi có dữ liệu mới** | online inference | Giữ tham số mô hình cố định, cập nhật 7 xác suất trạng thái sau mỗi quan sát mới. Đây là chức năng v0.6 cần xây dựng. |
| **Học tham số liên tục** | online learning | Tự thay các phân phối, ma trận chuyển hoặc bộ chuẩn hóa sau khi có dữ liệu mới. V0.6 hiện chưa có chức năng này. |
| **Bộ chuẩn hóa** | `scaler` | Phép biến đổi lợi suất đầu vào theo tham số đã học cùng mô hình; khi suy luận thì giữ cố định. |
| **Tệp mô hình đã lưu** | model artifact / `joblib` | Tệp chứa mô hình đã huấn luyện và thông tin cần để nạp lại; có tệp không đồng nghĩa đã kết nối dữ liệu trực tiếp. |
| **Chế độ quan sát** | shadow mode | Tính và lưu trạng thái để nghiên cứu, không tạo lệnh giao dịch. |

Ví dụ thống nhất: “Nguồn gửi **nến 1 phút** mới. V0.6 tính **lợi suất log** theo **khoảng tính lợi suất** đã chọn, rồi cập nhật **7 xác suất trạng thái**. Sang ngày mới có thể **đặt lại đầu chuỗi**; mô hình không tự **học tham số liên tục**.”
