# Hợp đồng đầu ra ARSH v1 cho Modus (chưa tích hợp)

ARSH xử lý mỗi lợi suất 1 phút hợp lệ theo đúng thứ tự thời gian và xuất một bản ghi sau khi nến đầu vào đã hoàn tất. Các cột ổn định trong states_strict.csv hoặc states_original.csv là:

| Trường | Ý nghĩa |
|---|---|
| timestamp_label | Dấu thời gian trong CSV lịch sử; nhà cung cấp hiện ghi đầu nến |
| event_bar_end | Thời điểm kết thúc nến dùng để tính lợi suất |
| available_at | Thời điểm dữ liệu thật sự đến và đủ điều kiện dùng; để trống khi chỉ có CSV lịch sử |
| availability_status | Nguồn chứng minh được available_at hay chưa |
| p_S0 đến p_S6 | Xác suất 7 trạng thái ở thứ tự ổn định trong phạm vi một model version |
| state_id, confidence | State có xác suất cao nhất và xác suất của nó |
| posterior_entropy_nats | Mức phân tán posterior, không phải xác suất state đúng |
| model_version, model_sha256 | Danh tính artifact; bắt buộc để tránh trộn hai lần fit |
| sequence_reset | True nếu nến này bắt đầu chuỗi mới sau ranh giới ngày hoặc gap |
| period_class | Phạm vi lịch sử để đánh giá; không dùng làm feature cho Modus |

Modus hiện ghép 5 nến 1 phút thành một nến 5 phút. Khi quyết định tại thời điểm t, nó chỉ được lấy bản ghi ARSH có available_at <= t và event_bar_end <= t, ưu tiên bản ghi mới nhất phù hợp. Nếu nến 1 phút cuối của bucket [09:00, 09:05) chưa đến lúc 09:05, quyết định lúc 09:05 không được dùng posterior của nến đó. Không tính trung bình 5 posterior vì bộ lọc ARSH đã tích lũy thông tin quá khứ.

Nếu available_at trống, trạng thái không được tự động gắn vào quyết định Modus như một feature nhân quả. Adapter nguồn trực tiếp phải điền available_at từ dấu thời gian nhận thực tế hoặc metadata có thể kiểm chứng, xử lý nến trễ/sửa lại và đánh dấu khoảng dữ liệu thiếu. Modus cần được huấn luyện/đánh giá lại với chính các feature có thể biết tại thời điểm quyết định; gói này không sửa Modus.

Số S0…S6 chỉ ổn định trong cùng artifact và phép ghép ID đã xác nhận. Khi chuyển từ original sang strict, phải gắn model version; không coi hai cột cùng tên là ý nghĩa kinh tế giống hệt nếu chưa kiểm định alignment.
