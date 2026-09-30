# Ngữ cảnh ARSH v1: kiểm kê dữ liệu lịch sử trên máy chạy

Ngày lập: 30/09/2026. Trước tiên hãy **kiểm kê chỉ đọc** để xác định đầu vào cho v1. Sau khi đã chọn đúng dữ liệu, người dùng yêu cầu chạy gói kiểm định v1 theo README.md và trả lại báo cáo. Không huấn luyện lại hàng loạt, thay model cuối ngày hoặc ghép ARSH vào Modus.

## Bối cảnh đã chốt với người dùng

- ARSH cung cấp thông tin về 7 trạng thái thị trường cho Modus trong tương lai. Không tiếp tục thử nghiệm KE và không làm v1.1 theo kế hoạch cũ.
- v1 sẽ kiểm định việc chia trạng thái bằng: (1) posterior entropy, (2) chồng lấn/JSD giữa các phân phối, (3) chuyển trạng thái, (4) thời lượng trạng thái, (5) đặc điểm kinh tế của từng trạng thái. Sau đó chuẩn hóa đầu ra từng nến 1 phút để Modus có thể đọc về sau và thử online learning bằng replay ở chế độ song song.
- Máy soạn tài liệu hiện không có nguồn dữ liệu phút trực tiếp và không có toàn bộ dữ liệu lịch sử đã dùng trên máy chạy. Không cần đưa toàn bộ dữ liệu gốc lên GitHub hoặc chép về máy soạn tài liệu nếu kiểm định có thể chạy tại máy đang giữ dữ liệu.
- Thí nghiệm K Stability của v0.6 đã hoàn thành. Ứng viên K=7 refit và model v0.6 đang báo cáo cuối ngày là **hai artifact khác nhau**; phải ghi rõ artifact nào dùng cho mỗi phép kiểm định. Dữ liệu đã dùng để chọn hoặc xem model không được gọi là kiểm tra tương lai độc lập.

## Câu hỏi máy chạy cần tự xác định

**Trên máy này hiện có những nguồn dữ liệu VN30F1M theo từng phút nào đủ để thực hiện 5 kiểm định v1?** Hãy kiểm tra nguồn thực tế (DB, CSV/Parquet và các bảng posterior/replay đã lưu), rồi trả lời:

1. Với từng nguồn: vị trí hoặc tên bảng (che thông tin đăng nhập), định dạng, khoảng ngày đầu/cuối, số ngày, số dòng, mã giao dịch và mức độ đầy đủ của dữ liệu. Nêu rõ nguồn nào tương ứng với snapshot dùng trong thí nghiệm K Stability và hash/số phiên nếu có.
2. Có những cột nào trong số: thời gian gốc, quy ước đầu/cuối nến, `bar_start`, `bar_end`, `available_at`, OHLC, volume, contract ID, log-return, posterior của 7 trạng thái, nhãn trạng thái, model/data version? Cột nào là dữ liệu gốc, cột nào tính lại được, cột nào hiện không có bằng chứng?
3. Có thể tái tạo hoặc trích xuất **chuỗi từng phút được ghép giữa dữ liệu thị trường và posterior** cho đúng từng artifact K=7 không? Nếu có, nêu đầu vào, script/đường dẫn, số dòng dự kiến và định dạng xuất phù hợp (CSV/Parquet). Chưa cần xuất toàn bộ dữ liệu ở bước kiểm kê.
4. Phân biệt rõ ba phạm vi: dữ liệu fit/chọn K, dữ liệu lịch sử đã dùng để replay/xem kết quả, và dữ liệu thực sự đến **sau khi ứng viên được khóa ngày 29/09/2026**. Báo các ngày còn thiếu, sửa dữ liệu, hoặc không chắc về dấu thời gian/rollover.
5. Với dữ liệu hiện có, phép nào trong 5 kiểm định có thể làm ngay, phép nào cần tái tạo posterior hoặc thêm dữ liệu/cột? Nêu nơi hợp lý để chạy từng phép (máy chạy hoặc máy soạn tài liệu) và kết quả gọn nào cần chuyển về để người dùng xem.

JSD/chồng lấn emission và chuyển trạng thái lý thuyết có thể tính từ tham số model. Posterior entropy cần chuỗi posterior; chuyển trạng thái và thời lượng **quan sát được** cần chuỗi posterior/nhãn theo thời gian; đặc điểm kinh tế cần ghép với dữ liệu thị trường. Không thay thế các chuỗi này bằng bảng tổng hợp nếu bảng tổng hợp không chứa thông tin cần tính.

## Dạng trả lời mong muốn

Trả lời bằng một bảng ngắn cho từng nguồn dữ liệu, kèm kết luận **đủ / thiếu gì** cho 5 kiểm định. Dẫn đường dẫn tệp hoặc tên bảng và bằng chứng như số dòng, phạm vi ngày, hash nếu có. Không in chuỗi kết nối, mật khẩu, token, dữ liệu thô toàn bộ hay đưa chúng lên GitHub. Nếu nguồn không truy cập được, ghi rõ lý do thay vì giả định có dữ liệu.
