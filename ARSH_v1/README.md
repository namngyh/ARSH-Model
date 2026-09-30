# ARSH v1 — gói kiểm định và replay để chuyển sang máy có dữ liệu

Gói này chưa phải bản tích hợp Modus hoặc bản ARSH đã kiểm chứng với dữ liệu tương lai. Nó thực hiện phần v1 có thể triển khai từ dữ liệu lịch sử: kiểm kê nguồn, kiểm định 5 đặc tính của 7 trạng thái, xuất xác suất từng phút theo một hợp đồng cố định và thử cập nhật tham số online ở chế độ nghiên cứu.

## Phạm vi và model

- strict_k7.joblib: ứng viên K=7 refit từ thí nghiệm 80 fit, khóa ngày 29/09/2026. Đây là ứng viên nghiên cứu, chưa thay bản cuối ngày.
- original_k7.joblib: model v0.6 gốc đang tạo báo cáo cuối ngày. Cả hai artifact được kiểm tra SHA-256 trước khi đọc.
- Dữ liệu được chấp nhận là CSV VN30F1M theo định dạng v0.5 (SYMBOL, TRADING_DATE, TRADING_TIME, OPEN_PX, HIGH_PX, LOW_PX, CLOSE_PX, VOL) hoặc thư mục các CSV ngày do v06_fetch_db.py tạo (VN30F1M_YYYY-MM-DD_vN.csv). Có thể đưa cả file lịch sử và thư mục ngày vào cùng một lệnh nếu không trùng nến.
- Lợi suất close-to-close 1 phút, phiên AM 09:00–11:30 và PM 13:00–14:30, không nối qua khoảng nghỉ/ATC, không nội suy. Mọi nến không đủ cửa sổ bị loại; chuỗi được reset theo ngày và sau gap.

## Chạy trên máy đang giữ dữ liệu

1. Chạy VERIFY_PACKAGE.bat, rồi đọc docs/NGU_CANH_BAN_GIAO.md để xác định nguồn và phạm vi dữ liệu thật. Không đưa chuỗi kết nối hoặc dữ liệu thô vào báo cáo công khai.
2. Chạy SETUP_V1.bat bằng Python 3.12. Script cài thư viện trong .venv riêng và chạy bộ kiểm tra.
3. Chạy RUN_INVENTORY.bat "C:\duong-dan\training_complete_days.csv" trước. Kiểm tra outputs/DATA_INVENTORY.json, số nến/ngày và SHA. Snapshot huấn luyện từng được báo cáo có 210.579 nến, 870 ngày từ 01/02/2023 đến 31/07/2026, SHA-256 d411a82cb27d453e0a5e0571589f0d0c7a8e1b6229678f14dc0bbc5fb0dd4374. Nếu nguồn khác, ghi rõ lý do.
4. Chạy RUN_V1.bat "C:\duong-dan\training_complete_days.csv"; nếu có các ngày mới dạng file ngày, thêm thư mục đó làm tham số thứ hai. Script chạy cả model strict và original, không sửa artifact.
5. Đọc outputs/REPORT_strict.md, REPORT_original.md, diagnostics_*.json và states_*.csv. Trả lại báo cáo cùng kiểm kê nguồn; không cần chuyển toàn bộ dữ liệu gốc.

Có thể chạy trực tiếp để chỉ dùng một model hoặc thử online learning song song:

    .venv\Scripts\python.exe run_v1.py --data "C:\duong-dan\training_complete_days.csv" --model strict --shadow-start 2026-09-30 --out outputs

Nếu chưa có dữ liệu từ 30/09/2026, bước shadow sẽ ghi rõ không có quan sát để cập nhật. Muốn kiểm tra cơ chế trên dữ liệu cũ, có thể dùng ngày bắt đầu cũ hơn nhưng phải gọi đó là replay mô tả, không phải kiểm định tương lai.

## Năm kiểm định

1. Posterior entropy: mức tập trung của xác suất 7 trạng thái, toàn mẫu và theo giai đoạn.
2. Emission overlap/JSD: 21 cặp Student-t ở scale chuẩn hóa, dùng prior bằng nhau; không dùng tần suất state để che chồng lấn.
3. Chuyển trạng thái: ma trận HMM lý thuyết và chuyển nhãn argmax quan sát được.
4. Thời lượng: kỳ vọng HMM 1/(1-P_ii) và phân phối chuỗi nhãn argmax thực tế.
5. Đặc điểm kinh tế: lợi suất, độ biến động, volume nếu có, tỷ trọng phiên AM và lợi suất ở quan sát liên tục kế tiếp theo posterior mềm.

Các chỉ số được tách thành dữ liệu đến hết 31/07/2026, dữ liệu tháng 8–9 đã xem trước khi khóa ứng viên, và dữ liệu từ 30/09/2026 trở đi. Trạng thái hiếm có thể thiếu mẫu; entropy thấp/JSD cao không đủ chứng minh trạng thái hữu ích trong tương lai.

## Đầu ra cho Modus về sau

states_*.csv chứa p_S0…p_S6, state_id, confidence, entropy, model_version, model_sha256, event_bar_end, available_at và cờ chất lượng. Dữ liệu CSV lịch sử không chứng minh thời điểm một nến thực sự sẵn dùng, nên available_at được để trống và availability_status=unknown_historical_csv. Trước khi Modus dùng trong phiên, adapter nguồn phải cung cấp thời điểm này; khi Modus đóng bar 5 phút, nó lấy posterior từ bar 1 phút mới nhất đã sẵn dùng tại thời điểm quyết định. Không lấy thông tin từ bar đến sau và không tính trung bình 5 posterior. Xem docs/MODUS_CONTRACT.md.

v1_shadow.py là một bộ cập nhật moment và chuyển trạng thái có chiết khấu theo thời gian, không phải online EM chính xác cho Student-t HMM. Nó tính điểm dự báo của quan sát trước rồi mới cập nhật bằng quan sát đó, giữ nguyên scaler và bậc tự do, giới hạn trôi tham số. Kết quả chỉ dùng để so sánh song song; không có lệnh thay model đang chạy.

## Giới hạn và an toàn dữ liệu

- Gói không chứa toàn bộ lịch sử giá và không tự truy vấn DB. Máy kia cần cung cấp CSV từ nguồn đã kiểm kê; có thể dùng bộ trích xuất v0.6 hiện có.
- Gói không sửa lịch trình EOD, không phát tín hiệu giao dịch, không tự push Git và không ghi đè model.
- Chỉ tải joblib từ hai tệp đã kiểm hash trong gói. Không dùng joblib từ nguồn không tin cậy.
- available_at trống làm cho CSV này chưa sẵn sàng để ghép as-of với Modus. Nó là định dạng và replay chuẩn để xây adapter khi có nguồn dữ liệu trực tiếp.
