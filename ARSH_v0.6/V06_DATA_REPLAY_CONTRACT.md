# Hợp đồng dữ liệu và replay nhân quả cho ARSH v0.6

Đây là **hợp đồng triển khai ban đầu**, chưa phải tuyên bố đã kết nối nguồn dữ liệu trực tiếp. Nguồn cụ thể và quy ước dấu thời gian (`timestamp`) của nhà cung cấp hiện chưa được chốt; bộ kết nối nguồn (`adapter`) phải khai báo chúng trước khi tính lợi suất. Các thuật ngữ nến, khoảng tính lợi suất, chạy lại dữ liệu và xác suất trạng thái được thống nhất trong `THUAT_NGU_V06.md`.

## Một nến đầu vào (`bar`)

Mỗi nến 1 phút cần lưu tối thiểu:

| Trường | Ý nghĩa |
|---|---|
| `symbol`, `contract_id` nếu có | Mã tổng hợp và hợp đồng gốc để audit rollover |
| `source_timestamp`, `timestamp_convention` | Nhãn thời gian gốc và quy ước đầu/cuối bar do nguồn công bố |
| `bar_start`, `bar_end` | Khoảng thời gian thực mà OHLCV đại diện; `bar_end > bar_start` |
| `first_seen_at` | Lần đầu hệ thống nhận bản ghi |
| `available_at` | Thời điểm bản ghi đầy đủ và dùng được cho suy luận |
| `open`, `high`, `low`, `close`, `volume` | Dữ liệu thị trường của bar đã đóng |
| `source_id`, `raw_record_hash` | Truy vết bản gốc; sửa dữ liệu là bản ghi mới có version mới |

Bộ kết nối nguồn không được suy `bar_end` từ `source_timestamp` khi chưa biết dấu thời gian của nguồn là đầu hay cuối phút. `available_at` phải không sớm hơn thời điểm nến thực sự hoàn thành. Chạy lại dữ liệu theo thời gian (`replay`) phải dựa vào **thời điểm thông tin có thể được dùng**, không sắp lại lịch sử theo dấu thời gian do nhà cung cấp chỉnh sửa sau.

## Quy tắc xử lý tuần tự

1. Giữ bản gốc bất biến, chuẩn hóa sau; bỏ qua nến chưa hoàn thành, giá không hợp lệ hoặc OHLC mâu thuẫn.
2. Xử lý tăng dần theo `available_at`. Tại quyết định thời điểm `t`, chỉ dùng nến có `available_at <= t`.
3. Kiểm tra trùng `symbol + bar_start + bar_end`. Bản sửa về sau không được hồi tố thay đổi xác suất trạng thái đã xuất trước đó; ghi phiên bản và chạy lại đoạn bị ảnh hưởng trong nghiên cứu nếu cần.
4. Tạo log-return chỉ khi đủ hai endpoint và mọi phút nội bộ bắt buộc của horizon. Không nội suy; không tạo return thông thường qua nghỉ trưa, ATC, qua đêm hoặc vùng rollover chưa được chứng minh liên tục.
5. Khi thiếu cửa sổ hoặc gặp ranh giới ngày/phiên theo policy đang thử, reset chuỗi đúng quy tắc. Nếu nguồn trực tiếp không cho biết gap nội bộ, dừng phát state và gắn cờ chất lượng dữ liệu thay vì giả định liên tục.
6. Đưa lợi suất mới vào bộ lọc nhân quả (`causal filter`) với tham số mô hình cố định. Lưu **toàn bộ** xác suất của 7 trạng thái (`posterior`), độ tin cậy, quy tắc chia chuỗi và đặt lại đầu chuỗi, phiên bản mô hình và dấu thời gian. Nhãn `argmax` chỉ là cách hiển thị.

Mỗi ngày có dữ liệu mới chỉ tiếp tục xử lý các nến mới. Với quy tắc chia chuỗi theo ngày (`daily_sequence`), quan sát đầu ngày dùng xác suất khởi đầu của mô hình; không huấn luyện lại mô hình. Nếu nguồn ghi sửa nến cũ, chạy lại đoạn bị ảnh hưởng theo bản dữ liệu mới và lưu phiên bản, không âm thầm ghi đè kết quả đã xuất.

## Bản ghi đầu ra tối thiểu

`model_version, data_version, horizon_min, event_bar_end, available_at, policy, sequence_reset, posterior[7], display_state, confidence, quality_flags`.

Trong v0.6, `posterior[7]` luôn chứa đúng 7 xác suất không âm có tổng bằng 1 (trong sai số số học). K=7 là quyết định cố định của phiên bản; thay số state cần quyết định phiên bản mới.

Trên máy chạy, mô hình K=7 và bộ chuẩn hóa chỉ được học từ dữ liệu hợp lệ có thời điểm **trước 01/08/2026**; cũng chỉ dùng giai đoạn này để chọn cấu hình và kiểm tra nội bộ. Ghi ngày cuối thực tế của dữ liệu học, hash và phiên bản. Giữ riêng 01/08–30/09/2026 để kiểm tra ngoài mẫu sau khi khóa mô hình và quy trình; không dùng giai đoạn này để chỉnh bộ chuẩn hóa, tham số hay lựa chọn ứng viên. Tính đến 24/09/2026, tháng 9 chưa đủ dữ liệu cả tháng. Các ngày về sau được lưu để suy luận và đánh giá; **không tự đưa vào huấn luyện hằng ngày**. Chính sách huấn luyện lại sẽ quyết định sau khi quan sát vận hành một thời gian.

**Báo cáo gửi người dùng vào cuối ngày:** tối thiểu nêu ngày dữ liệu, phiên bản mô hình/dữ liệu, 7 xác suất trạng thái ở quan sát hợp lệ cuối cùng, trạng thái có xác suất cao nhất, số quan sát đã xử lý và các cảnh báo dữ liệu thiếu hoặc chưa hoàn tất. Lưu bản ghi theo từng quan sát để truy vết, nhưng thời điểm gửi báo cáo là cuối ngày sau khi nguồn xác nhận dữ liệu đã đầy đủ; không suy ra giờ chốt cố định khi chưa kiểm tra nguồn.

State ID hiển thị phải đi cùng model version. Không so state số 0 của hai lần fit như cùng một trạng thái nếu chưa alignment. Không biến output v0.6 thành lệnh mua/bán.

## Kiểm tra trước khi dùng nguồn trực tiếp

- Replay cùng dữ liệu đầu vào phải tái hiện posterior offline của cùng model và policy trong sai số số học đã định.
- Không một posterior nào phụ thuộc bar có `available_at` sau thời điểm suy luận.
- Bar trùng, thiếu, tới trễ, sửa lại, qua trưa/đêm và gần rollover đều có kết quả xử lý xác định và được audit.
- Data/model hash, timestamp convention và nguồn bar được lưu cùng mỗi run.
- Nếu policy cuối của v0.5 chưa có, replay các ứng viên dưới nhãn **nghiên cứu/shadow**; không dùng một ứng viên tạm thời để công bố cấu hình sản xuất.

Việc tiếp theo phụ thuộc nguồn dữ liệu cập nhật mà người dùng chọn: viết một adapter vào hợp đồng này, rồi chạy replay trước khi kết nối liên tục.
