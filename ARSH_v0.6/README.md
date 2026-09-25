# ARSH v0.6 — điểm khởi đầu

**Mục tiêu hiện tại:** nghiên cứu ý nghĩa, thời gian duy trì và độ ổn định của 7 trạng thái thị trường; ưu tiên chạy lại dữ liệu theo thời gian (replay) và nhận dữ liệu cập nhật để kiểm tra trạng thái. V0.6 chỉ nhận diện trạng thái, không phát lệnh giao dịch, không học lại tham số sau mỗi nến và không coi các đoạn dữ liệu kiểm tra lịch sử đã xem là dữ liệu kiểm tra mới. Xem `THUAT_NGU_V06.md` để phân biệt nến, khoảng tính lợi suất, chạy lại dữ liệu và học tham số liên tục.

## Nguồn đang có

- Gói v0.5 tạm dừng: `../ARSH_v0.5_PAUSED_2026-09-24/`. Mã và dữ liệu trong gói khớp hash kế hoạch v0.5; kết quả daily/session đủ 10 fold, continuous chưa hoàn tất.
- **K=7 đã được cố định cho v0.6 theo quyết định dự án.** Các mô hình ứng viên: Gaussian HMM, Student-t HMM shared df, Student-t HMM state df. Khoảng tính lợi suất 1 phút là ứng viên mạnh khi so với 2 phút; họ HMM, khoảng tính lợi suất và quy tắc chia chuỗi chưa khóa.
- `audit_starting_point.py` tạo bảng đầu vào nghiên cứu v0.6 từ **validation và state profile đã lưu**, không fit model và không dùng test để chọn ứng viên.
- `K_DECISION.md` ghi quyết định K=7, bằng chứng validation và giới hạn của quyết định; `audit_k_shortlist.py` tái tạo bảng so K lịch sử.
- `DEFERRED_OPTIMIZATION_LOG.md` ghi các lựa chọn đã dùng dù chưa tối ưu, việc xem lại sau và mức kiểm tra tối thiểu để gọi v0.6 là chạy được.
- `THUAT_NGU_V06.md` là bảng thuật ngữ thống nhất cho trao đổi và tài liệu v0.6.
- `HANDOFF_OTHER_MACHINE.md` là hướng dẫn và danh sách câu hỏi kỹ thuật cho AI ở máy sẽ chạy v0.6.
- `ARSH_V06_LOGIC_V1.pdf` giải thích chi tiết logic, mục tiêu, lý do thiết kế và lộ trình từ v0.6 lên v1; file Markdown cùng tên là bản nội dung có thể sửa.
- `SETUP_V06.bat` và `RUN_V06_EOD.bat` chuẩn bị môi trường rồi chạy báo cáo cuối ngày bằng máy Windows; xem phần chạy thử bên dưới.

## Khi có dữ liệu mới

Thiết kế v0.6 xử lý tăng dần: khi đã có đủ nến đã đóng cho khoảng tính lợi suất được chọn, hệ thống tính một lợi suất log mới rồi cập nhật **7 xác suất trạng thái** bằng mô hình và bộ chuẩn hóa cố định. Với quy tắc chia chuỗi theo ngày (`daily_sequence`), đầu ngày mới đặt lại xác suất khởi đầu; **không chạy lại toàn bộ thí nghiệm hoặc huấn luyện lại mô hình hằng ngày**. Kết quả và trạng thái xử lý cuối được lưu để tiếp tục khi có quan sát kế tiếp. Nếu dữ liệu cũ bị sửa hoặc mô hình được thay phiên bản, chạy lại đoạn bị ảnh hưởng theo phiên bản tương ứng.

Đây là **suy luận khi có dữ liệu mới (online inference)**, chưa phải **học tham số liên tục (online learning)**: các phân phối của trạng thái, ma trận chuyển và bộ chuẩn hóa không tự thay đổi sau mỗi nến. Việc huấn luyện lại bằng dữ liệu mới là một quy trình riêng, có đánh giá và phiên bản mô hình mới. Phần nhận dữ liệu trực tiếp của v0.6 chưa triển khai xong; đoạn trên mô tả hành vi cần xây dựng.

**Quyết định đầu ra:** người dùng nhận báo cáo trạng thái **cuối ngày**, sau khi dữ liệu ngày đó đã hoàn tất theo nguồn. Hệ thống vẫn lưu kết quả từng quan sát để tính báo cáo và kiểm tra lại. Nguồn dữ liệu và cách xác định nến cuối ngày sẽ do AI ở máy chạy kiểm tra; xem `HANDOFF_OTHER_MACHINE.md`.

**Quyết định chia dữ liệu:** trên máy chạy, xây mô hình K=7 bằng dữ liệu hợp lệ **trước 01/08/2026** (đến hết 31/07/2026 nếu nguồn có đủ). Dữ liệu từ **01/08/2026 đến 30/09/2026** được giữ riêng làm tập kiểm tra chưa dùng (holdout). Chỉ dùng dữ liệu trước tháng 8 để học bộ chuẩn hóa, huấn luyện, chọn cấu hình và kiểm tra nội bộ; khóa toàn bộ quy trình trước khi đánh giá tháng 8–9. Không điều chỉnh mô hình theo kết quả tháng 8 rồi dùng tháng 9 để tuyên bố cả hai tháng là tập kiểm tra độc lập. Lưu mốc cuối thực tế của dữ liệu huấn luyện, hash nguồn, mã và phiên bản mô hình. Sau giai đoạn kiểm tra, dữ liệu ngày mới chỉ dùng để cập nhật xác suất trạng thái và đánh giá, **chưa tự đưa vào tập huấn luyện**; lịch huấn luyện lại sẽ quyết định sau. Bản `.bat` hiện tại vẫn dùng artifact v0.5 để chạy thử, chưa có mô hình mới theo cách chia này. Tính đến 24/09/2026, tháng 9 chưa hoàn tất nên chưa thể kết luận cho đủ hai tháng.

## Thứ tự công việc v0.6

1. **Kiểm kê ứng viên và state:** chạy audit từ các shard v0.5; tách số liệu validation dùng để chọn mô hình khỏi hồ sơ test chỉ dùng mô tả. Không coi ID state giữa hai shard là cùng một state.
2. **Định nghĩa hợp đồng dữ liệu cập nhật:** xem `V06_DATA_REPLAY_CONTRACT.md` cho timestamp của bar đã đóng, thời điểm hệ thống nhận, phiên, gap, trùng lặp, rollover và cách lưu bản gốc. Khi chưa có nguồn trực tiếp, replay dữ liệu phút lịch sử theo thứ tự thời gian với cùng giao diện.
3. **Replay nhân quả:** tại thời điểm `t` chỉ đưa return đã hoàn thành vào causal filter; lưu posterior đầy đủ, confidence, phiên bản model, dữ liệu/nguồn và lý do reset chuỗi. Chạy chế độ shadow, không tạo lệnh.
4. **Đánh giá 7 state:** occupancy mềm, phân phối lợi suất từng state, ma trận chuyển, thời lượng, one-bar runs, flicker, độ trễ xác nhận, độ ổn định theo seed/fold và ghép ID state. Dùng K khác làm đối chứng nghiên cứu khi cần; tách thay đổi cấu trúc dữ liệu khỏi thay đổi thị trường.
5. **Cổng chọn ứng viên:** chỉ đề xuất cấu hình khi cải thiện dự báo ngoài mẫu, state có ý nghĩa lặp lại và pipeline replay không nhìn trước. Nếu dữ liệu mới chưa đủ, tiếp tục theo dõi thay vì tuyên bố champion.

## Chạy audit đầu vào

```powershell
Set-Location 'D:\NCKHD\ARSH\ARSH v0.6'
python .\audit_starting_point.py
```

Output nằm trong `outputs/starting_point/`. Các file này là **bản kiểm kê**, không phải quyết định cuối cho v0.6. Chưa có nguồn dữ liệu cập nhật cụ thể nên giao diện kết nối trực tiếp sẽ được chốt khi biết nguồn và quy ước timestamp của nó.

`fold_champions.csv` và `validation_best_per_fold_horizon.csv` dùng điểm validation; `champion_state_profiles.csv` là mô tả trên test lịch sử và không được dùng để chọn policy hay siêu tham số.

## Chạy thử báo cáo cuối ngày trên máy Windows

Đặt thư mục `ARSH v0.6` cạnh bản clone `ARSH-Model` có đủ Git LFS. Chạy `SETUP_V06.bat` một lần để tạo môi trường Python 3.12 và cài thư viện; sau đó chạy `RUN_V06_EOD.bat YYYY-MM-DD` cho ngày muốn xử lý. Nếu không nhập ngày, chương trình chọn ngày mới nhất trong file CSV. Dữ liệu mặc định là `ARSH-Model/data/ohlc_export.csv`; báo cáo nằm trong `outputs/eod/`. Có thể đặt biến môi trường `ARSH_REPO`, `ARSH_DATA`, `ARSH_PYTHON` khi dùng đường dẫn khác.

Chương trình `v06_replay_eod.py` dùng **đúng định dạng CSV và cách tạo lợi suất của v0.5**, nạp tệp mô hình K=7 theo ngày, khoảng tính lợi suất 1 phút, họ Student-t bậc tự do theo trạng thái ở shard cuối. Đây là **ứng viên nghiên cứu tạm thời**, chưa phải quyết định chọn họ HMM cuối cùng. Mỗi lần chạy chỉ tính các quan sát của một ngày được chọn, không huấn luyện lại mô hình. Kết quả gồm CSV từng quan sát và JSON báo cáo cuối ngày. Vì CSV lịch sử không xác nhận được lúc nào ngày đã hoàn tất, báo cáo gắn cờ `source_day_completeness_unverified`. Nguồn trực tiếp khác định dạng này cần bộ kết nối nguồn riêng trên máy chạy trước khi sử dụng thực tế.

Tên file kết quả chứa mã băm của dữ liệu ngày và mô hình. Nếu nến cũ được sửa hoặc thay mô hình, lần chạy sau tạo bộ file khác để không âm thầm ghi đè bản cũ.

Kiểm tra trên máy soạn tài liệu: lệnh `.bat` đã xử lý 238 quan sát ngày 17/07/2026 và tạo hai file đầu ra. Chạy lại ngày 05/05/2026 cho 238 quan sát; so với `test_predictions_long.csv` của fold 9, thời điểm, độ tin cậy, ID trạng thái và điểm đặt lại chuỗi khớp **238/238**, sai khác độ tin cậy lớn nhất bằng 0. Đây là kiểm tra tính đúng của bản chạy lại dữ liệu lịch sử, không phải một kết quả dự báo ngoài mẫu mới.

## Chạy hằng ngày từ cơ sở dữ liệu (cập nhật 25/09/2026)

Máy chạy có nguồn cập nhật hằng ngày: PostgreSQL bảng `bars_1m`, thông tin đăng nhập nằm trong biến môi trường `PG_DSN` (không ghi trong mã). Chạy `RUN_V06_DAILY.bat` sau khi phiên đóng cửa:

1. `v06_fetch_db.py` lấy nến VN30F1M từ 06/11/2025 (mốc huấn luyện của artifact v0.5) đến hôm nay, ghi mỗi ngày **đã hoàn tất** thành `data/days/VN30F1M_<ngày>_v<N>.csv` theo định dạng CSV v0.5, kèm file JSON ghi hash, số nến, phút thiếu và thời điểm có dữ liệu. Nếu nhà cung cấp sửa một ngày cũ, tạo bản `_v2`, không ghi đè bản cũ.
2. `v06_daily.py` chỉ lập báo cáo cho ngày mới hoặc ngày có bản sửa (theo dõi trong `outputs/eod/processed.json`), gọi `v06_replay_eod.py` với `--source-meta` để báo cáo ghi `source_day_complete` và cảnh báo dữ liệu. Ngày chưa hoàn tất chỉ có ghi chú `pending_<ngày>.json`, chưa có báo cáo.

Quy ước thời gian của nguồn, quy tắc ngày hoàn tất và các giới hạn đã biết ghi ở mục D06 của `DEFERRED_OPTIMIZATION_LOG.md`. Kiểm tra ngày 05/05/2026: nến lấy từ DB trùng CSV v0.5 và xác suất trạng thái trùng 238/238, sai lệch 0.

**Cập nhật 26/09/2026:** đã huấn luyện và khóa model theo mốc 01/08/2026 (`V06_TRAINING_PLAN.md` → `V06_TRAINING_RESULT.md`, kiểm tra 03/08–25/09 đạt). `RUN_V06_DAILY.bat` tự dùng model này khi `models/v06_k7_cut20260801/artifact_manifest.json` tồn tại và ghi báo cáo vào `outputs/eod_v06/`. Windows Task Scheduler có tác vụ "ARSH v0.6 bao cao cuoi ngay" chạy 15:15 mỗi ngày: cuối tuần hoặc ngày trong `market_holidays.txt` thì không lập báo cáo; có phiên nhưng dữ liệu chưa hoàn tất thì kiểm tra lại mỗi 10 phút đến 18:00, quá hạn ghi `pending_<ngày>.json` và lần chạy sau lấy lại; ngày thường không có dữ liệu mã Việt Nam nào thì ghi "chưa xác định (ngày nghỉ hoặc nguồn lỗi)". Nhật ký: `outputs/eod_v06/daily_log.txt`.

**Quyết định 25/09/2026 về tập kiểm tra:** tập kiểm tra là từ 01/08/2026 **đến ngày cuối có dữ liệu hoàn tất tại lúc chạy xong mô hình**, không cần chờ hết tháng 9. Lần chạy ngày 25/09/2026 với artifact v0.5 có tập kiểm tra 01/08–25/09/2026 (37 phiên, 2 phiên có cảnh báo thiếu nến: 18/08, 25/09). Ghi ngày cuối thực tế vào mỗi lần đánh giá.

## Quyết định lộ trình

Báo cáo ngày 22/09/2026 tại `../PROJECT_STATUS_REPORTS/2026-09-22/ARSH_v0.5_Bao_cao_da_lam_va_chua_lam.md` ghi quyết định ngày 18/09: có thể bắt đầu v0.6 và ưu tiên dữ liệu cập nhật dù v0.5 còn thí nghiệm chưa chạy. Một số tài liệu v0.5 cũ còn câu “phải xong v0.5 mới sang v0.6”; quyết định mới thay thế điều kiện đó. Danh sách việc v0.5 vẫn được giữ riêng để tiếp tục sau.
