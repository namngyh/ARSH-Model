# ARSH v0.6: logic thực hiện, mục tiêu và đường lên v1

Ngày lập: 24/09/2026. Tài liệu này mô tả bản đã viết và chạy thử trên máy hiện tại, đồng thời tách rõ phần sẽ làm ở máy chạy thật. ARSH là hệ thống nhận diện trạng thái thị trường VN30F1M. Đây là tài liệu giải thích kỹ thuật, không phải tín hiệu mua hoặc bán.

## 1. Mục tiêu của v0.6

V0.6 dùng mô hình ARSH có **7 trạng thái thị trường (K=7)** để đọc dữ liệu giá theo thời gian, tính xác suất cho từng trạng thái và gửi một báo cáo vào **cuối ngày**. Mục tiêu nghiên cứu là biết 7 trạng thái xuất hiện khi nào, kéo dài bao lâu, chuyển qua lại ra sao và có đủ ổn định để dùng trong hệ thống lâu dài không.

V0.5 đã tạo các mô hình và kết quả nghiên cứu trên dữ liệu lịch sử. V0.6 là bước đưa một mô hình đã lưu vào quy trình xử lý dữ liệu theo thời gian. Làm vậy giúp kiểm tra cả mô hình lẫn chất lượng dữ liệu mới, trước khi phát triển hệ thống v1 vận hành liên tục.

**Tình trạng thực tế:** đã có chương trình Python và file `.bat` chạy báo cáo cuối ngày từ file CSV (bảng dữ liệu dạng văn bản) có định dạng v0.5. Chưa có bộ kết nối API (giao diện nhận dữ liệu tự động) hoặc nguồn trực tiếp của máy kia; chưa có báo cáo từ nguồn trực tiếp đó. Mô hình đang dùng là **ứng viên nghiên cứu tạm thời**, chưa phải mô hình cuối cùng cho vận hành.

**Quyết định chia dữ liệu:** xây mô hình K=7 chỉ với dữ liệu hợp lệ **trước 01/08/2026**. Giữ riêng tháng 8 và tháng 9/2026 làm tập kiểm tra chưa dùng. Trước khi xem kết quả của hai tháng này, cố định cấu hình, bộ chuẩn hóa và quy trình đánh giá. Mô hình theo mốc mới chưa được huấn luyện trong gói hiện tại; file `.bat` bên dưới vẫn chạy thử với mô hình v0.5 đã lưu. Dữ liệu mới không tự nhập vào tập huấn luyện.

## 2. Những từ cần hiểu trước

- **Nến 1 phút (bar):** một dòng giá mở, cao nhất, thấp nhất, đóng cửa và khối lượng của một phút. Chỉ dùng nến đã hoàn thành.
- **Khoảng tính lợi suất (horizon):** độ dài dùng để tính một thay đổi giá đưa vào mô hình. Bản chạy thử dùng 1 phút. Nến và khoảng tính lợi suất là hai khái niệm khác nhau; từ nến 1 phút vẫn có thể tính lợi suất 2 phút. Khoảng này không có nghĩa dự báo giá sau 1 hoặc 2 phút.
- **Lợi suất log:** `ln(giá đóng cửa cuối / giá đóng cửa đầu)` của một khoảng hợp lệ. Đây là quan sát mà ARSH đọc.
- **Trạng thái thị trường:** một nhóm ẩn của mô hình. K=7 nghĩa là mô hình có 7 nhóm, mỗi nhóm có phân phối lợi suất riêng. Tên số 0–6 chưa tự mang nghĩa “tốt”, “xấu”, “tăng” hay “giảm”.
- **Xác suất trạng thái (posterior):** bảy số không âm có tổng gần bằng 100%, biểu thị mức phù hợp của 7 trạng thái sau khi nhận quan sát mới. Nhãn hiển thị là trạng thái có xác suất cao nhất.
- **Chạy lại dữ liệu theo thời gian (replay):** cho máy đọc các nến lịch sử lần lượt như khi chúng đến, nhằm thử logic trước khi nối nguồn trực tiếp.
- **Suy luận khi có dữ liệu mới:** cập nhật xác suất trạng thái, giữ nguyên tham số mô hình. **Học tham số liên tục** là tự thay phân phối hoặc ma trận chuyển sau mỗi quan sát; v0.6 hiện chưa làm việc này.

## 3. Dòng xử lý của bản chạy thử

[[PIPELINE_DIAGRAM]]

### Bước 1 — Nạp mô hình và nến

File `RUN_V06_EOD.bat` gọi `v06_replay_eod.py` trên máy Windows. Chương trình nạp mô hình K=7 đã lưu từ kết quả v0.5, cùng bộ chuẩn hóa. Mô hình mặc định lấy từ thư mục kết quả `policy_daily__f08_09`: mô hình Markov ẩn (HMM) dùng phân phối Student-t cho lợi suất của từng trạng thái, khoảng tính lợi suất 1 phút, quy tắc chia chuỗi theo ngày. Student-t cho phép những thay đổi giá hiếm nhưng lớn xuất hiện nhiều hơn so với phân phối Gaussian. Mô hình này được huấn luyện từ dữ liệu trước 06/11/2025; các ngày được chạy lại phải nằm sau mốc đó.

Chương trình đọc CSV VN30F1M theo đúng các cột của v0.5: `SYMBOL`, `TRADING_DATE`, `TRADING_TIME`, `OPEN_PX`, `HIGH_PX`, `LOW_PX`, `CLOSE_PX`. Nó lọc mã VN30F1M, kiểm tra dấu thời gian trùng và giá không hợp lệ, rồi chỉ lấy **một ngày** được yêu cầu. Nếu không nhập ngày, chương trình chọn ngày mới nhất trong CSV. Nó ghi mã băm (dấu nhận dạng tính từ nội dung file) của nguồn và mô hình để truy vết phiên bản.

### Bước 2 — Tạo quan sát hợp lệ

Chương trình dùng quy tắc tạo lợi suất đã dùng trong v0.5: trong từng khoảng giao dịch liên tục được cấu hình, lấy hai giá đóng cửa cách nhau 1 phút, yêu cầu các nến cần thiết có mặt và không nội suy nến thiếu. Nó không tạo lợi suất thông thường xuyên nghỉ giữa phiên, phiên đóng cửa đặc biệt hoặc qua đêm. Một cửa sổ bị thiếu là ranh giới cứng: quan sát hợp lệ tiếp theo bắt đầu lại xác suất đầu chuỗi.

Ví dụ, nếu giá đóng cửa ở hai mốc liền nhau là 1.000 và 1.001, lợi suất log đầu vào xấp xỉ `ln(1001/1000) = 0,001`. Con số này chỉ là ví dụ để hiểu phép tính, không phải dự báo thị trường.

### Bước 3 — Chuẩn hóa và tính 7 xác suất

Lợi suất mới được biến đổi bằng **bộ chuẩn hóa đã học cùng mô hình**; bộ chuẩn hóa không học lại từ nến của ngày đang đọc. Mô hình Markov ẩn (HMM) dùng ba phần đã lưu: xác suất khởi đầu, ma trận chuyển giữa 7 trạng thái và phân phối lợi suất Student-t của từng trạng thái.

Ở quan sát đầu ngày hoặc sau khoảng thiếu, mô hình lấy xác suất khởi đầu. Với các quan sát liên tiếp, nó dùng 7 xác suất trước đó và ma trận chuyển để tính xác suất trước khi thấy lợi suất mới. Sau đó nó so lợi suất mới với phân phối của mỗi trạng thái và chuẩn hóa thành **7 xác suất trạng thái mới**. Viết gọn: `xác suất mới của trạng thái i ∝ xác suất trước của i × mức phù hợp của lợi suất với trạng thái i`.

Ví dụ giả định: `[5%, 10%, 60%, 5%, 10%, 5%, 5%]`. Máy hiển thị trạng thái thứ ba với độ tin cậy 60%, nhưng vẫn giữ sáu xác suất còn lại. Trạng thái này không có hai phân phối cùng lúc; **mỗi trạng thái có phân phối riêng, còn một nến có thể thuộc nhiều trạng thái với mức xác suất khác nhau**.

### Bước 4 — Tạo báo cáo cuối ngày

Chương trình lưu một CSV chứa kết quả của từng quan sát: thời điểm, lợi suất, ranh giới bắt đầu chuỗi, 7 xác suất, trạng thái hiển thị và độ tin cậy. Nó tạo thêm file JSON (bản ghi có cấu trúc) cuối ngày với 7 xác suất của **quan sát hợp lệ cuối cùng**, số quan sát, số cửa sổ bị loại, mã băm dữ liệu/mô hình và vị trí file chi tiết. Tên file chứa mã băm của dữ liệu ngày và mô hình, nên nếu nến cũ hoặc mô hình thay đổi thì tạo bộ file khác.

CSV lịch sử hiện tại không cho biết chắc chắn lúc nào dữ liệu ngày đã hoàn tất. Vì vậy báo cáo chạy thử ghi `source_day_completeness_unverified`. Ở máy kia, bộ kết nối nguồn phải xác định rõ ngày đã đủ dữ liệu trước khi gửi báo cáo cuối ngày cho người dùng.

## 4. Ngày mới có huấn luyện lại mô hình không?

**Không.** File `.bat` hiện chạy cho ngày được chọn: nạp cùng một mô hình đã huấn luyện, tính lại xác suất của các quan sát trong **ngày đó**, rồi tạo báo cáo. Nó không chạy lại toàn bộ thí nghiệm v0.5, không fit lại 7 phân phối và không đổi ma trận chuyển. Vì quy tắc hiện chọn là chia chuỗi theo ngày, đầu ngày mới bắt đầu lại từ xác suất khởi đầu; không cần mang xác suất cuối ngày hôm trước sang.

Lần chuẩn bị mô hình mới chỉ dùng lịch sử **trước 01/08/2026**. Cần ghi ngày cuối cùng thực tế đã đưa vào huấn luyện: CSV trong gói hiện mới đến 17/07/2026, nên máy chạy phải kiểm tra dữ liệu phần còn lại của tháng 7. Tháng 8–9 được giữ riêng để kiểm tra mô hình ngoài giai đoạn huấn luyện. Dự án sẽ quyết định có đưa dữ liệu mới vào lần huấn luyện tiếp theo hay không **sau một thời gian vận hành**, dựa trên chất lượng thực tế.

Mọi bước chọn họ HMM, khoảng tính lợi suất, quy tắc chia chuỗi, học bộ chuẩn hóa và chỉnh tham số đều dùng dữ liệu trước tháng 8. Sau khi khóa mô hình và cách chấm, chạy lần lượt dữ liệu tháng 8–9 mà **không học lại hay sửa cấu hình giữa hai tháng**. Có thể xem báo cáo tạm cho phần tháng 9 đã có, nhưng tính đến 24/09/2026 chưa thể kết luận cho cả tháng 9. Nếu sửa mô hình dựa trên kết quả hai tháng này, chúng không còn là tập kiểm tra độc lập của mô hình đã sửa; phải dùng một giai đoạn tương lai khác để kiểm tra tiếp.

Nếu sau này dữ liệu thị trường đổi nhiều đến mức mô hình cũ giảm chất lượng, việc huấn luyện lại phải là một công việc **riêng**: dùng dữ liệu phù hợp, đánh giá, ghi phiên bản mô hình mới và so sánh với mô hình cũ. Không tự đổi tham số sau từng nến rồi gọi đó là v0.6 hiện tại.

## 5. Điều đã xác nhận và giới hạn hiện nay

**Đã làm:** cố định K=7 cho v0.6; kiểm kê kết quả v0.5; viết hợp đồng dữ liệu và bảng thuật ngữ; tạo chương trình, file cài thư viện và file `.bat` chạy báo cáo cuối ngày; nạp được mô hình K=7 và tạo báo cáo cho ngày 17/07/2026 với 238 quan sát. Chạy lại ngày 05/05/2026 cho kết quả trùng **238/238** thời điểm, độ tin cậy, ID trạng thái và điểm đặt lại chuỗi so với kết quả v0.5 đã lưu. Phép đối chiếu này kiểm tra logic chương trình, không phải một chứng cứ dự báo mới.

**Chưa làm trên máy chạy:** huấn luyện một mô hình K=7 bằng dữ liệu trước 01/08/2026 rồi nối báo cáo với artifact mới; bổ sung hoặc xác nhận dữ liệu sau 17/07/2026; xác minh dấu thời gian của nhà cung cấp là đầu hay cuối nến, thời điểm nến sẵn dùng, cách báo ngày hoàn tất, nến thiếu/trùng/sửa lại, mã hợp đồng và chuyển hợp đồng; viết bộ kết nối nguồn; cấu hình lịch chạy tự động và theo dõi lỗi. Bản CSV lịch sử chỉ có mã tổng hợp VN30F1M, nên một số vấn đề chuyển hợp đồng không thể kết luận dứt điểm từ file này.

**Chưa tối ưu:** K=7 được chốt theo quyết định dự án và kết quả so sánh K=2–7, nhưng chưa thử biên K=8–9; họ HMM, khoảng tính lợi suất và quy tắc chia chuỗi chưa được chốt làm cấu hình cuối. Những mục này được ghi trong `DEFERRED_OPTIMIZATION_LOG.md` và không ngăn bản chạy thử nghiên cứu. Hiện chưa có học tham số liên tục, CNN đọc tin tức, học tăng cường hay giao dịch tự động trong bản này.

## 6. Vì sao làm theo thứ tự này?

Thứ nhất, mô hình đã có từ v0.5 nên có thể kiểm tra ngay cách nó phản ứng với từng quan sát mới mà không phải huấn luyện lại mỗi ngày. Thứ hai, chỉ đọc nến đã hoàn thành và giữ thứ tự thông tin sẵn dùng giúp tránh việc dùng dữ liệu tương lai. Thứ ba, kết quả theo từng quan sát và hash cho phép truy lại một báo cáo cuối ngày khi dữ liệu bị sửa. Thứ tư, chạy ở chế độ quan sát tạo dữ liệu mới để kiểm tra 7 trạng thái trước khi dùng chúng cho quyết định khác.

Mục tiêu của bước này là **biết hệ thống nhận diện trạng thái có chạy đúng và có ổn định trên dữ liệu mới không**. Chênh lệch lợi nhuận không phải tiêu chí duy nhất để gọi một trạng thái là tốt; v0.6 hiện không tạo lệnh mua bán.

## 7. Dự định đi từ v0.6 lên v1

**Phần còn lại của v0.6 trên máy kia:** AI ở máy chạy xác minh dữ liệu, khóa cấu hình bằng giai đoạn trước tháng 8 rồi huấn luyện K=7 chỉ bằng dữ liệu trước 01/08/2026; lưu bộ chuẩn hóa, mốc dữ liệu, phiên bản và mã băm. Sau đó kiểm tra ngoài mẫu bằng tháng 8–9, theo đúng thứ tự thời gian và không điều chỉnh mô hình giữa chừng. AI xác định nguồn dữ liệu, tài liệu dấu thời gian, cách biết ngày đã hoàn tất; viết bộ kết nối đưa nến mới vào cùng quy trình; chạy chế độ quan sát và lưu báo cáo cuối ngày. Nó kiểm tra mô hình mới tải đúng và kết quả có 7 xác suất hợp lệ, không dùng nến tương lai. Nếu phát hiện một lựa chọn tạm thời chưa tối ưu, ghi vào sổ thay vì tự đổi mô hình âm thầm.

**Điều kiện để gọi v1.0 là bước vận hành ổn định:** nguồn trực tiếp hoạt động có giám sát; xử lý nến thiếu/trùng/tới muộn và chuyển hợp đồng có quy tắc rõ; báo cáo chỉ phát sau khi ngày dữ liệu hoàn tất; dữ liệu và mô hình có phiên bản; có thể tái tạo báo cáo và phục hồi sau khi chương trình dừng; 7 trạng thái được theo dõi trên dữ liệu thực sự mới. Khi cần đổi mô hình, phải đánh giá và phát hành phiên bản rõ ràng. V1.0 tiếp tục là **nhận diện trạng thái**, chưa phát lệnh giao dịch.

**Sau v1.0:** lộ trình cũ dành v1.1 cho giao dịch thử trên giấy (paper trading), tức mô phỏng lệnh mà không dùng tiền thật. Chỉ nên thiết kế bước đó khi nhận diện trạng thái và dữ liệu đầu vào đã đủ tin cậy. CNN đọc tin tức hoặc học tham số liên tục là các nghiên cứu bổ sung tùy nhu cầu, không phải điều kiện mặc định để hoàn thành v1.0.

Kế hoạch cũ đặt lần đầu nối dữ liệu trực tiếp ở v1.0. Quyết định hiện tại ưu tiên thử nguồn dữ liệu ngay ở v0.6; vì vậy v1.0 được hiểu là bước **làm ổn định và chuẩn hóa việc vận hành trực tiếp**, không phải chỉ cắm dây nguồn lần đầu. Thời điểm lên v1 phụ thuộc bằng chứng vận hành trên máy kia, không phụ thuộc một ngày cố định.

## 8. Cách chạy bản hiện có và cách đọc kết quả

Đặt thư mục `ARSH v0.6` cạnh repo `ARSH-Model` đã tải đủ Git LFS. Chạy `SETUP_V06.bat` một lần trên máy Windows để cài thư viện. Khi có file CSV đúng định dạng v0.5, chạy `RUN_V06_EOD.bat YYYY-MM-DD`, hoặc bỏ ngày để dùng ngày mới nhất trong file. Máy tự tạo báo cáo trong `outputs/eod/`; mỗi lần chạy này không cần AI tiêu tốn token để tính xác suất.

Lưu ý: file `.bat` trong gói **vẫn nạp mô hình cũ từ v0.5**. Sau khi huấn luyện bằng dữ liệu trước tháng 8 trên máy kia, AI phải cấu hình chương trình nạp artifact mới, kiểm tra phiên bản và mốc dữ liệu. Chỉ những báo cáo sau bước đó mới được gọi là kết quả của mô hình theo mốc chia mới.

Đọc `report_*.json` trước: `final_posterior` là 7 xác suất của quan sát hợp lệ cuối cùng; `final_confidence` là xác suất lớn nhất; `observations` là số lợi suất hợp lệ; `rejected_windows` là số cửa sổ không đủ dữ liệu. Nếu thấy `source_day_completeness_unverified`, đó là **báo cáo chạy thử**, chưa xác nhận nến cuối ngày của nguồn trực tiếp. File `states_*.csv` giữ toàn bộ chuỗi để kiểm tra tại sao xác suất cuối ngày có giá trị đó.

## 9. Nguồn nội bộ dùng để lập tài liệu

Tài liệu dựa trên mã `v06_replay_eod.py`, `runtime.py` và `arsh_v05.py`; báo cáo/manifest của shard `policy_daily__f08_09`; `K_DECISION.md`, `V06_DATA_REPLAY_CONTRACT.md`, `DEFERRED_OPTIMIZATION_LOG.md`; và yêu cầu dự án `ARSH_PROJECT_REQUIREMENTS.md`. Các file này đi cùng gói bàn giao hoặc repo nguồn để AI máy kia kiểm tra lại.
