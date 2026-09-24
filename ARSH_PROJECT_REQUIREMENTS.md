# ARSH — Yêu cầu và quyết định dự án

Tài liệu này lưu các yêu cầu đã được xác nhận nhưng chưa nhất thiết triển khai ngay. Mục đích là tránh bỏ quên yêu cầu khi chuyển phiên bản.

## ARSH-REQ-TF-001 — Tìm khung thời gian và cách tạo lợi suất tối ưu

- **Trạng thái:** Đã kích hoạt trong ARSH v0.4.
- **Phạm vi áp dụng:** Một phiên bản nghiên cứu sau v0.3, khi mô hình regime cơ sở đã đủ ổn định để tiến hành nghiên cứu tần suất dữ liệu.
- **Quyết định hiện tại:** v0.3 tiếp tục dùng lợi suất log 60 phút ở ba cửa sổ 09:00–10:00, 10:00–11:00 và 13:00–14:00. Không thay tần suất trong cùng thí nghiệm so sánh Gaussian HMM và Student-t HMM.

### Yêu cầu bắt buộc khi được kích hoạt

Phải thực hiện một nghiên cứu ngoài mẫu để lựa chọn khung thời gian và cách tạo lợi suất phù hợp nhất cho nhận diện regime VN30F1M. Không được mặc định ba cửa sổ 60 phút hiện tại là tối ưu.

Theo quyết định mới nhất của người dùng, v0.4 so sánh các khung:

- 5 phút;
- 10 phút;
- 15 phút;
- 30 phút;
- 60 phút.

Các lợi suất so sánh phải được tạo theo quy tắc nhất quán:

- ưu tiên cửa sổ không chồng lấn;
- không tính một lợi suất thông thường qua nghỉ trưa, qua đêm, cuối tuần hoặc khoảng thiếu dữ liệu;
- ghi nhận riêng opening, closing, lunch gap và overnight gap;
- không nội suy giá nếu chưa có một thí nghiệm riêng chứng minh lợi ích;
- xác minh lịch giao dịch, quy ước timestamp và cơ chế rollover của VN30F1M trước khi kết luận;
- lưu số quan sát, tỷ lệ return bằng 0, missing window, outlier và thống kê phân phối cho từng khung.

### Thiết kế so sánh

Khi nghiên cứu tần suất, phải giữ cố định họ mô hình, cách chia walk-forward, cách chuẩn hóa, phạm vi K và ngân sách tối ưu trong cùng phép so sánh. Mỗi tần suất phải dùng train/validation/test theo cùng các mốc thời gian lịch để tránh lợi thế do khác giai đoạn thị trường.

Không chọn khung thời gian chỉ bằng kết quả trên một fold hoặc bằng in-sample likelihood. Dữ liệu test đã được xem trong quá trình nghiên cứu phải được gọi là research benchmark; cần dành dữ liệu tương lai làm final holdout.

### Tiêu chí lựa chọn

Không có một chỉ tiêu đơn lẻ tự động quyết định khung tối ưu. Quyết định phải xem đồng thời:

1. predictive log-density ngoài mẫu và paired bootstrap theo ngày;
2. độ ổn định của số state K giữa các fold;
3. occupancy, số state rỗng và mức độ tách biệt giữa các state;
4. độ ổn định của state sau khi căn chỉnh xuyên fold;
5. confidence và độ nhiễu/flicker của chuỗi posterior;
6. thời lượng regime quy đổi về phút, giờ và ngày thay vì chỉ số quan sát;
7. độ trễ phát hiện khi regime thay đổi;
8. độ nhạy với return bằng 0, microstructure noise, outlier và rollover;
9. chi phí tính toán và khả năng cập nhật khi paper trading;
10. tính hữu ích cho các mô hình downstream, chỉ đánh giá sau khi chất lượng regime đã đạt yêu cầu.

Khung được chọn phải đạt sự cân bằng giữa chất lượng dự báo phân phối, độ ổn định và tốc độ phát hiện. Lợi nhuận có thể là tiêu chí downstream sau này, nhưng không được dùng một mình để định nghĩa regime tốt.

### Đầu ra bắt buộc

Phiên bản nghiên cứu tần suất phải tạo:

- bảng audit dữ liệu cho từng khung;
- bảng kết quả từng fold và kết quả tổng hợp;
- bootstrap so sánh các khung;
- phân tích thời lượng và chuyển trạng thái theo thời gian thực;
- phân tích riêng theo thời điểm trong phiên;
- báo cáo HTML và PDF;
- file cấu hình, seed, phiên bản thư viện và SHA-256 của dữ liệu/mã/đầu ra chính;
- kết luận nêu rõ khung được chọn, mức độ chắc chắn và các trường hợp khung đó thất bại.

### Điều kiện kích hoạt

Chỉ bắt đầu yêu cầu này khi đã hoàn thành phép so sánh mô hình chính của v0.3 và có pipeline regime đủ ổn định để chạy cùng một mô hình trên nhiều tần suất. Nếu v0.3 cho thấy cấu trúc state chưa đáng tin, phải xử lý vấn đề mô hình trước khi tối ưu tần suất.

## ARSH-V03-SCOPE-001 — Phạm vi đã xác nhận cho v0.3

- **Trạng thái:** Đã xác nhận trước triển khai.
- **Vai trò phiên bản:** v0.3 là bước nghiên cứu offline tiếp theo và tạo lõi kỹ thuật tái sử dụng cho các phiên bản sau; chưa kết nối dữ liệu trực tiếp.
- **Mục tiêu:** so sánh Gaussian HMM với Student-t HMM và kiểm tra xem state có ổn định, có ý nghĩa phân phối và đủ điều kiện chuyển sang kiến trúc vận hành hay không.

### Mô hình và tham số

- Thử cả Student-t HMM dùng một bậc tự do chung và Student-t HMM dùng bậc tự do riêng cho từng state.
- Giữ các baseline Gaussian i.i.d., Student-t i.i.d., GMM và Gaussian HMM. Chúng là bộ đối chứng chính cho câu hỏi về Gaussian, đuôi dày, mixture và ký ức Markov.
- K tiếp tục được validation lựa chọn trong tập 2–5; chưa khóa K production trước khi có kết quả v0.3.
- Giữ cổng soft occupancy chính ở 1%; bổ sung phân tích độ nhạy tại 0,5%, 2% và 5%.
- Chưa dùng HireVAE và chưa dùng online parameter learning trong v0.3.

### Thời gian và đánh giá

- Giữ dữ liệu lịch sử, cách tạo lợi suất 60 phút và thiết kế walk-forward 3 năm train, 6 tháng validation, 6 tháng test, bước tiến 6 tháng để so sánh với v0.2.
- Đánh giá `continuous_carry`, `daily_sequence` và `session_sequence`; chính sách reset phải nhất quán giữa huấn luyện và filtering.
- Các test period đã được xem được gọi là research benchmark, không phải final untouched holdout.
- Giữ predictive log-density làm metric chính; bổ sung daily bootstrap và moving-block bootstrap 5, 10, 20 ngày.
- Quyết định có thay Gaussian HMM bằng Student-t HMM phải xét đồng thời log-density, bootstrap, kết quả theo fold, hội tụ, K, occupancy, state rỗng, độ tách biệt và độ ổn định state.

### State và khả năng tái sử dụng

- **Đã xác nhận:** căn chỉnh state xuyên fold và giữa các lần retrain; lưu cả nhãn nội bộ lẫn `stable_state_id`, cho phép state mới xuất hiện hoặc state cũ kết thúc.
- Trong v0.3 chưa truyền posterior qua mốc retrain; mô hình mới khởi tạo posterior theo chính sách chuỗi đã chọn.
- Không ép hai state khác bản chất phải dùng chung ID; state không có đối sánh đủ tốt phải được đánh dấu mới, state biến mất phải được đánh dấu kết thúc.
- V0.3 phải có giao diện huấn luyện, dự báo một quan sát theo causal filtering, save/load, lưu scaler, posterior, tham số, cấu hình, data/model hash và nhật ký kiểm toán.
- V0.3 và các phiên bản 0.x tiếp theo chỉ nhận diện state. V1.0 bắt đầu nhận dữ liệu trực tiếp và vẫn chỉ nhận diện; paper trading được hoãn đến v1.1.

### Lộ trình dự kiến sau v0.3

- v0.4 dành cho thí nghiệm còn thiếu trước dữ liệu trực tiếp, ưu tiên yêu cầu `ARSH-REQ-TF-001` về khung thời gian/cách tạo lợi suất.
- v0.5 chỉ được tạo nếu còn một thay đổi nghiên cứu lớn cần cô lập, chẳng hạn đặc trưng đa biến hoặc HireVAE. Không mặc định phải dùng HireVAE nếu mô hình đơn giản đã đủ tốt.
- v1.0 kết nối dữ liệu thật và vận hành state engine; v1.1 mới bổ sung paper trading.

## ARSH-V04-SCOPE-001 — Phạm vi đã xác nhận cho v0.4

- Tạo riêng chuỗi lợi suất log 5, 10, 15, 30 và 60 phút; không ghép đa biến.
- Dùng bar không chồng lấn và toàn bộ phần giao dịch liên tục 09:00–11:30, 13:00–14:30.
- Không tạo return qua nghỉ trưa, khoảng ATC, qua đêm hoặc cửa sổ thiếu phút; không nội suy trong thí nghiệm chính.
- Dùng `continuous_carry`; chỉ kiểm tra lại reset trên horizon thắng cuộc nếu cần robustness check.
- Sàng lọc tất cả horizon bằng Gaussian, Student-t, GMM và Gaussian HMM; kiểm tra hai horizon đầu bằng Student-t HMM shared-df và state-df.
- K=2–5; occupancy chính 1% và độ nhạy 0,5%/2%/5%.
- Giữ walk-forward 3 năm/6 tháng/6 tháng và cùng mốc lịch giữa các horizon.
- Báo cáo thời lượng state bằng số bar, phút giao dịch và phút lịch; không gọi số bar là số giao dịch khớp lệnh.
- Chưa dùng online parameter learning, HireVAE, tín hiệu giao dịch hoặc paper trading.

## ARSH-V05-SCOPE-001 — Phạm vi đã xác nhận cho v0.5

- **Vai trò phiên bản:** khóa lớp dữ liệu, cách tạo lợi suất, khung lợi suất, họ phân phối và phạm vi K trước khi v0.6 tập trung sâu vào cấu trúc regime.
- **Vị trí lưu:** toàn bộ code, checkpoint, output và báo cáo của phiên bản được lưu tại `D:\NCKH\ARSH v0.5` để dùng dung lượng ổ D; dữ liệu nguồn lịch sử vẫn được đọc từ vị trí hiện tại và phải ghi SHA-256 vào audit.
- Tiếp tục nghiên cứu offline bằng file lịch sử hiện tại; chưa kết nối trực tiếp, online parameter learning, tín hiệu hoặc paper trading.
- Không đưa lợi suất ngày vào v0.5. Nghiên cứu lợi suất ngày được hoãn đến sau v1.1.
- Khung intraday dự kiến: 1, 2, 3, 5, 10, 15, 20, 30, 45, 60 và 90 phút.
- Log-return close-to-close là dữ liệu chính; non-overlapping là thiết kế chính và overlapping chỉ là sensitivity.
- Không nối return thông thường qua nghỉ trưa, ATC, qua đêm, ngày nghỉ, dữ liệu thiếu hoặc rollover. Gap được audit riêng trong v0.5 và chỉ được cân nhắc làm thông tin regime trong v0.6.
- Không nội suy dữ liệu chính; kiểm tra sensitivity cho cửa sổ thiếu đúng một phút.
- So sánh raw return với return đã điều chỉnh mùa vụ biến động trong ngày; mọi tham số điều chỉnh chỉ được fit trên train.
- Giữ return bằng 0; không winsorize dữ liệu chính; chạy sensitivity riêng cho zero và outlier.
- Baseline phân phối: Gaussian, Student-t, GMM và skewed Student-t nếu triển khai ổn định. Mô hình state: Gaussian HMM, Student-t HMM shared-df và Student-t HMM state-specific-df.
- Thử đầy đủ K=2–7. Không loại K nhỏ; K=2 là baseline, K=3 là ứng viên gọn, K=4 là ứng viên cân bằng và K=5 là champion validation hiện tại.
- Cổng soft occupancy chính 1%, sensitivity 0,5%/2%/5%. Chọn K bằng predictive density, occupancy, convergence, seed/fold stability, flicker, duration, alignment và mức cải thiện so với K nhỏ hơn; ưu tiên K nhỏ hơn nếu nằm trong vùng kết quả tương đương.
- Ngân sách dự kiến: screening 3 seed và 100 vòng; refit 5 seed và 300 vòng; ca khó được audit đến 500 vòng. Nhiều seed dùng khi huấn luyện/nghiên cứu, không chạy lại cho mỗi bar trực tiếp.
- Giữ walk-forward 3 năm train/6 tháng validation/6 tháng test/bước 6 tháng; kiểm tra độ nhạy train 1 năm, 2 năm và expanding window. Cửa sổ có thể thay đổi khi lịch sử trực tiếp dài hơn.
- Research test đã được xem không được gọi là untouched holdout. Khóa confirmation set cho v0.5–v0.6 và dùng dữ liệu phát sinh sau khi v1 vận hành làm holdout thực sự mới.
- Lưu metric, cấu hình, seed, tham số, convergence, occupancy, dynamics, stability và lý do chọn/loại cho mọi ứng viên. Chỉ lưu posterior đầy đủ cho champion, ứng viên gần tương đương và đối chứng quan trọng.
- V0.5 lưu toàn bộ bằng chứng của cấu hình bị loại; v0.6 chỉ phân tích sâu cấu hình chính và đối chứng, không xóa lịch sử nghiên cứu.

### Audit nguồn đã thực hiện trước v0.5

- File có 525.635 dòng, 2.171 ngày, một mã tổng hợp duy nhất `VN30F1M`, không có mã hợp đồng gốc hoặc cờ rollover; vì vậy không thể xác minh tuyệt đối quy tắc nối chỉ từ file này.
- Không có timestamp trùng và không phát hiện vi phạm quan hệ OHLC cơ bản.
- Gap tuyệt đối sau ngày thứ Năm thứ ba có median khoảng 0,636%, so với khoảng 0,214% ở các ngày khác; đây là bằng chứng cần tách/audit rollover, chưa phải xác nhận đầy đủ cơ chế nối của nhà cung cấp.
- Trước 05/05/2025, mẫu phổ biến có 243 dòng/ngày và gồm các mốc 11:30, 14:30, 14:45. Từ 05/05/2025, mẫu ổn định có 241 dòng/ngày, không còn bar 11:30 và 14:30 nhưng vẫn có bar 14:45.
- Mốc thay đổi 05/05/2025 trùng ngày hệ thống KRX chính thức vận hành. V0.5 phải coi đây là thay đổi chế độ dữ liệu và kiểm tra ảnh hưởng tới cách tạo cửa sổ.
- Bar 14:45 có OHLC phẳng trong toàn bộ mẫu quan sát và volume trung vị rất lớn, phù hợp với một bản ghi giá khớp ATC riêng. Ý nghĩa start-time/end-time chính xác của bar phút vẫn cần tài liệu nhà cung cấp hoặc tick data để xác nhận tuyệt đối.
