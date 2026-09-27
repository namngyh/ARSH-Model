# Ngữ cảnh bàn giao ARSH v0.6 — 26/09/2026

**AI hoặc người vận hành máy kia phải đọc file này trước khi chạy.** Nhiệm vụ hiện tại là thực hiện thí nghiệm K và độ ổn định, không phải thay model cuối ngày đang hoạt động. Thư mục trong ZIP là một dự án thử nghiệm độc lập.

## 1. Đã đạt được gì?

### Kết quả nghiên cứu đã có từ máy kia

Repo: `https://github.com/namngyh/ARSH-Model.git`, nhánh `v0.6`, commit tải về `e8c924bd53c1411c18aa05560ed754a34b0c55bc`.

Model gốc: `ARSH-v0.6-k7-h1-daily-cut20260801`, Student-t HMM shared df, K=7, horizon 1 phút, policy theo ngày. Model đã khóa lúc 26/09/2026 02:02:44, SHA-256:

`9071bb38709abb3ba91012b1d4a7e18d3285e0e1de7bf2c37e8a186091cf46e1`.

Model gốc dùng dữ liệu cuối 01/08/2023–31/07/2026, 177.695 quan sát. Huấn luyện gốc không phải một seed: sàng lọc 3 họ ×3 seed=9 fit; fit cuối họ chọn với 5 seed=5 fit. Artifact giữ lại là seed 145. Cả 5 seed cuối chỉ đạt hội tụ thực dụng, chưa đạt hội tụ chặt; screening 9 fit chưa hội tụ.

Kiểm tra đã xem: 03/08–25/09/2026, 37 phiên, 8.796 quan sát. Log density trung bình v0.6 là 6,148601; artifact v0.5 cùng quan sát là 6,139291; Student-t độc lập là 6,116191. Gain so với v0.5 khoảng +0,009310 và so với Student-t +0,032410 mỗi quan sát. Model đạt hai ngưỡng gốc: gain>0 và occupancy mềm nhỏ nhất ≥1%.

Kết quả tích cực nhưng cần xem độ ổn định: khoảng 58,90% chuỗi nhãn chỉ dài một quan sát; S6 có occupancy mềm gộp 1,1974%, tháng 8 khoảng 1,62% nhưng tháng 9 đến 25/09 chỉ khoảng 0,70%. S6 chỉ là nhãn argmax ở 31 quan sát thuộc 7 phiên. Ít xuất hiện chưa đủ kết luận state sai: thị trường có thể ít biến động phù hợp với nó.

### Việc thực tế đã chạy trên máy chuẩn bị ZIP

- Tải toàn bộ thư mục v0.6: 557 file khớp Git blob hash.
- Đọc và phân tích báo cáo, kế hoạch, diagnostics, mã nguồn và các CSV từng quan sát.
- Tính lại filter từ lợi suất đã lưu; posterior khớp CSV với sai khác tối đa khoảng `9,21e-13`, các số tổng hợp khớp. Chưa tái tạo từ nến nguồn hoặc DB.
- Viết bộ thử nghiệm độc lập cho K=5,6,7,8, nhiều seed, hội tụ chặt hơn, checkpoint trong seed, audit ghép state và quy tắc xác nhận nhãn.
- Kiểm tra hành vi trên CPU bằng dữ liệu giả lập: nhân quả, reset chuỗi, hai điều kiện hội tụ, checkpoint/resume, số likelihood khớp tham số hiện tại, ghép ID bị hoán vị, chọn K và bảo vệ model đã khóa. Xem `outputs/verification/tests.txt` và `LOCAL_VERIFICATION.json` để biết trạng thái kiểm tra cuối của đúng bộ mã trong ZIP.
- Chạy thử quy tắc nhãn trên **model K=7 gốc và dữ liệu tháng 8–9 đã xem**, không fit lại model.

| Quy tắc | Lần đổi bộ nhớ nhãn | Có nhãn đủ điều kiện hiển thị | Chưa xác nhận |
|---|---:|---:|---:|
| Raw argmax | 3.844 | 100% — không dùng bộ lọc | Không áp dụng ngưỡng |
| Chờ 2 quan sát, p≥0,60, hơn nhãn giữ ≥0,10 | 276 | 33,99% | 66,01% |
| Chờ 3 quan sát, p≥0,70, hơn nhãn giữ ≥0,15 | 44 | 20,62% | 79,38% |

Số lần đổi nhãn giảm mạnh có đánh đổi về độ phủ và độ trễ. Không được gọi đây là cải thiện độ chính xác hay bằng chứng regime bền hơn. Model vẫn xuất đủ 7 posterior và điểm dự báo nguyên gốc. Khi nhãn cũ yếu, đầu ra hiện là `UNCERTAIN`, không hiển thị nhãn cũ như một kết luận được xác nhận.

Thời gian chờ sau quan sát đầu tiên đủ điều kiện: quy tắc 2 là 1 quan sát; quy tắc 3 là 2 quan sát. Trên replay gốc, thời gian đồng hồ của quy tắc 3 có thể đến 93 phút khi chuỗi đi qua nghỉ trưa; trung bình khoảng 4,07 phút. Đây không phải độ trễ so với trạng thái thị trường thật vì chưa có nhãn thật để đối chiếu. Chi tiết: `outputs/reference_label_audit/`.

## 2. Còn thiếu gì?

**Chưa có kết quả huấn luyện đầy đủ mới cho K=5,6,8 hoặc K=7 đối chứng cùng ngân sách.** Các kiểm tra fit ở máy chuẩn bị dùng dữ liệu giả lập nhỏ và ngưỡng nới lỏng để kiểm tra chương trình; không phải kết quả nghiên cứu.

Máy chuẩn bị chỉ có Intel HD Graphics 620, không có CUDA/PyTorch CUDA; không có biến kết nối `PG_DSN`. CSV địa phương kết thúc 17/07/2026, thiếu phần đến 31/07. Vì vậy không bắt đầu 80 fit thật tại đây và không rút ngắn cửa sổ để tạo kết quả không tương đương.

Trên máy kia còn phải:

1. Xác nhận Python/CUDA và các thư viện; chạy kiểm tra tương đương CPU/CUDA của fitter mới. **Nhánh CUDA chưa được thực thi trên máy chuẩn bị ZIP.** Mã sẽ chặn full run nếu gate không đạt.
2. Cung cấp dữ liệu huấn luyện hoàn tất đúng cửa sổ, từ DB hoặc CSV huấn luyện gốc. SHA-256 CSV gốc:
   `d411a82cb27d453e0a5e0571589f0d0c7a8e1b6229678f14dc0bbc5fb0dd4374`.
3. Chạy selection/final với cấu hình đã bàn giao; báo rõ fit nào chưa hội tụ hoặc không đạt occupancy.
4. So độ ổn định giữa seed bằng ghép state, không đồng nhất các ID chỉ vì cùng số thứ tự.
5. Đánh giá chất lượng nhãn xác nhận trên validation: đổi nhãn, độ phủ, nhãn giữ yếu, thời gian chờ; không chỉ tối thiểu hóa số lần chuyển nhãn.
6. Sau khóa model, chờ và đánh giá trên dữ liệu tương lai. Chưa thể có kết quả này lúc tạo ZIP.

Chưa có đầy đủ bằng chứng về state qua nhiều fold, mã hợp đồng gốc/rollover hoặc độ trễ và bản sửa dữ liệu trong vận hành trực tiếp. Những phần này không tự được giải quyết bằng thêm seed.

## 3. Người dùng muốn làm gì ở bước này?

Yêu cầu được chốt trong cuộc trao đổi:

> Thử K=5,6,8; K=7 đã có sẵn; chạy nhiều seed hơn; huấn luyện ứng viên đến mức hội tụ rõ hơn; giảm việc đổi nhãn khi model chưa chắc chắn. Chuẩn bị ZIP để chạy trên máy kia, kèm ngữ cảnh đã đạt/còn thiếu/muốn làm/mục tiêu hoàn thành.

Phạm vi thực hiện:

- Giữ nguyên model K=7 gốc làm mốc. Thêm K=7 refit với cùng ngân sách như K khác để tránh so K mới với K=7 ít seed/chưa hội tụ.
- Chỉ thay K, ngân sách seed và tiêu chí hội tụ trong thí nghiệm này. Cố định họ `student_t_shared`, lợi suất raw không chồng lấn 1 phút, policy theo ngày, quy tắc gap và không điều chỉnh intraday/outlier.
- Mỗi K có **10 seed selection (62–71)** và **10 seed final (142–151)**: tổng **80 fit**. Tối đa 1.500 cập nhật EM/fit; checkpoint mỗi 25 cập nhật.
- Hội tụ: ít nhất 50 cập nhật; 5 cải thiện liên tiếp đều nhỏ, đồng thời delta LL ≤0,01 và delta/quan sát ≤1e-7; cho phép sai số giảm LL tối đa 1e-6. Không dùng fallback lấy seed chưa hội tụ để công bố ứng viên.
- Selection train 01/02/2023–31/01/2026, validation 01/02–31/07/2026. Seed trong mỗi K chọn theo likelihood train trong nhóm hội tụ, không theo seed có validation tốt nhất.
- K đủ điều kiện cần mọi state có occupancy mềm validation ≥1%. Trong các K đạt điều kiện, dùng điểm validation; biên hòa 0,002 và ưu tiên K nhỏ hơn. Không cam kết rằng K sẽ phải giảm hoặc tăng.
- Final train 01/08/2023–31/07/2026. Xuất artifact riêng theo K, baseline Student-t và manifest khóa.
- So raw và hai quy tắc xác nhận nhân quả. Chúng là các cấu hình thử nghiệm, **chưa được chọn để triển khai**. Không sửa posterior để giả tạo confidence cao.
- Replay 03/08–25/09 là **mô tả trên dữ liệu đã xem**, không dùng chọn K, chọn seed hoặc chọn ngưỡng mới.
- Chưa triển khai Sticky HMM, HSMM, thêm feature hoặc thay model sản xuất; những việc đó nằm ngoài yêu cầu hiện tại.

## 4. Mục tiêu xong khi làm xong là gì?

### Hoàn thành thí nghiệm trên dữ liệu lịch sử

Thí nghiệm hoàn thành khi có đủ báo cáo dưới đây, hoặc ghi rõ lý do không thể hoàn tất một phần; không đồng nghĩa bắt buộc tìm được model tốt hơn:

1. Có dữ liệu cố định, hash nguồn/cấu hình/mã/backend và kết quả gate CPU/CUDA. Dữ liệu train/validation đúng mốc, không dùng tháng 8 trở đi để fit/chọn.
2. Tất cả **80 fit được thực hiện đến điều kiện dừng**, có số vòng, likelihood, delta, thời gian và lý do dừng; số seed hội tụ của mỗi K được ghi rõ. Fit không hội tụ được báo như vậy.
3. Có bảng K=5,6,7,8 trên cùng validation, nêu seed được chọn, log density/gain, occupancy nhỏ nhất và đủ/không đủ điều kiện. Nếu không K nào đạt, kết luận **chưa có ứng viên đủ điều kiện**, không tự hạ tiêu chí.
4. Có audit ghép state giữa seed của cùng K: chi phí ghép tham số, sai khác posterior, tỷ lệ nhãn khác sau ghép, state hiếm. Báo mức bằng chứng, không gọi tự động ghép ID là chứng minh ý nghĩa ổn định.
5. Có bảng raw/confirm_2/confirm_3 với số lần đổi nhãn, tỷ lệ có nhãn xác nhận, mức bất định, nhãn giữ có xác suất thấp và thời gian chờ. Không chấp nhận giảm đổi nhãn là mục tiêu duy nhất; nếu bộ lọc bỏ quá nhiều nhãn hoặc giữ nhãn yếu, kết luận chưa phù hợp.
6. Có artifact riêng theo K hội tụ, Student-t baseline, hash và ngày khóa; giữ nguyên K=7 gốc. Nếu K nào không có seed cuối hội tụ, ghi thiếu artifact với nguyên nhân.
7. Viết báo cáo bàn giao kết quả thật: đã chạy gì, mất bao lâu, K nào đủ điều kiện, những đánh đổi và phần còn thiếu. Không ghi "đã cải thiện ổn định" chỉ từ dữ liệu giả lập hoặc làm chậm nhãn.

### Hoàn thành kiểm chứng model mới

Sau giai đoạn trên, model vẫn là ứng viên nghiên cứu. Muốn xác nhận chất lượng mới cần đánh giá trên dữ liệu sau ngày model khóa, không trước **28/09/2026**, theo tiêu chí đã chốt trước khi xem giai đoạn đó. Nếu fit kết thúc muộn hơn, ngày bắt đầu holdout phải sau ngày khóa thực tế.

Theo dõi log density so với Student-t/K=7 gốc, occupancy theo cửa sổ thời gian, độ bền và độ phủ nhãn. Báo khoảng bất định và số phiên. Không dùng lại tháng 8–9 đã xem như một phép kiểm tra độc lập. **Không cần chờ dữ liệu tương lai để hoàn thành phần thử nghiệm lịch sử; phải tách trạng thái hoàn thành hai phần này.**

## Hướng dẫn và điểm đọc đầu tiên trên máy kia

- Đọc `README.md`, kiểm tra `PACKAGE_MANIFEST_SHA256.json` bằng `verify_package.py`.
- Không sửa model, kế hoạch, Task Scheduler hoặc báo cáo cuối ngày hiện tại.
- Dùng `RUN_K_STABILITY.bat`, có thể tách `-Phase selection` và `-Phase final`.
- Khi dừng, chạy lại với cùng dữ liệu/config/backend/output. Thay bất kỳ phần nào trong identity cần thư mục output mới, không dùng checkpoint cũ cho cấu hình mới.
- DB dùng `PG_DSN` đã có, không in hoặc đưa thông tin đăng nhập vào ZIP, Git hoặc báo cáo.
- Mỗi output chỉ do một worker ghi. Không tự gửi email/tin nhắn, push Git hoặc phát hành model.
- Nếu cần thay phạm vi/tiêu chí sau khi xem kết quả, tạo kế hoạch phiên bản mới và ghi lý do. Không sửa các kế hoạch đã khóa của model gốc.
