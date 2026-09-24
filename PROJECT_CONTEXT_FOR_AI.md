# ARSH — ngữ cảnh bàn giao cho AI/máy mới

## 1. Người dùng và mục tiêu

ARSH là **Adaptive Regime-Switching — Hieu**, dự án nghiên cứu nhận diện trạng thái thị trường cho hợp đồng tương lai gần nhất VN30F1M. Chữ H là tên Hieu, không phải tên một thuật toán. Mục tiêu hiện tại là làm tốt phần nhận diện regime; không ép mô hình theo long/short. KE chỉ có thể là một mô hình downstream nhỏ cho một hoặc vài regime về sau.

Lộ trình đã chốt:

- v0.5 khóa dữ liệu, cách tạo return, horizon, phân phối và phạm vi K.
- v0.6 tập trung sâu vào số regime, ý nghĩa state, độ dai dẳng và ổn định.
- v1.0 kết nối dữ liệu cập nhật trực tiếp và chỉ nhận diện regime.
- v1.1 mới paper trading.
- Lợi suất ngày chỉ nghiên cứu sau v1.1.

Không tự ý thêm tín hiệu giao dịch, tối ưu lợi nhuận, HireVAE hoặc online parameter learning vào v0.5.

## 2. Dữ liệu

File nghiên cứu: `ohlc_export.csv`, 525.635 dòng phút của một mã tổng hợp `VN30F1M`, từ 2017-11-06 08:59 đến 2026-07-17 14:45, 2.171 ngày, không có ô thiếu. SHA-256 của bản đã dùng:

`bf84b23d6fa48b9fd90c477ef6788cc0aca19e86a0480d91600fa77363e71096`

Cột: `SYMBOL, TRADING_DATE, OPEN_PX, HIGH_PX, LOW_PX, CLOSE_PX, VOL, TRADING_TIME, BUY_VOL, BUY_VAL, SELL_VOL, SELL_VAL`.

Audit quan trọng:

- File chỉ có mã tổng hợp, không có mã hợp đồng gốc/cờ rollover. Không thể chứng minh tuyệt đối quy tắc nối hợp đồng chỉ từ file.
- Gap tuyệt đối sau ngày thứ Năm thứ ba có median khoảng 0,636%; ngày khác khoảng 0,214%. Phải đánh dấu/audit rollover và không trộn gap vào return intraday.
- Trước 2025-05-05, mẫu phổ biến có 243 dòng/ngày và có 11:30, 14:30, 14:45.
- Từ 2025-05-05, mẫu ổn định có 241 dòng/ngày, không còn 11:30 và 14:30 nhưng vẫn có 14:45. Mốc này trùng ngày hệ thống KRX vận hành; đó là bằng chứng về thay đổi chế độ dữ liệu, chưa chứng minh nguyên nhân từ file.
- Bar 14:45 có OHLC phẳng và volume lớn, phù hợp bản ghi ATC riêng. Không dùng nó như bar liên tục.
- Timestamp start/end convention chưa thể xác nhận tuyệt đối nếu thiếu tài liệu nhà cung cấp hoặc tick data.

## 3. Các phiên bản trước

### v0.1

- Prototype Gaussian HMM với K=4.
- Ba return log 60 phút mỗi ngày: 09–10, 10–11, 13–14.
- Chia 60/20/20; dùng nhiều seed và EM.
- Giá trị chính: chứng minh pipeline từ dữ liệu thật tới state/report chạy được.
- Hạn chế: một split, Gaussian đơn giản, K cố định, chưa có benchmark thời gian và causality đầy đủ.

### v0.2

- Walk-forward 3 năm train, 6 tháng validation, 6 tháng test, bước 6 tháng.
- So sánh Gaussian, Student-t, GMM, Gaussian HMM carry và daily reset; K=2–5.
- HMM carry vượt baseline không ký ức; paired bootstrap test-day gain trung bình khoảng 0,05417, CI95% `[0,02032; 0,08123]`, xác suất dương 0,9985.
- Câu hỏi trung tâm được ủng hộ: phân phối return có phụ thuộc trạng thái ẩn dai dẳng theo thời gian; mixture không ký ức chưa giải thích đủ.

### v0.3

- So sánh Gaussian HMM, Student-t HMM shared df và state-specific df.
- So sánh continuous carry, daily sequence và session sequence.
- Thêm occupancy sensitivity, block bootstrap, causal runtime và state alignment.
- Student-t shared có mean log-density cao nhất nhưng lợi thế so Gaussian HMM chưa chắc: gain 0,00838, block-20 CI `[-0,00130; 0,02123]`.
- HMM champion vượt non-HMM rõ: gain 0,08033, CI `[0,04996; 0,11373]`.
- Quyết định: chưa thay Gaussian HMM bằng Student-t HMM; giữ Student-t làm challenger.

### v0.4

- Chuỗi non-overlapping 5, 10, 15, 30, 60 phút; không xuyên nghỉ trưa, ATC, qua đêm; không nội suy.
- Gaussian HMM gain so Student-t iid: 5m `0,104771` (10/10 fold dương), 10m `0,088182`, 15m `0,080230`, 30m `0,078218`, 60m `0,072162` (7/10).
- 5m trừ 10m: mean `0,016654`, moving-block bootstrap 20 ngày CI `[0,011684; 0,021830]`.
- 5m là horizon ưu tiên nghiên cứu, chưa khóa cho v1 vì K=5 chạm biên trên ở 8/10 fold.
- Gaussian HMM 5m: khoảng 22 lần đổi state/100 bar; mean run 4,72 bar ≈23,6 phút; one-bar run rate 36,5%; proxy confirmation delay khoảng 3 phút.
- Practical convergence Gaussian HMM 5m: 8/10 refit.

## 4. Câu hỏi của v0.5

1. Horizon intraday nào biểu hiện memory/regime ngoài mẫu tốt và ổn định nhất?
2. K nhỏ 2–4 có gần tương đương và ổn định hơn K lớn không?
3. Khi mở K=6,7, gain có tiếp tục tăng hay chỉ tạo state nhỏ/phân mảnh?
4. Gaussian, Student-t, skew-t và GMM không ký ức giải thích được bao nhiêu trước khi cần HMM?
5. Kết quả có còn sau khi loại mùa vụ volatility theo giờ trong ngày?
6. Kết quả nhạy thế nào với overlapping return, độ dài train, zero, outlier, missing window và rollover?

## 5. Cấu hình chính v0.5

- Horizon: 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90 phút.
- Main: log close-to-close return, non-overlapping, raw.
- Phiên liên tục: 09:00–11:30 và 13:00–14:30 theo grid; cửa sổ chỉ nhận khi đủ `h+1` mức giá thực tế.
- Không nội suy; không return xuyên trưa/ATC/overnight/rollover.
- Baseline: Gaussian iid, Student-t iid, GMM.
- State models: Gaussian HMM, Student-t HMM shared df, Student-t HMM state-specific df.
- K=2–7.
- Soft occupancy chính 1%; sensitivity 0,5%, 2%, 5%.
- Walk-forward: rolling train 3 năm, validation 6 tháng, test 6 tháng, bước 6 tháng.
- Screening: 3 seed, tối đa 100 vòng. Refit: 5 seed, tối đa 300 vòng. Trường hợp khó có thể audit 500 vòng sau, không tự tăng mọi fit.
- Top 3 horizon theo Gaussian-HMM validation được chạy hai Student-t HMM sâu hơn.
- Continuous carry posterior qua khoảng nghỉ; tham số không online-update.
- Predictive log-density là metric chính, luôn báo gain so Student-t iid cùng horizon.

## 6. Quy tắc lựa chọn

Không chọn K/horizon chỉ bằng likelihood cao nhất. Phải xét đồng thời:

- test predictive log-density và paired moving-block bootstrap;
- occupancy và state rỗng;
- convergence;
- ổn định qua seed và fold;
- state alignment;
- duration, one-bar runs và flicker;
- confidence và confirmation delay;
- dữ liệu bị loại/zero/outlier;
- tính đơn giản và khả năng diễn giải.

Nếu K nhỏ nằm trong vùng thống kê gần tương đương K lớn, ưu tiên K nhỏ hơn khi nó ổn định và dễ diễn giải hơn. K=2 vẫn là baseline; K=3 ứng viên gọn; K=4 ứng viên cân bằng; K=5 champion validation v0.4; K=6,7 dùng để kiểm tra biên.

Không diễn giải log-density gain thành lợi nhuận.

## 7. Sensitivity đã mã hóa

`arsh_v05.py` hỗ trợ:

- `--return-variant raw|intraday_adjusted`; intraday factor chỉ fit từ train và có shrinkage.
- `--overlapping`; chỉ chạy sensitivity, không thay main.
- `--train-years 1|2|3`.
- `--expanding-window`.
- `--horizons`, `--k-values`, `--max-folds`.
- checkpoint/resume đến từng seed HMM; ghi nguyên tử.
- SHA-256 dữ liệu và code trong run identity; không cho trộn checkpoint khác cấu hình/code.

`distribution_diagnostics.py` so sánh Gaussian, Student-t và Jones-Faddy skew-t ngoài mẫu. Chỉ giữ skew-t làm challenger nếu fit ổn định và gain lặp lại qua fold.

`run_sensitivities.ps1` chỉ chạy sau khi main hoàn thành; nó lấy winner/runner từ main rồi chạy adjusted, overlapping, train 1y/2y và expanding trong output riêng.

## 8. Trạng thái code khi bàn giao

- Main runner, checkpoint/resume, live progress, data audit, model comparison, bootstrap, state profiles/alignment/dynamics, HTML report và runtime research candidate đã có.
- Portable setup/run/status scripts đã có.
- Unit tests không dùng full experiment.
- Không mang checkpoint thử nghiệm cũ sang máy mới; chạy sạch để code hash/config thống nhất.
- Cần chạy `setup.ps1`, unit tests và smoke test trên máy mới trước full run.
- PDF dùng Edge/Chrome headless qua `export_report_pdf.py`.
- `--jobs` hiện giữ ở 1 để tránh trộn checkpoint và bùng RAM. AI trên máy mới chỉ nên parallel hóa sau khi đo RAM; cách an toàn là song song các fold/output độc lập, mỗi worker khóa BLAS một thread.

### Kiến trúc revised CPU/CUDA

- `outputs_main_legacy` là main run cũ, chỉ dùng làm đối chứng; không resume bằng code revised.
- Sequence policy hiện được áp dụng nhất quán cho EM, validation filtering, test filtering và runtime. Cửa sổ bị loại luôn tạo hard boundary.
- `--folds` tạo shard độc lập; `merge_shards.py` chỉ ghép khi hash code/data/config khớp và không trùng fold.
- `--backend cuda` chỉ chuyển EM/forward-backward/M-step HMM sang PyTorch CUDA; dữ liệu, GMM, bootstrap, alignment và báo cáo vẫn ở CPU.
- `--finalize-only` cho phép Colab CPU tổng hợp checkpoint CUDA và cấm âm thầm refit khi thiếu fold.
- Gói CUDA mang theo đúng `data/ohlc_export.csv` cùng manifest; `verify_cuda.py` phải đạt parity CPU/CUDA trước full run.
- Protocol đầy đủ chạy policy trước, rồi mới tạo post-policy plan cho main revised và mọi sensitivity bắt buộc, gồm cả cửa sổ có đúng một phút nội bộ bị thiếu nhưng vẫn đủ hai endpoint (không nội suy).
- Cổng hoàn tất còn yêu cầu distribution diagnostics, timestamp-regime audit và targeted 500-iteration convergence audit; thiếu một mục thì không được chuyển sang v0.6.

## 9. Chạy trên máy mới

1. Giải nén thư mục.
2. Chép `ohlc_export.csv` vào `data/` và xác nhận SHA-256 đúng nếu dùng cùng dữ liệu.
3. Chạy `setup.ps1`.
4. Chạy smoke test trong README.
5. Xóa `outputs_smoke` sau khi kiểm tra.
6. Chạy `run_v05.ps1` cho main.
7. Sau khi main hoàn tất, chạy `run_sensitivities.ps1`.
8. Không dùng smoke output để kết luận.

## 10. Những điều AI bên máy mới không được tự suy diễn

- State không phải lệnh giao dịch và số bar không phải số giao dịch khớp lệnh.
- `student_t_state` là Student-t HMM có df riêng theo state, không phải một state tên Student.
- `test_share` là tỷ lệ hard assignment; validation occupancy dùng mean posterior mềm.
- State ID nội bộ không ổn định giữa fit; phải alignment trước khi so xuyên fold.
- Test đã được xem qua các phiên bản là research benchmark. Dữ liệu mới sau khi v1 vận hành mới là holdout thực sự mới.
- Không gọi cấu hình production/champion cuối nếu K còn chạm biên, convergence yếu hoặc state không ổn định.

## 11. Tài liệu đi kèm

- `ARSH_PROJECT_REQUIREMENTS.md`: quyết định toàn dự án.
- `history/`: source, README và báo cáo chính v0.1–v0.4; không chứa prediction lớn.
- Các PDF gốc người dùng đã cung cấp có thể được sao chép riêng nếu cần đối chiếu lý thuyết: `Advanced_ARS_Framework-v2.pdf` và `2306.02848v1.pdf`.
