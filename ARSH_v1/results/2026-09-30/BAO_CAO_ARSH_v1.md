# Báo cáo ARSH v1 — kiểm kê dữ liệu và 5 kiểm định lịch sử

Ngày chạy: 30/09/2026. Máy chạy: Windows, Python 3.12.10, RTX 4060 (không dùng GPU cho v1).
Gói: nhánh `v1` của `namngyh/ARSH-Model`, thư mục `ARSH_v1`, 21/21 file đúng SHA-256, 6/6 kiểm thử đạt.

**Kết luận ngắn**

- Đã chạy đủ kiểm kê, 5 kiểm định cho cả `strict_k7` và `original_k7`, và online shadow (replay).
- **Lần chạy đầu (mục 1–7) dùng dữ liệu đến 04/09/2026.** Lần chạy thứ hai (mục 8, lúc 20:21–20:23) thêm 13 ngày từ 07/09 đến 30/09, nên có **1 ngày sau khóa (30/09, 238 quan sát)**. Một ngày chưa đủ để kết luận gì về tương lai. Mọi con số là mô tả, không phải kiểm định tương lai.
- **Không tái tạo được đúng snapshot huấn luyện** (210.579 nến, 870 ngày, SHA `d411a82c…`). Nguồn dùng khác snapshot (xem mục 2).
- Hai model cho kết quả rất gần nhau (nhãn argmax trùng 96,8%). Bảy state thực chất gồm 3 state bền (S0, S5, S6) và 4 state ngắn (S1–S4).
- Online shadow cải thiện log density rất nhỏ nhưng đều đặn. Không dùng để thay model đang chạy.

---

## 1. Việc đã làm và không làm

| Việc | Trạng thái |
|---|---|
| Kiểm tra hash gói, cài `.venv`, chạy 6 test | Đã làm (thư mục chạy `C:\arsh_v1_run`, do đường dẫn phiên quá dài cho pip) |
| Kiểm kê chỉ đọc các nguồn dữ liệu VN30F1M trên máy | Đã làm |
| Chạy 5 kiểm định cho `strict` và `original` | Đã làm |
| Online shadow từ 01/08/2026 (replay) | Đã làm, chỉ cho `strict` |
| Bật `qp-timescaledb` ở chế độ chỉ đọc để tìm dữ liệu mới | Đã làm: bật 20:17:16, tắt lại 20:21:02 (xem mục 8) |
| Sửa artifact, đẩy Git, thay model cuối ngày, ghép Modus | Không làm |

## 2. Kiểm kê nguồn dữ liệu

### 2.1 Các nguồn tìm thấy

| Nguồn | Vị trí | Định dạng | Khoảng ngày | Ghi chú |
|---|---|---|---|---|
| **CSV dùng để chạy** | `Desktop\laplace\ohlc_export.csv` (lọc từ 01/02/2023) | CSV phút, 12 cột | 01/02/2023 – 04/09/2026 | 533.344 dòng toàn file (từ 06/11/2017); bản lọc 216.845 dòng, 896 ngày |
| CSV Modus 1.3 | `D:\systemModus-ver_1.3\data\raw\ohlc_export.csv` | như trên | đến 03/09/2026 | Phần 01/02/2023–31/07/2026 **giống hệt** CSV laplace (211.546 dòng, cùng hash nội dung) |
| CSV cũ hơn | `QP PLATFORM`, `Downloads`, `systemVBhedge…` | như trên | đến 17/07, 15/06, 03/08/2026 | Không dùng (ngắn hơn) |
| Parquet của ingestion | `D:\Database - QuantPercent\data\raw` | Parquet | chỉ 23/07/2026 | Quá ít |
| **TimescaleDB** | Docker `qp-timescaledb` (bảng `bars_1m`, `ticks`, `predictions`) | DB | chưa truy vấn | Container **đã tắt từ khoảng 31 giờ trước** |
| Dump DB | `D:\Database - QuantPercent\backup` | `.dump` | 23/07 và 04/08/2026 | Cũ hơn CSV |

Không thấy trên máy: thư mục dự án ARSH v0.6, `v06_fetch_db.py`, `training_complete_days.csv`. File ghi chú cũ cho thấy máy này đã chạy các shard CUDA của v0.5.

### 2.2 Vì sao dữ liệu dừng ở 04/09/2026

Đây là tình trạng ở lần chạy đầu. Sau đó tôi đã bật DB và kiểm tra (mục 8): DB cũng chỉ có dữ liệu đến 12/08/2026, còn dữ liệu mới nằm trong file Parquet của ingestion.

### 2.3 Khác biệt với snapshot huấn luyện

| | Snapshot huấn luyện (theo tài liệu) | CSV dùng ở đây (01/02/2023–31/07/2026) |
|---|---|---|
| Số nến | 210.579 | 211.546 (hơn 967) |
| Số ngày | 870 | 874 (hơn 4) |
| SHA-256 | `d411a82c…dd4374` | không so được (khác định dạng, khác phạm vi) |

Có 6 ngày bất thường về số nến: 28/11/2023 (239), 01/12/2023 (244), 02/01/2025 (224), 25/03/2025 (233), 10/04/2025 (115), 10/07/2025 (239). Tổng 4 ngày bị loại khỏi snapshot không khớp với bất kỳ tổ hợp nào trong 6 ngày này, nên tôi **không đoán** được quy tắc "ngày đầy đủ". Vì không có `v06_fetch_db.py`, tôi không tái tạo được snapshot. Bộ lọc cửa sổ của v0.5 đã loại 939 cửa sổ thiếu phút khi dựng lợi suất. Mức ảnh hưởng của 4 ngày chênh lên kết quả chưa được định lượng.

### 2.4 Cột và bằng chứng

| Nhóm | Có trong CSV? |
|---|---|
| OHLC, VOL, BUY/SELL VOL/VAL | Có (nguồn gốc) |
| Mã giao dịch | Chỉ có `VN30F1M`; **không có mã hợp đồng**, không xác minh được rollover |
| Thời gian | `TRADING_DATE` + `TRADING_TIME` (nhà cung cấp ghi đầu nến); không có `bar_end`, `available_at` |
| Log-return, posterior, nhãn state, model/data version | Không có sẵn; tính lại được (đã tính trong `states_*.csv`) |
| `available_at` | **Không có bằng chứng** nên để trống |

### 2.5 Ba phạm vi dữ liệu (theo tài liệu v1)

| Phạm vi | Số quan sát (lợi suất 1 phút) | Số ngày |
|---|---:|---:|
| Đến 31/07/2026 (phát triển model) | 208.871 | 874 |
| 03/08 – 04/09/2026 (đã xem trước khi khóa) | 5.230 | 22 |
| Từ 30/09/2026 | 0 (lần chạy đầu); 238 ở lần chạy thứ hai, xem mục 8 | 0 → 1 |

Trong giai đoạn thứ hai thiếu 3 ngày làm việc: 31/08, 01/09, 02/09. Có thể do nghỉ lễ Quốc khánh nhưng tôi chưa xác nhận.

## 3. Kết quả 5 kiểm định

Toàn bộ dựa trên 214.101 lợi suất 1 phút hợp lệ (939 cửa sổ bị loại), từ 01/02/2023 09:01 đến 04/09/2026 14:29. Lợi suất có độ nhọn dư 22,9 và 766 điểm vượt 5 độ lệch chuẩn. Đó là lý do dùng phân phối Student-t.

### 3.1 Entropy posterior

| | strict | original |
|---|---:|---:|
| Entropy chuẩn hóa trung bình (0 = chắc chắn, 1 = đều) | 0,448 | 0,448 |
| Phân vị 90 | 0,679 | 0,673 |
| Tỷ lệ > 0,8 | 0,30% | 0,25% |
| Entropy TB đến 31/07 | 0,446 | 0,446 |
| Entropy TB 08–09/2026 | 0,537 | 0,530 |

Posterior hiếm khi phân tán hẳn, nhưng cũng không sắc nét (trung vị khoảng 1,0 nat trên tối đa 1,95). Entropy cao hơn khoảng 0,09 ở giai đoạn 08–09/2026, tức model kém chắc chắn hơn trên dữ liệu mới. Entropy thấp không chứng minh state đúng.

### 3.2 Chồng lấn emission và JSD (5 cặp chồng lấn nhất)

| Cặp | strict overlap / JSD | original overlap / JSD |
|---|---:|---:|
| S4–S5 | 0,790 / 0,033 | 0,702 / 0,067 |
| S0–S1 | 0,710 / 0,059 | 0,689 / 0,067 |
| S2–S4 | 0,662 / 0,092 | 0,661 / 0,091 |
| S5–S6 | 0,646 / 0,100 | 0,648 / 0,099 |
| S0–S2 | 0,644 / 0,089 | 0,659 / 0,081 |

S4 và S5 là cặp khó tách nhất, và **strict tách kém hơn original** ở cặp này. Ít chồng lấn nhất là S0–S6 (khoảng 0,29). Đủ 21 cặp nằm trong `diagnostics_*.json`.

### 3.3 và 3.4 Chuyển trạng thái và thời lượng (strict)

| State | P(ở lại) | Thời lượng kỳ vọng HMM (phút) | Số chuỗi argmax | TB chuỗi argmax | Trung vị | Tỷ lệ chuỗi dài 1 phút |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0,984 | 62,4 | 4.844 | 14,2 | 2 | 37,6% |
| S1 | 0,407 | 1,7 | 25.359 | 1,66 | 1 | 59,1% |
| S2 | 0,367 | 1,6 | 26.176 | 1,56 | 1 | 62,9% |
| S3 | 0,321 | 1,5 | 4.766 | 1,31 | 1 | 74,8% |
| S4 | 0,556 | 2,3 | 2.803 | 1,57 | 1 | 62,8% |
| S5 | 0,987 | 75,0 | 3.484 | 13,0 | 4 | 23,2% |
| S6 | 0,979 | 46,6 | 818 | 8,4 | 3 | 28,2% |

- **S0, S5, S6 là state bền** về mặt tham số (kỳ vọng 47–75 phút). **S1–S4 là state ngắn** (kỳ vọng 1,5–2,3 phút).
- Nhãn argmax thực tế **ngắn hơn nhiều** so với kỳ vọng HMM, kể cả ở state bền: S0 trung bình 14 phút so với 62 phút, trung vị chỉ 2 phút. Nếu dùng nhãn argmax trực tiếp cho Modus, nhãn sẽ nhấp nháy. Điều này khớp với kết luận v0.6 rằng quy tắc xác nhận 2–3 quan sát chưa phù hợp, nên cần cách khác (ví dụ dùng xác suất thay vì nhãn cứng).
- Original có cùng hình dạng (kỳ vọng S4 là 1,8 thay vì 2,3). Ma trận chuyển đầy đủ nằm trong JSON.

### 3.5 Đặc điểm kinh tế (strict, posterior mềm)

| State | Tỷ trọng | Độ lệch chuẩn (%/phút) | TB cùng phút (%) | TB phút kế tiếp (%) | Tỷ trọng phiên sáng | Volume TB |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 29,5% | 0,024 | 0,0000 | −0,0000 | 75% | 493 |
| S1 | 18,0% | 0,029 | −0,0176 | −0,0005 | 66% | 726 |
| S2 | 18,1% | 0,030 | +0,0223 | +0,0004 | 65% | 763 |
| S3 | 5,4% | 0,047 | −0,0457 | −0,0005 | 62% | 1.169 |
| S4 | 5,5% | 0,064 | +0,0320 | +0,0008 | 61% | 1.277 |
| S5 | 19,9% | 0,078 | −0,0002 | +0,0001 | 47% | 1.344 |
| S6 | 3,8% | 0,183 | −0,0032 | −0,0017 | 29% | 2.291 |

- Thứ tự biến động (độ lệch chuẩn) và volume tăng đồng thời từ S0 đến S6. Đây là đặc điểm kinh tế rõ nhất: S0 yên tĩnh, tập trung buổi sáng. S6 biến động gấp khoảng 7,6 lần S0, volume gấp khoảng 4,6 lần, chủ yếu ở phiên chiều.
- S1 và S2 (cũng như S3 và S4) gần như đối xứng theo dấu của lợi suất cùng phút. **Đây là hệ quả cơ học**, vì state được suy ra từ chính lợi suất đó; không phải tín hiệu dự báo.
- Lợi suất phút kế tiếp đều rất nhỏ: lớn nhất khoảng 0,0017% (S6), tức khoảng 0,03 điểm chỉ số, nhỏ hơn bước giá 0,1 điểm. Không có bằng chứng về khả năng dự báo có thể giao dịch.
- Model original cho profile rất gần strict (ví dụ S6: độ lệch chuẩn 0,182%, volume 2.279), nên nhãn S0–S6 của hai model có vẻ cùng nghĩa kinh tế. Nhãn argmax trùng 96,8% trên toàn mẫu. **Chưa có phép kiểm định alignment chính thức**, nên vẫn phải gắn `model_version` như hợp đồng Modus yêu cầu.

## 4. Online learning shadow (replay mô tả)

Bắt đầu từ 01/08/2026 (quan sát đầu tiên 03/08 09:01), 5.230 lần cập nhật, cho `strict`. Đây là replay trên dữ liệu đã xem, không phải kiểm định tương lai. Bộ cập nhật là nguyên mẫu nghiên cứu, không phải online EM chính xác.

| Chỉ số | Giá trị |
|---|---:|
| Log density TB, cố định | 6,0803 |
| Log density TB, shadow | 6,0837 |
| Chênh lệch TB (shadow − cố định) | +0,0034 nat/phút |
| Sai số chuẩn theo phút / theo ngày | 0,00063 / 0,00066 |
| Số ngày shadow tốt hơn | 21 / 22 |
| Tỷ lệ quan sát shadow tốt hơn | 51,6% |
| Dịch trung bình tối đa | 0,036 độ lệch chuẩn chuẩn hóa |
| Tỷ lệ scale | 0,83 – 1,10 |
| Nhãn argmax trùng bản cố định | 95,1% |

Shadow tốt hơn rất nhỏ nhưng đều (21/22 ngày; thống kê theo ngày khoảng 5 sai số chuẩn). Độ lớn +0,0034 nat/phút nhỏ so với mức chênh giữa hai giai đoạn (6,31 so với 6,08). Điều này gợi ý dữ liệu 08–09/2026 hơi lệch so với model cố định, nhưng với 22 ngày và các tham số bị chặn trôi, **chưa đủ cơ sở để đổi cơ chế cập nhật**. Bộ shadow không ghi đè artifact nào.

## 5. Mỗi kiểm định làm được gì, cần gì thêm

| Kiểm định | Làm được ngay? | Cần thêm |
|---|---|---|
| 1. Entropy | Có (đã xong) | Dữ liệu từ 30/09 để gọi là kiểm tra tương lai |
| 2. Chồng lấn/JSD | Có, chỉ cần tham số model | — |
| 3. Chuyển trạng thái | Lý thuyết và quan sát đã xong | Dữ liệu tương lai |
| 4. Thời lượng | Đã xong | Cân nhắc nhãn làm mượt (ngoài phạm vi v1) |
| 5. Đặc điểm kinh tế | Đã xong, mô tả | Dữ liệu tương lai; kiểm định ý nghĩa thống kê của lợi suất phút kế tiếp |

## 6. Hạn chế và việc còn dang dở

1. **Dữ liệu 07–30/09/2026 còn thiếu 5 ngày** (15, 18, 22, 24, 25/09) vì file Parquet hỏng, và nguồn dữ liệu này chưa được đối chiếu với nguồn chính thức (xem mục 8).
2. **Snapshot huấn luyện chưa đối chiếu được.** Cần `v06_fetch_db.py` hoặc `training_complete_days.csv` (nếu còn) để xác minh hash và số 210.579 nến.
3. **Ngày khóa không thống nhất**: manifest ghi khóa 29/09 05:52 và holdout từ 28/09, code phân giai đoạn "sau khóa" từ 30/09. Dữ liệu 28–29/09 (nếu có) sẽ rơi vào nhóm "đã xem trước khi khóa".
4. `available_at` để trống: output **chưa dùng được** để ghép as-of với Modus (`asof_join_ready: False`).
5. Không có mã hợp đồng nên không kiểm tra được ảnh hưởng của rollover lên các state biến động cao.

## 7. File đầu ra

Trong `C:\arsh_v1_run\outputs` (bản sao các file nhỏ nằm cạnh báo cáo này):

- `DATA_INVENTORY.json`, `REPORT_strict.md`, `REPORT_original.md`, `diagnostics_strict.json`, `diagnostics_original.json`.
- Chuỗi từng phút (khoảng 98 MB mỗi file, chỉ có ở `C:\arsh_v1_run\outputs`): `states_strict.csv`, `states_original.csv`, `states_shadow_experimental.csv`.
- Dữ liệu đã lọc làm đầu vào: `C:\arsh_v1_run\data\vn30f1m_20230201_20260904.csv` (SHA-256 `847426b7…f7184ebe8`). Nó được tạo từ CSV laplace, chỉ lọc dòng, không sửa giá trị.

Không có chuỗi kết nối, mật khẩu hay token nào được đọc hoặc ghi vào báo cáo này. Tôi không đọc file `.env` của dự án database.

---

## 8. Cập nhật: bật DB chỉ đọc và chạy lại với dữ liệu đến 30/09/2026

### 8.1 Mốc thời gian (giờ Việt Nam, UTC+7, ngày 30/09/2026)

| Giờ | Việc |
|---|---|
| 29/09 13:24 | Container `qp-timescaledb` tắt lần trước (trước khi tôi làm gì) |
| 20:17:16 | `docker start qp-timescaledb`; báo healthy sau khoảng 15 giây |
| 20:17:39 – 20:18:21 | Các truy vấn SELECT (phiên `default_transaction_read_only=on`, xác nhận trong phiên). Không ghi gì vào DB |
| 20:20:56 | Dựng 13 file CSV ngày từ Parquet |
| 20:21:02 | `docker stop qp-timescaledb` (đưa về trạng thái ban đầu: đã tắt) |
| 20:21:12 – 20:23 | Chạy lại v1 cho cả hai model (2 phút 17 giây) |

`qp-ingestion` và `qp-redis` giữ nguyên, tôi không chạm vào chúng.

### 8.2 Nội dung DB

| Bảng | Kết quả |
|---|---|
| `bars_1m` (`VN30F1M`) | 529.960 dòng, từ 06/11/2017 đến **12/08/2026 14:17**, cập nhật lần cuối cùng lúc đó |
| `ticks` (`VN30F1M`) | 158.552 dòng, đến 12/08/2026 |
| `predictions` | 149 dòng (chưa xem nội dung) |
| `quant.*` | 8 bảng, đều 0 dòng |
| `gap_log` | dừng ở id 87, trong khi log của ingestion báo id 617–618 |

**DB này là bản cũ** và không chứa dữ liệu tháng 9. Ingestion không ghi vào nó sau khi bật (không có dòng mới trong khoảng 1 phút tôi quan sát). Tôi chưa xác định ingestion thực sự ghi dữ liệu phút vào đâu ngoài Parquet.

### 8.3 Dữ liệu ghép từ Parquet (`D:\Database - QuantPercent\data\raw\bars_1m`)

- Có 13 ngày đọc được: 07, 08, 09, 10, 11, 14, 16, 17, 21, 23, 28, 29, 30/09/2026. Mỗi ngày đúng 241 nến, từ 09:00 đến 14:45.
- **Thiếu 5 ngày làm việc:** 15, 18, 22, 24, 25/09, vì file Parquet của các ngày này hỏng (không có footer). Cộng với 3 ngày thiếu 31/08 – 02/09, chuỗi có nhiều khoảng trống. Bộ lọc của v1 reset chuỗi sau mỗi khoảng trống, nên không nối sai qua ngày.
- 10 file Parquet không đọc được; danh sách nằm trong `data\days_from_parquet_manifest.json`.
- Đối chiếu với CSV lịch sử trên 5.519 phút trùng nhau: giá và volume khớp 99,98% (1 nến lệch).
- Quy tắc cho thời điểm trùng: lấy bản có volume lớn nhất, nếu bằng thì bản ở file sau. Trong tháng 9 chỉ có 1 thời điểm mâu thuẫn (14/09 09:09, hai bản volume 471 và 570), chọn bản 570.
- File Parquet không có cột `is_final`, nên không xác minh được bản nào là bản đã chốt ngoài quy tắc trên. Các file CSV ngày chỉ giữ OHLC và volume (đủ cho v1); các cột mua/bán có trong Parquet nhưng tôi không dùng.
- **Nguồn này chưa đối chiếu với nguồn chính thức.** Dùng tạm cho replay mô tả.

### 8.4 Kết quả lần chạy thứ hai (`C:\arsh_v1_run\outputs_to_0930`)

217.195 lợi suất hợp lệ (965 cửa sổ bị loại), đến 30/09/2026 14:29.

| Giai đoạn | Quan sát | Ngày | Entropy chuẩn hóa (strict / original) | Log density TB (strict / original) |
|---|---:|---:|---:|---:|
| Đến 31/07/2026 | 208.871 | 874 | 0,446 / 0,446 | 6,305 / 6,305 |
| 03/08 – 29/09/2026 (đã xem trước khi khóa) | 8.086 | 34 | 0,541 / 0,536 | 6,149 / 6,148 |
| **30/09/2026 (sau khóa)** | **238** | **1** | **0,579 / 0,578** | **6,384 / 6,383** |

- Giai đoạn đã xem trước khi khóa vẫn có entropy cao hơn giai đoạn phát triển (khoảng 0,09 đến 0,10). Kết luận ở mục 3.1 không đổi.
- Ngày 30/09: entropy cao hơn, nhưng log density (6,384) lại ngang hoặc cao hơn mức phát triển (6,305). Một ngày chỉ có 238 quan sát nên **không đủ để kết luận gì**.
- Cơ cấu state ngày 30/09 (strict, posterior mềm): S1 28,6%; S2 24,6%; S0 21,0%; S5 11,0%; S3 8,6%; S4 6,0%; S6 0,1%. S6 gần như không xuất hiện, nên các thống kê theo state của ngày này (đặc biệt S6, S4) không có ý nghĩa.
- Chuỗi argmax ngày 30/09 vẫn ngắn: S1 trung bình 2,0, S2 1,3 quan sát. Khớp với mục 3.4.
- Các bảng ở mục 3 (chạy trên dữ liệu đến 04/09) gần như không đổi với dữ liệu mới: ví dụ chuỗi argmax trung bình S0 là 14,0 (trước: 14,2), S5 là 12,9 (trước: 13,0). Toàn bộ số liệu mới nằm trong `outputs_to_0930\diagnostics_*.json`.

### 8.5 Online shadow (replay) với dữ liệu mới

| Chỉ số | Đến 04/09 (lần 1) | Đến 30/09 (lần 2) |
|---|---:|---:|
| Số lần cập nhật (từ 03/08) | 5.230 | 8.324 |
| Chênh log density TB (shadow − cố định) | +0,0034 | +0,0027 |
| Số ngày shadow tốt hơn | 21/22 | 31/35 |
| Tỷ lệ quan sát shadow tốt hơn | 51,6% | 50,1% |
| Dịch trung bình tối đa | 0,036 | 0,031 |
| Tỷ lệ scale | 0,83 – 1,10 | 0,81 – 1,16 |

Phần 07/09 – 30/09 (3.094 quan sát): chênh +0,0015. Riêng ngày 30/09: +0,0002 (238 quan sát). Lợi thế của shadow **nhỏ dần** khi dữ liệu mới đến, nên vẫn chưa đủ cơ sở để đổi cơ chế cập nhật. Sai số chuẩn theo ngày khoảng 0,0006. Đây là replay mô tả, không phải kiểm định tương lai. Không có artifact nào bị ghi đè.

### 8.6 File đầu ra của lần chạy thứ hai

- `C:\arsh_v1_run\outputs_to_0930\`: `DATA_INVENTORY.json`, `REPORT_strict.md`, `REPORT_original.md`, `diagnostics_*.json`, `states_strict.csv`, `states_original.csv`, `states_shadow_experimental.csv` (khoảng 100 MB mỗi file states).
- `C:\arsh_v1_run\data\days_from_parquet\` (13 CSV ngày) và `data\days_from_parquet_manifest.json`.
- Bản sao các file nhỏ nằm cạnh báo cáo này, trong thư mục `outputs_to_0930`.
