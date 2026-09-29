# Kết quả thí nghiệm K Stability — ARSH v0.6

Chạy trên máy có GPU RTX 4060 và kết nối DB, từ 27/09/2026 16:22 đến 29/09/2026 05:52 (có hai lần tạm dừng theo yêu cầu người dùng). Gói: `ARSH_v0.6_K_STABILITY_HANDOFF_2026-09-26.zip`, SHA-256 `0a794799…6869a`, tải từ nhánh `v0.6` commit `522f0e5`. Cấu hình chạy nguyên bản `experiment_config.json`, không sửa file nào trong gói. Kết quả gốc: `outputs/full/`.

**Tóm tắt:** cả 80 fit hội tụ chặt. Trên validation 02–07/2026, K=7 được chọn (hòa với K=8 trong biên 0,002, ưu tiên K nhỏ hơn); K=5 và K=6 kém rõ hơn. Đây là **ứng viên nghiên cứu**; model báo cáo cuối ngày không bị thay. Quy tắc xác nhận nhãn giảm số lần đổi nhãn khoảng 12 lần (`confirm_2`) nhưng đổi lại hơn một nửa số quan sát thành `UNCERTAIN`; chưa đủ cơ sở chọn quy tắc nào để triển khai.

## 1. Dữ liệu, mã, backend, kiểm tra CPU/CUDA

- Dữ liệu từ DB (`--fetch-db`), 870 ngày hoàn tất 01/02/2023–31/07/2026, 210.579 nến. SHA-256 bản xuất `d411a82c…4374` **trùng từng byte** với CSV huấn luyện của model K=7 gốc (`matches_original_training_export_bytes: true`). Không có dữ liệu từ 01/08/2026 trong fit hay chọn K.
- Quan sát: train chọn 178.884, validation 29.036, train cuối 177.695; 880 cửa sổ bị loại theo quy tắc gap v0.5.
- Kiểm tra trước khi chạy: `verify_package.py` 38/38 file khớp; `unittest` 9/9 đạt; `plan` đạt; **gate CPU/CUDA đạt** trên RTX 4060, torch 2.14.0+cu130, CUDA 13.0 (K=5 và K=8: sai khác posterior tối đa 2,1e-14, sai khác likelihood ≤1,1e-13). Đây là lần đầu nhánh CUDA của fitter mới được chạy.
- Hash mã khớp `LOCAL_VERIFICATION.json`; `arsh_v05.py` = `b5e19665…` (không đổi).
- Sai khác môi trường: `threadpoolctl` 3.6.0 thay vì 3.7.0 (chỉ giới hạn luồng CPU; 9/9 test đạt với bản này; không nâng cấp để khỏi phải chạy lại kiểm tra).

## 2. 80 fit và điều kiện dừng

Mọi fit dừng vì **hội tụ chặt** (`strict_converged`), không fit nào chạm trần 1.500 cập nhật. Chi tiết từng seed: `selection_seed_metrics.csv`, `final_seed_metrics.csv`.

| K | Selection: hội tụ | Số cập nhật | Giờ GPU | Final: hội tụ | Số cập nhật | Giờ GPU |
|---|---|---|---:|---|---|---:|
| 5 | 10/10 | 353–399 | 2,0 | 10/10 | 348–356 | 1,4 |
| 6 | 10/10 | 408–1.038 | 3,2 | 10/10 | 292–1.058 | 3,0 |
| 7 | 10/10 | 663–1.457 | 6,7 | 10/10 | 719–1.216 | 4,4 |
| 8 | 10/10 | 717–752 | 3,6 | 10/10 | 430–655 | 1,8 |

Tổng thời gian tính: selection 15,5 giờ, final 10,7 giờ, **26,2 giờ**. K=7 selection có 7/10 seed cần 1.343–1.457 cập nhật, sát trần 1.500; nếu trần thấp hơn, K=7 có thể đã bị loại vì chưa hội tụ.

## 3. So sánh K trên validation 01/02–31/07/2026

| K | Seed chọn (LL train cao nhất trong nhóm hội tụ) | Log density | Gain so Student-t | Occupancy mềm nhỏ nhất | Kém tốt nhất | Kết luận |
|---|---:|---:|---:|---:|---:|---|
| 5 | 71 | 6,16625 | 0,0920 | 4,8% | 0,0033 | đủ điều kiện, ngoài biên hòa |
| 6 | 66 | 6,16665 | 0,0924 | 4,8% | 0,0029 | đủ điều kiện, ngoài biên hòa |
| **7** | 65 | 6,16909 | 0,0948 | 3,8% | 0,0004 | **hòa → chọn** |
| 8 | 67 | 6,16951 | 0,0953 | 3,9% | 0 | hòa, cao nhất |

K=8 cao nhất nhưng chỉ hơn K=7 0,0004, trong biên 0,002 → chọn K nhỏ hơn. Khác biệt K=7/K=8 quá nhỏ để coi K=8 tốt hơn; khác biệt K=7 so với K=5/K=6 (~0,003) vượt biên nhưng vẫn là một đoạn validation 6 tháng, không có khoảng tin cậy.

## 4. Ổn định giữa các seed (ghép state trong cùng K)

Trung vị / lớn nhất qua 9 cặp (seed được chọn so với từng seed khác). "Nhãn khác" là tỷ lệ quan sát có argmax khác nhau sau khi ghép ID.

| K | Selection: chi phí ghép | Selection: nhãn khác | Final: chi phí ghép | Final: nhãn khác |
|---|---|---|---|---|
| 5 | 0,0003 / 0,0010 | 0,1% / 0,3% | 0,0003 / 0,0005 | 0,1% / 0,1% |
| 6 | 0,0008 / 0,0213 | 0,2% / 5,0% | **0,2008 / 0,3738** | **28,3% / 46,9%** |
| 7 | 0,0185 / 0,0186 | 1,7% / 1,8% | 0,0348 / 0,0349 | 2,4% / 2,4% |
| 8 | 0,0006 / 0,0012 | 0,1% / 0,2% | 0,0020 / 0,0562 | 0,3% / 12,3% |

- K=5 và K=8 gần như cùng một nghiệm ở mọi seed.
- K=7 ổn định nhưng mọi cặp đều lệch cùng một mức (~1,7–2,4% nhãn): các seed khác hội tụ về một nghiệm gần kề, khác seed được chọn.
- **K=6 ở fit cuối không ổn định:** các seed rơi vào ít nhất hai nghiệm khác nhau (đến 47% nhãn khác). Seed được chọn theo LL train, nhưng ý nghĩa từng state của K=6 phụ thuộc seed.
- Ghép ID tự động là bằng chứng về độ lặp lại của nghiệm, **không** chứng minh state có ý nghĩa kinh tế ổn định.

## 5. Quy tắc nhãn: raw / confirm_2 / confirm_3

Trên validation trước tháng 8 (29.036 quan sát), model K=7 selection:

| Quy tắc | Đổi nhãn | Có nhãn xác nhận | `UNCERTAIN` | Nhãn giữ có p<10% | Chờ (quan sát) | Chờ TB / tối đa (phút) |
|---|---:|---:|---:|---:|---:|---:|
| raw | 10.184 | 100% | 0% | 0% | 0 | 0 / 0 |
| confirm_2 | 865 | 47,3% | 52,7% | 18,5% | 1 | 1,4 / 92 |
| confirm_3 | 190 | 34,1% | 65,9% | 38,5% | 2 | 3,0 / 93 |

Các K khác cùng xu hướng: K lớn hơn → ít nhãn xác nhận hơn và nhiều nhãn giữ yếu hơn (K=8 `confirm_3`: 27,0% có nhãn, 51,1% nhãn giữ p<10%). Mức chờ tối đa 92–93 phút là chuỗi đi qua nghỉ trưa.

**Kết luận mục 5: chưa phù hợp để triển khai.** Giảm đổi nhãn 12–54 lần nhưng hơn một nửa đến hai phần ba quan sát không có nhãn, và nhãn đang giữ thường không còn được posterior ủng hộ (18–39% dưới 10%). Đây đúng là trường hợp gói cảnh báo: "bộ lọc bỏ quá nhiều nhãn hoặc giữ nhãn yếu". Không kết luận regime bền hơn.

Replay mô tả trên 03/08–25/09/2026 đã xem (`seen_replay/`, `frozen_k7_reference/`): model K=7 gốc cho 3.844 / 276 / 44 lần đổi nhãn và 34,0% / 20,6% có nhãn — **khớp** bảng trong `NGU_CANH_BAN_GIAO.md`. K=7 refit cho 3.871 / 278 / 44. Không dùng số liệu này để chọn K, seed hay ngưỡng.

## 6. Artifact

`final_models/k5 … k8/`: `runtime_research_candidate.joblib`, `student_baseline.json`, `model_manifest.json`. Khóa lúc 29/09/2026 05:52. Mọi K có seed cuối hội tụ nên không thiếu artifact.

| K | Seed cuối | Log density trong mẫu (audit train, không phải validation) |
|---|---:|---:|
| 5 | 144 | 6,27578 |
| 6 | 146 | 6,27632 |
| 7 | 148 | 6,27958 |
| 8 | 143 | 6,27993 |

K=7 (ứng viên chọn): artifact SHA-256 `4c67a274092bdcde4cd7c00af89b7308268ac8360285c3346450b30e3c86d69f`. Model K=7 gốc (`9071bb38…`) giữ nguyên và vẫn là model báo cáo cuối ngày (`production_model_replaced: false`).

## 7. Đánh đổi và phần còn thiếu

- **Thí nghiệm lịch sử: hoàn thành.** Kiểm chứng tương lai: **chưa**. Theo gói, holdout mới bắt đầu sau ngày khóa (29/09/2026), tức từ phiên 30/09/2026; cần chốt tiêu chí trước khi xem giai đoạn đó.
- K=7 thắng K=5/6 với biên nhỏ và hòa K=8 trên một đoạn validation duy nhất; chưa có khoảng tin cậy hay nhiều fold.
- K=6 không ổn định giữa seed ở fit cuối; K=7 nhiều seed cần gần trần cập nhật.
- Quy tắc xác nhận nhãn hiện tại đánh đổi quá nhiều độ phủ; nếu muốn giảm đổi nhãn, cần thiết kế khác (ngưỡng khác, hoặc hướng mô hình như Sticky HMM/HSMM mà gói ghi là ngoài phạm vi) và một kế hoạch mới.
- Chưa giải quyết: rollover/mã hợp đồng gốc, độ trễ và bản sửa dữ liệu khi vận hành trực tiếp.

## Ghi chú vận hành

- `RUN_K_STABILITY.ps1` dừng ngay trên **Windows PowerShell 5.1** khi Python ghi DeprecationWarning ra stderr (`$ErrorActionPreference='Stop'` + `2>&1`). Đã chạy cùng script bằng **PowerShell 7** (`pwsh`), không sửa file. Nên sửa launcher ở phiên bản gói sau.
- Chạy qua `outputs/run_selection_then_final.ps1` (selection rồi final, final chỉ chạy khi có `selection_decision.json`); tạm dừng/chạy tiếp bằng `outputs/TAM_DUNG.bat`, `outputs/CHAY_TIEP.bat`. Mỗi lần chạy tiếp mất thêm ~13 phút đọc lại fit đã xong; kết quả không đổi.
- Không push Git, không sửa model/kế hoạch/Task Scheduler/báo cáo cuối ngày hiện tại, theo yêu cầu của gói.
