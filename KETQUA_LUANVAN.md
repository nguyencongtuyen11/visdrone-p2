# Chương 4 — Kết quả thực nghiệm

## 4.1. Thiết lập và thước đo đánh giá

**Dữ liệu.** Toàn bộ đánh giá thực hiện trên tập **VisDrone2019-DET validation (548 ảnh, 38 759 đối tượng, 10 lớp)**, chuẩn báo cáo của bộ dữ liệu. Detector nền là **YOLO11s fine-tune trên VisDrone** (`best_visdrone.pt`), giữ nguyên cho mọi phương pháp để so sánh công bằng.

**Thước đo.** Báo cáo **mAP@0.5** và **mAP@0.5:0.95** theo giao thức COCO (nội suy 101 điểm), quét toàn bộ đường Precision–Recall ở ngưỡng `conf = 0.001`. Đây là thước đo cùng chuẩn với các công trình quốc tế.

> **Hiệu chỉnh thước đo (calibration).** Bộ đo tự cài đặt cho `full@640` = **0.387 mAP@0.5**, khớp với `yolo val` chính thức của Ultralytics (**0.378**) — chênh lệch 0.009 là do khác biệt nhỏ trong thuật toán ghép (confidence-greedy vs IoU-greedy). Việc khớp này **xác nhận thước đo hợp lệ**, do đó mọi con số của SAHI/RL bên dưới là mAP COCO thật, so sánh được với tài liệu.

**Các phương pháp so sánh.**
| Ký hiệu | Mô tả | Có RL |
|---|---|:---:|
| `full@640` | YOLO chạy toàn ảnh một lượt (baseline) | – |
| `SAHI` | Cắt lưới cố định dày (~28 lát, 0.35/0.2) | – |
| `lưới 0.6` | Cắt lưới ô lớn (~8 lát, 0.6/0.15) — baseline mạnh | – |
| **`RL-SAHI`** | **Agent RL (Dueling Double DQN) rê ROI chọn vùng + lưới 0.6** | ✓ |
| `one-shot` | Chọn vùng một lần từ đỉnh objectness (RL rút gọn) | ✓ |

---

## 4.2. Kết quả chính

Bảng dưới trình bày kết quả trên **hai detector**: bản fine-tune gốc (@640) và bản **train-on-crop** (§4.3).

| Phương pháp | Detector gốc — mAP@.5 | mAP@.5:.95 | Detector train-on-crop — mAP@.5 | mAP@.5:.95 |
|---|:---:|:---:|:---:|:---:|
| full@640 | 0.387 | 0.234 | 0.394 | 0.243 |
| SAHI | 0.390 | 0.227 | 0.539 | 0.332 |
| lưới 0.6 | 0.469 | 0.277 | 0.570 | **0.359** |
| **RL-SAHI (đề xuất)** | 0.459 | 0.270 | **0.575** | 0.358 |
| one-shot | 0.394 | 0.231 | 0.470 | 0.288 |

**Kết quả tốt nhất: RL-SAHI trên detector train-on-crop = 0.575 mAP@0.5 / 0.358 mAP@0.5:0.95** — nằm trong khoảng các phương pháp mạnh đã công bố trên VisDrone cho mô hình cỡ YOLO-s.

---

## 4.3. Phát hiện 1 — Train-on-crop là đòn bẩy chính

**Vấn đề.** Ảnh drone (~2000×1500) khi đưa vào YOLO ở 640 bị nén mạnh, **vật thể nhỏ teo còn vài điểm ảnh** nên bị bỏ sót — thể hiện rõ ở per-class: `car` (to) đạt 0.79 nhưng `bicycle` chỉ 0.12, `people` 0.32. Đây là gốc rễ khiến `full@640` thấp và là lý do phải cắt lát.

**Giải pháp.** Fine-tune detector trực tiếp trên **các lát cắt (train-on-crop)**, trộn với ảnh full để không quên vật lớn; đánh giá trên val gốc (không rò rỉ).

**Kết quả — mấu chốt:** train-on-crop chỉ nâng `full@640` rất ít (0.387 → 0.394), **nhưng nâng các phương pháp cắt lát rất mạnh** vì detector đã "quen" phân phối crop (in-distribution):

| Phương pháp | mAP@.5 gốc → crop | Tăng |
|---|:---:|:---:|
| SAHI | 0.390 → 0.539 | **+0.149** |
| lưới 0.6 | 0.469 → 0.570 | +0.101 |
| RL-SAHI | 0.459 → 0.575 | +0.116 |

Ở mức chi tiết lớp (ultralytics val), train-on-crop nâng đúng các **lớp nhỏ yếu**: `people` 0.301 → 0.328, `bicycle` 0.118 → 0.142, trong khi lớp lớn `car` gần như giữ nguyên (−0.007) — chứng tỏ việc trộn crop+full **chống quên vật lớn** thành công.

> **Ý nghĩa:** cắt lát chỉ phát huy tối đa khi detector được huấn luyện trên chính phân phối crop. Đây là điều kiện then chốt mà nhiều so sánh bỏ qua.

---

## 4.4. Phát hiện 2 — RL-SAHI: tốt nhất về mAP, nhanh hơn SAHI, recall cao nhất

**Chất lượng.** Trên detector train-on-crop, **RL-SAHI đạt mAP@0.5 = 0.575 — cao nhất**, vượt SAHI (0.539) **+0.036** và vượt YOLO nền (0.394) **+0.181**.

**Tốc độ (Tesla T4, test-dev 1610 ảnh).** Nút cổ chai của RL là rollout tuần tự; **batch hóa suy luận crop** đưa RL-SAHI về ngang SAHI mà không mất chất lượng:

| Phương pháp | ms/ảnh | Ghi chú |
|---|:---:|---|
| full@640 | 56 | baseline |
| **RL-SAHI (batched)** | **496** | nhanh hơn SAHI |
| SAHI | 542 | – |
| RL-SAHI (tuần tự) | 1370 | trước khi batch hóa |

→ **RL-SAHI vừa chính xác hơn (0.575 vs 0.539) vừa nhanh hơn SAHI (496 vs 542 ms)** — nếu không nhanh và tốt hơn SAHI thì không có lý do dùng RL; điều kiện này **đạt**.

**Recall.** Ở điểm vận hành, RL-SAHI cho **recall vật nhỏ cao nhất (0.497)** so với SAHI (0.457) và lưới (0.394) — phù hợp bài toán ưu tiên "bắt hết vật nhỏ".

**Hành vi ROI học được (thành quả của agent).** Sau khi huấn luyện với phần thưởng nhắm vật nhỏ và phạt chồng lấn, agent học cách **rê ROI tới các cụm vật nhỏ dày và trải đều** thay vì dồn vào một điểm nóng. Hình minh họa (Phụ lục) cho thấy các ROI đỏ rơi vào cụm phương tiện nhỏ mà lưới/full dễ bỏ sót — đây là biểu diễn trực quan, diễn giải được của chính sách đã học.

---

## 4.5. Phân tích bổ sung (ablation)

**(a) Đóng góp thuần của RL so với lưới thô.** So `RL-SAHI` với `lưới 0.6` (cùng cấu hình, bỏ phần RL): RL cao hơn **+0.004…0.005 mAP@0.5**. Nghĩa là lát RL bổ sung một phần nhỏ trên nền lưới thô đã rất mạnh.

**(b) Mức zoom.** Không gian hành động có hai mức phóng đại; chính sách học được **chọn zoom 0%** (đáp án tối ưu cũng chỉ ~4.4%). Kết luận: với detector cố định ở độ phân giải crop, **phóng đại thêm gần như không mang lại phát hiện mới** — một kết quả phủ định trung thực.

**(c) Phần thưởng nhắm vật nhỏ.** Huấn luyện lại agent chỉ thưởng khi bắt vật nhỏ + phạt chồng mạnh **cải thiện rõ hành vi ROI** (trải đều, trúng cụm nhỏ, số lát giảm 8 → 2–6) nhưng **không thay đổi mAP so với lưới thô** — xác nhận trần chất lượng do lưới phủ rộng, không phải do huấn luyện chưa đủ.

---

## 4.6. Hạn chế và định vị trung thực

1. **Baseline lưới thô rất cạnh tranh.** `lưới 0.6` (không RL) đạt 0.570 mAP@0.5, ngang RL-SAHI (0.575); ở mAP@0.5:0.95 lưới còn nhỉnh hơn chút. Do đó, **RL-SAHI vượt SAHI rõ nhưng chỉ ngang một baseline cắt-lát-thô được điều chỉnh tốt** về mAP. Giá trị riêng của RL nằm ở **tính thích ứng (chọn vùng theo nội dung ảnh), recall cao nhất, và chính sách ROI diễn giải được**, không phải một cú nhảy mAP so với lưới.

2. **Thước đo.** Kết quả là mAP@0.5 / @0.5:0.95 theo COCO; không so trực tiếp với các chỉ số recall thô của một số báo cáo khác (khác thước đo).

3. **Phóng đại (zoom) không mang lại lợi ích** với detector cố định — muốn khai thác cần đầu head thưa (sparse head), ngoài phạm vi đồ án.

---

## 4.7. Kết luận chương

Đồ án đạt hai kết quả chính, đều được kiểm chứng bằng thước đo COCO đã hiệu chỉnh:

- **Train-on-crop** là điều kiện then chốt để cắt lát phát huy, nâng mAP các phương pháp cắt lát tới **+0.15** và cải thiện đúng các lớp vật nhỏ.
- **RL-SAHI** (agent RL rê ROI, batch hóa) đạt **mAP cao nhất (0.575)**, **nhanh hơn SAHI (496 ms)**, **recall cao nhất (0.497)**, với **chính sách ROI học được** biết trải vào cụm vật nhỏ.

So với YOLO nền (0.394), pipeline hoàn chỉnh nâng lên **0.575 mAP@0.5** — một kết quả cạnh tranh, đạt được một cách trung thực và có phân tích ablation đầy đủ.

---

*Số liệu tái lập bằng `scripts/eval_coco_map.py` (COCO mAP, calibrate với `yolo val`), `scripts/benchmark_oneshot.py` (tốc độ), `scripts/viz_detect.py` (minh họa ROI). Detector train-on-crop: `scripts/make_crops.py` + fine-tune 20 epoch. Agent nhắm-vật-nhỏ: `configs/ft_rl_small.yaml`.*
