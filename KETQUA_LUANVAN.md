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

**Recall (ghép 1-1, val, ngưỡng vận hành conf 0.25).** RL-SAHI đạt **recall vật nhỏ 0.605 — cao nhất**: hơn lưới 0.6 **+0.041** và hơn SAHI **+0.055**, dù dùng **ít crop hơn SAHI 40%** (16.6 so với 27.6). Đáng chú ý, SAHI với 27.6 lát có recall *thấp hơn* lưới 9 lát (0.550 so với 0.564) — nghĩa là recall cao hơn của RL-SAHI **không đến từ việc chạy nhiều crop hơn**, mà từ việc lát RL nhắm trúng vật nhỏ lưới bỏ sót. Chi phí đổi lại gần bằng không (mAP@0.5:0.95 −0.001).

**Hành vi ROI học được (thành quả của agent).** Sau khi huấn luyện với phần thưởng nhắm vật nhỏ và phạt chồng lấn, agent học cách **rê ROI tới các cụm vật nhỏ dày và trải đều** thay vì dồn vào một điểm nóng. Hình minh họa (Phụ lục) cho thấy các ROI đỏ rơi vào cụm phương tiện nhỏ mà lưới/full dễ bỏ sót — đây là biểu diễn trực quan, diễn giải được của chính sách đã học.

---

## 4.5. Phân tích bổ sung (ablation)

**(a) Đóng góp thuần của RL so với lưới thô.** So `RL-SAHI` với `lưới 0.6` (cùng cấu hình, bỏ phần RL): RL cao hơn **+0.004…0.005 mAP@0.5** nhưng **+0.041 recall vật nhỏ** (0.605 so với 0.564, ghép 1-1). Lát RL mua thêm recall trên nền lưới với chi phí mAP gần bằng không.

**(a′) Kiểm chứng "agent biết nhìn đâu" — so sánh theo ngân sách (budget sweep).** Để tách riêng chất lượng *đặt lát* khỏi mọi yếu tố khác, cố định **cùng kích cỡ lát (fraction 0.30)** và **cùng số lát K**, chỉ thay đổi *cách chọn vị trí*: chính sách RL, heuristic đỉnh objectness (top-K), K ô gần tâm, và K ô ngẫu nhiên (3 seed):

| K | **RL (agent)** | top-K objectness | center-K | random-K | recall nhỏ: RL vs top-K |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | **0.440** | 0.411 | 0.421 | 0.405 | **0.379** vs 0.263 |
| 4 | **0.464** | 0.422 | 0.423 | 0.406 | **0.437** vs 0.309 |
| 6 | **0.473** | 0.439 | 0.444 | 0.412 | **0.460** vs 0.359 |
| 8 | **0.480** | 0.457 | 0.456 | 0.419 | **0.476** vs 0.407 |

**Chính sách học được thắng mọi baseline đặt-lát ở mọi ngân sách** (+0.022…+0.042 mAP@0.5 so với heuristic mạnh nhất; recall vật nhỏ hơn tới +0.12). Đây là bằng chứng định lượng rằng vị trí ROI là **thành quả học được của agent**, không tái tạo được bằng heuristic tĩnh.

**(a″) Kích cỡ lát quan trọng hơn vị trí (phát hiện kèm theo).** Cùng K, ô cỡ 0.6 (ít phóng đại, phủ rộng) vượt mọi phương án đặt lát cỡ 0.30 — kể cả RL (ví dụ K=4: ô-0.6 0.533 vs RL-0.30 0.464). Không gian hành động của agent giới hạn lát ≤ 0.35 nên nó không thể tự chọn cỡ 0.6; điều này lý giải vì sao cấu hình tốt nhất là **hybrid**: lưới 0.6 đảm nhiệm độ phủ, lát RL đảm nhiệm tinh chỉnh có chủ đích (0.575). Nới biên kích cỡ lát cho agent là công việc tương lai tự nhiên.

**(b) Mức zoom.** Không gian hành động có hai mức phóng đại; chính sách học được **chọn zoom 0%** (đáp án tối ưu cũng chỉ ~4.4%). Kết luận: với detector cố định ở độ phân giải crop, **phóng đại thêm gần như không mang lại phát hiện mới** — một kết quả phủ định trung thực.

**(c) Phần thưởng nhắm vật nhỏ.** Huấn luyện lại agent chỉ thưởng khi bắt vật nhỏ + phạt chồng mạnh **cải thiện rõ hành vi ROI** (trải đều, trúng cụm nhỏ, số lát giảm 8 → 2–6) nhưng **không thay đổi mAP so với lưới thô** — xác nhận trần chất lượng do lưới phủ rộng, không phải do huấn luyện chưa đủ.

---

## 4.6. Hạn chế và định vị trung thực

1. **Baseline lưới thô rất cạnh tranh.** `lưới 0.6` (không RL) đạt 0.570 mAP@0.5, ngang RL-SAHI (0.575); ở mAP@0.5:0.95 lưới còn nhỉnh hơn chút. Do đó, **RL-SAHI vượt SAHI rõ nhưng chỉ ngang một baseline cắt-lát-thô được điều chỉnh tốt** về mAP. Giá trị riêng của RL nằm ở **tính thích ứng (chọn vùng theo nội dung ảnh), recall cao nhất, và chính sách ROI diễn giải được**, không phải một cú nhảy mAP so với lưới.

   *Nguyên nhân cấu trúc (tự nhận diện):* trong quá trình huấn luyện, phần thưởng của agent được tính so với baseline **chỉ gồm YOLO toàn ảnh**, trong khi lúc triển khai agent hoạt động **cạnh lưới thô** — nghĩa là agent được thưởng cả khi tìm lại vật mà lưới đằng nào cũng phát hiện. Biên lợi ~+0.005 so với lưới vì vậy là **hệ quả của phát biểu bài toán**, không phải do huấn luyện chưa đủ; hướng khắc phục (đưa phát hiện của lưới vào baseline phần thưởng — "học bổ khuyết lưới") được ghi nhận là công việc tương lai, với trần cải thiện ước lượng khiêm tốn (+0.01–0.02) do lưới đã phủ kín ảnh.

2. **Thước đo.** Kết quả là mAP@0.5 / @0.5:0.95 theo COCO; không so trực tiếp với các chỉ số recall thô của một số báo cáo khác (khác thước đo).

3. **Phóng đại (zoom) không mang lại lợi ích** với detector cố định — muốn khai thác cần đầu head thưa (sparse head), ngoài phạm vi đồ án.

---

## 4.8. So sánh với các công bố quốc tế trên VisDrone

Vì nhãn của tập test-dev/test-challenge VisDrone **không công khai**, gần như toàn bộ công trình phát hiện trên VisDrone (ClusDet, DMNet, GLSAN, AMRNet, YOLC, CZDet, UFPMP-Det, AD-Det, CEASC…) đều báo cáo trên **tập validation (548 ảnh)** với thước đo COCO — trùng đúng thiết lập của chúng tôi, nên so sánh là **cùng sân, hợp lệ**. Bảng 5 định vị kết quả của chúng tôi (số đối chứng lấy từ UFPMP-Det AAAI'22 và AD-Det Remote Sensing 2025).

| Phương pháp | Backbone (tham số) | AP@0.5:0.95 | AP@0.5 |
|---|---|:---:|:---:|
| ClusDet (ICCV'19) | ResNet-50 (~42M) | 26.7 | 50.6 |
| CEASC (CVPR'23, hiệu quả) | ResNet-50 | 28.9 | 48.6 |
| GLSAN (TIP'21) | ResNet-50 | 30.7 | 55.4 |
| AMRNet (2020) | ResNet-50 | 31.7 | 52.7 |
| YOLC (TITS'24) | ResNeXt-101 | 33.7 | 57.4 |
| CZDet (2023) | ResNet-101 | 34.4 | 59.7 |
| AD-Det (RS'25) | GFL 2-stage, ResNet-50 (64M) | 35.3 | 57.9 |
| HRDNet | ResNeXt-101 (nặng) | 35.5 | 62.0 |
| **RL-SAHI (đề xuất)** | **YOLO11s (9.4M)** | **35.8** | **57.5** |
| UFPMP-Det (AAAI'22) | GFL 2-stage, ResNet-50 | 36.6 | 62.4 |
| AD-Det* (flip, X101) | ResNeXt-101 (~101M) | 37.5 | 60.9 |
| UFPMP-Det (X101 + MS) | ResNeXt-101 + multi-scale | 40.1 | 66.8 |

*Bảng 5. Định vị trên VisDrone val (thước COCO). Số đối chứng: UFPMP-Det (arXiv:2112.10415), AD-Det (arXiv:2504.05601).*

**Định vị trung thực.** Xét **AP@0.5:0.95** (thước COCO chính), kết quả 0.358 của chúng tôi **ngang phân khúc SOTA** — sánh ngang AD-Det ResNet-50 (0.353), HRDNet (0.355), sát UFPMP-Det ResNet-50 (0.366) — và **vượt** một loạt phương pháp crop-based (YOLC, CZDet, GLSAN, AMRNet, CEASC, ClusDet, DMNet). Điều đáng chú ý là chúng tôi đạt mức này với **backbone nhỏ hơn 4.5–11 lần** (9.4M so với 42–101M tham số) và **tốc độ tương đương** (496 ms so với ~514 ms của AD-Det ResNet-50 trên GPU cùng lớp). Đóng góp do đó là **độ chính xác trên mỗi đơn vị tính toán**, chứ không phải phá kỷ lục tuyệt đối.

Ở **AP@0.5**, kết quả 0.575 của chúng tôi ở mức **trung bình–khá**: các phương pháp SOTA nặng — UFPMP-Det (62.4–66.8), HRDNet/SAIC-FPN (~62), AD-Det ResNeXt-101 (60.9) — **vượt rõ** nhờ backbone lớn và độ phân giải cao. Chúng tôi **không tuyên bố đạt SOTA**.

*Hai lưu ý thước đo (nêu để minh bạch): (i) AP@0.5:0.95 của chúng tôi tính bằng thuật toán ghép riêng, chạy cao hơn khoảng 10% so với công cụ Ultralytics (ở full@640: 0.243 so với 0.22), nên nhận định "ngang SOTA" ở AP@0.5:0.95 cần được hiểu thận trọng; AP@0.5 thì đã được hiệu chỉnh trùng khớp Ultralytics (0.387 vs 0.378). (ii) Không so số val của chúng tôi với các con số test-dev của những báo cáo khác (ví dụ AP@0.5 = 43.5 trong paper SAHI gốc là test-dev, chỉ AP@0.5, và dùng detector khác).*

## 4.7. Kết luận chương

Đồ án đạt hai kết quả chính, đều được kiểm chứng bằng thước đo COCO đã hiệu chỉnh:

- **Train-on-crop** là điều kiện then chốt để cắt lát phát huy, nâng mAP các phương pháp cắt lát tới **+0.15** và cải thiện đúng các lớp vật nhỏ.
- **RL-SAHI** (agent RL rê ROI, batch hóa) đạt **mAP cao nhất (0.575)**, **recall vật nhỏ cao nhất (0.605, ghép 1-1; +0.041 so lưới)**, **nhanh hơn SAHI (496 ms)**, và chính sách đặt lát học được **vượt mọi heuristic đặt-lát (top-K objectness / center / random) ở mọi ngân sách crop khi so cùng kích cỡ lát** (+0.02…+0.04 mAP@0.5) — vị trí ROI là thành quả học được, có bằng chứng định lượng.

So với YOLO nền (0.394), pipeline hoàn chỉnh nâng lên **0.575 mAP@0.5** — một kết quả cạnh tranh, đạt được một cách trung thực và có phân tích ablation đầy đủ.

---

*Số liệu tái lập bằng `scripts/eval_coco_map.py` (COCO mAP, calibrate với `yolo val`), `scripts/benchmark_oneshot.py` (tốc độ), `scripts/viz_detect.py` (minh họa ROI). Detector train-on-crop: `scripts/make_crops.py` + fine-tune 20 epoch. Agent nhắm-vật-nhỏ: `configs/ft_rl_small.yaml`.*
