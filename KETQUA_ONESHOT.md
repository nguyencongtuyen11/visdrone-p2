# Kết quả thực nghiệm: RL-ONESHOT

*(Dự thảo mục Kết quả cho luận văn — VisDrone2019-DET test-dev, 1610 ảnh, Tesla T4, YOLO11s fine-tune VisDrone `best_visdrone.pt`, base@640, crop@640, k=8 vùng/ảnh. Recall = recall vật thể nhỏ, GT ≤ percentile-40 diện tích.)*

## 1. Thiết lập so sánh

Bốn phương pháp cùng chung detector nền và cùng gộp bằng class-aware NMS, chỉ khác cách chọn vùng để cắt:

- **full@640** — YOLO chạy 1 lượt trên toàn ảnh (detector nền, không cắt lát).
- **STATIC top-K KEEP** — sinh K=8 vùng từ đỉnh objectness, **giữ hết** (không zoom, không bỏ). Baseline *không có RL* để đo giá trị của quyết định học được.
- **RL-ONESHOT** — cùng K vùng ứng viên; agent RL (REINFORCE) chọn *một phát* mỗi vùng một hành động ∈ {DROP, KEEP, ZOOM×1.5, ZOOM×2}, không rollout.
- **SAHI** — cắt lưới cố định phủ toàn ảnh (~27.6 lát/ảnh), làm mốc "cắt vét cạn".

## 2. Bảng kết quả chính (test-dev 1610 ảnh)

| Phương pháp | mAP@0.5 | Recall (vật nhỏ) | FP/ảnh | Số crop/ảnh | ms/ảnh |
|---|:---:|:---:|:---:|:---:|:---:|
| full@640 (detector nền) | 0.270 | 0.217 | 33.9 | 1.0 | 56 |
| STATIC top-K KEEP | 0.299 | 0.394 | 64.1 | 9.0 | 205 |
| **RL-ONESHOT (đề xuất)** | **0.295** | **0.346** | **45.3** | **3.6** | **154** |
| SAHI (vét cạn) | 0.303 | 0.458 | 98.7 | 27.6 | 515 |

## 3. Phân tích

**(a) So với detector nền — RL-ONESHOT tăng phát hiện vật nhỏ đáng kể với chi phí thấp.**
Recall vật nhỏ tăng từ 0.217 lên 0.346 (**+59%** tương đối) và mAP@0.5 tăng từ 0.270 lên 0.295, chỉ với **3.6 crop/ảnh**. Đây là mức cải thiện lớn so với việc chạy YOLO một lượt.

**(b) So với SAHI — bằng chất lượng, rẻ hơn nhiều bậc.**
RL-ONESHOT đạt mAP 0.295, **chỉ kém SAHI 0.7 điểm** (0.303), nhưng dùng **ít hơn 7.7× số crop** (3.6 so với 27.6), **giảm 54% false positive** (45.3 so với 98.7) và **nhanh gấp 3.3×** (154 ms so với 515 ms). Nói cách khác, phần lớn các lát của SAHI là **dư thừa**: agent đạt gần như cùng mAP mà chỉ cần một phần nhỏ số lát.

**(c) So với STATIC keep-all — vai trò của RL là tỉa, không phải tăng recall.**
Ba phương pháp cắt lát bám sát nhau về mAP (STATIC 0.299, RL 0.295, SAHI 0.303 — chênh trong 0.008). Điểm khác biệt là **hiệu quả**: agent học **bỏ 68% vùng ứng viên** (phân bố hành động: DROP 68%, KEEP 32%), nhờ đó cắt số crop từ 9.0 xuống 3.6, giảm FP từ 64.1 xuống 45.3 (**−29%**) và giảm thời gian từ 205 ms xuống 154 ms — trong khi recall chỉ giảm 0.048 (0.394 → 0.346). Đây là một **đánh đổi recall lấy độ chính xác và tốc độ** do agent tự học, không cần luật thủ công.

**Định vị đóng góp.** RL-ONESHOT là một **điểm vận hành Pareto**: chất lượng (mAP) ngang SAHI/STATIC nhưng **ít crop nhất, ít FP nhất trong nhóm cắt lát, và nhanh nhất**. Đóng góp nằm ở trục **hiệu quả – độ chính xác**, không phải ở recall tuyệt đối.

## 4. Ablation: mức zoom

Không gian hành động cho phép 4 lựa chọn gồm hai mức phóng đại (ZOOM×1.5 tạo crop 23% cạnh ảnh, ZOOM×2 tạo crop 17.5%). **Chính sách học được chọn zoom 0%** — không bao giờ phóng đại, chỉ dùng DROP hoặc KEEP (crop 35%).

Đây **không** phải lỗi huấn luyện: bảng phần thưởng "oracle" (hành động tối ưu theo phần thưởng thực trên tập train) cũng chỉ ưu tiên zoom **4.4%** số vùng (2266/51768). Nghĩa là **với detector đóng băng ở độ phân giải crop 640, việc phóng đại thêm gần như không mang lại phát hiện mới** so với crop KEEP vốn đã phóng to vật nhỏ khi resize về 640. Kết quả phủ định này nhất quán với phân tích cắt-trên-đặc-trưng: trần lợi ích của việc "nhìn sát hơn" bị chặn bởi bản thân detector, không phải bởi thuật toán chọn vùng.

## 5. Hạn chế (trung thực)

1. **Không phải phương pháp recall-max.** Nếu mục tiêu là recall tối đa bất kể chi phí, SAHI (0.458) và cả STATIC keep-all (0.394) đều vượt RL-ONESHOT (0.346). Recall cao hơn đòi hỏi nhiều lát hơn hoặc **detector độ phân giải cao hơn (P2@1280)**, không phải chính sách chọn vùng tốt hơn.
2. **Con số là recall/mAP@0.5, không so trực tiếp được với AP@0.5:0.95 của các công trình SOTA** (khác đơn vị đo). Khoảng cách với SOTA chủ yếu đến từ độ phân giải huấn luyện và giao thức đánh giá của detector, không phải từ thuật toán RL.
3. **Trần recall của khung "top-K vùng + full" chính là baseline keep-all.** RL chỉ dịch chuyển trên đường đổi recall ↔ hiệu quả; muốn phá trần phải can thiệp vào detector.

---
*Số liệu tái lập bằng `scripts/benchmark_oneshot_rl.py --split test --limit 1610 --k 8 --with-sahi` trên `runs/oneshot/policy.pt` (REINFORCE, 6471 ảnh train, 20000 bước). Đã kiểm chứng trên hai cỡ mẫu (test-400 và test-1610) cho cùng kết luận.*
