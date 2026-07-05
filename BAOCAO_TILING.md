# So sánh: YOLO fine-tune vs SAHI thường vs Tiling (adaptive tile-coding)

## 1. Thiết lập
- **Detector nền:** YOLO11 **đã fine-tune trên VisDrone** (10 lớp 0–9: pedestrian…motor), weight `best_visdrone.pt`.
- **3 phương pháp so sánh** (cùng detector, cùng NMS, cùng ảnh — công bằng):
  1. **YOLO-ft full** — chạy detector 1 lần trên ảnh gốc (không cắt).
  2. **SAHI thường** (fixed-grid) — cắt lưới cố định phủ đều (~28 lát), gộp bằng class-aware NMS.
  3. **Tiling** — adaptive tile-coding (tác nhân RL biểu diễn state bằng tile coding + hàm Q tuyến tính,
     tự chọn vùng cắt; giữ nguyên MDP di chuyển ROI).
- **Đánh giá:** VisDrone test, 150 ảnh held-out, mAP@0.5 (VOC all-points), small-recall (vật ≤ p40 diện tích),
  FP/ảnh, số crop/ảnh. Ngưỡng output_conf = 0.10.
- **Ràng buộc phần cứng:** GTX 1650 4GB → quy mô nhỏ (train ≤1500 ảnh, tiling 2000 ep). Số mang tính so sánh
  tương đối; cần chạy full-scale để lấy số cuối.

## 2. Kết quả chính (TEST-150 held-out)

| Method | mAP50 | small_recall | FP/ảnh | crops/ảnh |
|---|---:|---:|---:|---:|
| YOLO-ft full | 0.316 | 0.373 | 64 | 0 |
| SAHI thường (fixed-grid) | 0.316 | **0.603** | 192 | 28 |
| **Tiling (tile-coding)** | 0.310 | 0.505 | 135 | **11.5** |
| *objectness-topk (tham chiếu)* | 0.323 | 0.570 | 145 | 14 |

**Quét ngân sách crop của Tiling (inference, không train lại):**

| Tiling max_slices | small_recall | FP/ảnh | crops/ảnh |
|---|---:|---:|---:|
| 14 (mặc định) | 0.505 | 135 | 11.5 |
| 16 | 0.506 | 135 | 12.4 |
| 20 (nới ngưỡng) | 0.393 | 67 | 4.05 |

→ **Recall của Tiling bão hòa ~0.505 ở ~11–12 lát.** Tăng budget không giúp (16≈14); ép 20 lát + nới
ngưỡng phản tác dụng (cắt marginal sớm → trùng lấp → dừng episode sớm). Khoảng cách tới SAHI (0.603) là
**cấu trúc** (SAHI phủ lưới dày đều; tiling thích ứng thưa hơn) — Tiling đổi coverage lấy hiệu quả.

## 3. Nhận định
1. **Tiling vs YOLO-ft full:** tiling nâng small-recall **0.373 → 0.505** (vớt thêm vật nhỏ), nhưng **mAP
   không đổi** (0.316 vs 0.310) — phần vớt thêm lẫn FP. Giá: +11.5 crop, FP 64→135.
2. **Tiling vs SAHI thường:** tiling đạt **~84% small-recall của SAHI (0.505 vs 0.603) nhưng chỉ dùng
   41% số crop (11.5 vs 28) và 70% FP (135 vs 192)** → **hiệu quả hơn nhiều**.
3. **Không method cắt lát nào tăng mAP@0.5** trên detector đã fine-tune (~0.31 phẳng): slicing đổi
   recall lấy FP, hai cái bù trừ. → Đóng góp của Tiling nằm ở **small-recall + hiệu quả crop/FP**,
   không phải mAP.
4. Tiling (tile-coding, classical linear FA) cho kết quả **ngang một agent Deep-DQN** (recall 0.505 vs
   0.504, crop 11.5 vs 11.6) → phương án đơn giản, chạy CPU, đủ tốt.

## 4. Định vị đóng góp
> *"Adaptive tiling (tile-coding) đạt gần trọn small-recall của SAHI dày ở ~40% chi phí crop và ít FP hơn,
> trên detector đã fine-tune — một điểm Pareto hiệu-quả-vs-recall."*

## 5. Giới hạn trung thực
- Không cải thiện mAP@0.5 (chỉ small-recall). Phải nêu rõ.
- FP còn cao (135) do output_conf thấp; là trade-off điều chỉnh được.
- Quy mô nhỏ / 1 seed / 150 ảnh; cần full-scale + đa seed cho số cuối.

## 6. Hình minh họa
- `figures_ft_tile/fig_fttile_grid.jpg` — full vs Tiling theo 3 mật độ (sparse/medium/dense).
- `figures_ft_tile/fig_fttile_dense.jpg`, `fig_fttile_roi_dense.jpg` — chi tiết ảnh dày + bản đồ ROI.
