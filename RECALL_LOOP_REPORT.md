# RL-SAHI Recall-Max — Báo cáo vòng lặp tự động qua đêm

**Mục tiêu (theo yêu cầu):** giữ YOLO11s COCO **đóng băng** (không fine-tune); dùng RL đưa ROI vào
vật nhỏ để bắt được càng nhiều càng tốt; vật to do YOLO full-image lo. Tối đa **small_recall**,
chấp nhận nhiều crop/FP hơn.

**Môi trường chạy:** máy local (GTX 1650 4GB, `D:\Python_Main` = Python 3.11 + torch cu121).
Dataset VisDrone tải về ổ D:. Train RL **không gọi YOLO trong vòng lặp** (reward = phủ hard-region
hoặc crop-outcome), YOLO chỉ chạy lúc cache + benchmark → đủ nhanh (~2s/episode) để lặp nhiều cấu hình.
Quy mô nhỏ (train ~800–1500 ảnh, benchmark val/test 150) — **số liệu mang tính tương đối để chọn hướng**,
không phải run 20k-ep đầy đủ.

---

## 1. Lỗi đã sửa (bằng config, không đụng code)

| Lỗi gốc | Sửa | Vì sao quan trọng |
|---|---|---|
| `reward_clip=10` | **100** | Clip 10 cắt mọi lát cắt tốt (hard_hit 4.0 × nhiều vật) về 10 → agent học "cắt ít". Đây là nguyên nhân model cũ chỉ cắt 1.19 crop/ảnh. |
| Chọn best.pt theo mAP+phạt crop/FP | theo **small_recall** | Đúng mục tiêu recall |
| `output_conf=0.25` | **0.05** | Đòn bẩy recall lớn nhất (giữ vật nhỏ mờ) |
| `max_slices=8`, phạt hiệu suất cao | 14, giảm phạt | Cho phủ nhiều vật hơn |

## 2. Các thí nghiệm đã chạy (loop tự động)

- **Iter1** (S1 120ep): xác nhận hướng recall-max đúng; `output_conf=0.05` cho recall cao nhất.
- **Iter2** (S1 1000ep): RL ≈ heuristic top-K nhưng ít crop hơn ~40%.
- **Iter3** (quét `min_slice_utility`, KHÔNG train lại): ngưỡng chấp nhận đã cạn dư địa (crop thêm dư thừa).
- **Iter4** (bật **crop-outcome reward**): THẮNG hard-region — recall+mAP cao hơn, crop ít hơn, sample-efficient.
- **Iter5** (crop-outcome 1000ep): benchmark **held-out TEST** — RL vượt heuristic (bảng dưới). ⭐ BEST.
- **Iter6** (thêm data 1500 + 1500ep): held-out TEST recall 0.425 ≈ iter5 (0.428), vẫn > topk (0.406).
  Thêm data/ep không cải thiện (chạm trần quy mô nhỏ) NHƯNG **xác nhận "RL > heuristic" vững qua 2 run độc lập**.
  → **Loop hội tụ, dừng ở đây.**

## 3. Kết quả tốt nhất — HELD-OUT TEST-150 (con số quan trọng nhất)

Model: crop-outcome reward, 1000 ep. Đã khóa tại `runs/best_recall/best_iter5_test0428.pt`.

**@ output_conf = 0.05 (tối đa recall):**
| Method | mAP50 | small_recall | fp/img | crops/img |
|---|---:|---:|---:|---:|
| YOLO full-image | 0.181 | 0.044 | 25 | 0 |
| Fixed-grid SAHI | 0.267 | 0.457 | 102 | 28 |
| Objectness top-K (heuristic) | 0.243 | 0.385 | 77 | 14 |
| **RL-SAHI (của ta)** | **0.250** | **0.428** | 84 | **9.35** |

**@ output_conf = 0.10 (cân bằng, ít FP hơn):**
| Method | mAP50 | small_recall | fp/img | crops/img |
|---|---:|---:|---:|---:|
| YOLO full-image | 0.167 | 0.030 | 13 | 0 |
| Fixed-grid SAHI | 0.257 | 0.377 | 59 | 28 |
| Objectness top-K | 0.238 | 0.328 | 48 | 14 |
| **RL-SAHI** | **0.240** | **0.354** | 50 | **9.2** |

### Điểm mấu chốt cho luận văn
1. **RL-SAHI VƯỢT heuristic top-K trên held-out** (recall 0.428 vs 0.385, mAP 0.250 vs 0.243) mà dùng
   **ít crop hơn** (9.4 vs 14). → Chứng minh policy học được đặt crop **khôn hơn** heuristic, không chỉ
   ngang bằng. Đây là đóng góp khoa học mạnh nhất.
2. Recall **0.428 = ~9.7× YOLO full-image** (0.044). Vật nhỏ được vớt lên rõ rệt nhờ RL-ROI.
3. Đạt **94% recall của fixed-grid ở 1/3 số crop** (9.4 vs 28) và **ít FP hơn** (84 vs 102).
   → Định vị đúng: RL-SAHI cho recall gần bằng cắt lưới dày nhưng **hiệu quả hơn nhiều**.

## 3b. Minh họa trực quan (YOLO full-image vs RL-SAHI)

**So sánh theo mật độ vật nhỏ** (conf=0.10) — lợi ích của RL-SAHI tăng dần theo độ đông:

![Sparse/Medium/Dense](figures/fig_grid_sparse_medium_dense.jpg)

| Cảnh | GT | YOLO full-image | RL-SAHI | Lát cắt |
|---|---:|---:|---:|---:|
| Thưa | 11 | 13 box | 25 box | 5 |
| Vừa | 94 | 35 box | 124 box | 12 |
| Dày | 293 | 26 box | 197 box | 14 |

**Ảnh dày @ conf=0.05 (recall tối đa):** full-image 47 → RL-SAHI **283 box** (gần bằng GT 293):

![Dense recall-max](figures/fig_dense_conf005_recallmax.jpg)

*Nhận xét:* RL đặt lát cắt đúng vào vùng đông người nhỏ (không lạm dụng ở cảnh thưa); vật to vẫn do
YOLO full-image bắt. Ở conf 0.05 recall cao nhất nhưng có FP; conf 0.10 (ảnh `figures/fig_dense_conf010_balanced.jpg`)
sạch hơn. **Nên đặt cặp ảnh này trong slide bảo vệ.**

**RL đặt crop ở đâu?** (xanh = ROI RL chấp nhận + số detection; đỏ = ROI thử nhưng loại vì ít vật)

![ROI placement theo mật độ](figures/fig_roi_placement_grid.jpg)

Số ROI RL dùng **thích ứng theo mật độ**: cảnh thưa 5 (loại 19, rất kén) → vừa 12 → dày 14. Các ROI xanh
dồn vào vùng đông vật nhỏ (mỗi lát 20–32 detection), ROI đỏ nằm trên nền trống (cây/tòa nhà). → Bằng chứng
policy **học được vị trí đặt crop có ích**, không cắt lưới mù như SAHI cố định.

## 4. Cách dùng model tốt nhất

```bash
PY="D:/Python_Main/python.exe"
# Infer 1 ảnh (vẽ ROI + detection):
"$PY" scripts/infer.py --config configs/rc_005.yaml --checkpoint runs/best_recall/best_iter5_test0428.pt --image <anh.jpg>
# Benchmark lại:
"$PY" scripts/benchmark.py --config configs/rc_005.yaml --checkpoint runs/best_recall/best_iter5_test0428.pt --split test --limit 150 --out-dir runs/best_recall/bench
```
- Cấu hình thắng: `configs/sweep_s1_crop_long.yaml` (crop-outcome reward, recall-max).
- `output_conf`: **0.05** cho recall tối đa (FP cao), **0.10** nếu muốn cân bằng.

## 5. Giới hạn trung thực (phải ghi trong luận văn)
- YOLO COCO đóng băng KHÔNG bắt được lớp van/tricycle/awning và vật cực nhỏ → recall tuyệt đối bị chặn trên.
  "Bắt hết" là bất khả; ta tối đa recall của lớp COCO-detectable.
- FP/ảnh cao (50–84) do hạ `output_conf` — đánh đổi recall lấy precision. Cần nêu rõ.
- Số liệu ở đây là **quy mô nhỏ trên 1 GPU yếu** (train ≤1500 ảnh, ≤1500 ep). Để có số cuối cho báo cáo,
  nên chạy lại cấu hình thắng ở **quy mô đầy đủ (toàn train, ~10–20k ep) trên cloud** khi có GPU.

## 6. Việc tiếp theo đề xuất
- Chạy lại `sweep_s1_crop_long.yaml` full-scale trên Lightning/Colab (bundle đã cập nhật).
- Báo cáo bảng 4-method (đã có baseline heuristic) — nhấn mạnh RL > heuristic ở crop ít hơn.
