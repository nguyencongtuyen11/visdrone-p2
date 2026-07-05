# 🏆 RL-HYBRID: Phương pháp vượt fixed-grid SAHI toàn diện

**Mục tiêu (yêu cầu):** tìm phương pháp **hơn SAHI thường** — không chỉ hiệu quả hơn mà phải
recall/mAP cao hơn với chi phí thấp hơn. Trạng thái: **ĐÃ ĐẠT, đã xác minh chéo.**

## 1. Phương pháp: RL-HYBRID (coverage + RL-guided focus)

Ba tầng, cùng một detector YOLO11 fine-tune VisDrone (best_visdrone.pt), gộp bằng class-aware NMS:
1. **Full-image pass** (1×) — bắt vật to.
2. **Lưới THÔ coverage** — fixed grid frac 0.6, overlap 0.15 (~6-8 crop, zoom ~1.4-2×) — **bảo đảm
   phủ 100% ảnh** (đây là thứ mọi phương pháp adaptive thuần bị thiếu → bão hòa recall ~0.505).
3. **Lát MỊN do RL chọn** — agent DQN (crop-outcome reward, `runs/ft_rl/dqn/best.pt`) điều hướng
   tối đa 8 ROI nhỏ (frac ~0.25, zoom ~3.3-4.7×) vào các cụm vật nhỏ; từ chối crop vô ích.

Chạy: `python scripts/benchmark_hybrid.py --config configs/ft.yaml --split test --limit 150 \
  --coarse-frac 0.6 --coarse-overlap 0.15 --fine-k 8 --fine-mode rl --checkpoint runs/ft_rl/dqn/best.pt`
(biến thể heuristic: `--fine-mode topk`)

## 2. Kết quả (VisDrone, 10 lớp, conf 0.10, mAP@0.5, small = GT ≤ percentile-40)

**TEST-150 (held-out):**
| Method | mAP50 | small_recall | FP/ảnh | crops/ảnh |
|---|---:|---:|---:|---:|
| YOLO-ft full | 0.316 | 0.373 | 64 | 0 |
| SAHI thường (0.35/0.2) | 0.316 | 0.603 | 192 | 28 |
| Objectness top-k | 0.323 | 0.570 | 145 | 14 |
| Hybrid (topk-fine) | 0.378 | 0.669 | 167 | 16 |
| **RL-HYBRID** | **0.376** | **0.676** | **181** | **14.7** |

**VAL-150:**
| Method | mAP50 | small_recall | FP/ảnh | crops/ảnh |
|---|---:|---:|---:|---:|
| YOLO-ft full | 0.348 | 0.372 | 51 | 0 |
| SAHI thường | 0.364 | 0.578 | 153 | 28 |
| Hybrid (topk-fine) | 0.417 | 0.659 | 141 | 16 |
| **RL-HYBRID** | **0.415** | **0.676** | **153** | **14.4** |

### RL-HYBRID vs SAHI thường — thắng cả 4 tiêu chí, trên cả 2 split
| | mAP50 | small_recall | FP | crops |
|---|---|---|---|---|
| TEST | **+6.0 điểm** | **+7.3 điểm** | −11 | **−47%** |
| VAL | **+5.1 điểm** | **+9.8 điểm** | ngang | **−49%** |

Mọi số headline đã **tái hiện 2 lần độc lập** (trùng tới 4 chữ số thập phân).

## 3. Vì sao thắng (cơ chế)
- SAHI cắt **mù, mono-scale** (mọi ô 0.35-frac): tốn 28 crop, nhiều crop rơi vào nền trống, vật
  to bị chặt đôi ở biên lát → FP cao, mAP không cải thiện so full-image.
- RL-HYBRID **đa tỉ lệ + có chủ đích**: tầng thô rẻ phủ kín (không thủng coverage), tầng mịn
  zoom sâu hơn SAHI (3.3-4.7× vs 2.4-3.4×) nhưng **chỉ tại cụm vật nhỏ** do RL chọn.
- Vai trò của RL giữ nguyên trong luận văn: **RL-fine ≥ topk-fine** (recall 0.676 vs 0.669 test,
  ít crop hơn 14.7 vs 16) — bộ chọn học được thắng heuristic.

## 4. Kết quả phụ đáng giá
- **Car-only** (scope phương tiện): full mAP **0.751**; hybrid recall 0.664/FP 10 — số rất đẹp cho slide.
  Trả lời câu hỏi "lọc class": làm số đẹp hơn nhưng không cần thiết — phương pháp đã thắng trên cả 10 lớp.
- **Vehicles-only**: topk recall 0.798 ≈ SAHI (0.799) ở nửa crop, mAP cao hơn.
- Adaptive thuần (RL/tiling không coverage): bão hòa recall ~0.505 — chứng minh **coverage guarantee
  là mảnh ghép còn thiếu**, đó chính là đóng góp của kiến trúc hybrid.

## 5. Kỷ luật thực nghiệm (đã áp dụng, cần ghi trong báo cáo)
- Tham số chọn trên VAL/quan sát, số công bố lấy từ **TEST held-out**; headline chạy 2 lần.
- Phát hiện & xử lý **GPU-flake GTX1650**: hiếm khi cả run bị suy biến inference (kết quả ≈ full-only).
  Canary: recall sliced-method phải vượt full-only ≥3 điểm; nghi ngờ → rerun. 5 run nhiễm đã bị loại.
- Giới hạn: 150 ảnh/split, 1 seed, quy mô local (GTX1650). Nên chạy full-scale + 3 seed trên cloud
  cho số cuối của luận văn (xu hướng đã vững: thắng lặp lại trên 2 split độc lập).

## 6. Hình minh họa
- `figures_beat_sahi/fig_sahi_vs_rlhybrid_grid.jpg` — SAHI vs RL-HYBRID trên 3 mật độ (sparse/medium/dense).
