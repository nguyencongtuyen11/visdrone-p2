# ONE-SHOT RL — thiết kế (nhánh `oneshot-rl`)

Bỏ **rollout di chuyển** (8 hướng × 10 bước × 14 lát — 22% thời gian, không batch được).
Thay bằng **one-shot chọn vùng + zoom-level** = đúng ý "zoom nhiều/ít/không zoom/bỏ".
Giữ RL làm đích. Bám: Uzkent (one-shot grid), AdaZoom (fixation+scale), AD-Det (K-means P3 1-shot), EVORL (per-region zoom).

## Luồng mới (1 ảnh)
```
YOLO full 1 lần → objectness heatmap 16×16 (đã có sẵn trong DetectionCache)
   │
   ├─ 1. SINH ỨNG VIÊN 1 LẦN: top-K đỉnh objectness (K~12), KHÔNG rollout
   │
   ├─ 2. AGENT QUYẾT ĐỊNH 1 PHÁT / ứng viên (batch 1 forward Q-net nhỏ):
   │      với mỗi vùng chọn 1 action ∈ { DROP · KEEP(no-zoom) · ZOOM_1.5 · ZOOM_2 }
   │
   ├─ 3. Gom tất cả vùng KEEP/ZOOM → BATCH YOLO 1 LƯỢT  (đánh vào 75% chi phí)
   │
   └─ 4. Merge NMS + discard-boundary (đã fix ở main)
```

## Vì sao vẫn là RL (giữ đích đồ án)
Agent **học** chọn zoom-level tối ưu mỗi vùng để **tối đa recall** — reward = crop-outcome (TP thật, đối chiếu GT).
Đây là **contextual bandit 1 bước** (mỗi vùng = 1 quyết định độc lập) — đơn giản & train nhanh hơn MDP tuần tự,
nhưng vẫn là "agent học where/how to look". Không còn 22% rollout, và biết trước hết crop → batch được.

## Kiến trúc (nhỏ hơn nhiều so với 5660-chiều)
- **State/vùng** (local, ~vài trăm chiều): cắt patch quanh vùng từ các map 16×16 (objectness, detection 4 kênh, small)
  + đặc trưng vùng (objectness peak, mật độ proposal, kích thước vật nhỏ ước lượng) + toạ độ/scale chuẩn hoá.
- **Policy**: MLP nhỏ (Dueling giữ cho quen) → Q trên 4 action.
- **Train (bandit)**: Q(s,a) ≈ reward(s,a). Với mỗi vùng thử action → chạy YOLO crop (cache) → reward crop-outcome.
  Loss = MSE(Q, reward). Không bootstrap (1 bước). Nhanh, ổn định.

## Baseline bắt buộc (honest — plan #2)
- **Static**: top-K objectness, KEEP-all, no-zoom (không RL). Để biết RL zoom-decision có ĐÁNG không.
- So với RL-hybrid cũ (rollout) + SAHI.

## Tốc độ kỳ vọng (T4)
Bỏ rollout (~360ms) + batch 1 lượt (đánh 75%): **1636ms → ~400–600ms** (ngang/hơn SAHI) mà giữ RL + recall.
(Không hứa real-time <100ms — cái đó cần sparse-head ESOD, đã chốt là ảo tưởng với detector frozen.)

## Files (nhánh này)
- `src/rl_sahi/rl/oneshot.py` — sinh ứng viên + local state + apply-action + policy net
- `scripts/benchmark_oneshot_rl.py` — chạy one-shot (static baseline TRƯỚC, rồi RL) + đo recall/mAP/FP/latency + so baseline
- `scripts/train_oneshot.py` — bandit training (làm sau khi baseline chạy)

## Thứ tự build
1. ⬜ `oneshot.py`: proposal + apply-action + local-state (test được ngay, không cần train)
2. ⬜ `benchmark_oneshot_rl.py`: chạy **static baseline** one-shot (KEEP-all) — ra số baseline + scaffold
3. ⬜ policy net + `train_oneshot.py` (bandit) → train → benchmark RL vs baseline
