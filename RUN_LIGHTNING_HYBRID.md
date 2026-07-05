# Chạy RL-HYBRID full-scale trên Lightning.ai (GPU T4)

Mục tiêu: lấy **số chính thức cho luận văn** — train agent RL (crop-outcome) trên FULL VisDrone,
**3 seeds**, benchmark full val — thay cho số quy-mô-nhỏ đã chạy local (GTX 1650, 150 ảnh).

> Lightning Studio có ổ đĩa bền: Stop studio để nhả GPU (ngừng tính tiền), file vẫn còn, Start lại chạy tiếp.
> Mọi lệnh dưới đây dùng `--config configs/ft_cloud...` — KHÔNG dùng ft.yaml (bản đó trỏ cache sang ổ D: của Windows).

## 0. Tạo Studio
- Studio kiểu AI development / Python, gắn **GPU T4**.
- Upload `rl_sahi_lightning_hybrid.zip`, rồi:
```bash
unzip rl_sahi_lightning_hybrid.zip -d rl_sahi && cd rl_sahi
pip install -r requirements.txt
```

## 1. Dataset (CHỈ 1 LẦN — nằm lại trên đĩa bền)
```bash
python scripts/download_visdrone.py
```
Kỳ vọng: train 6471 · val 548 · test 1610 ảnh trong `data/raw/`.

## 2. Cache với detector FINE-TUNE (CHỈ 1 LẦN, ~30-45 phút T4)
```bash
python scripts/detect.py      --config configs/ft_cloud.yaml --split train
python scripts/detect.py      --config configs/ft_cloud.yaml --split val
python scripts/detect.py      --config configs/ft_cloud.yaml --split test
python scripts/hard_region.py --config configs/ft_cloud.yaml --split train
python scripts/hard_region.py --config configs/ft_cloud.yaml --split val
```

## 3. TRAIN 3 SEEDS — chọn 1 trong 2 mức ngân sách

**Về số episode:** dữ liệu thực nghiệm cho thấy agent học bão hòa sau ~500 ep
(400 ep ≈ 1000 ep ≈ 1500 ep trong các test trước; và hybrid rất khoan dung với chất lượng agent
vì lưới thô đã gánh coverage). Nên **2000 ep gần như chắc chắn ngang 6000 ep**.

**Khuyến nghị:** chạy seed 42 bản FULL trước, nhìn cột `val_recall` trong log —
nếu hết tăng từ trước ep ~2000 (rất có thể) thì 2 seed còn lại chạy bản FAST cho đỡ tốn.

Bản FULL (6000 ep, ~3-4h/seed — chắc ăn tuyệt đối):
```bash
python scripts/train.py --config configs/ft_rl_cloud_s42.yaml --split train
python scripts/train.py --config configs/ft_rl_cloud_s43.yaml --split train
python scripts/train.py --config configs/ft_rl_cloud_s44.yaml --split train
```
Bản FAST (2000 ep, ~1-1.5h/seed — tiết kiệm 2/3 giờ GPU):
```bash
python scripts/train.py --config configs/ft_rl_cloud_fast_s42.yaml --split train
python scripts/train.py --config configs/ft_rl_cloud_fast_s43.yaml --split train
python scripts/train.py --config configs/ft_rl_cloud_fast_s44.yaml --split train
```
- Ngắt (Ctrl+C / Stop studio) rồi chạy lại ĐÚNG lệnh đó → tự resume. KHÔNG thêm `--no-resume`.
- Checkpoint: FULL → `runs/ft_rl_s{42,43,44}/dqn/best.pt` · FAST → `runs/ft_rl_fast_s{42,43,44}/dqn/best.pt`
  (nhớ trỏ `--checkpoint` đúng thư mục ở bước 4).

## 4. BENCHMARK — số chính thức
4a. Baseline 3 phương pháp (full/SAHI/topk) trên **full val 548** và test 500:
```bash
python scripts/benchmark_detector.py --config configs/ft_cloud.yaml --split val  --limit 548
python scripts/benchmark_detector.py --config configs/ft_cloud.yaml --split test --limit 500
```
4b. **RL-HYBRID** cho từng seed (val 548 + test 500):
```bash
for S in 42 43 44; do
  python scripts/benchmark_hybrid.py --config configs/ft_cloud.yaml --split val  --limit 548 \
    --coarse-frac 0.6 --coarse-overlap 0.15 --fine-k 8 --fine-mode rl \
    --checkpoint runs/ft_rl_s$S/dqn/best.pt
  python scripts/benchmark_hybrid.py --config configs/ft_cloud.yaml --split test --limit 500 \
    --coarse-frac 0.6 --coarse-overlap 0.15 --fine-k 8 --fine-mode rl \
    --checkpoint runs/ft_rl_s$S/dqn/best.pt
done
```
4c. (khuyến nghị) Ablation heuristic để giữ luận điểm "RL > topk":
```bash
python scripts/benchmark_hybrid.py --config configs/ft_cloud.yaml --split test --limit 500 \
  --coarse-frac 0.6 --coarse-overlap 0.15 --fine-k 8 --fine-frac 0.25 --fine-mode topk
```

## 5. ⚠️ CANARY — kiểm run hỏng (BẮT BUỘC đọc)
Trên GPU đôi khi cả run bị "suy biến im lặng": kết quả method-có-cắt ≈ full-image.
**Quy tắc:** small_recall của SAHI/hybrid phải CAO HƠN yolo_full ≥ 3 điểm.
Nếu xấp xỉ bằng → run hỏng, chạy lại lệnh đó (không cần cache lại).
Số headline (bảng luận văn) nên chạy 2 lần — hai lần phải khớp nhau.

## 6. Tổng hợp báo cáo
- Ghi các dòng `[bench_det]` và `[hybrid]` vào bảng; tính **mean ± std theo 3 seed** cho RL-HYBRID.
- Kỳ vọng (từ kết quả local): RL-HYBRID vượt SAHI cả 4 tiêu chí
  (tham chiếu local test-150: mAP .376 vs .316 · recall .676 vs .603 · FP 181 vs 192 · crops 14.7 vs 28).

## Tham số công thức thắng (đừng đổi nếu không có lý do)
- Hybrid: `coarse 0.6/0.15 + fine 8 (RL) @ slice_imgsz 640, output_conf 0.10, merge_iou 0.5`
- Ngưỡng nhận crop RL: ≥1 detection mới & utility ≥ 0.2 (`min_slice_detections/min_slice_utility`)
- Reward: TP +3.0 · FP −0.75 · crop rỗng −1.2 · hard-hit +4.0 · reward_clip 100
- Chi tiết phương pháp: mở `RL_HYBRID_EXPLAINER.html` (trang minh họa tương tác).
