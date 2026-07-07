# Train lại RL trên cloud (reward ĐÃ FIX) — Lightning T4

Mục tiêu: train lại agent với **reward đã sửa** (audit MAJOR) → kỳ vọng recall cao hơn checkpoint cũ (.691).
Reward fix nằm trong `src/rl_sahi/rl/trainer.py` (dùng chung train + batched_trainer) nên **tự động áp** — chỉ cần chạy lại.

## Chuỗi lệnh (Lightning, GPU T4 phải bật)

```bash
git pull                                    # -> commit a52c6a3+ (co reward + boundary fix)
python scripts/check_status.py              # xac nhan T4 / data / checkpoint

# 1. Data (bo qua neu da co data/raw)
python scripts/download_visdrone.py

# 2. Cache detection (backbone features cho train split) — vai phut
python scripts/detect.py --config configs/ft_rl_cloud_s42.yaml --split train

# 3. Cache hard-region (vung kho = tin hieu reward)
python scripts/hard_region.py --config configs/ft_rl_cloud_s42.yaml --split train

# 4. TRAIN RL 6000 ep (crop-outcome, reward da fix). CHAY LAU (~qua dem) -> dung tmux/nohup:
nohup python scripts/train.py --config configs/ft_rl_cloud_s42.yaml --split train --device 0 > train_s42.log 2>&1 &
tail -f train_s42.log                       # xem tien do; Ctrl-C de thoat tail (train van chay)
```

Ket qua: `runs/ft_rl_s42/dqn/best.pt`

## Sau khi train xong — đo recall THẬT
```bash
python scripts/benchmark_oneshot.py --limit 100 --checkpoint runs/ft_rl_s42/dqn/best.pt --with-seq
```
So `small_recall` cua checkpoint MOI vs cu (.691). Reward fix nham lam agent bat nhieu TP that hon
(khong con bi phat khi tang recall) → ky vong recall len.

## Ghi chú
- Config `ft_rl_cloud_s42.yaml`: 6000 ep · seed 42 · crop-outcome · max_steps 10 · max_slices 14
  (giong checkpoint cu de so cong bang, chi khac reward da fix + train dai hon).
- Muon chon best.pt theo RECALL THAT (thay vi proxy hard-region): sua `eval_benchmark_images: 60`
  trong config + cache them val (`detect.py`/`hard_region.py --split val`). Cham hon nhung best.pt sat muc tieu hon.
- Seed 43/44 (`ft_rl_cloud_s43/s44.yaml`) de chay lap lai kiem tra on dinh neu con GPU.
