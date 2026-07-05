# Chạy RL-SAHI trên Lightning.ai (train chia đợt, resume tự động)

Lightning Studio có **ổ đĩa bền**: Stop studio để nhả GPU (ngừng tính tiền) nhưng mọi file
(code, dataset, cache, `resume.pt`, `best.pt`) vẫn còn. Start lại → chạy tiếp được ngay.

## GPU nên chọn
- **L4** (đáng tiền nhất) hoặc **T4** (rẻ nhất). KHÔNG cần A100/H100 — workload này nghẽn ở
  latency/CPU, không phải FLOPS.

## 1. Tạo Studio + đưa code lên
- Tạo Studio (AI development / Python), gắn GPU L4 hoặc T4.
- Upload `rl_sahi_lightning.zip` vào Studio, rồi giải nén:
```bash
unzip rl_sahi_lightning.zip -d rl_sahi
cd rl_sahi
```

## 2. Cài thư viện
```bash
pip install -r requirements.txt
```

## 3. Tải dataset (CHỈ 1 LẦN — sau đó nằm trên đĩa bền)
```bash
python scripts/download_visdrone.py
```

## 4. Cache (CHỈ 1 LẦN)
```bash
python scripts/detect.py --split train
python scripts/detect.py --split val
python scripts/detect.py --split test
python scripts/hard_region.py --split train
python scripts/hard_region.py --split val
```

## 5. TRAIN — chia đợt tới đích 20.000
**Luôn dùng đúng lệnh này mọi đợt** (đích cố định 20.000):
```bash
python scripts/train.py --split train --episodes 20000
```
- Chạy tới ~5.000 ep thì bấm **Ctrl+C** để dừng đợt (hoặc Stop studio).
- Đợt sau: Start studio → `cd rl_sahi` → chạy **đúng lệnh trên** → tự resume.
- Đầu mỗi đợt phải thấy: `[batched_train] resumed ... (completed=N ...)`.
- KHÔNG thêm `--no-resume` (sẽ train lại từ đầu). KHÔNG đổi số `--episodes`.

Thời gian / ETA in trong log (`elapsed`, `eta~`, `s/ep`); tổng thời gian ghi vào
`runs/dqn/train_time.txt`.

## 6. INFER / BENCHMARK (sau khi có best.pt)
```bash
python scripts/infer.py --split test --limit 20
python scripts/benchmark.py --split test --limit 100 --out-dir runs/benchmark/test100
```

## Mẹo
- Theo dõi `benchmark_score` (eval mỗi 1.000 ep). 2-3 mốc liên tiếp không tăng → có thể dừng
  sớm, `best.pt` luôn giữ checkpoint tốt nhất.
- Khi Stop studio để nghỉ: nhớ Stop để khỏi tốn tiền GPU; file vẫn còn cho lần sau.
