# RL-SAHI

Adaptive image slicing for small object detection with a Dueling Double DQN policy and YOLO11.

This project explores a reinforcement-learning alternative to fixed-grid SAHI-style slicing. Instead of running YOLO on every fixed tile, the agent observes full-image detections and cached feature/state maps, then selects regions of interest that are likely to recover missed small objects.

## Highlights

- Built an adaptive slicing pipeline with a Dueling Double DQN agent for ROI selection.
- Integrated YOLO11 full-image detection, crop-level inference, class-aware NMS, prediction export, and visualization.
- Implemented batched DQN training with parallel environments, Double DQN targets, prioritized replay, n-step returns, curriculum scheduling, and cached YOLO state features.
- Benchmarked three modes: full-image YOLO, fixed-grid SAHI-style slicing, and RL-SAHI.

## Benchmark

Latest local benchmark:

```powershell
conda run -n doan python scripts\benchmark.py --split test --limit 100 --out-dir runs\benchmark\cv_test_100
```

Results on 100 test images:

| Method | mAP50 | Small Recall | FP / Image | Crops / Image | Latency / Image |
|---|---:|---:|---:|---:|---:|
| YOLO full image | 0.1648 | 0.0047 | 3.41 | 0.00 | 0.2 ms |
| Fixed-grid SAHI-style | 0.2570 | 0.2538 | 26.14 | 28.00 | 1722.9 ms |
| RL-SAHI | 0.2267 | 0.1928 | 16.14 | 5.35 | 1811.2 ms |

Compared with full-image YOLO, RL-SAHI improved mAP50 from `0.1648` to `0.2267`. Compared with fixed-grid slicing, RL-SAHI used `80.9%` fewer crops and produced `38.3%` fewer false positives, while fixed-grid still had higher raw mAP50 and small-object recall in this run.

Benchmark outputs are stored in:

- `runs/benchmark/cv_test_100/benchmark.csv`
- `runs/benchmark/cv_test_100/benchmark.json`

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLO11
- OpenCV
- NumPy
- CUDA-capable PyTorch
- PyYAML

## Repository Layout

```text
configs/                 YAML configuration for paths, detection, RL, and inference
scripts/
  detect.py              Cache full-image YOLO detections and features
  hard_region.py         Cache hard small-object regions for RL reward/training
  train.py               Train the Dueling Double DQN slicing policy
  infer.py               Run adaptive slicing inference and save visualizations
  benchmark.py           Compare YOLO full image, fixed-grid slicing, and RL-SAHI
src/rl_sahi/
  common/                Boxes, NMS, config, data, device helpers
  detection/             YOLO detection and feature cache utilities
  hard_region/           Hard-region cache generation
  inference/             Adaptive slicing inference pipeline
  rl/                    Environment, state builder, replay, DQN, trainer
  eval/                  Benchmark and evaluation utilities
tests/                   Focused unit tests for RL and detection utilities
```

## Setup

Create an environment and install dependencies:

```powershell
conda create -n rl-sahi python=3.11
conda activate rl-sahi
pip install -r requirements.txt
```

The default config targets CUDA, which is the expected backend for Kaggle T4:

```yaml
infer:
  device: "cuda"
train:
  device: "cuda"
```

For a local non-CUDA machine, set the device values in `configs/detection.yaml`, `configs/inference.yaml`, and `configs/rl.yaml` to `""` for auto selection or to a valid PyTorch device such as `cpu`.

## Data Layout

The code expects YOLO-format images and labels:

```text
data/raw/images/
  train/*.jpg
  val/*.jpg
  test/*.jpg
data/raw/labels/
  train/*.txt
  val/*.txt
  test/*.txt
```

Each label file should use normalized YOLO boxes:

```text
class_id center_x center_y width height
```

Project paths are configured in `configs/paths.yaml`.

## Workflow

1. Cache full-image YOLO detections and feature states:

```powershell
python scripts\detect.py --split train
python scripts\detect.py --split val
python scripts\detect.py --split test
```

2. Cache hard small-object regions:

```powershell
python scripts\hard_region.py --split train
python scripts\hard_region.py --split val
```

3. Train the RL slicing policy with the batched trainer:

```powershell
python scripts\train.py --split train
```

The default checkpoint path is:

```text
runs/dqn/best.pt
```

Training also writes a resume checkpoint to:

```text
runs/dqn/resume.pt
```

If training is interrupted, rerun the same train command to continue from the last saved resume checkpoint. To start a fresh run instead, pass:

```powershell
python scripts\train.py --split train --no-resume
```

4. Run inference on a split:

```powershell
python scripts\infer.py --split test --limit 20
```

Or run inference on a single image:

```powershell
python scripts\infer.py --image data\raw\images\test\example.jpg
```

Outputs are written under:

```text
runs/infer/
```

5. Run benchmark:

```powershell
python scripts\benchmark.py --split test --limit 100 --out-dir runs\benchmark\cv_test_100
```

## Configuration

Main configuration entry point:

```text
configs/default.yaml
```

It includes:

- `configs/paths.yaml` for dataset, cache, weights, checkpoints, and output paths.
- `configs/detection.yaml` for YOLO cache generation and hard-region mining.
- `configs/rl.yaml` for environment, state, and DQN training settings.
- `configs/inference.yaml` for adaptive slicing inference and benchmark settings.

## Notes

- The project uses a SAHI-style fixed-grid baseline for comparison; it does not require the external SAHI package.
- `data/cache/`, `data/raw/images/`, `runs/`, and generated artifacts should generally stay out of Git.
- The included benchmark should be reported honestly: RL-SAHI improves over full-image YOLO and is much more crop-efficient than fixed-grid slicing, but fixed-grid currently has higher raw mAP50 and small-object recall on the 100-image test run.
