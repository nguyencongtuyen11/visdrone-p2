"""Train ONE-SHOT RL policy (bandit full-info, nhanh oneshot-rl).

Bo rollout di chuyen. Voi moi anh:
  - propose_regions (top-K dinh objectness, 1 lan)
  - moi vung: cham reward cho CA 4 action { DROP=0 | KEEP | ZOOM_1.5 | ZOOM_2 }
    bang CropOutcomeEvaluator (chay YOLO tren crop, doi chieu GT -> tp/fp, co cache).
  - luu (region_state, reward[4]) -> hoi quy Q-net (MLP) MSE(Q, reward).
DROP reward = 0 -> agent tu hoc bo vung vo ich (crop khong TP se cho reward am < 0).

Chay (Lightning, sau khi da co cache detection — xem RUN_TRAIN_CLOUD.md buoc 2):
  python scripts/train_oneshot.py --config configs/ft_cloud.yaml --split train --limit 1500 --device cuda
Ket qua: runs/oneshot/policy.pt
"""
import sys, time, argparse
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").addFilter(lambda r: "deprecated" not in r.getMessage())
from pathlib import Path
import numpy as np, torch

from rl_sahi.common.config import load_default_config
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import get_initial_detection
from rl_sahi.rl.crop_outcome import CropOutcomeEvaluator
from rl_sahi.rl.oneshot import (propose_regions, action_to_roi, region_local_state, objectness_grid,
                                KEEP, ZOOM_1_5, ZOOM_2, NUM_ACTIONS, ACTION_NAMES)
from rl_sahi.rl.oneshot_policy import OneShotPolicy, save_policy

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--split", default="train")
ap.add_argument("--limit", type=int, default=1500, help="so anh dung xay reward cache")
ap.add_argument("--k", type=int, default=8, help="so ung vien / anh")
ap.add_argument("--epochs", type=int, default=200, help="epoch hoi quy MLP (nhanh)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", type=Path, default=ROOT / "runs" / "oneshot" / "policy.pt")
args = ap.parse_args()

dev = args.device
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"; WEIGHTS = ROOT / "best_visdrone.pt"
cfg = load_default_config(args.config, ROOT)
tc = tuple(cfg.section("infer")["target_classes"]); cm = ClassMapping.from_config(cfg.section("classes"))
icfg = InferenceConfig(full_imgsz=640, slice_imgsz=640, full_conf=0.01, output_conf=0.10, iou=0.7, merge_iou=0.5,
    max_det=3000, device=dev, feature_layers=(16,), target_classes=tc, class_mapping=cm, min_slice_detections=1,
    min_slice_utility=0.2, duplicate_iou=0.5, max_slice_attempts=14, require_stop_for_acceptance=True)
model = load_yolo(str(WEIGHTS), device=dev)
ev = CropOutcomeEvaluator(model, IR, LR, CR, args.split, icfg, weights=WEIGHTS,
                          tp_reward=3.0, fp_penalty=0.75, detection_reward=0.75, empty_penalty=1.2, no_gain_penalty=1.2)

images = iter_images(IR, split=args.split, limit=args.limit)
if not images:
    sys.exit(f"[oneshot-train] khong thay anh o {IR}/{args.split}")
print(f"[oneshot-train] xay reward cache tu {len(images)} anh, k={args.k}/anh ...")

X, Y = [], []
t0 = time.perf_counter()
for i, img in enumerate(images):
    if i % 100 == 0 and i:
        el = time.perf_counter() - t0
        print(f"  {i}/{len(images)} | {el:.0f}s | eta~{el/i*(len(images)-i):.0f}s | mau={len(X)}", flush=True)
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=640, full_conf=0.01,
        full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16, spatial_feature_channels=4,
        cache_root=CR, split=args.split, use_cache=True)
    grid = objectness_grid(det)
    fb, fs, fc = ev.full_predictions(det)
    for region in propose_regions(det, k=args.k):
        state = region_local_state(det, region, grid)
        rewards = [0.0] * NUM_ACTIONS  # DROP = 0
        for a in (KEEP, ZOOM_1_5, ZOOM_2):
            roi = action_to_roi(region, a, det.image_shape)
            if roi is None:
                continue
            out = ev.evaluate(img, det, roi, fb, fs, fc, [], [], [], 0)
            rewards[a] = float(out.reward)
        X.append(state); Y.append(rewards)

X = np.asarray(X, dtype=np.float32); Y = np.asarray(Y, dtype=np.float32)
print(f"[oneshot-train] xong reward cache: {X.shape[0]} mau, state_dim={X.shape[1]}")
# thong ke: action tot nhat trung binh
best = Y.argmax(axis=1)
dist = {ACTION_NAMES[a]: int((best == a).sum()) for a in range(NUM_ACTIONS)}
print(f"[oneshot-train] action TOT NHAT theo reward (oracle): {dist}")

# --- train MLP hoi quy Q -> reward ---
dt = resolve_torch_device(dev)
policy = OneShotPolicy(state_dim=X.shape[1]).to(dt)
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
xt = torch.from_numpy(X).to(dt); yt = torch.from_numpy(Y).to(dt)
n = xt.shape[0]; bs = min(2048, n)
for ep in range(args.epochs):
    perm = torch.randperm(n, device=dt)
    tot = 0.0
    for j in range(0, n, bs):
        idx = perm[j:j + bs]
        q = policy(xt[idx]); loss = torch.nn.functional.mse_loss(q, yt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    if ep % 50 == 0 or ep == args.epochs - 1:
        with torch.no_grad():
            pred = policy(xt).argmax(1)
            acc = (pred == torch.from_numpy(best).to(dt)).float().mean().item()
        print(f"  ep {ep}: mse={tot/n:.4f} | khop action-oracle={acc*100:.1f}%")

args.out.parent.mkdir(parents=True, exist_ok=True)
save_policy(policy, args.out, meta={"k": args.k, "limit": args.limit, "action_dist_oracle": dist})
print(f"[oneshot-train] LUU {args.out}")
print("[oneshot-train] Buoc ke: benchmark one-shot RL vs static baseline (dang lam scripts/benchmark_oneshot_rl.py)")
