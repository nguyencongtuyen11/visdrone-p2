"""Train ONE-SHOT RL policy — REINFORCE (policy gradient), nhanh oneshot-rl.

RL DUNG NGHIA (giong Uzkent 2020 / AdaZoom cho one-shot patch selection):
  - policy pi(a|s) = softmax(logits) tren 4 action { DROP | KEEP | ZOOM_1.5 | ZOOM_2 }.
  - moi vung = 1 buoc: LAY MAU action ~ pi (KHAM PHA), chi nhan reward cua action DA CHON,
    cap nhat bang policy gradient: loss = -log pi(a|s) * (r - baseline) - beta*entropy.
  - reward = crop-outcome (chay YOLO tren crop cua action do, doi chieu GT -> TP/FP).
DROP reward = 0 -> agent tu hoc bo vung vo ich. Reward table cache 1 lan (hieu qua),
NHUNG thuat toan chi dung reward cua action DA LAY MAU moi buoc = RL that.

Chay (Lightning, da co cache detection):
  python scripts/train_oneshot.py --config configs/ft_cloud.yaml --split train --limit 2000 --k 8 --device cuda
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
ap.add_argument("--limit", type=int, default=2000, help="so anh xay reward cache")
ap.add_argument("--k", type=int, default=8, help="so ung vien / anh")
ap.add_argument("--steps", type=int, default=4000, help="so buoc cap nhat policy-gradient")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--entropy", type=float, default=0.02, help="he so thuong entropy (kham pha)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", type=Path, default=ROOT / "runs" / "oneshot" / "policy.pt")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
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

# ---------- PHASE A: xay reward table (cache YOLO 1 lan) ----------
images = iter_images(IR, split=args.split, limit=args.limit)
if not images:
    sys.exit(f"[oneshot-rl] khong thay anh o {IR}/{args.split}")
print(f"[oneshot-rl] PHASE A — reward table tu {len(images)} anh, k={args.k}/anh ...")
X, R = [], []
t0 = time.perf_counter()
for i, img in enumerate(images):
    if i % 100 == 0 and i:
        el = time.perf_counter() - t0
        print(f"  {i}/{len(images)} | {el:.0f}s | eta~{el/i*(len(images)-i):.0f}s | vung={len(X)}", flush=True)
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=640, full_conf=0.01,
        full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16, spatial_feature_channels=4,
        cache_root=CR, split=args.split, use_cache=True)
    grid = objectness_grid(det)
    fb, fs, fc = ev.full_predictions(det)
    for region in propose_regions(det, k=args.k):
        rew = [0.0] * NUM_ACTIONS  # DROP = 0
        for a in (KEEP, ZOOM_1_5, ZOOM_2):
            roi = action_to_roi(region, a, det.image_shape)
            if roi is not None:
                rew[a] = float(ev.evaluate(img, det, roi, fb, fs, fc, [], [], [], 0).reward)
        X.append(region_local_state(det, region, grid)); R.append(rew)

X = np.asarray(X, np.float32); R = np.asarray(R, np.float32)
oracle = R.argmax(1)
odist = {ACTION_NAMES[a]: int((oracle == a).sum()) for a in range(NUM_ACTIONS)}
print(f"[oneshot-rl] {X.shape[0]} vung | action TOT NHAT theo reward (oracle, chi de tham chieu): {odist}")

# ---------- PHASE B: REINFORCE (policy gradient RL) ----------
dt = resolve_torch_device(dev)
policy = OneShotPolicy(state_dim=X.shape[1]).to(dt)
opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
xt = torch.from_numpy(X).to(dt); rt = torch.from_numpy(R).to(dt)
n = xt.shape[0]; baseline = 0.0
print(f"[oneshot-rl] PHASE B — REINFORCE {args.steps} buoc (lay mau action + policy gradient) ...")
for step in range(args.steps):
    idx = torch.randint(0, n, (min(args.batch, n),), device=dt)
    s = xt[idx]
    logits = policy(s)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()                                  # KHAM PHA: lay mau tu policy
    r = rt[idx].gather(1, a.unsqueeze(1)).squeeze(1)   # CHI reward cua action DA CHON
    adv = r - baseline
    loss = -(dist.log_prob(a) * adv.detach()).mean() - args.entropy * dist.entropy().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    baseline = 0.98 * baseline + 0.02 * float(r.mean().item())
    if step % 500 == 0 or step == args.steps - 1:
        with torch.no_grad():
            greedy = policy(xt).argmax(1)
            gdist = {ACTION_NAMES[k2]: int((greedy == k2).sum()) for k2 in range(NUM_ACTIONS)}
            realized = rt.gather(1, greedy.unsqueeze(1)).mean().item()
        print(f"  step {step}: baseline={baseline:.3f} | policy greedy dist={gdist} | reward TB(greedy)={realized:.3f}")

args.out.parent.mkdir(parents=True, exist_ok=True)
save_policy(policy, args.out, meta={"k": args.k, "limit": args.limit, "algo": "REINFORCE",
                                    "oracle_dist": odist, "steps": args.steps})
print(f"[oneshot-rl] LUU {args.out}")
print("[oneshot-rl] So sanh 'policy greedy dist' (RL hoc) vs 'oracle' — RL co bat kip oracle khong.")
