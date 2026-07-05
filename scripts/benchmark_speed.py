"""Do LATENCY + accuracy THAT tren GPU hien tai (T4/A10/...) cho 3 phuong phap:
   full@base (1 luot) | +SAHI (~grid) | +RL-HYBRID (coarse + RL fine-slice).
Tra loi: SAHI/RL-SAHI thuc su nhanh/cham the nao tren GPU manh (khong phai GTX1650 4GB yeu)?

CHAY tren Lightning — T4 phai RANH (tam dung P2 bang Ctrl+C truoc):
  python scripts/benchmark_speed.py --limit 100 --base 640
Xem moc 1-luot @1280:  python scripts/benchmark_speed.py --limit 100 --base 1280
"""
import sys, time, argparse
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np, torch
from rl_sahi.common.config import load_default_config
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import (_fixed_grid_rois, _full_predictions, _merge_predictions,
    _predict_fixed_sahi, _evaluate_method, _read_gt, _image_shape, _small_area_threshold, BenchmarkConfig)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.inference.pipeline import (_filter_classes, _new_detection_gain, _new_detection_utility,
    _attempt_overlap, get_initial_detection)
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft.yaml")
ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "ft_rl" / "dqn" / "best.pt")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--base", type=int, default=640, help="imgsz anh goc (1 luot)")
ap.add_argument("--slice", type=int, default=640, help="imgsz moi crop")
ap.add_argument("--max-fine", type=int, default=8, help="so lat min RL toi da")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

BASE, SLI, dev = args.base, args.slice, args.device
# ep MOI path ve relative theo repo (tranh path Windows trong ft.yaml)
IR = ROOT / "data" / "raw" / "images"
LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"
WEIGHTS = ROOT / "best_visdrone.pt"

cfg = load_default_config(args.config, ROOT)
tc = tuple(cfg.section("infer")["target_classes"])
cm = ClassMapping.from_config(cfg.section("classes"))
icfg = InferenceConfig(full_imgsz=BASE, slice_imgsz=SLI, full_conf=0.01, output_conf=0.10, iou=0.7,
    merge_iou=0.5, max_det=3000, device=dev, feature_layers=(16,), target_classes=tc, class_mapping=cm,
    min_slice_detections=1, min_slice_utility=0.2, duplicate_iou=0.5, max_slice_attempts=14,
    require_stop_for_acceptance=True)
bcfg = BenchmarkConfig(fixed_slice_fraction=0.35, fixed_overlap=0.2, target_classes=tc, class_mapping=cm)
model = load_yolo(str(WEIGHTS), device=dev)
dt = resolve_torch_device(dev)
policy, ck = load_policy(args.checkpoint, dt)
env_cfg = ck["env_cfg_obj"]; env_cfg.max_slices = args.max_fine
state_cfg = ck.get("state_cfg_obj", StateConfig())

def sync():
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
def nw(): sync(); return time.perf_counter()

images = iter_images(IR, split=args.split, limit=args.limit)
if not images:
    sys.exit(f"[speed] khong thay anh o {IR}/{args.split} — chay download_visdrone.py truoc.")
small_thr = _small_area_threshold(images, IR, LR, tc, 40.0, cm)
gt = {}
for img in images:
    gb, gc = _read_gt(img, IR, LR, tc, cm); gt[img.stem] = (gb, gc, _image_shape(img))

pB, pS, pH = {}, {}, {}
tB = tS = tH = 0.0; scrops = 0; hcrops = 0
# warmup
get_initial_detection(model=model, weights=str(WEIGHTS), image_path=images[0], weights_imgsz=BASE,
    full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
    spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False); sync()

for img in images:
    iid = img.stem
    t = nw()
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=BASE,
        full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    base_t = nw() - t
    fb, fs, fc = _full_predictions(det, icfg)
    pB[iid] = (fb, fs, fc); tB += base_t
    # +SAHI
    t = nw(); sahi_out = _predict_fixed_sahi(model, img, det, icfg, bcfg); tS += base_t + (nw() - t)
    pS[iid] = (sahi_out[0], sahi_out[1], sahi_out[2])
    scrops += 1 + len(_fixed_grid_rois(det.image_shape, bcfg.fixed_slice_fraction, bcfg.fixed_overlap))
    # +RL-HYBRID (coarse grid + RL fine-slices)
    t = nw()
    acc, att, xb, xs, xc = [], [], [], [], []
    for _ in range(int(icfg.max_slice_attempts)):
        if len(acc) >= env_cfg.max_slices: break
        hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
        ov = np.stack(acc).astype("f4") if acc else np.zeros((0, 4), "f4")
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                       overlap_rois=ov, target_classes=tc, class_mapping=cm)
        roi, _a, info = rollout_one_slice(policy, env, dt)
        rej = info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap") or \
              (icfg.require_stop_for_acceptance and (info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi")))
        if rej:
            att.append(roi)
            if _attempt_overlap(roi, att) >= 0.95: break
            continue
        b, s, c = run_yolo_on_crop(model, img, roi, imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev)
        att.append(roi); c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        g = _new_detection_gain(fb, fs, fc, xb, xs, xc, b, s, c, det.image_shape, 0.5, 0.5)
        u = _new_detection_utility(fb, fs, fc, xb, xs, xc, b, s, c, det.image_shape, 0.5, 0.5)
        if g < 1 or u < 0.2: continue
        acc.append(roi); xb.append(b); xs.append(s); xc.append(c)
    coarse = _fixed_grid_rois(det.image_shape, 0.6, 0.15)
    pb, ps, pc = [fb, *xb], [fs, *xs], [fc, *xc]
    for roi in coarse:
        b, s, c = run_yolo_on_crop(model, img, roi, imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev)
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    hb, hs, hc = _merge_predictions(det.image_shape, 0.5, pb, ps, pc)
    tH += base_t + (nw() - t)
    pH[iid] = (hb, hs, hc); hcrops += 1 + len(coarse) + len(acc)

def ev(p): return _evaluate_method(p, gt, tc, 0.5, small_thr)
mB, mS, mH = ev(pB), ev(pS), ev(pH)
N = len(images)
gpu = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else "CPU"
print(f"\n===== LATENCY + ACCURACY (test-{N}, {gpu}) base@{BASE} crop@{SLI} =====")
print(f"  {'method':14s}{'mAP50':>8s}{'s_recall':>9s}{'FP/img':>8s}{'crops':>7s}{'ms/anh':>9s}")
for name, m, tt, cN in [(f"full@{BASE}", mB, tB, N), ("+SAHI", mS, tS, scrops), ("+RL-HYBRID", mH, tH, hcrops)]:
    print(f"  {name:14s}{m['mAP50']:8.4f}{m['small_recall']:9.4f}{m['fp_per_image']:8.1f}{cN / N:7.1f}{tt / N * 1000:9.0f}")
print(f"\n  => RL-HYBRID vs SAHI:  recall {mH['small_recall']:.3f} vs {mS['small_recall']:.3f}  |  "
      f"{tH / N * 1000:.0f}ms vs {tS / N * 1000:.0f}ms  |  {hcrops / N:.0f} vs {scrops / N:.0f} crop")
print(f"  => full@{BASE} (1 luot) = {tB / N * 1000:.0f} ms  <-- moc toc do 1 luot")
