"""Do mAP theo CHUAN COCO/ultralytics cho tat ca method — de so voi paper/ban ban (0.378).

Khac benchmark cu: KHONG cat conf 0.10. Chay o conf ~0.001 -> duong PR DAY DU,
AP 101-point, ca mAP50 va mAP50-95, tat ca 10 lop VisDrone. Co dong CALIBRATION:
  full@640 (COCO mAP cua script nay) PHAI ~ ultralytics `yolo val` -> xac nhan thuoc do dung.

Chay tren Lightning (co data + 2 checkpoint):
  python scripts/eval_coco_map.py --split test --map-conf 0.001 --device cuda
Doi chung ultralytics cung split:
  yolo val model=best_visdrone.pt data=VisDrone.yaml split=test imgsz=640 conf=0.001 iou=0.7
"""
import sys, argparse, time
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
from rl_sahi.common.data import iter_images, image_to_label_path, read_yolo_labels
from rl_sahi.common.boxes import iou_matrix
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.pipeline import _filter_classes, _attempt_overlap, get_initial_detection
from rl_sahi.eval.benchmark import _fixed_grid_rois, _merge_predictions, _image_shape
from rl_sahi.rl.oneshot import propose_regions, action_to_roi, region_local_state, objectness_grid, KEEP
from rl_sahi.rl.oneshot_policy import load_oneshot_policy

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "ft_rl" / "dqn" / "best.pt", help="policy RL di chuyen")
ap.add_argument("--policy", type=Path, default=ROOT / "runs" / "oneshot" / "policy.pt", help="policy one-shot")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=0, help="0 = het anh")
ap.add_argument("--map-conf", type=float, default=0.001, help="conf THAP de duong PR day du (nhu ultralytics)")
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--max-fine", type=int, default=8)
ap.add_argument("--max-attempts", type=int, default=14)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--max-det", type=int, default=300, help="ultralytics val mac dinh 300")
ap.add_argument("--weights", type=Path, default=ROOT / "best_visdrone.pt", help="doi weight moi sau khi fine-tune crop")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

BASE, SLI, dev, MC = args.base, args.slice, args.device, args.map_conf
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"; WEIGHTS = args.weights
CLASSES = list(range(10))
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
IOUV = np.linspace(0.5, 0.95, 10)

cfg = load_default_config(args.config, ROOT)
cm = ClassMapping.from_config(cfg.section("classes"))
tc = tuple(CLASSES)
model = load_yolo(str(WEIGHTS), device=dev)
dt = resolve_torch_device(dev)

# RL di chuyen (tuy chon)
policy_mv = env_cfg = state_cfg = None
if args.checkpoint.exists():
    from rl_sahi.rl.checkpoint import load_policy
    from rl_sahi.rl.slice_env import SliceEnv
    from rl_sahi.rl.state_config import StateConfig
    from rl_sahi.inference.rollout import rollout_one_slice
    policy_mv, ckm = load_policy(args.checkpoint, dt)
    env_cfg = ckm["env_cfg_obj"]; env_cfg.max_slices = args.max_fine
    state_cfg = ckm.get("state_cfg_obj", StateConfig())
# one-shot (tuy chon)
policy_os = None
if args.policy.exists():
    policy_os, _ = load_oneshot_policy(args.policy, dt)

def crop_parts(img, rois):
    outs = []
    for i in range(0, len(rois), args.chunk):
        part = rois[i:i + args.chunk]
        outs.extend(run_yolo_on_crops(model, [img] * len(part), part,
                    imgsz=SLI, conf=MC, iou=0.7, max_det=3000, device=dev))
    pb, ps, pc = [], [], []
    for (b, s, c) in outs:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    return pb, ps, pc

def merge_full(shape, full, parts):
    fb, fs, fc = full; pb, ps, pc = parts
    return _merge_predictions(shape, 0.5, [fb, *pb], [fs, *ps], [fc, *pc])

def select_rois_moving(det):
    kept, att = [], []
    for _ in range(int(args.max_attempts)):
        if len(kept) >= env_cfg.max_slices: break
        hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
        ov = np.stack(kept).astype("f4") if kept else np.zeros((0, 4), "f4")
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                       overlap_rois=ov, target_classes=tc, class_mapping=cm)
        roi, _a, info = rollout_one_slice(policy_mv, env, dt)
        att.append(roi)
        rejected = bool(info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap") or
                        (info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi")))
        if rejected:
            if _attempt_overlap(roi, att[:-1]) >= 0.95: break
            continue
        kept.append(roi)
    return kept

# ---------------- COCO mAP (101-point, IoU 0.5:0.95), tu chua, khong phu thuoc ultralytics ----------------
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # numpy 2.x doi ten trapz -> trapezoid

def _ap101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    return float(_TRAPZ(np.interp(x, mrec, mpre), x))

def eval_map(preds, gts):
    """preds/gts: dict iid -> (boxes, scores, cls) / (boxes, cls). Tra mAP50, mAP50-95, ap50 moi lop."""
    ap = np.full((len(CLASSES), len(IOUV)), np.nan)
    for ci, c in enumerate(CLASSES):
        n_gt = 0
        per_img = {}  # iid -> (scores[np], iou[preds,gts], n_gt_img)
        entries = []  # (score, iid, local_pred_idx)
        for iid, (gb, gc) in gts.items():
            gm = gc.astype(np.int64) == c
            gbi = gb[gm]; n_gt += int(gm.sum())
            pb, ps, pc = preds[iid]
            pm = pc.astype(np.int64) == c
            pbi = pb[pm]; psi = ps[pm]
            if len(pbi) == 0 and len(gbi) == 0:
                continue
            iou = iou_matrix(pbi, gbi) if (len(pbi) and len(gbi)) else np.zeros((len(pbi), len(gbi)), np.float32)
            per_img[iid] = (psi, iou, len(gbi))
            for k, s in enumerate(psi):
                entries.append((float(s), iid, k))
        if n_gt == 0:
            continue
        if not entries:
            ap[ci, :] = 0.0; continue
        entries.sort(key=lambda e: -e[0])
        for ti, thr in enumerate(IOUV):
            matched = {iid: np.zeros(v[2], bool) for iid, v in per_img.items()}
            tp = np.zeros(len(entries)); fp = np.zeros(len(entries))
            for rank, (s, iid, k) in enumerate(entries):
                _sc, iou, ng = per_img[iid]
                if ng == 0:
                    fp[rank] = 1.0; continue
                row = iou[k]
                j = int(row.argmax())
                if float(row[j]) >= thr and not matched[iid][j]:
                    tp[rank] = 1.0; matched[iid][j] = True
                else:
                    fp[rank] = 1.0
            tpc = np.cumsum(tp); fpc = np.cumsum(fp)
            rec = tpc / n_gt
            prec = tpc / np.maximum(tpc + fpc, 1e-16)
            ap[ci, ti] = _ap101(rec, prec)
    map50 = float(np.nanmean(ap[:, 0]))
    map5095 = float(np.nanmean(ap))
    return map50, map5095, ap[:, 0]

# ---------------- chay ----------------
images = iter_images(IR, split=args.split, limit=(args.limit or None))
if not images: sys.exit(f"[coco] khong thay anh o {IR}/{args.split}")
print(f"[coco] {len(images)} anh, split={args.split}, map-conf={MC}, max_det={args.max_det}")
print(f"[coco] RL di chuyen: {'CO' if policy_mv else 'THIEU'} | one-shot: {'CO' if policy_os else 'THIEU'}")

gts = {}
P = {"full": {}, "sahi": {}, "coarse": {}, "rl_move": {}, "oneshot": {}}
t0 = time.perf_counter()
for idx, img in enumerate(images):
    if idx % 50 == 0 and idx:
        el = time.perf_counter() - t0
        print(f"  {idx}/{len(images)} | {el:.0f}s | eta~{el/idx*(len(images)-idx):.0f}s", flush=True)
    iid = img.stem
    shape = _image_shape(img)
    # GT
    cls, boxes = read_yolo_labels(image_to_label_path(img, IR, LR), shape)
    cls = cm.map_label_classes(cls)
    gts[iid] = (np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(cls, np.float32).reshape(-1))
    # base detection (full, conf thap) + feature cho RL/one-shot
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=BASE,
        full_conf=MC, full_iou=0.7, max_det=args.max_det, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    fb = np.asarray(det.boxes, np.float32).reshape(-1, 4)
    fs = np.asarray(det.scores, np.float32).reshape(-1)
    fc = cm.map_model_classes(det.classes)
    fb, fs, fc = _filter_classes(fb, fs, fc, tc)
    full = (fb, fs, fc)
    P["full"][iid] = full
    # SAHI (luoi 0.35/0.2)
    P["sahi"][iid] = merge_full(shape, full, crop_parts(img, _fixed_grid_rois(shape, 0.35, 0.2)))
    # ABLATION: luoi tho 0.6 KHONG co lat RL — crop dung lai cho rl_move
    coarse_parts = crop_parts(img, _fixed_grid_rois(shape, 0.6, 0.15))
    P["coarse"][iid] = merge_full(shape, full, coarse_parts)
    # RL di chuyen (fine RL + luoi tho 0.6, dung lai coarse_parts)
    if policy_mv is not None:
        fp = crop_parts(img, select_rois_moving(det))
        P["rl_move"][iid] = _merge_predictions(shape, 0.5,
            [fb, *fp[0], *coarse_parts[0]], [fs, *fp[1], *coarse_parts[1]], [fc, *fp[2], *coarse_parts[2]])
    # one-shot
    if policy_os is not None:
        grid = objectness_grid(det); regions = propose_regions(det, k=args.k)
        rois = []
        if regions:
            states = np.stack([region_local_state(det, r, grid) for r in regions]).astype(np.float32)
            with torch.no_grad():
                acts = policy_os(torch.from_numpy(states).to(dt)).argmax(1).cpu().numpy()
            for r, a in zip(regions, acts):
                roi = action_to_roi(r, int(a), det.image_shape)
                if roi is not None: rois.append(roi)
        P["oneshot"][iid] = merge_full(shape, full, crop_parts(img, rois)) if rois else full

print(f"\n===== COCO mAP (split={args.split}, {len(images)} anh, conf={MC}) =====")
print(f"  {'method':16s}{'mAP50':>9s}{'mAP50-95':>10s}")
order = [("full", "YOLO full@640"), ("sahi", "SAHI"), ("coarse", "luoi 0.6 (khong RL)"),
         ("rl_move", "RL di chuyen"), ("oneshot", "one-shot")]
map50_full = None
for key, name in order:
    if not P[key]: continue
    m50, m5095, per = eval_map(P[key], gts)
    if key == "full": map50_full = m50; per_full = per
    print(f"  {name:16s}{m50:9.4f}{m5095:10.4f}")
print(f"\n  [CALIBRATION] full@640 mAP50 = {map50_full:.4f}  — PHAI ~ ultralytics `yolo val split={args.split}` (val 548 = 0.378)")
print("  Neu khop -> thuoc do dung -> so SAHI/RL o tren la COCO mAP that, so voi paper duoc.")
print("\n  full@640 per-class mAP50:")
for ci, nm in enumerate(CLASS_NAMES):
    print(f"    {nm:18s}{per_full[ci]:.4f}")
