"""VE SO SANH: ban DI CHUYEN (rollout DQN) vs ban ONE-SHOT (policy REINFORCE).

Voi moi anh test, xuat 1 anh ghep 2 panel:
  TRAI  = ONE-SHOT : ROI DO = vung policy GIU (KEEP/ZOOM), cham xam = vung DROP.
  PHAI  = DI CHUYEN: ROI DO = vung rollout chon.
Hop xanh la = detection cuoi cung (full + crop, merge NMS). Chu tren moi ROI = hanh dong.

Chay tren Lightning (co data + 2 checkpoint):
  python scripts/viz_compare.py --split test --limit 8 --k 8 --device cuda
Anh luu o runs/viz/*.jpg  (tai ve / mo bang file browser cua Lightning de xem).
Neu thieu checkpoint DI CHUYEN (runs/ft_rl/dqn/best.pt) -> chi ve ONE-SHOT, panel phai bao thieu.
"""
import sys, argparse
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").addFilter(lambda r: "deprecated" not in r.getMessage())
from pathlib import Path
import numpy as np, torch, cv2

from rl_sahi.common.config import load_default_config
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _full_predictions, _merge_predictions
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.pipeline import _filter_classes, _attempt_overlap, get_initial_detection
from rl_sahi.rl.oneshot import (propose_regions, action_to_roi, region_local_state, objectness_grid,
                                DROP, KEEP, ACTION_NAMES)
from rl_sahi.rl.oneshot_policy import load_oneshot_policy

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--policy", type=Path, default=ROOT / "runs" / "oneshot" / "policy.pt")
ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "ft_rl" / "dqn" / "best.pt")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=8, help="so anh ve")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--max-fine", type=int, default=8)
ap.add_argument("--max-attempts", type=int, default=14)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", type=Path, default=ROOT / "runs" / "viz")
ap.add_argument("--out-width", type=int, default=0, help="neu >0: thu nho anh ghep ve chieu rong nay (tiet kiem dia khi chay ca bo test)")
ap.add_argument("--jpg-quality", type=int, default=90, help="chat luong JPEG 1-100")
args = ap.parse_args()

BASE, SLI, dev = args.base, args.slice, args.device
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"; WEIGHTS = ROOT / "best_visdrone.pt"

cfg = load_default_config(args.config, ROOT)
tc = tuple(cfg.section("infer")["target_classes"]); cm = ClassMapping.from_config(cfg.section("classes"))
icfg = InferenceConfig(full_imgsz=BASE, slice_imgsz=SLI, full_conf=0.01, output_conf=0.10, iou=0.7,
    merge_iou=0.5, max_det=3000, device=dev, feature_layers=(16,), target_classes=tc, class_mapping=cm,
    min_slice_detections=1, min_slice_utility=0.2, duplicate_iou=0.5, max_slice_attempts=args.max_attempts,
    require_stop_for_acceptance=True)
model = load_yolo(str(WEIGHTS), device=dev)
dt = resolve_torch_device(dev)

if not args.policy.exists():
    sys.exit(f"[viz] thieu policy one-shot {args.policy}")
policy_os, _ = load_oneshot_policy(args.policy, dt)

# --- ban DI CHUYEN (tuy chon: co the thieu checkpoint) ---
policy_mv = env_cfg = state_cfg = None
if args.checkpoint.exists():
    from rl_sahi.rl.checkpoint import load_policy
    from rl_sahi.rl.slice_env import SliceEnv
    from rl_sahi.rl.state_config import StateConfig
    from rl_sahi.inference.rollout import rollout_one_slice
    policy_mv, ckm = load_policy(args.checkpoint, dt)
    env_cfg = ckm["env_cfg_obj"]; env_cfg.max_slices = args.max_fine
    state_cfg = ckm.get("state_cfg_obj", StateConfig())
    print(f"[viz] co checkpoint DI CHUYEN: {args.checkpoint}")
else:
    print(f"[viz] THIEU checkpoint DI CHUYEN {args.checkpoint} -> chi ve ONE-SHOT")

RED, GREEN, GRAY, WHITE, BLACK = (0, 0, 255), (0, 200, 0), (150, 150, 150), (255, 255, 255), (0, 0, 0)

def batch_crops(img, rois):
    outs = []
    for i in range(0, len(rois), args.chunk):
        part = rois[i:i + args.chunk]
        outs.extend(run_yolo_on_crops(model, [img] * len(part), part,
                    imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev))
    return outs

def merged_dets(det, img, fb, fs, fc, rois):
    outs = batch_crops(img, rois) if rois else []
    pb, ps, pc = [fb], [fs], [fc]
    for (b, s, c) in outs:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    return _merge_predictions(det.image_shape, 0.5, pb, ps, pc)

def draw_box(im, box, color, th, label=None):
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(im, (x0, y0), (x1, y1), color, th)
    if label:
        (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (x0, max(0, y0 - thh - 6)), (x0 + tw + 4, y0), color, -1)
        cv2.putText(im, label, (x0 + 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

def draw_dets(im, boxes, th):
    for b in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
        cv2.rectangle(im, (x0, y0), (x1, y1), GREEN, th)

def banner(im, text):
    h, w = im.shape[:2]
    bar = int(max(34, h * 0.045))
    cv2.rectangle(im, (0, 0), (w, bar), BLACK, -1)
    cv2.putText(im, text, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.6, w / 1400), WHITE, 2, cv2.LINE_AA)

def select_rois_moving(det):
    acc, att = [], []
    for _ in range(int(args.max_attempts)):
        if len(acc) >= env_cfg.max_slices: break
        hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
        ov = np.stack(acc).astype("f4") if acc else np.zeros((0, 4), "f4")
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                       overlap_rois=ov, target_classes=tc, class_mapping=cm)
        roi, _a, info = rollout_one_slice(policy_mv, env, dt)
        att.append(roi)
        rejected = bool(info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap") or
                        (icfg.require_stop_for_acceptance and (info.get("stop_due_to_max_steps") or
                         info.get("stop_due_to_stalled_roi"))))
        if rejected:
            if _attempt_overlap(roi, att[:-1]) >= 0.95: break
            continue
        acc.append(roi)
    return acc

args.out.mkdir(parents=True, exist_ok=True)
images = iter_images(IR, split=args.split, limit=args.limit)
if not images: sys.exit(f"[viz] khong thay anh o {IR}/{args.split}")

for _idx, img in enumerate(images):
    im0 = cv2.imread(str(img))
    if im0 is None:
        print(f"[viz] bo qua (khong doc duoc) {img.name}"); continue
    H, W = im0.shape[:2]
    th_roi = max(2, W // 500); th_det = max(1, W // 900)
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=BASE,
        full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    fb, fs, fc = _full_predictions(det, icfg)

    # ---------- panel ONE-SHOT ----------
    im_os = im0.copy()
    grid = objectness_grid(det)
    regions = propose_regions(det, k=args.k)
    kept, n_drop = [], 0
    if regions:
        states = np.stack([region_local_state(det, r, grid) for r in regions]).astype(np.float32)
        with torch.no_grad():
            acts = policy_os(torch.from_numpy(states).to(dt)).argmax(1).cpu().numpy()
        for r, a in zip(regions, acts):
            a = int(a)
            roi = action_to_roi(r, a, det.image_shape)
            if roi is None:  # DROP -> cham xam tai tam vung
                n_drop += 1
                cx, cy = r["center"]
                cv2.circle(im_os, (int(cx), int(cy)), max(4, W // 250), GRAY, -1)
            else:
                kept.append(roi)
                draw_box(im_os, roi, RED, th_roi, ACTION_NAMES[a])
    ob, os_, oc = merged_dets(det, img, fb, fs, fc, kept)
    draw_dets(im_os, ob, th_det)
    banner(im_os, f"ONE-SHOT | GIU {len(kept)}/{len(regions)} vung (DROP {n_drop}) | {len(ob)} det")

    # ---------- panel DI CHUYEN ----------
    im_mv = im0.copy()
    if policy_mv is not None:
        rois_mv = select_rois_moving(det)
        for i, roi in enumerate(rois_mv):
            draw_box(im_mv, roi, RED, th_roi, f"MOVE{i+1}")
        mb, ms, mc = merged_dets(det, img, fb, fs, fc, rois_mv)
        draw_dets(im_mv, mb, th_det)
        banner(im_mv, f"DI CHUYEN (rollout) | {len(rois_mv)} ROI | {len(mb)} det")
    else:
        im_mv[:] = (40, 40, 40)
        banner(im_mv, "DI CHUYEN | THIEU checkpoint runs/ft_rl/dqn/best.pt")

    # ---------- ghep ngang ----------
    sep = np.full((H, max(4, W // 300), 3), 255, np.uint8)
    combo = np.hstack([im_os, sep, im_mv])
    if args.out_width and combo.shape[1] > args.out_width:
        sc = args.out_width / combo.shape[1]
        combo = cv2.resize(combo, (args.out_width, int(round(combo.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    out_path = args.out / f"{_idx:04d}_{img.stem}_cmp.jpg"
    cv2.imwrite(str(out_path), combo, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)])
    print(f"[viz] {out_path.name}: one-shot giu {len(kept)}/{len(regions)}"
          + (f", di-chuyen {len(rois_mv)} ROI" if policy_mv is not None else ", (thieu ban di chuyen)"), flush=True)

print(f"\n[viz] XONG. {len(images)} anh o {args.out}")
print("[viz] DO = ROI (RL chon) | XANH LA = detection | CHAM XAM = vung one-shot DROP")
print("[viz] Tai thu muc runs/viz/ ve xem, hoac mo bang file browser cua Lightning.")
