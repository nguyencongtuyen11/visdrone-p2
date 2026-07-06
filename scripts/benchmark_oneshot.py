"""RL-ONESHOT-BATCH: go YOLO khoi vong lap RL -> chon het lat truoc, batch 1 chuyen.

Van de: pipeline cu CHAY YOLO TRONG vong lap chon lat (cho ket qua tung lat de gate)
        -> tuan tu, khong batch duoc -> T4 van 1636ms.
Giai phap (giu nguyen agent da train):
  1) SELECT : agent rollout chon toi da K lat, KHONG cham YOLO (luat chong trung dung
              attempted/kept ROIs — von khong can detection cua lat truoc).
  2) BATCH  : tat ca crop (lat RL + luoi tho hybrid) di 1 chuyen run_yolo_on_crops theo chunk.
  3) GATE   : cong gain>=1 & utility>=0.2 loc SAU tren box (numpy, ~0ms) roi merge.

Chay tren Lightning T4:
  python scripts/benchmark_oneshot.py --limit 100 --base 640 --chunk 16
So sanh cung run voi ban tuan tu cu:  them --with-seq
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
from rl_sahi.inference.crops import run_yolo_on_crop, run_yolo_on_crops
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
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--max-fine", type=int, default=8, help="so lat RL toi da")
ap.add_argument("--max-attempts", type=int, default=14)
ap.add_argument("--chunk", type=int, default=16, help="batch size khi gom crop (T4: 16 ok, 4GB local: 6)")
ap.add_argument("--with-seq", action="store_true", help="chay them RL-HYBRID tuan tu cu de doi chieu cung run")
ap.add_argument("--device", default="cuda")
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
bcfg = BenchmarkConfig(fixed_slice_fraction=0.35, fixed_overlap=0.2, target_classes=tc, class_mapping=cm)
model = load_yolo(str(WEIGHTS), device=dev)
dt = resolve_torch_device(dev)
policy, ck = load_policy(args.checkpoint, dt)
env_cfg = ck["env_cfg_obj"]; env_cfg.max_slices = args.max_fine
state_cfg = ck.get("state_cfg_obj", StateConfig())

def sync():
    if dev == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize()
def nw(): sync(); return time.perf_counter()

images = iter_images(IR, split=args.split, limit=args.limit)
if not images: sys.exit(f"[oneshot] khong thay anh o {IR}/{args.split}")
small_thr = _small_area_threshold(images, IR, LR, tc, 40.0, cm)
gt = {}
for img in images:
    gb, gc = _read_gt(img, IR, LR, tc, cm); gt[img.stem] = (gb, gc, _image_shape(img))

def is_rejected(info):
    return bool(info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap") or
                (icfg.require_stop_for_acceptance and (info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi"))))

def select_rois_oneshot(det):
    """Agent chon lat KHONG cham YOLO: kept ROIs dong vai 'accepted' cho luat chong trung."""
    kept, att = [], []
    for _ in range(int(args.max_attempts)):
        if len(kept) >= env_cfg.max_slices: break
        hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
        ov = np.stack(kept).astype("f4") if kept else np.zeros((0, 4), "f4")
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                       overlap_rois=ov, target_classes=tc, class_mapping=cm)
        roi, _a, info = rollout_one_slice(policy, env, dt)
        att.append(roi)
        if is_rejected(info):
            if _attempt_overlap(roi, att[:-1]) >= 0.95: break
            continue
        kept.append(roi)
    return kept

def batch_crops(img, rois):
    """1 chuyen batch cho toan bo crop, chunk theo VRAM."""
    outs = []
    for i in range(0, len(rois), args.chunk):
        part = rois[i:i + args.chunk]
        outs.extend(run_yolo_on_crops(model, [img] * len(part), part,
                    imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev))
    return outs

pF, pS, pO, pQ = {}, {}, {}, {}
tF = tS = tO = tQ = 0.0
oSel = oBat = oGate = 0.0
scrops = ocrops = qcrops = 0
ogate_drop = 0

# warmup
get_initial_detection(model=model, weights=str(WEIGHTS), image_path=images[0], weights_imgsz=BASE,
    full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
    spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
run_yolo_on_crops(model, [images[0]] * 2, [np.array([0, 0, 320, 320], "f4"), np.array([100, 100, 500, 500], "f4")],
    imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev); sync()

for img in images:
    iid = img.stem
    t = nw()
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=BASE,
        full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    base_t = nw() - t
    fb, fs, fc = _full_predictions(det, icfg)
    pF[iid] = (fb, fs, fc); tF += base_t

    # ---- SAHI (batched luoi co dinh — cung duoc huong loi batch cho cong bang) ----
    t = nw()
    srois = _fixed_grid_rois(det.image_shape, bcfg.fixed_slice_fraction, bcfg.fixed_overlap)
    souts = batch_crops(img, srois)
    spb, sps, spc = [fb], [fs], [fc]
    for (b, s, c) in souts:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        spb.append(b); sps.append(s); spc.append(c)
    sb, ss, sc2 = _merge_predictions(det.image_shape, 0.5, spb, sps, spc)
    tS += base_t + (nw() - t); pS[iid] = (sb, ss, sc2); scrops += 1 + len(srois)

    # ---- RL-ONESHOT-BATCH (hybrid: lat RL + luoi tho, TAT CA batch 1 chuyen) ----
    t = nw()
    fine = select_rois_oneshot(det)
    t_sel = nw(); oSel += t_sel - t
    coarse = _fixed_grid_rois(det.image_shape, 0.6, 0.15)
    all_rois = fine + coarse
    outs = batch_crops(img, all_rois) if all_rois else []
    t_bat = nw(); oBat += t_bat - t_sel
    # cong gain/utility loc SAU cho lat RL (numpy, gan nhu mien phi)
    xb, xs, xc = [], [], []
    for (b, s, c) in outs[:len(fine)]:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        g = _new_detection_gain(fb, fs, fc, xb, xs, xc, b, s, c, det.image_shape, 0.5, 0.5)
        u = _new_detection_utility(fb, fs, fc, xb, xs, xc, b, s, c, det.image_shape, 0.5, 0.5)
        if g < 1 or u < 0.2:
            ogate_drop += 1; continue
        xb.append(b); xs.append(s); xc.append(c)
    pb, ps, pc = [fb, *xb], [fs, *xs], [fc, *xc]
    for (b, s, c) in outs[len(fine):]:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    ob, os_, oc = _merge_predictions(det.image_shape, 0.5, pb, ps, pc)
    oGate += nw() - t_bat
    tO += base_t + (nw() - t); pO[iid] = (ob, os_, oc); ocrops += 1 + len(all_rois)

    # ---- (tuy chon) RL-HYBRID tuan tu cu de doi chieu cung run ----
    if args.with_seq:
        t = nw()
        acc, att, yb, ys, yc = [], [], [], [], []
        for _ in range(int(args.max_attempts)):
            if len(acc) >= env_cfg.max_slices: break
            hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
            ov = np.stack(acc).astype("f4") if acc else np.zeros((0, 4), "f4")
            env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                           overlap_rois=ov, target_classes=tc, class_mapping=cm)
            roi, _a, info = rollout_one_slice(policy, env, dt)
            if is_rejected(info):
                att.append(roi)
                if _attempt_overlap(roi, att[:-1]) >= 0.95: break
                continue
            b, s, c = run_yolo_on_crop(model, img, roi, imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev)
            att.append(roi); c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
            g = _new_detection_gain(fb, fs, fc, yb, ys, yc, b, s, c, det.image_shape, 0.5, 0.5)
            u = _new_detection_utility(fb, fs, fc, yb, ys, yc, b, s, c, det.image_shape, 0.5, 0.5)
            if g < 1 or u < 0.2: continue
            acc.append(roi); yb.append(b); ys.append(s); yc.append(c)
        qb, qs2, qc2 = [fb, *yb], [fs, *ys], [fc, *yc]
        for roi in _fixed_grid_rois(det.image_shape, 0.6, 0.15):
            b, s, c = run_yolo_on_crop(model, img, roi, imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev)
            c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
            qb.append(b); qs2.append(s); qc2.append(c)
        hb, hs, hc = _merge_predictions(det.image_shape, 0.5, qb, qs2, qc2)
        tQ += base_t + (nw() - t); pQ[iid] = (hb, hs, hc)
        qcrops += 1 + len(_fixed_grid_rois(det.image_shape, 0.6, 0.15)) + len(acc)

def ev(p): return _evaluate_method(p, gt, tc, 0.5, small_thr)
N = len(images)
gpu = torch.cuda.get_device_name(0) if dev == "cuda" and torch.cuda.is_available() else "CPU"
mF, mS, mO = ev(pF), ev(pS), ev(pO)
print(f"\n===== ONESHOT-BATCH (test-{N}, {gpu}) base@{BASE} crop@{SLI} chunk={args.chunk} =====")
print(f"  {'method':22s}{'mAP50':>8s}{'s_recall':>9s}{'FP/img':>8s}{'crops':>7s}{'ms/anh':>9s}")
print(f"  {'full@'+str(BASE):22s}{mF['mAP50']:8.4f}{mF['small_recall']:9.4f}{mF['fp_per_image']:8.1f}{1.0:7.1f}{tF/N*1000:9.0f}")
print(f"  {'SAHI (batched)':22s}{mS['mAP50']:8.4f}{mS['small_recall']:9.4f}{mS['fp_per_image']:8.1f}{scrops/N:7.1f}{tS/N*1000:9.0f}")
print(f"  {'RL-ONESHOT-BATCH':22s}{mO['mAP50']:8.4f}{mO['small_recall']:9.4f}{mO['fp_per_image']:8.1f}{ocrops/N:7.1f}{tO/N*1000:9.0f}")
if args.with_seq:
    mQ = ev(pQ)
    print(f"  {'RL-HYBRID tuan tu':22s}{mQ['mAP50']:8.4f}{mQ['small_recall']:9.4f}{mQ['fp_per_image']:8.1f}{qcrops/N:7.1f}{tQ/N*1000:9.0f}")
print(f"\n  ONESHOT breakdown: select(rollout)={oSel/N*1000:.0f}ms | batch-YOLO={oBat/N*1000:.0f}ms | gate+merge={oGate/N*1000:.0f}ms | base={tF/N*1000:.0f}ms")
print(f"  gate loc sau: bo {ogate_drop/N:.1f} lat/anh (lat chay YOLO nhung khong dong gop box moi)")
