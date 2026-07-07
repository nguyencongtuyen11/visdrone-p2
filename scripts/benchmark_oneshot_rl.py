"""ONE-SHOT RL benchmark: do policy da train (REINFORCE) vs STATIC baseline vs full YOLO.

Tra loi cau hoi "y one-shot co on khong":
  - STATIC top-K KEEP : sinh K vung tu dinh objectness, GIU HET (khong zoom/khong bo) — KHONG RL.
  - RL-ONESHOT        : cung K vung, POLICY chon {DROP|KEEP|ZOOM1.5|ZOOM2} moi vung (greedy).
  - full@base         : YOLO full 1 luot (nen tham chieu).
Ca 3 gom crop -> BATCH YOLO 1 chuyen -> merge class-aware NMS -> do mAP50/recall/FP + THOI GIAN.

Tin hieu can nhin (in cuoi):
  - RL action dist: policy chon gi. Toan DROP => slicing vo ich (honest null).
  - RL recall vs STATIC recall: bang recall + IT crop/IT FP hon  => zoom/drop CO ich (thang).
                                cao hon recall                    => zoom giup bat them (thang manh).
                                y het STATIC                      => policy khong hoc them gi (null).

Chay tren Lightning T4 (sau khi train xong runs/oneshot/policy.pt):
  python scripts/benchmark_oneshot_rl.py --split test --limit 200 --k 8 --chunk 16 --device cuda
Them SAHI de doi chieu:  --with-sahi
"""
import sys, time, argparse
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# --- chan warning spam ("'half' is deprecated") de terminal doc duoc; KHONG doi hanh vi ---
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
from rl_sahi.eval.benchmark import (_fixed_grid_rois, _full_predictions, _merge_predictions,
    _evaluate_method, _read_gt, _image_shape, _small_area_threshold, BenchmarkConfig)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.pipeline import _filter_classes, get_initial_detection
from rl_sahi.rl.oneshot import (propose_regions, action_to_roi, region_local_state, objectness_grid,
                                KEEP, NUM_ACTIONS, ACTION_NAMES)
from rl_sahi.rl.oneshot_policy import load_oneshot_policy

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--policy", type=Path, default=ROOT / "runs" / "oneshot" / "policy.pt")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--k", type=int, default=8, help="so vung ung vien/anh (nen KHOP voi luc train)")
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--chunk", type=int, default=16, help="batch size gom crop (T4:16, local 4GB:6)")
ap.add_argument("--with-sahi", action="store_true", help="chay them SAHI luoi co dinh de doi chieu")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

BASE, SLI, dev = args.base, args.slice, args.device
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"; WEIGHTS = ROOT / "best_visdrone.pt"

cfg = load_default_config(args.config, ROOT)
tc = tuple(cfg.section("infer")["target_classes"]); cm = ClassMapping.from_config(cfg.section("classes"))
icfg = InferenceConfig(full_imgsz=BASE, slice_imgsz=SLI, full_conf=0.01, output_conf=0.10, iou=0.7,
    merge_iou=0.5, max_det=3000, device=dev, feature_layers=(16,), target_classes=tc, class_mapping=cm,
    min_slice_detections=1, min_slice_utility=0.2, duplicate_iou=0.5, max_slice_attempts=14,
    require_stop_for_acceptance=True)
bcfg = BenchmarkConfig(fixed_slice_fraction=0.35, fixed_overlap=0.2, target_classes=tc, class_mapping=cm)

if not args.policy.exists():
    sys.exit(f"[oneshot-rl] khong thay policy o {args.policy} — train_oneshot.py chua chay xong?")
model = load_yolo(str(WEIGHTS), device=dev)
dt = resolve_torch_device(dev)
policy, ck = load_oneshot_policy(args.policy, dt)
pmeta = ck.get("meta", {})

def sync():
    if dt.type == "cuda" and torch.cuda.is_available(): torch.cuda.synchronize()
def nw(): sync(); return time.perf_counter()

def batch_crops(img, rois):
    """1 chuyen batch cho toan bo crop, chunk theo VRAM."""
    outs = []
    for i in range(0, len(rois), args.chunk):
        part = rois[i:i + args.chunk]
        outs.extend(run_yolo_on_crops(model, [img] * len(part), part,
                    imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev))
    return outs

def merge_with_full(det, fb, fs, fc, crop_outs):
    """Gom full + cac crop -> class-aware NMS."""
    pb, ps, pc = [fb], [fs], [fc]
    for (b, s, c) in crop_outs:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    return _merge_predictions(det.image_shape, 0.5, pb, ps, pc)

images = iter_images(IR, split=args.split, limit=args.limit)
if not images: sys.exit(f"[oneshot-rl] khong thay anh o {IR}/{args.split}")
small_thr = _small_area_threshold(images, IR, LR, tc, 40.0, cm)
gt = {}
for img in images:
    gb, gc = _read_gt(img, IR, LR, tc, cm); gt[img.stem] = (gb, gc, _image_shape(img))

pF, pSt, pO, pSa = {}, {}, {}, {}
tF = tSt = tO = tSa = 0.0
stcrops = ocrops = sacrops = 0
oProp = oPol = oBat = oMrg = 0.0
action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
rl_regions_total = 0

# warmup (dung crop that de nap kernel, khong tinh gio)
get_initial_detection(model=model, weights=str(WEIGHTS), image_path=images[0], weights_imgsz=BASE,
    full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
    spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
run_yolo_on_crops(model, [images[0]] * 2, [np.array([0, 0, 320, 320], "f4"), np.array([100, 100, 500, 500], "f4")],
    imgsz=SLI, conf=0.1, iou=0.7, max_det=3000, device=dev); sync()

t0_all = time.perf_counter()
for _idx, img in enumerate(images):
    if _idx % 10 == 0 and _idx > 0:
        _el = time.perf_counter() - t0_all
        print(f"[oneshot-rl] {_idx}/{len(images)} anh | {_el:.0f}s troi qua | uoc con ~{_el/_idx*(len(images)-_idx):.0f}s", flush=True)
    iid = img.stem
    # ---- base: YOLO full 1 luot (dung chung cho ca 3 method) ----
    t = nw()
    det = get_initial_detection(model=model, weights=str(WEIGHTS), image_path=img, weights_imgsz=BASE,
        full_conf=0.01, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    base_t = nw() - t
    fb, fs, fc = _full_predictions(det, icfg)
    pF[iid] = (fb, fs, fc); tF += base_t

    # ---- sinh ung vien 1 lan (objectness + proposal = CHUNG cho STATIC + RL) ----
    t = nw()
    grid = objectness_grid(det)
    regions = propose_regions(det, k=args.k)
    prop_t = nw() - t
    rl_regions_total += len(regions)
    # build state = RL-ONLY (STATIC khong dung) -> tinh gio RIENG, chi tru vao RL
    t = nw()
    states = (np.stack([region_local_state(det, r, grid) for r in regions]).astype(np.float32)
              if regions else None)
    state_t = nw() - t

    # ---- STATIC top-K KEEP (khong RL): giu het vung, crop 35% (KEEP), khong zoom/khong bo ----
    t = nw()
    st_rois = []
    for r in regions:
        roi = action_to_roi(r, KEEP, det.image_shape)
        if roi is not None: st_rois.append(roi)
    souts = batch_crops(img, st_rois) if st_rois else []
    sb, ss, sc = merge_with_full(det, fb, fs, fc, souts)
    tSt += base_t + prop_t + (nw() - t); pSt[iid] = (sb, ss, sc); stcrops += 1 + len(st_rois)

    # ---- RL-ONESHOT: policy chon action moi vung (greedy) ----
    t = nw()
    if states is not None:
        with torch.no_grad():
            acts = policy(torch.from_numpy(states).to(dt)).argmax(1).cpu().numpy()
    else:
        acts = np.zeros((0,), dtype=np.int64)
    rl_rois = []
    for r, a in zip(regions, acts):
        action_counts[int(a)] += 1
        roi = action_to_roi(r, int(a), det.image_shape)
        if roi is not None: rl_rois.append(roi)
    pol_t = nw() - t
    t = nw()
    outs = batch_crops(img, rl_rois) if rl_rois else []
    bat_t = nw() - t
    t = nw()
    ob, os_, oc = merge_with_full(det, fb, fs, fc, outs)
    mrg_t = nw() - t
    oProp += prop_t; oPol += pol_t + state_t; oBat += bat_t; oMrg += mrg_t
    tO += base_t + prop_t + state_t + pol_t + bat_t + mrg_t
    pO[iid] = (ob, os_, oc); ocrops += 1 + len(rl_rois)

    # ---- (tuy chon) SAHI luoi co dinh, cung batch de cong bang ----
    if args.with_sahi:
        t = nw()
        srois = _fixed_grid_rois(det.image_shape, bcfg.fixed_slice_fraction, bcfg.fixed_overlap)
        sa_outs = batch_crops(img, srois)
        ab, as_, ac = merge_with_full(det, fb, fs, fc, sa_outs)
        tSa += base_t + (nw() - t); pSa[iid] = (ab, as_, ac); sacrops += 1 + len(srois)

def ev(p): return _evaluate_method(p, gt, tc, 0.5, small_thr)
N = len(images)
gpu = torch.cuda.get_device_name(0) if dt.type == "cuda" and torch.cuda.is_available() else "CPU"
mF, mSt, mO = ev(pF), ev(pSt), ev(pO)
print(f"\n===== ONESHOT-RL (test-{N}, {gpu}) base@{BASE} crop@{SLI} k={args.k} =====")
print(f"  {'method':22s}{'mAP50':>8s}{'s_recall':>9s}{'FP/img':>8s}{'crops':>7s}{'ms/anh':>9s}")
print(f"  {'full@'+str(BASE):22s}{mF['mAP50']:8.4f}{mF['small_recall']:9.4f}{mF['fp_per_image']:8.1f}{1.0:7.1f}{tF/N*1000:9.0f}")
print(f"  {'STATIC top-K KEEP':22s}{mSt['mAP50']:8.4f}{mSt['small_recall']:9.4f}{mSt['fp_per_image']:8.1f}{stcrops/N:7.1f}{tSt/N*1000:9.0f}")
print(f"  {'RL-ONESHOT (policy)':22s}{mO['mAP50']:8.4f}{mO['small_recall']:9.4f}{mO['fp_per_image']:8.1f}{ocrops/N:7.1f}{tO/N*1000:9.0f}")
if args.with_sahi:
    mSa = ev(pSa)
    print(f"  {'SAHI (batched)':22s}{mSa['mAP50']:8.4f}{mSa['small_recall']:9.4f}{mSa['fp_per_image']:8.1f}{sacrops/N:7.1f}{tSa/N*1000:9.0f}")

tot_act = int(action_counts.sum())
adist = {ACTION_NAMES[a]: int(action_counts[a]) for a in range(NUM_ACTIONS)}
apct = {ACTION_NAMES[a]: f"{100*action_counts[a]/max(tot_act,1):.0f}%" for a in range(NUM_ACTIONS)}
print(f"\n  RL action dist (policy chon tren {tot_act} vung): {adist}")
print(f"  RL action %:                                  {apct}")
print(f"  RL breakdown: propose={oProp/N*1000:.0f}ms | state+policy={oPol/N*1000:.1f}ms | batch-YOLO={oBat/N*1000:.0f}ms | merge={oMrg/N*1000:.0f}ms | base={tF/N*1000:.0f}ms")
print(f"  policy meta: {pmeta}")
print(f"  * full@{BASE} o day = detection NEN cua pipeline (co trich feature/objectness), KHONG phai plain YOLO;"
      f" plain full@640 sach ~45ms xem benchmark_speed.py.")
print(f"\n  DOC: RL recall >= STATIC recall & IT crop/FP hon  => zoom/drop CO ich."
      f"\n       Toan DROP hoac y het STATIC                     => slicing khong them gia tri (ket qua honest).")
