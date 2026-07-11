"""BUDGET SWEEP — bang chung "agent BIET NHIN DAU": RL-only vs baseline o CUNG so crop K.

Cau hoi doc cua hoi dong: "luoi 0.6 khong hoc gi = .570 @ 9 crop; RL them gi?"
Tra loi bang cach doi truc: mAP THEO NGAN SACH crop (K=2/4/6/8). O K nho, baseline
co dinh phai rai mu / bam heuristic, con agent dat lat co dieu kien theo anh.

Cac canh tay (arm) moi K — TAT CA + full@640, cung merge/NMS/conf:
  rl_K    : prefix K lat dau cua agent DQN (chay 1 lan max_slices=8; hop le vi
            rollout greedy deterministic & state khong ma hoa budget)
  topk_K  : top-K dinh objectness, fraction 0.30 (heuristic manh nhat, CO SAN trong repo
            — PHAI dua vao, giau no nguy hiem hon thua no)
  cent_K  : K o gan tam anh nhat tu luoi fraction 0.30 (deterministic, khong RL)
  rand_K  : K o ngau nhien tu luoi fraction 0.30 (N seed, mean±std) — cung SCALE voi RL
            de triet confound zoom-vs-placement
  g06c_K  : K o gan tam tu luoi san pham 0.6/0.15 (arm PHU — khac scale, chi tham khao)
Moc: full (0 crop) va grid06 day du (~8 crop, luoi san pham).

Metric: COCO mAP50 / mAP50-95 (conf 0.001, 101-point) + small-recall@op-conf
(GHEP 1-1 greedy, nguong vat nho = percentile-40 dien tich GT toan split) + SO CROP THUC.

Chay tren Lightning (~10-20 phut GPU val-548):
  python scripts/eval_budget_sweep.py --split val --device cuda \
      --weights runs/detect/ft_crop/weights/best.pt --checkpoint runs/ft_rl_small/dqn/best.pt
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
from rl_sahi.common.data import iter_images, image_to_label_path, read_yolo_labels
from rl_sahi.common.boxes import area, iou_matrix
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _fixed_grid_rois, _merge_predictions, _image_shape, _objectness_grid, _topk_peak_rois
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.pipeline import _filter_classes, _attempt_overlap, get_initial_detection
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.inference.rollout import rollout_one_slice

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detect" / "ft_crop" / "weights" / "best.pt")
ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "ft_rl_small" / "dqn" / "best.pt")
ap.add_argument("--split", default="val")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--map-conf", type=float, default=0.001)
ap.add_argument("--op-conf", type=float, default=0.25, help="operating point cho small-recall")
ap.add_argument("--ks", default="2,4,6,8")
ap.add_argument("--rand-seeds", type=int, default=3)
ap.add_argument("--frac", type=float, default=0.30, help="fraction cua baseline cung scale voi RL")
ap.add_argument("--small-pct", type=float, default=40.0)
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--max-attempts", type=int, default=14)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--max-det", type=int, default=300)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

KS = sorted(int(x) for x in args.ks.split(","))
KMAX = max(KS)
BASE, SLI, dev, MC = args.base, args.slice, args.device, args.map_conf
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"
CLASSES = list(range(10))
IOUV = np.linspace(0.5, 0.95, 10)

cfg = load_default_config(args.config, ROOT)
cm = ClassMapping.from_config(cfg.section("classes"))
tc = tuple(CLASSES)
model = load_yolo(str(args.weights), device=dev)
dt = resolve_torch_device(dev)
if not args.checkpoint.exists():
    sys.exit(f"[sweep] thieu checkpoint RL {args.checkpoint}")
policy_mv, ckm = load_policy(args.checkpoint, dt)
env_cfg = ckm["env_cfg_obj"]; env_cfg.max_slices = KMAX
state_cfg = ckm.get("state_cfg_obj", StateConfig())

# ---------------- helpers ----------------
def select_rois_rl(det):
    kept, att = [], []
    for _ in range(int(args.max_attempts)):
        if len(kept) >= KMAX: break
        hist = np.stack(att).astype("f4") if att else np.zeros((0, 4), "f4")
        ov = np.stack(kept).astype("f4") if kept else np.zeros((0, 4), "f4")
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=hist,
                       overlap_rois=ov, target_classes=tc, class_mapping=cm)
        roi, _a, info = rollout_one_slice(policy_mv, env, dt)
        att.append(roi)
        rejected = bool(info.get("stop_due_to_old_overlap") or info.get("stop_due_to_attempted_overlap") or
                        info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi"))
        if rejected:
            if _attempt_overlap(roi, att[:-1]) >= 0.95: break
            continue
        kept.append(roi)
    return kept  # thu tu giu = thu tu agent chon -> prefix K hop le

def center_sorted(pool, shape):
    h, w = shape
    cxy = np.array([[(r[0] + r[2]) / 2 - w / 2, (r[1] + r[3]) / 2 - h / 2] for r in pool])
    order = np.argsort((cxy ** 2).sum(1))
    return [pool[i] for i in order]

def roi_key(r):
    return tuple(round(float(v), 1) for v in np.asarray(r).reshape(4))

def eval_ap_all(preds, gts):
    ap = np.full((len(CLASSES), len(IOUV)), np.nan)
    for ci, c in enumerate(CLASSES):
        n_gt = 0; per_img = {}; entries = []
        for iid, (gb, gc) in gts.items():
            gm = gc.astype(np.int64) == c
            gbi = gb[gm]; n_gt += int(gm.sum())
            pb, ps, pc = preds[iid]
            pm = pc.astype(np.int64) == c
            pbi = pb[pm]; psi = ps[pm]
            if len(pbi) == 0 and len(gbi) == 0: continue
            iou = iou_matrix(pbi, gbi) if (len(pbi) and len(gbi)) else np.zeros((len(pbi), len(gbi)), np.float32)
            per_img[iid] = (psi, iou, len(gbi))
            for k2, s in enumerate(psi): entries.append((float(s), iid, k2))
        if n_gt == 0: continue
        if not entries: ap[ci, :] = 0.0; continue
        entries.sort(key=lambda e: -e[0])
        for ti, thr in enumerate(IOUV):
            matched = {iid: np.zeros(v[2], bool) for iid, v in per_img.items()}
            tp = np.zeros(len(entries)); fp = np.zeros(len(entries))
            for rank, (s, iid, k2) in enumerate(entries):
                _sc, iou, ng = per_img[iid]
                if ng == 0: fp[rank] = 1.0; continue
                row = iou[k2]; j = int(row.argmax())
                if float(row[j]) >= thr and not matched[iid][j]:
                    tp[rank] = 1.0; matched[iid][j] = True
                else: fp[rank] = 1.0
            tpc = np.cumsum(tp); fpc = np.cumsum(fp)
            rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-16)
            mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([1.0], prec, [0.0]))
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            x = np.linspace(0, 1, 101)
            trapz = getattr(np, "trapezoid", None) or np.trapz
            ap[ci, ti] = float(trapz(np.interp(x, mrec, mpre), x))
    return float(np.nanmean(ap[:, 0])), float(np.nanmean(ap))

def small_recall(preds, gts, small_thr, op_conf):
    """GHEP 1-1 greedy theo score, class-aware, IoU>=0.5 — chi dem GT NHO (dien tich <= thr)."""
    hit = tot = 0
    for iid, (gb, gc) in gts.items():
        if not len(gb): continue
        sm = area(gb) <= small_thr
        tot += int(sm.sum())
        if not sm.any(): continue
        pb, ps, pc = preds[iid]
        keep = ps >= op_conf
        pb, ps, pc = pb[keep], ps[keep], pc[keep]
        if not len(pb): continue
        order = np.argsort(-ps)
        matched = np.zeros(len(gb), bool)
        for i in order:
            same = gc.astype(np.int64) == int(pc[i])
            cand = np.flatnonzero(same & ~matched)
            if not len(cand): continue
            ious = iou_matrix(pb[i].reshape(1, 4), gb[cand])[0]
            j = int(ious.argmax())
            if float(ious[j]) >= 0.5:
                matched[cand[j]] = True
        hit += int((matched & sm).sum())
    return hit / max(tot, 1)

# ---------------- main loop ----------------
images = iter_images(IR, split=args.split, limit=(args.limit or None))
if not images: sys.exit(f"[sweep] khong thay anh o {IR}/{args.split}")
print(f"[sweep] {len(images)} anh | K={KS} | frac baseline={args.frac} | seeds={args.rand_seeds}")
print(f"[sweep] weights={args.weights.name} | checkpoint={args.checkpoint}")

ARMS = ["full", "grid06"]
for K in KS:
    ARMS += [f"rl_{K}", f"topk_{K}", f"cent_{K}", f"g06c_{K}"]
    ARMS += [f"rand_{K}_s{s}" for s in range(args.rand_seeds)]
P = {a: {} for a in ARMS}
crops_used = {a: 0 for a in ARMS}
gts = {}
all_gt_areas = []
t0 = time.perf_counter()

for idx, img in enumerate(images):
    if idx % 50 == 0 and idx:
        el = time.perf_counter() - t0
        print(f"  {idx}/{len(images)} | {el:.0f}s | eta~{el/idx*(len(images)-idx):.0f}s", flush=True)
    iid = img.stem
    shape = _image_shape(img)
    cls, boxes = read_yolo_labels(image_to_label_path(img, IR, LR), shape)
    cls = cm.map_label_classes(cls)
    gb = np.asarray(boxes, np.float32).reshape(-1, 4); gc = np.asarray(cls, np.float32).reshape(-1)
    gts[iid] = (gb, gc)
    if len(gb): all_gt_areas.append(area(gb))

    det = get_initial_detection(model=model, weights=str(args.weights), image_path=img, weights_imgsz=BASE,
        full_conf=MC, full_iou=0.7, max_det=args.max_det, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    fb = np.asarray(det.boxes, np.float32).reshape(-1, 4)
    fs = np.asarray(det.scores, np.float32).reshape(-1)
    fc = cm.map_model_classes(det.classes)
    fb, fs, fc = _filter_classes(fb, fs, fc, tc)

    # --- sinh ROI cho moi arm (prefix-consistent) ---
    rl_rois = select_rois_rl(det)                                          # <= KMAX, thu tu chon
    ogrid = _objectness_grid(det)
    topk_rois = _topk_peak_rois(ogrid, det.image_shape, KMAX, args.frac, 2)  # prefix-consistent
    pool03 = _fixed_grid_rois(det.image_shape, args.frac, 0.0)             # luoi cung scale RL
    cent03 = center_sorted(pool03, det.image_shape)
    g06 = _fixed_grid_rois(det.image_shape, 0.6, 0.15)                     # luoi san pham (~8 o)
    g06c = center_sorted(g06, det.image_shape)
    rand_lists = []
    for s in range(args.rand_seeds):
        rng = np.random.RandomState(100003 * (s + 1) + idx)
        pm = rng.permutation(len(pool03))
        rand_lists.append([pool03[i] for i in pm])

    arm_rois = {"grid06": g06}
    for K in KS:
        arm_rois[f"rl_{K}"] = rl_rois[:K]
        arm_rois[f"topk_{K}"] = topk_rois[:K]
        arm_rois[f"cent_{K}"] = cent03[:K]
        arm_rois[f"g06c_{K}"] = g06c[:K]
        for s in range(args.rand_seeds):
            arm_rois[f"rand_{K}_s{s}"] = rand_lists[s][:K]

    # --- batch YOLO 1 lan tren cac ROI DUY NHAT ---
    uniq = {}
    for rois in arm_rois.values():
        for r in rois: uniq.setdefault(roi_key(r), np.asarray(r, np.float32).reshape(4))
    keys = list(uniq.keys())
    outs = {}
    for i in range(0, len(keys), args.chunk):
        part = keys[i:i + args.chunk]
        res = run_yolo_on_crops(model, [img] * len(part), [uniq[k2] for k2 in part],
                                imgsz=SLI, conf=MC, iou=0.7, max_det=3000, device=dev)
        for k2, (b, s, c) in zip(part, res):
            c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
            outs[k2] = (b, s, c)

    # --- merge tung arm ---
    P["full"][iid] = (fb, fs, fc)
    for name, rois in arm_rois.items():
        pb, ps, pc = [fb], [fs], [fc]
        for r in rois:
            b, s, c = outs[roi_key(r)]
            pb.append(b); ps.append(s); pc.append(c)
        P[name][iid] = _merge_predictions(det.image_shape, 0.5, pb, ps, pc)
        crops_used[name] += len(rois)

small_thr = float(np.percentile(np.concatenate(all_gt_areas), args.small_pct)) if all_gt_areas else 1e9
N = len(images)

# ---------------- bao cao ----------------
print(f"\n===== BUDGET SWEEP (split={args.split}, {N} anh, conf={MC}, small<=p{args.small_pct:.0f}) =====")
print(f"  {'arm':16s}{'K':>3s}{'mAP50':>9s}{'mAP50-95':>10s}{'s_recall':>10s}{'crops':>7s}")

def row(name, label, K):
    m50, m5095 = eval_ap_all(P[name], gts)
    sr = small_recall(P[name], gts, small_thr, args.op_conf)
    print(f"  {label:16s}{K:>3s}{m50:9.4f}{m5095:10.4f}{sr:10.4f}{crops_used[name]/N:7.1f}")
    return m50, m5095, sr

row("full", "full", "-")
row("grid06", "luoi-0.6 (8 o)", "-")
res = {}
for K in KS:
    print(f"  {'-'*55}")
    res[("rl", K)] = row(f"rl_{K}", "RL (agent)", str(K))
    res[("topk", K)] = row(f"topk_{K}", "topK objectness", str(K))
    res[("cent", K)] = row(f"cent_{K}", "center-K", str(K))
    m50s, m5095s, srs = [], [], []
    for s in range(args.rand_seeds):
        m50, m5095 = eval_ap_all(P[f"rand_{K}_s{s}"], gts)
        m50s.append(m50); m5095s.append(m5095)
        srs.append(small_recall(P[f"rand_{K}_s{s}"], gts, small_thr, args.op_conf))
    cavg = np.mean([crops_used[f"rand_{K}_s{s}"] for s in range(args.rand_seeds)]) / N
    print(f"  {'random-K':16s}{K:>3d}{np.mean(m50s):9.4f}{np.mean(m5095s):10.4f}{np.mean(srs):10.4f}{cavg:7.1f}"
          f"   (±{np.std(m50s):.4f} mAP50, {args.rand_seeds} seed)")
    res[("rand", K)] = (float(np.mean(m50s)), float(np.mean(m5095s)), float(np.mean(srs)))
    res[("g06c", K)] = row(f"g06c_{K}", "luoi-0.6 lay K o", str(K))

print("\n  DOC KET QUA (moi K, so mAP50):")
for K in KS:
    rl = res[("rl", K)][0]
    rivals = {"topK": res[("topk", K)][0], "random": res[("rand", K)][0],
              "center": res[("cent", K)][0], "g06-K": res[("g06c", K)][0]}
    best_name = max(rivals, key=rivals.get)
    verdict = "RL THANG het baseline" if rl > max(rivals.values()) else f"RL {'hoa' if abs(rl-rivals[best_name])<0.003 else 'THUA'} {best_name} ({rivals[best_name]:.4f})"
    print(f"    K={K}: RL={rl:.4f} | {verdict}")
print("\n  * RL thang topK+random o K nho => 'agent hoc duoc nhin dau' co bang chung so.")
print("  * RL~topK: fallback = RL hoc JOINT (dat+co+chong+dung) khong can tay chinh separation/fraction.")
print("  * so crop = so THUC (RL co the giu < K). Latency do rieng neu can (prefix trick khong dung cho ms).")
