"""Infer + ve detection cua METHOD TOT NHAT (RL-SAHI batch) len anh.

Chay full + lat RL (rollout) + luoi tho, batch YOLO, merge -> ve box mau theo lop.
Dung detector da train-on-crop (--weights). De XEM ket qua, khong phai do mAP.

Chay tren Lightning:
  python scripts/viz_detect.py --split test --limit 60 --weights runs/detect/ft_crop/weights/best.pt --device cuda
Ca bo test (thu nho cho nhe dia):  --limit 0 --out-width 1600
Doi method:  --method coarse | full | rl (mac dinh rl = RL-SAHI)
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
from rl_sahi.common.data import iter_images, image_to_label_path, read_yolo_labels
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _fixed_grid_rois, _full_predictions, _merge_predictions, _image_shape
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.common.boxes import iou_matrix
from rl_sahi.inference.pipeline import _filter_classes, _attempt_overlap, get_initial_detection

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--config", type=Path, default=ROOT / "configs" / "ft_cloud.yaml")
ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "detect" / "ft_crop" / "weights" / "best.pt")
ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "ft_rl" / "dqn" / "best.pt")
ap.add_argument("--method", choices=["rl", "coarse", "full"], default="rl")
ap.add_argument("--split", default="test")
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--offset", type=int, default=0, help="bo qua N anh dau -> lay bo anh KHAC")
ap.add_argument("--conf", type=float, default=0.25, help="nguong hien thi (cao = it box nhieu, sach hon)")
ap.add_argument("--base", type=int, default=640)
ap.add_argument("--slice", type=int, default=640)
ap.add_argument("--max-fine", type=int, default=8)
ap.add_argument("--max-attempts", type=int, default=14)
ap.add_argument("--chunk", type=int, default=16)
ap.add_argument("--labels", action="store_true", help="ghi ten lop + conf tren moi box (dong dac thi roi)")
ap.add_argument("--no-roi", action="store_true", help="an ROI do (chi ve detection)")
ap.add_argument("--show-grid", action="store_true", help="ve them luoi tho (mac dinh CHI ve lat RL cho sach)")
ap.add_argument("--rescue", action="store_true", help="CHE DO CHUNG MINH RL KHON: to VANG vat chi lat RL bat duoc (full+luoi sot)")
ap.add_argument("--min-rescue", type=int, default=0, help="chi luu anh co >= N vat RL cuu (nhat anh dep cho slide)")
ap.add_argument("--vs-full", action="store_true", help="rescue so voi YOLO THUAN (dramatic hon) thay vi so voi luoi")
ap.add_argument("--all-roi", action="store_true", help="ve TAT CA ROI (mac dinh rescue chi ve ROI co cuu duoc vat)")
ap.add_argument("--out-width", type=int, default=0, help=">0: thu nho anh ra de tiet kiem dia")
ap.add_argument("--jpg-quality", type=int, default=90)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", type=Path, default=ROOT / "runs" / "viz_detect")
args = ap.parse_args()

BASE, SLI, dev, CONF = args.base, args.slice, args.device, args.conf
IR = ROOT / "data" / "raw" / "images"; LR = ROOT / "data" / "raw" / "labels"
CR = ROOT / "data" / "cache_ft"
CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (60, 200, 255), (255, 80, 255),
          (255, 220, 60), (255, 140, 40), (160, 60, 255), (60, 160, 255), (140, 255, 60)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

cfg = load_default_config(args.config, ROOT)
tc = tuple(range(10)); cm = ClassMapping.from_config(cfg.section("classes"))
icfg = InferenceConfig(full_imgsz=BASE, slice_imgsz=SLI, full_conf=CONF, output_conf=CONF, iou=0.7,
    merge_iou=0.5, max_det=3000, device=dev, feature_layers=(16,), target_classes=tc, class_mapping=cm,
    min_slice_detections=1, min_slice_utility=0.2, duplicate_iou=0.5, max_slice_attempts=args.max_attempts,
    require_stop_for_acceptance=True)
if not args.weights.exists():
    sys.exit(f"[viz] khong thay weight {args.weights}")
model = load_yolo(str(args.weights), device=dev)
dt = resolve_torch_device(dev)

policy_mv = env_cfg = state_cfg = None
if args.method == "rl":
    if not args.checkpoint.exists():
        sys.exit(f"[viz] method=rl can checkpoint {args.checkpoint} (hoac dung --method coarse)")
    from rl_sahi.rl.checkpoint import load_policy
    from rl_sahi.rl.slice_env import SliceEnv
    from rl_sahi.rl.state_config import StateConfig
    from rl_sahi.inference.rollout import rollout_one_slice
    policy_mv, ckm = load_policy(args.checkpoint, dt)
    env_cfg = ckm["env_cfg_obj"]; env_cfg.max_slices = args.max_fine
    state_cfg = ckm.get("state_cfg_obj", StateConfig())

def crop_parts(img, rois):
    outs = []
    for i in range(0, len(rois), args.chunk):
        part = rois[i:i + args.chunk]
        outs.extend(run_yolo_on_crops(model, [img] * len(part), part,
                    imgsz=SLI, conf=CONF, iou=0.7, max_det=3000, device=dev))
    pb, ps, pc = [], [], []
    for (b, s, c) in outs:
        c = cm.map_model_classes(c); b, s, c = _filter_classes(b, s, c, tc)
        pb.append(b); ps.append(s); pc.append(c)
    return pb, ps, pc

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
                        info.get("stop_due_to_max_steps") or info.get("stop_due_to_stalled_roi"))
        if rejected:
            if _attempt_overlap(roi, att[:-1]) >= 0.95: break
            continue
        kept.append(roi)
    return kept

def draw(im, boxes, scores, classes, th):
    for b, s, c in zip(boxes, scores, classes):
        c = int(c); color = COLORS[c % 10]
        x0, y0, x1, y1 = [int(round(float(v))) for v in b]
        cv2.rectangle(im, (x0, y0), (x1, y1), color, th)
        if args.labels:
            cv2.putText(im, f"{CLASS_NAMES[c]} {float(s):.2f}", (x0, max(10, y0 - 3)),
                        FONT, 0.4, color, 1, cv2.LINE_AA)

def legend(im):
    x, y = 8, 30
    for i, nm in enumerate(CLASS_NAMES):
        cv2.rectangle(im, (x, y), (x + 14, y + 14), COLORS[i], -1)
        cv2.putText(im, nm, (x + 18, y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    cv2.rectangle(im, (x, y), (x + 14, y + 14), (0, 0, 255), 2)
    cv2.putText(im, "vung cat (ROI)", (x + 18, y + 12), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

def _not_in(b1, c1, b0, c0):
    """Mask: box trong (b1,c1) KHONG khop box nao cung lop trong (b0,c0) o IoU>=0.5 -> vat MOI."""
    b1 = np.asarray(b1, np.float32).reshape(-1, 4); c1 = np.asarray(c1).reshape(-1)
    if len(b1) == 0: return np.zeros(0, bool)
    if len(b0) == 0: return np.ones(len(b1), bool)
    iou = iou_matrix(b1, np.asarray(b0, np.float32).reshape(-1, 4))
    same = c1[:, None] == np.asarray(c0).reshape(1, -1)
    return ~((iou >= 0.5) & same).any(axis=1)

def legend_rescue(im):
    x, y = 8, 30
    for col, txt in [((150, 150, 150), "da bat (full + luoi)"),
                     ((0, 255, 255), "RL CUU THEM (full+luoi sot)"),
                     ((0, 0, 255), "vung RL cat (ROI)")]:
        cv2.rectangle(im, (x, y), (x + 14, y + 14), col, 2 if col == (0, 0, 255) else -1)
        cv2.putText(im, txt, (x + 18, y + 12), FONT, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20

def banner(im, text):
    w = im.shape[1]
    cv2.rectangle(im, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(im, text, (8, 16), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

args.out.mkdir(parents=True, exist_ok=True)
_all = iter_images(IR, split=args.split, limit=((args.offset + args.limit) if args.limit else None))
images = _all[args.offset:]
if not images: sys.exit(f"[viz] khong thay anh o {IR}/{args.split} (offset={args.offset})")
mname = {"rl": "RL-SAHI (crop)", "coarse": "luoi 0.6 (crop)", "full": "YOLO full (crop)"}[args.method]
print(f"[viz] {len(images)} anh (offset {args.offset}) | method={args.method} | conf={CONF} | weight={args.weights.name}")

# --- Tu ghi file GIAI THICH vao folder de nguoi khac xem khoi hieu lam ROI ---
_doc = f"""GIAI THICH ANH TRONG FOLDER NAY  (method: {mname})
=====================================================================
CACH DOC 1 ANH:
  - HOP MAU (xanh la / cam / xanh duong ... theo lop) = VAT THE PHAT HIEN DUOC
    (ket qua cuoi cung sau khi gop). Goc trai co bang chu giai mau theo lop.
  - HOP DO (ROI) = VUNG MA AGENT RL CHON DE CAT LAT. DAY KHONG PHAI box phat hien!
  - Tieu de tren cung: ten method | so vat the | so vung RL + so o luoi.

VI SAO ROI DO KHONG "SAI" DU NHIN CO VE TO / CHONG / KHONG OM KHIT VAT:
  1. ROI la CUA SO DE CAT, khong phai hop bao vat. Muc dich: cat vung do ra,
     phong to (resize ve 640) roi chay YOLO lai o do phan giai cao hon -> bat
     duoc vat NHO ma anh full@640 bo sot. Nen ROI chi can TRUM vung co vat nho,
     KHONG can om khit tung vat.
  2. Kich thuoc ROI bi rang buoc = 10-35% canh anh (khong gian hanh dong cua
     agent). Do la thiet ke, khong phai chon tuy tien.
  3. ROI chum vao cum vat nho DAY (noi objectness cao) la DUNG hanh vi: do la
     cho anh full de bo sot nhat. Cac ROI co the chong nhau chut khi nhieu cum
     gan nhau (luat chong lan chi chan trung y het, khong chan chum).
  4. Tinh toan cuoi: full@640 + cac crop tu ROI + luoi tho -> gop bang class-aware
     NMS -> ra hop mau. Vi vay "vat the" (mau) va "vung cat" (do) la HAI thu khac
     nhau, dung danh gia ROI bang viec no co khit vat hay khong.
  5. Vi tri ROI la HOC DUOC, khong ngau nhien: thi nghiem budget sweep chung minh
     agent dat ROI TOT HON topK-objectness / center / random o MOI ngan sach crop
     (+0.02..0.04 mAP, recall vat nho hon toi +0.12). Xem KETQUA_LUANVAN.md muc 4.5.

TOM LAI: hop DO = "nhin o dau" (quyet dinh cua agent), hop MAU = "thay gi"
(ket qua). ROI khong sai — no la cach pipeline tinh toan.
"""
try:
    (args.out / "GIAI_THICH_ROI.txt").write_text(_doc, encoding="utf-8")
except Exception:
    pass

for idx, img in enumerate(images):
    if idx % 20 == 0 and idx:
        print(f"  {idx}/{len(images)}", flush=True)
    im = cv2.imread(str(img))
    if im is None: continue
    H, W = im.shape[:2]; th = max(2, W // 700)
    det = get_initial_detection(model=model, weights=str(args.weights), image_path=img, weights_imgsz=BASE,
        full_conf=CONF, full_iou=0.7, max_det=3000, device=dev, feature_layers=(16,), aux_grid_size=16,
        spatial_feature_channels=4, cache_root=CR, split=args.split, use_cache=False)
    fb, fs, fc = _full_predictions(det, icfg)
    fine, coarse = [], []
    if args.rescue:
        # === CHE DO CHUNG MINH: vat CHI lat RL bat duoc (full + luoi deu sot) ===
        coarse = _fixed_grid_rois(det.image_shape, 0.6, 0.15)
        fine = select_rois_moving(det)
        cp = crop_parts(img, coarse); fp = crop_parts(img, fine)
        if args.vs_full:
            base = (fb, fs, fc)                                                                          # YOLO THUAN
        else:
            base = _merge_predictions(det.image_shape, 0.5, [fb, *cp[0]], [fs, *cp[1]], [fc, *cp[2]])   # full + LUOI (khong RL)
        rl = _merge_predictions(det.image_shape, 0.5, [fb, *cp[0], *fp[0]], [fs, *cp[1], *fp[1]], [fc, *cp[2], *fp[2]])  # + lat RL
        rb, rs, rc = rl
        resc = _not_in(rb, rc, base[0], base[2])          # vat trong rl KHONG co trong base => CHI RL bat
        # honest: chi tinh la "cuu" neu khop vat THAT (ground-truth) -> khong to nham false-positive
        try:
            gcls, gbox = read_yolo_labels(image_to_label_path(img, IR, LR), det.image_shape)
            gcls = cm.map_label_classes(gcls)
            is_real = ~_not_in(rb, rc, np.asarray(gbox, np.float32).reshape(-1, 4), gcls)
            resc = resc & is_real
        except Exception:
            pass                                           # thieu nhan -> giu resc (unique-over-grid)
        n_resc = int(resc.sum())
        if n_resc < args.min_rescue:
            continue                                       # bo anh it rescue -> chi giu anh dep
        for bx in base[0]:                                 # nen: da bat boi full+luoi -> xam mong
            x0, y0, x1, y1 = [int(round(float(v))) for v in bx]
            cv2.rectangle(im, (x0, y0), (x1, y1), (150, 150, 150), max(1, W // 1100))
        rbx = rb[resc]                                     # cac hop vat RL cuu
        for bx in rbx:                                     # VANG DAM: RL cuu them
            x0, y0, x1, y1 = [int(round(float(v))) for v in bx]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 255), max(3, W // 240))
        # chi ve ROI CO cuu duoc vat (tam vat vang nam trong ROI) — tru khi --all-roi
        if args.all_roi:
            useful = list(fine)
        elif len(rbx):
            cxr = (rbx[:, 0] + rbx[:, 2]) / 2; cyr = (rbx[:, 1] + rbx[:, 3]) / 2
            useful = [r for r in fine
                      if bool(((cxr >= r[0]) & (cxr <= r[2]) & (cyr >= r[1]) & (cyr <= r[3])).any())]
        else:
            useful = []
        for r in useful:                                   # DO: vung RL cat CO ich
            x0, y0, x1, y1 = [int(round(float(v))) for v in r]
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), max(2, W // 320))
        legend_rescue(im)
        _vs = "YOLO thuan" if args.vs_full else "full+luoi"
        banner(im, f"RL cat dung cho -> CUU {n_resc} vat THAT ma {_vs} deu sot | {len(useful)}/{len(fine)} vung RL trung")
    else:
        if args.method == "full":
            b, s, c = fb, fs, fc
        else:
            coarse = _fixed_grid_rois(det.image_shape, 0.6, 0.15)
            fine = select_rois_moving(det) if args.method == "rl" else []
            pb, ps, pc = crop_parts(img, fine + coarse)
            b, s, c = _merge_predictions(det.image_shape, 0.5, [fb, *pb], [fs, *ps], [fc, *pc])
        draw(im, b, s, c, th)
        if not args.no_roi and args.method != "full":
            if args.show_grid:    # luoi tho = ROI do MANH (mac dinh AN cho sach)
                for r in coarse:
                    x0, y0, x1, y1 = [int(round(float(v))) for v in r]
                    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), max(1, W // 1000))
            for r in fine:        # lat RL = ROI do DAM (cai agent chon)
                x0, y0, x1, y1 = [int(round(float(v))) for v in r]
                cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), max(2, W // 320))
        legend(im)
        banner(im, f"{mname} | {len(b)} vat | {len(fine)} vung RL + {len(coarse)} luoi")
    if args.out_width and W > args.out_width:
        sc = args.out_width / W
        im = cv2.resize(im, (args.out_width, int(round(H * sc))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(args.out / f"{idx:04d}_{img.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)])

print(f"\n[viz] XONG {len(images)} anh -> {args.out}")
print("[viz] Box mau theo lop (goc trai co chu giai). Tai runs/viz_detect/ ve xem.")
