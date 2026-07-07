"""ONE-SHOT action space (nhanh oneshot-rl): bo rollout di chuyen.

Thay vi agent re ROI tung buoc, ta:
  1) SINH ung vien 1 lan tu dinh objectness (propose_regions) — khong rollout.
  2) Voi MOI ung vien, agent chon 1 action { DROP | KEEP | ZOOM_1.5 | ZOOM_2 }
     (action_to_roi) = dung y "zoom nhieu/it/khong/bo".
  3) Gom cac ROI giu lai -> batch YOLO 1 luot (o script benchmark/train).

Reward (train bandit 1 buoc) = crop-outcome (TP that doi chieu GT) — giu RL lam dich.
"""
from __future__ import annotations

import numpy as np

from rl_sahi.common.box_transforms import box_from_center

# --- Action space nho: dung y user "zoom nhieu/it/khong zoom/bo" ---
DROP, KEEP, ZOOM_1_5, ZOOM_2 = 0, 1, 2, 3
NUM_ACTIONS = 4
ACTION_NAMES = ("DROP", "KEEP", "ZOOM_1.5", "ZOOM_2")
# zoom "nhieu" = crop NHO hon (vat phong to hon khi resize ve 640). factor = do phong dai.
_ZOOM_FACTOR = {KEEP: 1.0, ZOOM_1_5: 1.5, ZOOM_2: 2.0}


def objectness_grid(det) -> np.ndarray:
    """Heatmap objectness 2D (max qua kenh neu 3D). Nguon 'noi nghi co vat'."""
    obj = np.nan_to_num(np.asarray(det.objectness_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if obj.ndim == 3:
        return obj.max(axis=0).astype(np.float32)
    if obj.ndim == 2:
        return obj.astype(np.float32)
    return np.zeros((0, 0), dtype=np.float32)


def propose_regions(det, k: int = 12, separation: int = 2) -> list[dict]:
    """Sinh toi da K ung vien tu K dinh objectness (1 lan, khong rollout).
    Moi ung vien: {center:(cx,cy) px anh goc, obj:gia-tri-dinh, cell:(gy,gx)}."""
    grid = objectness_grid(det)
    if grid.size == 0 or int(k) <= 0:
        return []
    h, w = det.image_shape
    gy, gx = grid.shape
    work = grid.astype(np.float32).copy()
    sep = max(int(separation), 0)
    out: list[dict] = []
    for _ in range(int(k)):
        idx = int(np.argmax(work))
        val = float(work.flat[idx])
        if not np.isfinite(val) or val <= 0.0:
            break
        y, x = np.unravel_index(idx, work.shape)
        cx = (float(x) + 0.5) * w / max(gx, 1)
        cy = (float(y) + 0.5) * h / max(gy, 1)
        out.append({"center": (float(cx), float(cy)), "obj": val, "cell": (int(y), int(x))})
        y0, y1 = max(0, int(y) - sep), min(gy, int(y) + sep + 1)
        x0, x1 = max(0, int(x) - sep), min(gx, int(x) + sep + 1)
        work[y0:y1, x0:x1] = -np.inf
    return out


def action_to_roi(region: dict, action: int, image_shape: tuple[int, int],
                  base_frac: float = 0.35, min_frac: float = 0.08) -> np.ndarray | None:
    """DROP -> None (bo vung). KEEP/ZOOM -> box vuong tam vung, canh = base/factor.
    Zoom cang nhieu -> crop cang nho -> vat phong to cang manh khi resize ve slice_imgsz."""
    if int(action) == DROP:
        return None
    h, w = image_shape
    factor = _ZOOM_FACTOR.get(int(action), 1.0)
    side = max(min(h, w) * float(min_frac), min(h, w) * float(base_frac) / float(factor))
    cx, cy = region["center"]
    return box_from_center(float(cx), float(cy), float(side), image_shape)


def region_local_state(det, region: dict, grid: np.ndarray | None = None, patch: int = 3) -> np.ndarray:
    """Vector state NHO cho policy (RL): cua so patch x patch quanh cell tu objectness
    + vai scalar (dinh objectness, tam chuan hoa). Nho hon nhieu so voi state 5660 cu."""
    if grid is None:
        grid = objectness_grid(det)
    gy, gx = grid.shape if grid.size else (16, 16)
    y, x = region["cell"]
    half = int(patch) // 2
    if grid.size:
        padded = np.pad(grid, half, mode="constant", constant_values=0.0)
        win = padded[y:y + 2 * half + 1, x:x + 2 * half + 1].reshape(-1).astype(np.float32)
    else:
        win = np.zeros((patch * patch,), dtype=np.float32)
    h, w = det.image_shape
    cx, cy = region["center"]
    peak = float(grid[y, x]) if grid.size else 0.0
    scalars = np.array([float(region.get("obj", peak)), cx / max(w, 1), cy / max(h, 1), peak], dtype=np.float32)
    return np.concatenate([win, scalars]).astype(np.float32)


def region_state_dim(patch: int = 3) -> int:
    return int(patch) * int(patch) + 4
