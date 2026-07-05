"""Huấn luyện tác nhân TILE-CODING (linear Q) trên CÙNG MDP di chuyển của RL-SAHI.

Đối chứng với Deep DQN: giữ nguyên SliceEnv (action dịch/zoom/stop), reward hard-region,
vòng multi-slice và tiêu chí chấp nhận; chỉ thay Q-network sâu bằng tile coding + Q tuyến tính.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.actions import NUM_ACTIONS
from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.cache import detection_cache_metadata
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.tile_coding import TileQAgent
from rl_sahi.rl.tile_state import NUM_TILE_FEATURES, tile_features
from rl_sahi.rl.trainer import _attempt_overlap, _max_slice_attempts, _stack_rois


def _lin(step, total, start, end):
    frac = min(float(step) / max(total, 1), 1.0)
    return start + frac * (end - start)


def _run_episode(agent, det, hard, env_cfg, state_cfg, tc, cm, epsilon, guide_prob, rng, learn):
    """Chạy 1 ảnh: vòng multi-slice. Trả (covered, total, accepted)."""
    previous, attempted = [], []
    prev_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
    accepted = 0
    max_attempts = _max_slice_attempts(env_cfg, None)
    for _ in range(max_attempts):
        if accepted >= env_cfg.max_slices:
            break
        env = SliceEnv(det, hard, env_cfg=env_cfg, state_cfg=state_cfg,
                       previous_rois=_stack_rois(attempted), overlap_rois=_stack_rois(previous),
                       previous_covered=prev_covered, target_classes=tc, class_mapping=cm)
        env.reset()
        f = tile_features(env)
        info = {}
        for _step in range(env_cfg.max_steps + 1):
            valid = env.valid_actions()
            a = agent.act(f, valid, epsilon, rng)
            if learn and guide_prob > 0.0 and rng.random() < guide_prob:
                ga = int(env.guided_action())
                if ga < len(valid) and valid[ga]:
                    a = ga
            result = env.step(a)
            f2 = tile_features(env)
            if learn:
                agent.update(f, a, result.reward, f2, result.done, env.valid_actions())
            f = f2
            info = result.info
            if result.done:
                break
        new_hits = int((env.covered & ~prev_covered).sum())
        repeat = _attempt_overlap(env.roi, attempted)
        attempted.append(env.roi.copy())
        rejected = (info.get("stop_due_to_old_overlap", False) or info.get("stop_due_to_attempted_overlap", False)
                    or info.get("stop_due_to_max_steps", False) or info.get("stop_due_to_stalled_roi", False)
                    or new_hits < env_cfg.min_new_hits_to_accept)
        if rejected:
            if repeat >= 0.95:
                break
            continue
        previous.append(env.roi.copy())
        prev_covered = env.covered.copy()
        accepted += 1
        if prev_covered.all() and len(prev_covered) > 0:
            break
    return int(prev_covered.sum()), int(len(prev_covered)), accepted


def _eval_recall(agent, dataset, env_cfg, state_cfg, tc, cm, episodes, rng):
    cov = tot = sl = 0
    n = min(episodes, len(dataset))
    for _ in range(n):
        det, hard = dataset.random_episode()
        c, t, s = _run_episode(agent, det, hard, env_cfg, state_cfg, tc, cm, 0.0, 0.0, rng, learn=False)
        cov += c; tot += t; sl += s
    return cov / max(tot, 1), sl / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train tile-coding linear-Q agent on the RL-SAHI movement MDP.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--num-tilings", type=int, default=8)
    ap.add_argument("--tiles-per-dim", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "tile" / "agent.pkl")
    args = ap.parse_args()

    cfg = load_default_config(args.config, ROOT)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in cfg.section("infer").get("target_classes", (0, 2, 3, 5, 8, 9)))
    cm = ClassMapping.from_config(cfg.section("classes"))
    dc, sc = cfg.section("detect"), cfg.section("state")
    meta = detection_cache_metadata(weights=cfg.path_value("weights"), imgsz=int(dc["imgsz"]),
        conf=float(dc["conf"]), iou=float(dc["iou"]), max_det=int(dc["max_det"]),
        feature_layers=cfg.feature_layers("detect"), aux_grid_size=int(sc["grid_size"]),
        spatial_feature_channels=int(sc.get("spatial_feature_channels", 4)))

    train_ds = CachedEpisodeDataset(cfg.path_value("image_root"), cfg.path_value("cache_root"),
                                    args.split, limit=args.limit, preload=True, detection_metadata=meta)
    try:
        val_ds = CachedEpisodeDataset(cfg.path_value("image_root"), cfg.path_value("cache_root"),
                                      "val", limit=args.limit, preload=True, detection_metadata=meta)
    except FileNotFoundError:
        val_ds = None

    rng = np.random.default_rng(42)
    agent = TileQAgent(NUM_ACTIONS, num_tilings=args.num_tilings, tiles_per_dim=args.tiles_per_dim,
                       alpha=args.alpha, gamma=float(cfg.section("train").get("gamma", 0.95)))
    print(f"[tile] features={NUM_TILE_FEATURES} tilings={args.num_tilings} tiles/dim={args.tiles_per_dim} "
          f"alpha={args.alpha} episodes={args.episodes}")

    eps_decay = max(int(args.episodes * 0.6), 1)
    best_recall = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.episodes + 1):
        det, hard = train_ds.random_episode()
        epsilon = _lin(ep, eps_decay, 1.0, 0.05)
        guide = _lin(ep, eps_decay, 0.25, 0.05)
        _run_episode(agent, det, hard, env_cfg, state_cfg, tc, cm, epsilon, guide, rng, learn=True)

        if ep == 1 or ep % max(args.episodes // 20, 1) == 0 or ep == args.episodes:
            if val_ds is not None:
                recall, avg_sl = _eval_recall(agent, val_ds, env_cfg, state_cfg, tc, cm, 60, rng)
                print(f"[tile] ep={ep}/{args.episodes} eps={epsilon:.3f} val_recall={recall:.4f} "
                      f"avg_slices={avg_sl:.2f} iht_used={len(agent.iht.dictionary)}")
                if recall > best_recall:
                    best_recall = recall
                    agent.save(args.out)
    # luôn lưu bản cuối nếu chưa có best
    if best_recall < 0:
        agent.save(args.out)
    print(f"[tile] best val_recall={best_recall:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
