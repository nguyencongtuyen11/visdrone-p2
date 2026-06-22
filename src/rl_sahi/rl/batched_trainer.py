from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_sahi.common.actions import ACTION_NAMES, Action
from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import configure_torch_runtime
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import BenchmarkConfig, evaluate_rl_sahi_policy
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.checkpoint import save_checkpoint
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import PrioritizedReplayBuffer, ReplayBuffer
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_layout import state_layout_from_detection
from rl_sahi.rl.crop_outcome import CropOutcome
from rl_sahi.rl.trainer import (
    TrainConfig,
    _attempt_overlap,
    _max_slice_attempts,
    _stack_rois,
    _terminal_reward_with_crop_outcome,
    benchmark_score,
    epsilon_by_step,
    evaluate_policy,
    guide_prob_by_step,
    make_crop_outcome_evaluator,
    optimize,
    soft_update,
)


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_resume_checkpoint(
    path: Path,
    policy: QNetwork,
    target_net: QNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    state_dim: int,
    train_cfg: TrainConfig,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    layout,
    detection_metadata: dict[str, Any] | None,
    global_step: int,
    episodes_started: int,
    episodes_completed: int,
    best_score: float,
    best_reward: float,
    optimizer_steps: int,
    scheduler_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_type": "rl_sahi_train_resume",
            "version": 1,
            "policy": policy.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "replay": replay,
            "state_dim": int(state_dim),
            "state_layout": asdict(layout) if layout is not None else None,
            "train_cfg": asdict(train_cfg),
            "env_cfg": asdict(env_cfg),
            "state_cfg": asdict(state_cfg),
            "detection_metadata": detection_metadata,
            "actions": {int(k): v for k, v in ACTION_NAMES.items()},
            "global_step": int(global_step),
            "episodes_started": int(episodes_started),
            "episodes_completed": int(episodes_completed),
            "best_score": float(best_score),
            "best_reward": float(best_reward),
            "optimizer_steps": int(optimizer_steps),
            "scheduler_steps": int(scheduler_steps),
            "rng_state": _rng_state(),
        },
        tmp_path,
    )
    tmp_path.replace(path)


@dataclass
class EnvWorker:
    episode: int
    det: Any
    hard: Any
    previous_rois: list[np.ndarray]
    attempted_rois: list[np.ndarray]
    previous_covered: np.ndarray
    full_boxes: np.ndarray
    full_scores: np.ndarray
    full_classes: np.ndarray
    slice_boxes_all: list[np.ndarray]
    slice_scores_all: list[np.ndarray]
    slice_classes_all: list[np.ndarray]
    accepted_new_count: int
    current_max_slices: int
    current_max_attempts: int
    slice_idx: int
    attempt_idx: int
    env: SliceEnv
    state: np.ndarray
    n_step_buffer: list
    total_reward: float
    total_steps: int
    accepted_slices: int
    rejected_slices: int
    crop_new_detection_gain_total: int
    crop_new_detection_utility_total: float
    crop_tp_gain_total: int
    crop_fp_gain_total: int
    crop_outcome_reward_total: float
    losses: list[float]
    info: dict
    done: bool

def batched_train_dqn(
    image_root: Path, cache_root: Path, split: str, out_dir: Path, cfg: TrainConfig, env_cfg: EnvConfig, state_cfg: StateConfig,
    limit: int | None = None, device_name: str | None = None, detection_metadata: dict[str, Any] | None = None,
    target_classes: tuple[int, ...] = (), class_mapping: ClassMapping | None = None, label_root: Path | None = None,
    eval_weights: Path | None = None, infer_cfg: InferenceConfig | None = None, bench_cfg: BenchmarkConfig | None = None,
    eval_use_cache: bool = True,
) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = CachedEpisodeDataset(image_root=image_root, cache_root=cache_root, split=split, limit=limit, preload=cfg.preload_cache, detection_metadata=detection_metadata)
    
    val_dataset = None
    if getattr(cfg, "val_split", ""):
        try:
            val_dataset = CachedEpisodeDataset(image_root=image_root, cache_root=cache_root, split=cfg.val_split, limit=limit, preload=cfg.preload_cache, detection_metadata=detection_metadata)
        except FileNotFoundError as exc:
            print(f"[batched_train] validation disabled: {exc}")

    inference_model = None
    benchmark_model = None
    benchmark_images: list[Path] = []
    if getattr(cfg, "eval_benchmark_images", 0) > 0:
        if eval_weights is None or label_root is None or infer_cfg is None or bench_cfg is None:
            raise RuntimeError("Benchmark validation requires weights, labels, inference config, and benchmark config")
        benchmark_images = iter_images(image_root, split=cfg.val_split, limit=cfg.eval_benchmark_images)
        if not benchmark_images:
            raise FileNotFoundError(f"No images found for benchmark validation split '{cfg.val_split}'")
        inference_model = load_yolo(eval_weights, device=infer_cfg.device)
        benchmark_model = inference_model
    elif cfg.use_crop_outcome_reward:
        if eval_weights is None or infer_cfg is None:
            raise RuntimeError("Crop outcome reward requires eval_weights and inference config")
        inference_model = load_yolo(eval_weights, device=infer_cfg.device)
    crop_evaluator = make_crop_outcome_evaluator(
        model=inference_model,
        image_root=image_root,
        label_root=label_root,
        cache_root=cache_root,
        split=split,
        cfg=cfg,
        infer_cfg=infer_cfg,
        bench_cfg=bench_cfg,
        eval_weights=eval_weights,
        eval_use_cache=eval_use_cache,
    )

    probe_det = dataset.first_detection()
    probe_env = SliceEnv(probe_det, None, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
    state_dim = int(probe_env.reset().shape[0])
    layout = state_layout_from_detection(probe_det, state_cfg)

    device = configure_torch_runtime(device_name)
    policy = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net.load_state_dict(policy.state_dict())
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.episodes, eta_min=1e-6)

    if cfg.use_per:
        replay: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(capacity=cfg.replay_size, alpha=cfg.per_alpha, beta_start=cfg.per_beta_start, beta_frames=cfg.per_beta_frames)
    else:
        replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    resume_path = out_dir / "resume.pt"

    best_score = -float("inf")
    best_reward = -float("inf")
    global_step = 0
    num_envs = getattr(cfg, "num_envs", 8)
    episodes_started = 0
    episodes_completed = 0
    optimizer_steps = 0
    scheduler_steps = 0
    resume_loaded = False

    if bool(getattr(cfg, "resume", True)) and resume_path.exists():
        resume_data = _torch_load_checkpoint(resume_path)
        if int(resume_data.get("state_dim", -1)) != state_dim:
            raise RuntimeError(
                f"Resume checkpoint state_dim={resume_data.get('state_dim')} does not match current state_dim={state_dim}. "
                f"Delete {resume_path} or run with --no-resume."
            )
        resume_actions = resume_data.get("actions")
        if isinstance(resume_actions, dict) and len(resume_actions) != len(ACTION_NAMES):
            raise RuntimeError(
                f"Resume checkpoint has {len(resume_actions)} actions, current code has {len(ACTION_NAMES)}. "
                f"Delete {resume_path} or run with --no-resume."
            )
        policy.load_state_dict(resume_data["policy"])
        target_net.load_state_dict(resume_data["target_net"])
        optimizer.load_state_dict(resume_data["optimizer"])
        _optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(resume_data["scheduler"])
        replay = resume_data["replay"]
        global_step = int(resume_data.get("global_step", 0))
        episodes_completed = int(resume_data.get("episodes_completed", 0))
        episodes_started = episodes_completed
        best_score = float(resume_data.get("best_score", best_score))
        best_reward = float(resume_data.get("best_reward", best_reward))
        optimizer_steps = int(resume_data.get("optimizer_steps", 0))
        scheduler_steps = int(resume_data.get("scheduler_steps", 0))
        _restore_rng_state(resume_data.get("rng_state", {}))
        resume_loaded = True
        print(
            f"[batched_train] resumed {resume_path} "
            f"(completed={episodes_completed}, global_step={global_step}, replay={len(replay)})"
        )

    print(f"[batched_train] num_envs={num_envs}, episodes={cfg.episodes}")
    if episodes_completed >= cfg.episodes:
        print(f"[batched_train] resume checkpoint already completed {episodes_completed}/{cfg.episodes} episodes")

    def reset_worker(episode: int) -> EnvWorker:
        det, hard = dataset.random_episode()
        current_max_slices = env_cfg.max_slices
        if cfg.use_curriculum:
            curriculum_frac = min(float(global_step) / max(cfg.curriculum_steps, 1), 1.0)
            current_max_slices = max(1, int(env_cfg.max_slices * curriculum_frac))
        previous_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
        if crop_evaluator is not None:
            full_boxes, full_scores, full_classes = crop_evaluator.full_predictions(det)
            accepted_new_count = crop_evaluator.initial_new_count(
                full_boxes,
                full_scores,
                full_classes,
                det.image_shape,
            )
        else:
            full_boxes = np.zeros((0, 4), dtype=np.float32)
            full_scores = np.zeros((0,), dtype=np.float32)
            full_classes = np.zeros((0,), dtype=np.float32)
            accepted_new_count = 0
        current_max_attempts = _max_slice_attempts(env_cfg, infer_cfg, current_max_slices)
        env = SliceEnv(
            det,
            hard,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=np.zeros((0, 4), dtype=np.float32),
            overlap_rois=np.zeros((0, 4), dtype=np.float32),
            previous_covered=previous_covered,
            target_classes=target_classes,
            class_mapping=class_mapping,
        )
        return EnvWorker(
            episode=episode, det=det, hard=hard, previous_rois=[], attempted_rois=[], previous_covered=previous_covered,
            full_boxes=full_boxes, full_scores=full_scores, full_classes=full_classes,
            slice_boxes_all=[], slice_scores_all=[], slice_classes_all=[],
            accepted_new_count=accepted_new_count, current_max_slices=current_max_slices,
            current_max_attempts=current_max_attempts, slice_idx=0, attempt_idx=0,
            env=env, state=env.reset(), n_step_buffer=[], total_reward=0.0, total_steps=0,
            accepted_slices=0, rejected_slices=0, crop_new_detection_gain_total=0,
            crop_new_detection_utility_total=0.0,
            crop_tp_gain_total=0, crop_fp_gain_total=0, crop_outcome_reward_total=0.0,
            losses=[], info={}, done=False
        )

    active_workers = []
    for _ in range(num_envs):
        if episodes_started < cfg.episodes:
            episodes_started += 1
            active_workers.append(reset_worker(episodes_started))

    append_log = resume_loaded and log_path.exists()
    with log_path.open("a" if append_log else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "global_step",
                "episodes_completed",
                "reward",
                "loss",
                "epsilon",
                "steps",
                "slices",
                "current_max_slices",
                "attempts",
                "rejected_slices",
                "covered",
                "hard_total",
                "crop_new_detection_gain",
                "crop_new_detection_utility",
                "crop_tp_gain",
                "crop_fp_gain",
                "crop_outcome_reward",
                "val_recall",
                "val_slices",
                "val_score",
                "val_mAP50",
                "val_small_recall",
                "val_fp_per_image",
                "val_crops",
                "val_benchmark_score",
            ],
        )
        if not append_log:
            writer.writeheader()

        while active_workers:
            states = [w.state for w in active_workers]
            valid_masks = [w.env.valid_actions() for w in active_workers]
            epsilons = [epsilon_by_step(global_step, cfg)] * len(active_workers)
            guide_probs = [guide_prob_by_step(global_step, cfg)] * len(active_workers)
            
            actions = [Action.STOP] * len(active_workers)
            nn_indices = []
            
            for i, w in enumerate(active_workers):
                valid_mask = valid_masks[i]
                valid_actions = np.flatnonzero(valid_mask)
                if len(valid_actions) == 0:
                    valid_actions = np.asarray([int(Action.STOP)], dtype=np.int64)
                if random.random() < guide_probs[i]:
                    action = w.env.guided_action()
                    if int(action) < len(valid_mask) and bool(valid_mask[int(action)]):
                        actions[i] = action
                        continue
                if random.random() < epsilons[i]:
                    actions[i] = Action(int(random.choice(valid_actions.tolist())))
                    continue
                nn_indices.append(i)
                
            if nn_indices:
                batch_states = np.stack([states[i] for i in nn_indices])
                with torch.no_grad():
                    x = torch.from_numpy(batch_states).float().to(device)
                    q = policy(x)
                    for j, i in enumerate(nn_indices):
                        valid = torch.from_numpy(valid_masks[i]).bool().to(device)
                        q[j, ~valid] = -torch.inf
                        actions[i] = Action(int(q[j].argmax().item()))
                        
            step_results = []
            terminal_outcomes: list[CropOutcome | None] = [None] * len(active_workers)
            crop_indices: list[int] = []
            crop_paths = []
            crop_rois = []
            for i, w in enumerate(active_workers):
                result = w.env.step(actions[i])
                step_results.append(result)
                if (
                    result.done
                    and crop_evaluator is not None
                    and not crop_evaluator.should_skip_terminal(result.info)
                ):
                    crop_indices.append(i)
                    crop_paths.append(w.det.image_path)
                    crop_rois.append(w.env.roi.copy())

            if crop_indices and crop_evaluator is not None:
                crop_predictions = crop_evaluator.crop_predictions_many(crop_paths, crop_rois)
                for i, (raw_boxes, raw_scores, raw_classes) in zip(crop_indices, crop_predictions):
                    w = active_workers[i]
                    outcome = crop_evaluator.evaluate_from_predictions(
                        image_path=w.det.image_path,
                        det=w.det,
                        full_boxes=w.full_boxes,
                        full_scores=w.full_scores,
                        full_classes=w.full_classes,
                        slice_boxes_parts=w.slice_boxes_all,
                        slice_scores_parts=w.slice_scores_all,
                        slice_classes_parts=w.slice_classes_all,
                        accepted_new_count=w.accepted_new_count,
                        raw_boxes=raw_boxes,
                        raw_scores=raw_scores,
                        raw_classes=raw_classes,
                    )
                    step_results[i].info.update(outcome.info())
                    terminal_outcomes[i] = outcome

            for i in range(len(active_workers)):
                w = active_workers[i]
                action = actions[i]
                result = step_results[i]
                terminal_outcome = terminal_outcomes[i]
                if result.done and terminal_outcome is not None:
                    result.reward = _terminal_reward_with_crop_outcome(result.reward, terminal_outcome)
                    w.crop_new_detection_gain_total += int(terminal_outcome.new_detection_gain)
                    w.crop_new_detection_utility_total += float(terminal_outcome.new_detection_utility)
                    w.crop_tp_gain_total += int(terminal_outcome.tp_gain)
                    w.crop_fp_gain_total += int(terminal_outcome.fp_gain)
                    w.crop_outcome_reward_total += float(terminal_outcome.reward)
                next_valid_actions = w.env.valid_actions().copy()

                w.n_step_buffer.append((w.state, action, result.reward, result.state, result.done, next_valid_actions))
                if len(w.n_step_buffer) >= getattr(cfg, "n_step", 1):
                    ret = 0.0
                    for k, (_, _, r, _, _, _) in enumerate(w.n_step_buffer):
                        ret += r * (cfg.gamma ** k)
                    s0, a0, _, _, _, _ = w.n_step_buffer[0]
                    _, _, _, sn, dn, nv = w.n_step_buffer[-1]
                    replay.push(s0, a0, ret, sn, dn, nv)
                    w.n_step_buffer.pop(0)
                    
                w.state = result.state
                w.total_reward += result.reward
                w.total_steps += 1
                w.info = result.info
                global_step += 1
                
                optimize_every = max(int(cfg.optimize_every), 1)
                if len(replay) >= cfg.min_replay and global_step % optimize_every == 0:
                    loss = optimize(policy, target_net, optimizer, replay, cfg.batch_size, cfg.gamma ** getattr(cfg, "n_step", 1), device, double_dqn=cfg.double_dqn, reward_clip=cfg.reward_clip)
                    if loss is not None:
                        optimizer_steps += 1
                        w.losses.append(loss)
                        if cfg.use_soft_update:
                            soft_update(policy, target_net, cfg.tau)
                if not cfg.use_soft_update and global_step % cfg.target_update == 0:
                    target_net.load_state_dict(policy.state_dict())
                    
                if result.done:
                    while len(w.n_step_buffer) > 0:
                        ret = 0.0
                        for k, (_, _, r, _, d, _) in enumerate(w.n_step_buffer):
                            ret += r * (cfg.gamma ** k)
                            if d: break
                        s0, a0, _, _, _, _ = w.n_step_buffer[0]
                        _, _, _, sn, dn, nv = w.n_step_buffer[-1]
                        replay.push(s0, a0, ret, sn, dn, nv)
                        w.n_step_buffer.pop(0)
                        
                    new_hits = int((w.env.covered & ~w.previous_covered).sum())
                    repeat_overlap = _attempt_overlap(w.env.roi, w.attempted_rois)
                    w.attempted_rois.append(w.env.roi.copy())
                    w.attempt_idx += 1
                    reject_slice = (
                        w.info.get("stop_due_to_old_overlap", False)
                        or w.info.get("stop_due_to_attempted_overlap", False)
                        or w.info.get("stop_due_to_max_steps", False)
                        or w.info.get("stop_due_to_stalled_roi", False)
                    )
                    if crop_evaluator is not None:
                        reject_slice = reject_slice or terminal_outcome is None or not terminal_outcome.accepted
                    else:
                        reject_slice = reject_slice or new_hits < env_cfg.min_new_hits_to_accept
                    if reject_slice:
                        w.rejected_slices += 1
                    stop_episode = bool(reject_slice and repeat_overlap >= 0.95)
                    if not reject_slice:
                        w.previous_rois.append(w.env.roi.copy())
                        w.previous_covered = w.env.covered.copy()
                        if terminal_outcome is not None:
                            w.slice_boxes_all.append(terminal_outcome.boxes)
                            w.slice_scores_all.append(terminal_outcome.scores)
                            w.slice_classes_all.append(terminal_outcome.classes)
                            w.accepted_new_count = terminal_outcome.accepted_new_count_after
                        w.accepted_slices += 1
                        w.slice_idx += 1
                        stop_episode = (
                            w.slice_idx >= w.current_max_slices
                            or w.attempt_idx >= w.current_max_attempts
                            or (w.previous_covered.all() and len(w.previous_covered) > 0)
                        )
                    elif w.attempt_idx >= w.current_max_attempts:
                        stop_episode = True
                    if stop_episode:
                        if optimizer_steps > 0:
                            scheduler.step()
                            scheduler_steps += 1
                        mean_loss = float(np.mean(w.losses)) if w.losses else 0.0
                        completed_episode = episodes_completed + 1
                        row = {
                            "episode": completed_episode, "global_step": global_step,
                            "episodes_completed": completed_episode,
                            "reward": round(w.total_reward, 6), "loss": round(mean_loss, 6),
                            "epsilon": round(epsilon_by_step(global_step, cfg), 6), "steps": w.total_steps,
                            "slices": w.accepted_slices, "current_max_slices": w.current_max_slices,
                            "attempts": w.attempt_idx, "rejected_slices": w.rejected_slices,
                            "covered": int(w.previous_covered.sum()),
                            "hard_total": int(len(w.previous_covered)),
                            "crop_new_detection_gain": w.crop_new_detection_gain_total,
                            "crop_new_detection_utility": round(w.crop_new_detection_utility_total, 6),
                            "crop_tp_gain": w.crop_tp_gain_total,
                            "crop_fp_gain": w.crop_fp_gain_total,
                            "crop_outcome_reward": round(w.crop_outcome_reward_total, 6),
                            "val_recall": "", "val_slices": "",
                            "val_score": "", "val_mAP50": "", "val_small_recall": "", "val_fp_per_image": "",
                            "val_crops": "", "val_benchmark_score": ""
                        }
                        
                        selected_score = None
                        if completed_episode == 1 or completed_episode % max(int(cfg.eval_interval), 1) == 0:
                            if val_dataset is not None:
                                metrics = evaluate_policy(policy, val_dataset, env_cfg, state_cfg, cfg, device, target_classes=target_classes, class_mapping=class_mapping)
                                row["val_recall"] = round(metrics["val_recall"], 6)
                                row["val_slices"] = round(metrics["val_slices"], 6)
                                row["val_score"] = round(metrics["val_score"], 6)
                                selected_score = metrics["val_score"]
                            if benchmark_model is not None and infer_cfg is not None and bench_cfg is not None and label_root is not None:
                                bench_metrics = evaluate_rl_sahi_policy(
                                    model=benchmark_model,
                                    policy=policy,
                                    device_t=device,
                                    weights=eval_weights,
                                    images=benchmark_images,
                                    image_root=image_root,
                                    label_root=label_root,
                                    cache_root=cache_root,
                                    split=cfg.val_split,
                                    infer_cfg=infer_cfg,
                                    bench_cfg=bench_cfg,
                                    env_cfg=env_cfg,
                                    state_cfg=state_cfg,
                                    use_cache=eval_use_cache,
                                )
                                selected_score = benchmark_score(bench_metrics, cfg, env_cfg)
                                row["val_mAP50"] = round(bench_metrics["mAP50"], 6)
                                row["val_small_recall"] = round(bench_metrics["small_recall"], 6)
                                row["val_fp_per_image"] = round(bench_metrics["fp_per_image"], 6)
                                row["val_crops"] = round(bench_metrics["crops_per_image"], 6)
                                row["val_benchmark_score"] = round(selected_score, 6)
                                
                        if selected_score is not None and selected_score > best_score:
                            best_score = selected_score
                            save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
                        elif val_dataset is None and w.total_reward > best_reward:
                            best_reward = w.total_reward
                            save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
                            
                        writer.writerow(row)
                        f.flush()
                        
                        if completed_episode % cfg.log_interval == 0 or completed_episode == 1:
                            val_msg = ""
                            if row["val_score"] != "":
                                val_msg = f" val_recall={row['val_recall']} val_slices={row['val_slices']} val_score={row['val_score']}"
                            if row["val_benchmark_score"] != "":
                                val_msg += (
                                    f" val_mAP50={row['val_mAP50']} "
                                    f"small_recall={row['val_small_recall']} "
                                    f"benchmark_score={row['val_benchmark_score']}"
                                )
                            print(
                                f"[batched_train] ep={completed_episode}/{cfg.episodes} reward={w.total_reward:.3f} "
                                f"loss={mean_loss:.4f} eps={epsilon_by_step(global_step, cfg):.3f} "
                                f"slices={w.accepted_slices}/{w.current_max_slices} "
                                f"rejected={w.rejected_slices} covered={row['covered']}/{row['hard_total']}{val_msg}"
                            )
                        
                        w.done = True
                        episodes_completed = completed_episode
                        resume_interval = max(int(getattr(cfg, "resume_interval", cfg.log_interval)), 1)
                        if bool(getattr(cfg, "resume", True)) and (
                            episodes_completed == 1 or episodes_completed % resume_interval == 0
                        ):
                            _save_resume_checkpoint(
                                resume_path,
                                policy,
                                target_net,
                                optimizer,
                                scheduler,
                                replay,
                                state_dim,
                                cfg,
                                env_cfg,
                                state_cfg,
                                layout,
                                detection_metadata,
                                global_step,
                                episodes_started,
                                episodes_completed,
                                best_score,
                                best_reward,
                                optimizer_steps,
                                scheduler_steps,
                            )
                    else:
                        w.env = SliceEnv(
                            w.det,
                            w.hard,
                            env_cfg=env_cfg,
                            state_cfg=state_cfg,
                            previous_rois=_stack_rois(w.attempted_rois),
                            overlap_rois=_stack_rois(w.previous_rois),
                            previous_covered=w.previous_covered,
                            target_classes=target_classes,
                            class_mapping=class_mapping,
                        )
                        w.state = w.env.reset()
                        w.n_step_buffer.clear()
            
            next_active_workers = []
            for w in active_workers:
                if w.done:
                    if episodes_started < cfg.episodes:
                        episodes_started += 1
                        next_active_workers.append(reset_worker(episodes_started))
                else:
                    next_active_workers.append(w)
            active_workers = next_active_workers
            
    save_checkpoint(last_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
    if bool(getattr(cfg, "resume", True)):
        _save_resume_checkpoint(
            resume_path,
            policy,
            target_net,
            optimizer,
            scheduler,
            replay,
            state_dim,
            cfg,
            env_cfg,
            state_cfg,
            layout,
            detection_metadata,
            global_step,
            episodes_started,
            episodes_completed,
            best_score,
            best_reward,
            optimizer_steps,
            scheduler_steps,
        )
    return best_path
