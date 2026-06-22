from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from rl_sahi.common.actions import Action
from rl_sahi.common.boxes import area, as_boxes, intersection_matrix
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import configure_torch_runtime
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import BenchmarkConfig, evaluate_rl_sahi_policy
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.checkpoint import save_checkpoint
from rl_sahi.rl.crop_outcome import CropOutcome, CropOutcomeEvaluator
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import PrioritizedReplayBuffer, ReplayBuffer
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_layout import state_layout_from_detection


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 20000
    num_envs: int = 1
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 0.95
    lr: float = 1e-4
    min_replay: int = 512
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15000
    guide_prob_start: float = 0.25
    guide_prob_end: float = 0.05
    guide_decay_steps: int = 15000
    n_step: int = 3
    hidden_dim: int = 512
    use_spatial_cnn: bool = True
    double_dqn: bool = True
    dueling: bool = True
    reward_clip: float = 10.0
    optimize_every: int = 2
    preload_cache: bool = True
    seed: int = 42
    log_interval: int = 25
    val_split: str = "val"
    eval_interval: int = 500
    eval_episodes: int = 256
    eval_slice_cost_weight: float = 0.05
    eval_benchmark_images: int = 0
    eval_map_weight: float = 1.0
    eval_small_recall_weight: float = 1.5
    eval_fp_cost_weight: float = 0.01
    resume: bool = True
    resume_interval: int = 25
    use_soft_update: bool = True
    tau: float = 0.005
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000
    use_curriculum: bool = True
    curriculum_steps: int = 15000
    use_crop_outcome_reward: bool = True
    crop_detection_reward: float = 0.5
    crop_tp_reward: float = 3.0
    crop_fp_penalty: float = 1.5
    crop_empty_penalty: float = 1.2
    crop_no_gain_penalty: float = 1.2

def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(float(step) / max(cfg.epsilon_decay_steps, 1), 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

def guide_prob_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(float(step) / max(cfg.guide_decay_steps, 1), 1.0)
    return cfg.guide_prob_start + frac * (cfg.guide_prob_end - cfg.guide_prob_start)


def select_action(
    policy: QNetwork,
    state: np.ndarray,
    epsilon: float,
    guide_prob: float,
    env: SliceEnv,
    device: torch.device,
) -> Action:
    valid_actions = np.flatnonzero(env.valid_actions())
    if len(valid_actions) == 0:
        valid_actions = np.asarray([int(Action.STOP)], dtype=np.int64)
    if random.random() < guide_prob:
        action = env.guided_action()
        if int(action) in set(int(x) for x in valid_actions):
            return action
    if random.random() < epsilon:
        return Action(int(random.choice(valid_actions.tolist())))
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = policy(x)
        valid = torch.from_numpy(env.valid_actions()).bool().to(device)
        q[:, ~valid] = -torch.inf
        return Action(int(q.argmax(dim=1).item()))


def soft_update(policy: QNetwork, target: QNetwork, tau: float) -> None:
    for tp, pp in zip(target.parameters(), policy.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)


def _empty_rois() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def _stack_rois(rois: list[np.ndarray]) -> np.ndarray:
    return np.stack(rois).astype(np.float32) if rois else _empty_rois()


def _attempt_overlap(roi: np.ndarray, attempted_rois: list[np.ndarray]) -> float:
    if not attempted_rois:
        return 0.0
    previous = _stack_rois(attempted_rois)
    roi_arr = np.asarray(roi, dtype=np.float32).reshape(1, 4)
    inter = intersection_matrix(roi_arr, previous)[0]
    current_area = max(float(area(roi_arr)[0]), 1.0)
    return float(np.clip(inter.max() / current_area, 0.0, 1.0))


def _max_slice_attempts(env_cfg: EnvConfig, infer_cfg: InferenceConfig | None, max_slices: int | None = None) -> int:
    if infer_cfg is not None and infer_cfg.max_slice_attempts > 0:
        return int(infer_cfg.max_slice_attempts)
    slice_budget = env_cfg.max_slices if max_slices is None else int(max_slices)
    return max(int(slice_budget) * 2, int(slice_budget), 1)


def _terminal_reward_with_crop_outcome(base_reward: float, outcome: CropOutcome) -> float:
    if outcome.accepted:
        return float(base_reward + outcome.reward)
    return float(min(base_reward, 0.0) + outcome.reward)


def optimize(
    policy: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer | PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    double_dqn: bool = True,
    reward_clip: float = 0.0,
) -> float | None:
    if len(replay) < batch_size:
        return None

    def tensor_from_numpy(value, dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=dtype)
        if device.type == "cuda":
            return tensor.pin_memory().to(device, non_blocking=True)
        return tensor.to(device)

    use_per = isinstance(replay, PrioritizedReplayBuffer)
    next_valid_actions = None
    if use_per:
        sample = replay.sample(batch_size)
        if len(sample) == 8:
            states, actions, rewards, next_states, dones, next_valid_actions, indices, weights = sample
        else:
            states, actions, rewards, next_states, dones, indices, weights = sample
        weights_t = tensor_from_numpy(weights, torch.float32)
    else:
        sample = replay.sample(batch_size)
        if len(sample) == 6:
            states, actions, rewards, next_states, dones, next_valid_actions = sample
        else:
            states, actions, rewards, next_states, dones = sample

    states_t = tensor_from_numpy(states, torch.float32)
    actions_t = tensor_from_numpy(actions, torch.long)
    rewards_t = tensor_from_numpy(rewards, torch.float32)
    if reward_clip and reward_clip > 0.0:
        rewards_t = rewards_t.clamp(-float(reward_clip), float(reward_clip))
    next_states_t = tensor_from_numpy(next_states, torch.float32)
    dones_t = tensor_from_numpy(dones, torch.float32)

    q_values = policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_valid_t = None
        if next_valid_actions is not None:
            next_valid_t = tensor_from_numpy(next_valid_actions, torch.bool)
            invalid_rows = ~next_valid_t.any(dim=1)
            if invalid_rows.any():
                next_valid_t[invalid_rows, int(Action.STOP)] = True
        if double_dqn:
            next_policy_q = policy(next_states_t)
            if next_valid_t is not None:
                next_policy_q = next_policy_q.masked_fill(~next_valid_t, -torch.inf)
            next_actions = next_policy_q.argmax(dim=1)
            next_q = target(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            next_target_q = target(next_states_t)
            if next_valid_t is not None:
                next_target_q = next_target_q.masked_fill(~next_valid_t, -torch.inf)
            next_q = next_target_q.max(dim=1).values
        target_q = rewards_t + gamma * next_q * (1.0 - dones_t)

    td_errors = q_values - target_q

    if use_per:
        element_loss = F.smooth_l1_loss(q_values, target_q, reduction="none")
        loss = (weights_t * element_loss).mean()
        replay.update_priorities(indices, td_errors.detach().cpu().numpy())
    else:
        loss = F.smooth_l1_loss(q_values, target_q)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def _greedy_eval_episode(
    policy: QNetwork,
    det,
    hard,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    device: torch.device,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
) -> tuple[int, int, int]:
    previous_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    previous_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
    accepted_slices = 0
    max_attempts = _max_slice_attempts(env_cfg, None)
    for _attempt_idx in range(max_attempts):
        if accepted_slices >= env_cfg.max_slices:
            break
        env = SliceEnv(
            det,
            hard,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=_stack_rois(attempted_rois),
            overlap_rois=_stack_rois(previous_rois),
            previous_covered=previous_covered,
            target_classes=target_classes,
            class_mapping=class_mapping,
        )
        state = env.reset()
        info = {}
        for _ in range(env_cfg.max_steps + 1):
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                valid = torch.from_numpy(env.valid_actions()).bool().to(device)
                q[:, ~valid] = -torch.inf
                action = Action(int(q.argmax(dim=1).item()))
            result = env.step(action)
            state = result.state
            info = result.info
            if result.done:
                break

        new_hits = int((env.covered & ~previous_covered).sum())
        repeat_overlap = _attempt_overlap(env.roi, attempted_rois)
        attempted_rois.append(env.roi.copy())
        rejected = (
            info.get("stop_due_to_old_overlap", False)
            or info.get("stop_due_to_attempted_overlap", False)
            or info.get("stop_due_to_max_steps", False)
            or info.get("stop_due_to_stalled_roi", False)
            or new_hits < env_cfg.min_new_hits_to_accept
        )
        if rejected:
            if repeat_overlap >= 0.95:
                break
            continue
        previous_rois.append(env.roi.copy())
        previous_covered = env.covered.copy()
        accepted_slices += 1
        if previous_covered.all() and len(previous_covered) > 0:
            break
    return int(previous_covered.sum()), int(len(previous_covered)), accepted_slices


def evaluate_policy(
    policy: QNetwork,
    dataset: CachedEpisodeDataset,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    cfg: TrainConfig,
    device: torch.device,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
) -> dict[str, float]:
    episodes = min(max(int(cfg.eval_episodes), 1), len(dataset))
    covered_total = 0
    hard_total = 0
    slices_total = 0
    for _ in range(episodes):
        det, hard = dataset.random_episode()
        covered, total, slices = _greedy_eval_episode(
            policy,
            det,
            hard,
            env_cfg,
            state_cfg,
            device,
            target_classes=target_classes,
            class_mapping=class_mapping,
        )
        covered_total += covered
        hard_total += total
        slices_total += slices
    recall = float(covered_total / max(hard_total, 1))
    avg_slices = float(slices_total / max(episodes, 1))
    score = recall - float(cfg.eval_slice_cost_weight) * avg_slices / max(float(env_cfg.max_slices), 1.0)
    return {
        "val_recall": recall,
        "val_slices": avg_slices,
        "val_score": score,
        "val_covered": float(covered_total),
        "val_hard_total": float(hard_total),
    }


def benchmark_score(metrics: dict[str, float], cfg: TrainConfig, env_cfg: EnvConfig) -> float:
    crop_cost = float(cfg.eval_slice_cost_weight) * metrics["crops_per_image"] / max(float(env_cfg.max_slices), 1.0)
    fp_cost = float(cfg.eval_fp_cost_weight) * metrics["fp_per_image"]
    return (
        float(cfg.eval_map_weight) * metrics["mAP50"]
        + float(cfg.eval_small_recall_weight) * metrics["small_recall"]
        - crop_cost
        - fp_cost
    )


def make_crop_outcome_evaluator(
    model,
    image_root: Path,
    label_root: Path | None,
    cache_root: Path,
    split: str,
    cfg: TrainConfig,
    infer_cfg: InferenceConfig | None,
    bench_cfg: BenchmarkConfig | None,
    eval_weights: Path | None,
    eval_use_cache: bool,
) -> CropOutcomeEvaluator | None:
    if not cfg.use_crop_outcome_reward:
        return None
    if model is None or infer_cfg is None:
        raise RuntimeError(
            "Crop outcome reward requires eval_weights and inference config. "
            "Disable train.use_crop_outcome_reward to use hard-region-only training."
        )
    return CropOutcomeEvaluator(
        model=model,
        image_root=image_root,
        label_root=label_root,
        cache_root=cache_root,
        split=split,
        infer_cfg=infer_cfg,
        weights=eval_weights,
        iou_threshold=float(bench_cfg.iou_threshold) if bench_cfg is not None else 0.5,
        use_cache=eval_use_cache,
        detection_reward=cfg.crop_detection_reward,
        tp_reward=cfg.crop_tp_reward,
        fp_penalty=cfg.crop_fp_penalty,
        empty_penalty=cfg.crop_empty_penalty,
        no_gain_penalty=cfg.crop_no_gain_penalty,
    )


def apply_terminal_crop_outcome(
    evaluator: CropOutcomeEvaluator | None,
    image_path: Path | str,
    det,
    env: SliceEnv,
    info: dict,
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_all: list[np.ndarray],
    slice_scores_all: list[np.ndarray],
    slice_classes_all: list[np.ndarray],
    accepted_new_count: int,
) -> CropOutcome | None:
    if evaluator is None or evaluator.should_skip_terminal(info):
        return None
    outcome = evaluator.evaluate(
        image_path=image_path,
        det=det,
        roi=env.roi,
        full_boxes=full_boxes,
        full_scores=full_scores,
        full_classes=full_classes,
        slice_boxes_parts=slice_boxes_all,
        slice_scores_parts=slice_scores_all,
        slice_classes_parts=slice_classes_all,
        accepted_new_count=accepted_new_count,
    )
    info.update(outcome.info())
    return outcome


def train_dqn(
    image_root: Path,
    cache_root: Path,
    split: str,
    out_dir: Path,
    cfg: TrainConfig,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    limit: int | None = None,
    device_name: str | None = None,
    detection_metadata: dict[str, Any] | None = None,
    target_classes: tuple[int, ...] = (),
    class_mapping: ClassMapping | None = None,
    label_root: Path | None = None,
    eval_weights: Path | None = None,
    infer_cfg: InferenceConfig | None = None,
    bench_cfg: BenchmarkConfig | None = None,
    eval_use_cache: bool = True,
) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = CachedEpisodeDataset(
        image_root=image_root,
        cache_root=cache_root,
        split=split,
        limit=limit,
        preload=cfg.preload_cache,
        detection_metadata=detection_metadata,
    )
    val_dataset = None
    try:
        val_dataset = CachedEpisodeDataset(
            image_root=image_root,
            cache_root=cache_root,
            split=cfg.val_split,
            limit=limit,
            preload=cfg.preload_cache,
            detection_metadata=detection_metadata,
        )
    except FileNotFoundError as exc:
        print(f"[train] validation disabled: {exc}")

    inference_model = None
    benchmark_model = None
    benchmark_images: list[Path] = []
    if cfg.eval_benchmark_images > 0:
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
    probe_env = SliceEnv(
        probe_det,
        None,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        target_classes=target_classes,
        class_mapping=class_mapping,
    )
    state_dim = int(probe_env.reset().shape[0])
    layout = state_layout_from_detection(probe_det, state_cfg)
    if layout.state_dim != state_dim:
        raise ValueError(f"State layout mismatch: layout has {layout.state_dim}, env produced {state_dim}")

    device = configure_torch_runtime(device_name)
    policy = QNetwork(
        state_dim,
        hidden_dim=cfg.hidden_dim,
        layout=layout,
        use_spatial_cnn=cfg.use_spatial_cnn,
        dueling=cfg.dueling,
    ).to(device)
    target_net = QNetwork(
        state_dim,
        hidden_dim=cfg.hidden_dim,
        layout=layout,
        use_spatial_cnn=cfg.use_spatial_cnn,
        dueling=cfg.dueling,
    ).to(device)
    target_net.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.episodes, eta_min=1e-6)

    if cfg.use_per:
        replay: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(
            capacity=cfg.replay_size,
            alpha=cfg.per_alpha,
            beta_start=cfg.per_beta_start,
            beta_frames=cfg.per_beta_frames,
        )
    else:
        replay = ReplayBuffer(cfg.replay_size)

    update_mode = "soft" if cfg.use_soft_update else "hard"
    print(f"[train] target update: {update_mode} (tau={cfg.tau})" if cfg.use_soft_update
          else f"[train] target update: {update_mode} (every {cfg.target_update} steps)")
    print(f"[train] dueling={cfg.dueling}, double_dqn={cfg.double_dqn}")
    print(f"[train] simplified_reward={env_cfg.use_simplified_reward}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_score = -float("inf")
    best_reward = -float("inf")
    global_step = 0
    optimizer_steps = 0
    with log_path.open("w", newline="", encoding="utf-8") as f:
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
        writer.writeheader()
        for episode in range(1, cfg.episodes + 1):
            det, hard = dataset.random_episode()
            previous_rois: list[np.ndarray] = []
            attempted_rois: list[np.ndarray] = []
            previous_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
            slice_boxes_all: list[np.ndarray] = []
            slice_scores_all: list[np.ndarray] = []
            slice_classes_all: list[np.ndarray] = []
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
            total_reward = 0.0
            total_steps = 0
            accepted_slices = 0
            attempt_count = 0
            rejected_slices = 0
            crop_new_detection_gain_total = 0
            crop_new_detection_utility_total = 0.0
            crop_tp_gain_total = 0
            crop_fp_gain_total = 0
            crop_outcome_reward_total = 0.0
            losses: list[float] = []
            info = {"covered": 0, "hard_total": len(previous_covered)}

            current_max_slices = env_cfg.max_slices
            if cfg.use_curriculum:
                curriculum_frac = min(float(global_step) / max(cfg.curriculum_steps, 1), 1.0)
                current_max_slices = max(1, int(env_cfg.max_slices * curriculum_frac))

            max_attempts = _max_slice_attempts(env_cfg, infer_cfg, current_max_slices)
            for _attempt_idx in range(max_attempts):
                if accepted_slices >= current_max_slices:
                    break
                env = SliceEnv(
                    det,
                    hard,
                    env_cfg=env_cfg,
                    state_cfg=state_cfg,
                    previous_rois=_stack_rois(attempted_rois),
                    overlap_rois=_stack_rois(previous_rois),
                    previous_covered=previous_covered,
                    target_classes=target_classes,
                    class_mapping=class_mapping,
                )
                state = env.reset()

                n_step_buffer: list[tuple[np.ndarray, Action, float, np.ndarray, bool, np.ndarray]] = []
                terminal_outcome: CropOutcome | None = None
                for _ in range(env_cfg.max_steps + 1):
                    epsilon = epsilon_by_step(global_step, cfg)
                    guide_prob = guide_prob_by_step(global_step, cfg)
                    action = select_action(policy, state, epsilon, guide_prob, env, device)
                    result = env.step(action)
                    if result.done:
                        terminal_outcome = apply_terminal_crop_outcome(
                            crop_evaluator,
                            det.image_path,
                            det,
                            env,
                            result.info,
                            full_boxes,
                            full_scores,
                            full_classes,
                            slice_boxes_all,
                            slice_scores_all,
                            slice_classes_all,
                            accepted_new_count,
                        )
                        if terminal_outcome is not None:
                            result.reward = _terminal_reward_with_crop_outcome(result.reward, terminal_outcome)
                            crop_new_detection_gain_total += int(terminal_outcome.new_detection_gain)
                            crop_new_detection_utility_total += float(terminal_outcome.new_detection_utility)
                            crop_tp_gain_total += int(terminal_outcome.tp_gain)
                            crop_fp_gain_total += int(terminal_outcome.fp_gain)
                            crop_outcome_reward_total += float(terminal_outcome.reward)
                    next_valid_actions = env.valid_actions().copy()

                    n_step_buffer.append((state, action, result.reward, result.state, result.done, next_valid_actions))
                    if len(n_step_buffer) >= cfg.n_step:
                        ret = 0.0
                        for i, (_, _, r, _, _, _) in enumerate(n_step_buffer):
                            ret += r * (cfg.gamma ** i)
                        s0, a0, _, _, _, _ = n_step_buffer[0]
                        _, _, _, sn, dn, nv = n_step_buffer[-1]
                        replay.push(s0, a0, ret, sn, dn, nv)
                        n_step_buffer.pop(0)
                        
                    state = result.state
                    total_reward += result.reward
                    total_steps += 1
                    info = result.info
                    global_step += 1

                    optimize_every = max(int(cfg.optimize_every), 1)
                    if len(replay) >= cfg.min_replay and global_step % optimize_every == 0:
                        loss = optimize(
                            policy,
                            target_net,
                            optimizer,
                            replay,
                            cfg.batch_size,
                            cfg.gamma ** cfg.n_step,
                            device,
                            double_dqn=cfg.double_dqn,
                            reward_clip=cfg.reward_clip,
                        )
                        if loss is not None:
                            optimizer_steps += 1
                            losses.append(loss)
                            if cfg.use_soft_update:
                                soft_update(policy, target_net, cfg.tau)
                    if not cfg.use_soft_update and global_step % cfg.target_update == 0:
                        target_net.load_state_dict(policy.state_dict())

                    if result.done:
                        while len(n_step_buffer) > 0:
                            ret = 0.0
                            for i, (_, _, r, _, d, _) in enumerate(n_step_buffer):
                                ret += r * (cfg.gamma ** i)
                                if d:
                                    break
                            s0, a0, _, _, _, _ = n_step_buffer[0]
                            _, _, _, sn, dn, nv = n_step_buffer[-1]
                            replay.push(s0, a0, ret, sn, dn, nv)
                            n_step_buffer.pop(0)
                        break

                new_hits = int((env.covered & ~previous_covered).sum())
                repeat_overlap = _attempt_overlap(env.roi, attempted_rois)
                attempted_rois.append(env.roi.copy())
                attempt_count += 1
                rejected = (
                    info.get("stop_due_to_old_overlap", False)
                    or info.get("stop_due_to_attempted_overlap", False)
                    or info.get("stop_due_to_max_steps", False)
                    or info.get("stop_due_to_stalled_roi", False)
                )
                if crop_evaluator is not None:
                    rejected = rejected or terminal_outcome is None or not terminal_outcome.accepted
                else:
                    rejected = rejected or new_hits < env_cfg.min_new_hits_to_accept
                if rejected:
                    rejected_slices += 1
                    if repeat_overlap >= 0.95:
                        break
                    continue
                previous_rois.append(env.roi.copy())
                previous_covered = env.covered.copy()
                if terminal_outcome is not None:
                    slice_boxes_all.append(terminal_outcome.boxes)
                    slice_scores_all.append(terminal_outcome.scores)
                    slice_classes_all.append(terminal_outcome.classes)
                    accepted_new_count = terminal_outcome.accepted_new_count_after
                accepted_slices += 1
                if previous_covered.all() and len(previous_covered) > 0:
                    break

            if optimizer_steps > 0:
                scheduler.step()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            row = {
                "episode": episode,
                "global_step": global_step,
                "episodes_completed": episode,
                "reward": round(total_reward, 6),
                "loss": round(mean_loss, 6),
                "epsilon": round(epsilon_by_step(global_step, cfg), 6),
                "steps": total_steps,
                "slices": accepted_slices,
                "current_max_slices": current_max_slices,
                "attempts": attempt_count,
                "rejected_slices": rejected_slices,
                "covered": int(previous_covered.sum()),
                "hard_total": int(len(previous_covered)),
                "crop_new_detection_gain": crop_new_detection_gain_total,
                "crop_new_detection_utility": round(crop_new_detection_utility_total, 6),
                "crop_tp_gain": crop_tp_gain_total,
                "crop_fp_gain": crop_fp_gain_total,
                "crop_outcome_reward": round(crop_outcome_reward_total, 6),
                "val_recall": "",
                "val_slices": "",
                "val_score": "",
                "val_mAP50": "",
                "val_small_recall": "",
                "val_fp_per_image": "",
                "val_crops": "",
                "val_benchmark_score": "",
            }

            selected_score: float | None = None
            if episode == 1 or episode % max(int(cfg.eval_interval), 1) == 0:
                if val_dataset is not None:
                    metrics = evaluate_policy(
                        policy,
                        val_dataset,
                        env_cfg,
                        state_cfg,
                        cfg,
                        device,
                        target_classes=target_classes,
                        class_mapping=class_mapping,
                    )
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
                    save_checkpoint(
                        best_path,
                        policy,
                        state_dim,
                        cfg,
                        env_cfg,
                        state_cfg,
                        layout,
                        detection_metadata=detection_metadata,
                    )
            elif val_dataset is None and benchmark_model is None and total_reward > best_reward:
                best_reward = total_reward
                save_checkpoint(
                    best_path,
                    policy,
                    state_dim,
                    cfg,
                    env_cfg,
                    state_cfg,
                    layout,
                    detection_metadata=detection_metadata,
                )
            writer.writerow(row)
            f.flush()
            if episode % cfg.log_interval == 0 or episode == 1:
                val_msg = ""
                if row["val_score"] != "":
                    val_msg = (
                        f" val_recall={row['val_recall']} "
                        f"val_slices={row['val_slices']} val_score={row['val_score']}"
                    )
                if row["val_benchmark_score"] != "":
                    val_msg += (
                        f" val_mAP50={row['val_mAP50']} "
                        f"small_recall={row['val_small_recall']} "
                        f"benchmark_score={row['val_benchmark_score']}"
                    )
                print(
                    f"[train] ep={episode}/{cfg.episodes} reward={total_reward:.3f} "
                    f"loss={mean_loss:.4f} eps={epsilon_by_step(global_step, cfg):.3f} "
                    f"slices={accepted_slices}/{current_max_slices} rejected={rejected_slices} "
                    f"covered={row['covered']}/{row['hard_total']}{val_msg}"
                )

    save_checkpoint(
        last_path,
        policy,
        state_dim,
        cfg,
        env_cfg,
        state_cfg,
        layout,
        detection_metadata=detection_metadata,
    )
    return best_path
