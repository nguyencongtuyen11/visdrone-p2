from __future__ import annotations

import numpy as np
import torch

from rl_sahi.common.actions import ACTION_NAMES, Action
from rl_sahi.rl.slice_env import SliceEnv


def rollout_one_slice(policy, env: SliceEnv, device: torch.device) -> tuple[np.ndarray, list[str], dict]:
    state = env.reset()
    expected_dim = int(getattr(policy, "input_dim", state.shape[0]))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"Checkpoint expects state_dim={expected_dim}, but current detection state has {state.shape[0]}. "
            "Regenerate detection caches and retrain the DQN with the current state configuration."
        )
    actions: list[str] = []
    info: dict = {}
    for _ in range(env.env_cfg.max_steps + 1):
        with torch.no_grad():
            q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
            valid = torch.from_numpy(env.valid_actions()).bool().to(device)
            q[:, ~valid] = -torch.inf
            action = Action(int(q.argmax(dim=1).item()))
        actions.append(ACTION_NAMES[action])
        result = env.step(action)
        state = result.state
        info = result.info
        if result.done:
            break
    return env.roi.copy(), actions, info
