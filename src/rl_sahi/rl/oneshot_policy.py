"""Policy net cho ONE-SHOT RL (nhanh oneshot-rl): MLP nho, state/vung -> Q tren 4 action."""
from __future__ import annotations

import torch
import torch.nn as nn

from rl_sahi.rl.oneshot import NUM_ACTIONS, region_state_dim


class OneShotPolicy(nn.Module):
    def __init__(self, state_dim: int | None = None, hidden: int = 64, num_actions: int = NUM_ACTIONS) -> None:
        super().__init__()
        state_dim = int(state_dim or region_state_dim())
        self.state_dim = state_dim
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_policy(policy: OneShotPolicy, path, meta: dict | None = None) -> None:
    torch.save({
        "model": policy.state_dict(),
        "state_dim": policy.state_dim,
        "num_actions": policy.num_actions,
        "meta": meta or {},
        "kind": "oneshot",
    }, str(path))


def load_oneshot_policy(path, device) -> tuple[OneShotPolicy, dict]:
    ck = torch.load(str(path), map_location=device, weights_only=False)
    p = OneShotPolicy(state_dim=int(ck["state_dim"]), num_actions=int(ck["num_actions"]))
    p.load_state_dict(ck["model"])
    p.to(device).eval()
    return p, ck
