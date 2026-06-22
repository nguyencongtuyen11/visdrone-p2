from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.rl.replay import ReplayBuffer
from rl_sahi.rl.trainer import optimize


class StaticQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current = torch.nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        for row in x:
            if float(row[0].item()) < 0.5:
                rows.append(self.current)
            else:
                rows.append(torch.tensor([0.0, 10.0, 2.0], device=x.device) + self.current * 0.0)
        return torch.stack(rows, dim=0)


class DqnTargetMaskTest(unittest.TestCase):
    def test_optimize_masks_invalid_next_actions(self) -> None:
        replay = ReplayBuffer(4)
        replay.push(
            np.array([0.0], dtype=np.float32),
            0,
            0.0,
            np.array([1.0], dtype=np.float32),
            False,
            np.array([True, False, True], dtype=bool),
        )
        policy = StaticQ()
        target = StaticQ()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

        loss = optimize(
            policy,
            target,
            optimizer,
            replay,
            batch_size=1,
            gamma=1.0,
            device=torch.device("cpu"),
            double_dqn=True,
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss), 1.5, places=5)


if __name__ == "__main__":
    unittest.main()
