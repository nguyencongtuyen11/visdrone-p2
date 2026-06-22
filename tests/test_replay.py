from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.rl.replay import PrioritizedReplayBuffer


class PrioritizedReplayBufferTest(unittest.TestCase):
    def test_new_items_receive_raw_max_priority(self) -> None:
        replay = PrioritizedReplayBuffer(capacity=4, alpha=0.5)
        state = np.array([0.0], dtype=np.float32)
        replay.push(state, 0, 0.0, state, False)
        replay.update_priorities(np.array([0]), np.array([100.0], dtype=np.float32))
        replay.push(state, 0, 0.0, state, False)

        self.assertAlmostEqual(float(replay._priorities[1]), float(replay._max_priority), places=6)


if __name__ == "__main__":
    unittest.main()
