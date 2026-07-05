from __future__ import annotations

import random
from collections import deque

import numpy as np


# giải thích: Lớp ReplayBuffer thực hiện bộ nhớ trải nghiệm thông thường (đều nhau)
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    # giải thích: Lưu một bộ dữ liệu chuyển đổi trạng thái (transition) vào bộ nhớ đệm
    def push(self, state, action, reward, next_state, done, next_valid_actions=None) -> None:
        next_valid = None if next_valid_actions is None else np.asarray(next_valid_actions, dtype=bool)
        self.buffer.append((state, int(action), float(reward), next_state, bool(done), next_valid))

    # giải thích: Lấy ngẫu nhiên đều một lô (batch) mẫu chuyển đổi để huấn luyện
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid = zip(*batch)
        result = (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )
        if any(mask is not None for mask in next_valid):
            default_mask = next(mask for mask in next_valid if mask is not None)
            masks = [
                np.asarray(mask, dtype=bool)
                if mask is not None
                else np.ones_like(np.asarray(default_mask, dtype=bool))
                for mask in next_valid
            ]
            return (*result, np.stack(masks))
        return result

    def __len__(self) -> int:
        return len(self.buffer)


# giải thích: Lớp PrioritizedReplayBuffer thực hiện bộ nhớ trải nghiệm có ưu tiên (Prioritized Experience Replay - PER)
class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self._frame = 0

        self._buffer: list = []
        self._priorities = np.zeros((capacity,), dtype=np.float64)
        self._pos = 0
        self._max_priority = 1.0

    # giải thích: Hàm tính toán hệ số beta điều khiển tỷ lệ khử lệch Importance Sampling theo thời gian huấn luyện
    @property
    def beta(self) -> float:
        frac = min(float(self._frame) / max(self.beta_frames, 1), 1.0)
        return self.beta_start + frac * (1.0 - self.beta_start)

    # giải thích: Đưa dữ liệu mới vào bộ đệm vòng tròn và gán độ ưu tiên lớn nhất hiện tại
    def push(self, state, action, reward, next_state, done, next_valid_actions=None) -> None:
        next_valid = None if next_valid_actions is None else np.asarray(next_valid_actions, dtype=bool)
        data = (state, int(action), float(reward), next_state, bool(done), next_valid)
        if len(self._buffer) < self.capacity:
            self._buffer.append(data)
        else:
            self._buffer[self._pos] = data
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    # giải thích: Lấy mẫu có trọng số dựa trên độ ưu tiên và tính trọng số Importance Sampling
    def sample(self, batch_size: int):
        self._frame += 1
        n = len(self._buffer)
        if n == 0 or batch_size <= 0:
            raise ValueError("Cannot sample from empty buffer")

        # giải thích: Tính toán xác suất chọn mẫu probs dựa trên độ ưu tiên lũy thừa alpha
        priorities = np.power(self._priorities[:n], self.alpha)
        priority_sum = priorities.sum()
        if priority_sum <= 0.0:
            probs = np.full((n,), 1.0 / n, dtype=np.float64)
        else:
            probs = priorities / priority_sum

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        # giải thích: Tính Importance Sampling weights để bù trừ cho việc lấy mẫu lệch
        beta = self.beta
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        states, actions, rewards, next_states, dones, next_valid = [], [], [], [], [], []
        for idx in indices:
            s, a, r, ns, d, nv = self._buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            next_valid.append(nv)

        result = (
            np.stack(states),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )
        if any(mask is not None for mask in next_valid):
            default_mask = next(mask for mask in next_valid if mask is not None)
            masks = [
                np.asarray(mask, dtype=bool)
                if mask is not None
                else np.ones_like(np.asarray(default_mask, dtype=bool))
                for mask in next_valid
            ]
            return (*result, np.stack(masks), indices, weights)
        return (*result, indices, weights)

    # giải thích: Cập nhật độ ưu tiên của các mẫu chuyển đổi bằng lỗi TD (Temporal Difference) mới
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64).reshape(-1)) + 1e-6
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self._priorities[indices] = priorities
        if priorities.size:
            self._max_priority = max(self._max_priority, float(priorities.max()))

    def __len__(self) -> int:
        return len(self._buffer)
