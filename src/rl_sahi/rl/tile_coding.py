"""Tile coding (Albus CMAC / Sutton-Barto) + linear Q-agent cho MDP di chuyển.

Đây là NHÁNH SO SÁNH: giữ nguyên bài toán gốc (SliceEnv: action dịch/zoom/stop,
reward hard-region), nhưng thay Deep Q-Network bằng tile coding + hàm Q TUYẾN TÍNH.
Mục đích: đối chứng "deep function approximation" vs "classical linear FA" trên cùng MDP.
"""
from __future__ import annotations

import pickle
from math import floor
from pathlib import Path

import numpy as np


# ---- tiles3 (port chuẩn của Sutton) -----------------------------------------
class IHT:
    """Index-Hash-Table: băm tọa độ ô về chỉ số trong bảng kích thước cố định."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.overfull_count = 0
        self.dictionary: dict = {}

    def get_index(self, obj, read_only: bool = False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        if read_only:
            return None
        if len(d) >= self.size:
            self.overfull_count += 1
            return hash(obj) % self.size
        d[obj] = len(d)
        return d[obj]


def _hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    return coordinates


def tiles(iht: IHT, num_tilings: int, floats, ints=(), read_only=False) -> list[int]:
    """Trả về danh sách chỉ số ô kích hoạt (1 ô / tiling). `floats` đã scale theo #tiles."""
    q = [floor(f * num_tilings) for f in floats]
    result = []
    for tiling in range(num_tilings):
        tiling_x2 = tiling * 2
        coords = [tiling]
        b = tiling
        for qi in q:
            coords.append((qi + b) // num_tilings)
            b += tiling_x2
        coords.extend(ints)
        result.append(_hash_coords(coords, iht, read_only))
    return result


# ---- Linear Q-agent với tile coding -----------------------------------------
class TileQAgent:
    """Q(s,a) = tổng trọng số của các ô kích hoạt cho (features, action).

    - state được đưa vào dưới dạng vector features chuẩn hóa [0,1]^d.
    - action đưa vào dạng int (tile riêng cho mỗi action).
    - cập nhật: semi-gradient Q-learning tuyến tính (không cần replay/target net).
    """

    def __init__(
        self,
        num_actions: int,
        num_tilings: int = 8,
        tiles_per_dim: int = 8,
        iht_size: int = 2 ** 20,
        alpha: float = 0.1,
        gamma: float = 0.95,
    ) -> None:
        self.num_actions = int(num_actions)
        self.num_tilings = int(num_tilings)
        self.tiles_per_dim = float(tiles_per_dim)
        self.iht = IHT(iht_size)
        self.weights = np.zeros(iht_size, dtype=np.float64)
        self.alpha = float(alpha) / float(num_tilings)  # chia cho #tilings (chuẩn)
        self.gamma = float(gamma)

    def _active(self, features: np.ndarray, action: int, read_only: bool = False) -> list[int]:
        scaled = [float(x) * self.tiles_per_dim for x in np.asarray(features, dtype=np.float64).reshape(-1)]
        return tiles(self.iht, self.num_tilings, scaled, ints=(int(action),), read_only=read_only)

    def q(self, features: np.ndarray, action: int, read_only: bool = False) -> float:
        active = [i for i in self._active(features, action, read_only) if i is not None]
        if not active:
            return 0.0
        return float(self.weights[np.asarray(active, dtype=np.int64)].sum())

    def q_all(self, features: np.ndarray, read_only: bool = False) -> np.ndarray:
        return np.array([self.q(features, a, read_only) for a in range(self.num_actions)], dtype=np.float64)

    def act(self, features: np.ndarray, valid_mask: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        valid = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
        if len(valid) == 0:
            return 0
        if rng.random() < epsilon:
            return int(rng.choice(valid))
        qs = self.q_all(features, read_only=True)
        qs_masked = np.full(self.num_actions, -np.inf)
        qs_masked[valid] = qs[valid]
        return int(np.argmax(qs_masked))

    def update(self, features, action, reward, next_features, done, next_valid_mask) -> float:
        active = self._active(features, action)
        q_sa = float(self.weights[active].sum())
        target = float(reward)
        if not done:
            nq = self.q_all(next_features, read_only=True)
            valid = np.flatnonzero(np.asarray(next_valid_mask, dtype=bool))
            best = np.max(nq[valid]) if len(valid) else 0.0
            target += self.gamma * float(best)
        delta = target - q_sa
        self.weights[active] += self.alpha * delta
        return abs(delta)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "num_actions": self.num_actions,
                    "num_tilings": self.num_tilings,
                    "tiles_per_dim": self.tiles_per_dim,
                    "gamma": self.gamma,
                    "iht": self.iht,
                    "weights": self.weights,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TileQAgent":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        agent = cls(
            num_actions=data["num_actions"],
            num_tilings=data["num_tilings"],
            tiles_per_dim=int(data["tiles_per_dim"]),
            iht_size=data["iht"].size,
            gamma=data["gamma"],
        )
        agent.iht = data["iht"]
        agent.weights = data["weights"]
        return agent
