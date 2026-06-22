from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _int_mapping(raw: Any) -> dict[int, int]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Class mapping must be a YAML mapping of source_id: target_id")
    return {int(src): int(dst) for src, dst in raw.items()}


def _apply_mapping(classes: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    classes_i = np.asarray(classes, dtype=np.int64).reshape(-1)
    if not mapping or len(classes_i) == 0:
        return classes_i.astype(np.float32)
    mapped = classes_i.copy()
    for src, dst in mapping.items():
        mapped[classes_i == int(src)] = int(dst)
    return mapped.astype(np.float32)


@dataclass(slots=True)
class ClassMapping:
    model_to_label: dict[int, int] = field(default_factory=dict)
    label_to_eval: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> "ClassMapping":
        raw = raw or {}
        return cls(
            model_to_label=_int_mapping(raw.get("model_to_label")),
            label_to_eval=_int_mapping(raw.get("label_to_eval")),
        )

    def map_model_classes(self, classes: np.ndarray) -> np.ndarray:
        label_classes = _apply_mapping(classes, self.model_to_label)
        return _apply_mapping(label_classes, self.label_to_eval)

    def map_label_classes(self, classes: np.ndarray) -> np.ndarray:
        return _apply_mapping(classes, self.label_to_eval)

