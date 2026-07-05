# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from dataclasses import dataclass, field  # Hỗ trợ định nghĩa lớp dữ liệu ngắn gọn và thiết lập giá trị mặc định (field)
from typing import Any                   # Định nghĩa kiểu dữ liệu bất kỳ cho type hinting

import numpy as np                       # Thư viện tính toán mảng NumPy


def _int_mapping(raw: Any) -> dict[int, int]:
    """
    Chuyển đổi dữ liệu ánh xạ thô (thường đọc từ cấu hình YAML/INI) thành dictionary có kiểu dữ liệu int:int.
    Ví dụ: {"0": 1, "2": 2} -> {0: 1, 2: 2}
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Class mapping must be a YAML mapping of source_id: target_id")
    # Ép cả khóa và giá trị sang kiểu số nguyên int
    return {int(src): int(dst) for src, dst in raw.items()}


def _apply_mapping(classes: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """
    Áp dụng ánh xạ class ID lên mảng chứa nhãn các lớp đối tượng.
    Ví dụ: Nếu mapping chứa {0: 10}, tất cả phần tử có giá trị 0 trong classes sẽ được đổi thành 10.
    """
    classes_i = np.asarray(classes, dtype=np.int64).reshape(-1)
    if not mapping or len(classes_i) == 0:
        return classes_i.astype(np.float32)
    mapped = classes_i.copy()
    # Lặp qua từng quy tắc ánh xạ và thay thế các phần tử tương ứng trong mảng
    for src, dst in mapping.items():
        mapped[classes_i == int(src)] = int(dst)
    return mapped.astype(np.float32)


@dataclass(slots=True)
class ClassMapping:
    """
    Lớp quản lý việc chuyển đổi ID của các lớp đối tượng (classes).
    Hữu ích khi các nhãn đầu ra của mô hình (model classes), nhãn dữ liệu gốc (labels),
    và nhãn đánh giá (eval) không trùng khớp ID với nhau.
    """
    model_to_label: dict[int, int] = field(default_factory=dict)  # Ánh xạ từ class của mô hình sang nhãn thực tế
    label_to_eval: dict[int, int] = field(default_factory=dict)   # Ánh xạ từ nhãn thực tế sang nhãn dùng để tính toán điểm (mAP)

    @classmethod
    def from_config(cls, raw: dict[str, Any] | None) -> "ClassMapping":
        """
        Khởi tạo ClassMapping từ dữ liệu cấu hình thô.
        """
        raw = raw or {}
        return cls(
            model_to_label=_int_mapping(raw.get("model_to_label")),
            label_to_eval=_int_mapping(raw.get("label_to_eval")),
        )

    def map_model_classes(self, classes: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi nhãn lớp từ đầu ra mô hình sang nhãn đánh giá (qua 2 bước ánh xạ).
        """
        label_classes = _apply_mapping(classes, self.model_to_label)
        return _apply_mapping(label_classes, self.label_to_eval)

    def map_label_classes(self, classes: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi trực tiếp nhãn gốc (label) sang nhãn đánh giá.
        """
        return _apply_mapping(classes, self.label_to_eval)


